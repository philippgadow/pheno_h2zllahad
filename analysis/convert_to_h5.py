#!/usr/bin/env python3
"""
Delphes ROOT to ML pipeline for jet tagging and mass regression.

Outputs an HDF5 with
- jet_features: (N, 6)
- constituent_features: (N, max_constituents, 6)
- constituent_mask: (N, max_constituents)
- labels: (N,) integer truth flavor tag
- targets: (N,) float truth mass, NaN when unmatched

Usage:
    python delphes_pipeline.py --input /path/to/root_or_dir --output /path/to/outdir
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import uproot
import h5py
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm.rich import tqdm

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("delphes_pipeline")

@dataclass
class JetConstituent:
    pt: float
    eta: float
    phi: float
    mass: float
    pid: int
    charge: int

@dataclass
class JetInfo:
    pt: float
    eta: float
    phi: float
    mass: float
    btag: int
    tau_tag: int
    constituents: List[JetConstituent]
    truth_flavor: int
    truth_mass: float  # NaN if unmatched

class DelphesProcessor:
    def __init__(self, max_constituents: int = 100):
        self.max_constituents = max_constituents
        self.jet_features = ["PT", "Eta", "Phi", "Mass", "BTag", "TauTag"]
        self.constituent_features = ["PT", "Eta", "Phi", "Mass", "PID", "Charge"]

    def read_delphes_file(self, filepath: str) -> Tuple[List[JetInfo], Dict]:
        logger.info(f"Processing file: {filepath}")
        try:
            with uproot.open(filepath) as file:
                tree_name = "Delphes"
                if tree_name not in file:
                    keys = list(file.keys())
                    candidates = [k for k in keys if "delphes" in k.lower() or "tree" in k.lower()]
                    if candidates:
                        tree_name = candidates[0]
                    else:
                        raise ValueError(f"No suitable tree in {filepath}")
                tree = file[tree_name]

                jets_data = self._extract_jets(tree)
                truth_data = self._extract_truth_info(tree)
                jet_list = self._combine_jet_truth(jets_data, truth_data)

                meta = {
                    "filename": os.path.basename(filepath),
                    "n_events": len(tree),
                    "n_jets": len(jet_list),
                }
                return jet_list, meta
        except Exception as e:
            logger.error(f"Failed on {filepath}: {e}")
            return [], {}

    def _extract_jets(self, tree) -> List[Dict]:
        try:
            jet_branch_name = "Jet"
            if f"{jet_branch_name}.PT" not in tree:
                for alt in ["Jets", "AntiKtJet", "Jet_"]:
                    if f"{alt}.PT" in tree:
                        jet_branch_name = alt
                        break
                else:
                    raise ValueError("No jet branch found")

            jet_branches: Dict[str, Optional[np.ndarray]] = {}
            for feat in self.jet_features:
                bname = f"{jet_branch_name}.{feat}"
                jet_branches[feat] = tree[bname].array(library="np") if bname in tree else None

            # optional stored flavor per jet
            jet_flavor = tree[f"{jet_branch_name}.Flavor"].array(library="np") if f"{jet_branch_name}.Flavor" in tree else None

            # placeholder, not associating real constituents here
            constituent_data: Dict[int, Dict[int, List[Dict]]] = {}

            jets_data: List[Dict] = []
            n_events = len(jet_branches["PT"]) if jet_branches["PT"] is not None else 0
            for evt in range(n_events):
                if jet_branches["PT"] is None:
                    continue
                n_j = len(jet_branches["PT"][evt])
                for j in range(n_j):
                    jd: Dict = {"event_idx": evt}
                    for feat in self.jet_features:
                        arr = jet_branches[feat]
                        jd[feat] = float(arr[evt][j]) if arr is not None else 0.0
                    if jet_flavor is not None:
                        jd["Flavor"] = int(jet_flavor[evt][j])
                    jd["constituents"] = constituent_data.get(evt, {}).get(j, [])
                    jets_data.append(jd)
            return jets_data
        except Exception as e:
            logger.error(f"Error extracting jets: {e}")
            return []

    def _extract_truth_info(self, tree) -> Dict[int, List[Dict]]:
        truth: Dict[int, List[Dict]] = {}
        try:
            needed = ["Particle.PID", "Particle.Status", "Particle.Eta", "Particle.Phi"]
            if not all(k in tree for k in needed):
                logger.warning("Missing Particle branches, no truth available")
                return truth
            pids = tree["Particle.PID"].array(library="np")
            status = tree["Particle.Status"].array(library="np")
            etas = tree["Particle.Eta"].array(library="np")
            phis = tree["Particle.Phi"].array(library="np")
            masses = tree["Particle.Mass"].array(library="np") if "Particle.Mass" in tree else None

            for i in range(len(pids)):
                ev = []
                pid_arr = pids[i]
                st_arr = status[i]
                eta_arr = etas[i]
                phi_arr = phis[i]
                m_arr = masses[i] if masses is not None else np.zeros_like(pid_arr, dtype=np.float32)
                for pid, st, eta, phi, m in zip(pid_arr, st_arr, eta_arr, phi_arr, m_arr):
                    if st in (1, 2, 22) and pid in (36, 35, 25, 443, 441):
                        ev.append({"PID": int(pid), "Status": int(st), "Eta": float(eta), "Phi": float(phi), "Mass": float(m)})
                truth[i] = ev
        except Exception as e:
            logger.warning(f"Truth extraction failed: {e}")
        return truth

    @staticmethod
    def _delta_r(eta1, phi1, eta2, phi2) -> float:
        dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
        deta = eta1 - eta2
        return float(np.sqrt(deta * deta + dphi * dphi))

    def _match_truth(self, jet_dict: Dict, truth_data: Dict) -> Tuple[int, float]:
        jet_eta = jet_dict.get("Eta", 0.0)
        jet_phi = jet_dict.get("Phi", 0.0)
        evt = jet_dict.get("event_idx", None)
        if evt is not None and evt in truth_data:
            for p in truth_data[evt]:
                if self._delta_r(jet_eta, jet_phi, p["Eta"], p["Phi"]) < 0.3 and p["Status"] in (1, 2, 22):
                    if p["PID"] == 36:
                        return 36, p.get("Mass", np.nan)
                    if p["PID"] in (35, 25):
                        return 25, p.get("Mass", np.nan)
                    if p["PID"] == 443:
                        return 443, p.get("Mass", np.nan)
                    if p["PID"] == 441:
                        return 441, p.get("Mass", np.nan)
        if jet_dict.get("Flavor", 0) > 0:
            return int(jet_dict.get("Flavor", 0)), np.nan
        return 0, np.nan

    def _combine_jet_truth(self, jets_data: List[Dict], truth_data: Dict) -> List[JetInfo]:
        out: List[JetInfo] = []
        for jd in jets_data:
            cons: List[JetConstituent] = []
            flavor, tmass = self._match_truth(jd, truth_data)
            out.append(
                JetInfo(
                    pt=float(jd.get("PT", 0.0)),
                    eta=float(jd.get("Eta", 0.0)),
                    phi=float(jd.get("Phi", 0.0)),
                    mass=float(jd.get("Mass", 0.0)),
                    btag=int(jd.get("BTag", 0)),
                    tau_tag=int(jd.get("TauTag", 0)),
                    constituents=cons[: self.max_constituents],
                    truth_flavor=int(flavor),
                    truth_mass=float(tmass),
                )
            )
        return out

    def process_files(self, input_path: str, file_pattern: str = "*.root") -> Tuple[List[JetInfo], List[Dict]]:
        if os.path.isfile(input_path):
            files = [input_path]
        elif os.path.isdir(input_path):
            files = glob.glob(os.path.join(input_path, file_pattern))
            if not files:
                raise ValueError(f"No ROOT files in {input_path} with pattern {file_pattern}")
        else:
            raise ValueError(f"Invalid input {input_path}")

        jets_all: List[JetInfo] = []
        metas: List[Dict] = []
        for fp in tqdm(files, desc="Processing ROOT"):
            jets, meta = self.read_delphes_file(fp)
            jets_all.extend(jets)
            if meta:
                metas.append(meta)
        logger.info(f"Processed {len(jets_all)} jets from {len(files)} file(s)")
        return jets_all, metas

class MLDataConverter:
    def __init__(self, max_constituents: int = 100):
        self.max_constituents = max_constituents

    def jets_to_arrays(self, jets: List[JetInfo]) -> Dict[str, np.ndarray]:
        n = len(jets)
        jet_features = np.zeros((n, 6), dtype=np.float32)
        constituent_features = np.zeros((n, self.max_constituents, 6), dtype=np.float32)
        constituent_mask = np.zeros((n, self.max_constituents), dtype=bool)
        labels = np.zeros((n,), dtype=np.int32)
        targets = np.full((n,), np.nan, dtype=np.float32)

        for i, j in enumerate(jets):
            jet_features[i] = [j.pt, j.eta, j.phi, j.mass, j.btag, j.tau_tag]
            # no real constituent fill here
            labels[i] = int(j.truth_flavor)
            targets[i] = np.float32(j.truth_mass)

        return {
            "jet_features": jet_features,
            "constituent_features": constituent_features,
            "constituent_mask": constituent_mask,
            "labels": labels,
            "targets": targets,
        }

    def save_hdf5(self, data: Dict[str, np.ndarray], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v, compression="gzip")
        logger.info(f"Saved HDF5 to {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Convert Delphes ROOT to HDF5 for ML")
    ap.add_argument("--input", "-i", required=True, help="ROOT file or directory")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--max-constituents", type=int, default=100)
    ap.add_argument("--file-pattern", default="*.root")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    proc = DelphesProcessor(max_constituents=args.max_constituents)
    jets, metas = proc.process_files(args.input, args.file_pattern)
    if not jets:
        logger.error("No jets found")
        return

    conv = MLDataConverter(max_constituents=args.max_constituents)
    data = conv.jets_to_arrays(jets)
    h5_out = os.path.join(args.output, "jet_data.h5")
    conv.save_hdf5(data, h5_out)

    if metas:
        pd.DataFrame(metas).to_csv(os.path.join(args.output, "metadata.csv"), index=False)
    logger.info("Done")

if __name__ == "__main__":
    main()
