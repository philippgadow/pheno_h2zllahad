#!/usr/bin/env python3
"""
Delphes ROOT to ML pipeline for jet tagging and mass regression with ghost track association.

Outputs an HDF5 with
- jet_features: (N, 6)
- constituent_features: (N, max_constituents, 6)
- constituent_mask: (N, max_constituents)
- track_features: (N, max_tracks, 8)
- track_mask: (N, max_tracks)
- labels: (N,) integer truth flavor tag
- targets: (N,) float truth mass, NaN when unmatched

Usage:
    python delphes_pipeline.py --input /path/to/root_or_dir --output /path/to/outdir
"""

import argparse
import os
import glob
import re
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import uproot
import h5py
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm.rich import tqdm

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("delphes_pipeline")


def deterministic_class_id(key: str) -> int:
    """Return a stable 32-bit integer derived from the class key."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=True)
    if value < 0:
        value = -value
    if value == -1:
        value = 1
    return value


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


@dataclass
class TrackInfo:
    pt: float
    eta: float
    phi: float
    charge: int
    pid: int
    d0: float
    dz: float
    mass: float

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
    tracks: List[TrackInfo]  # Delta-R matched tracks
    truth_flavor: int
    truth_mass: float  # NaN if unmatched
    # Track-based jet substructure variables
    ghost_track_vars: Dict[str, float] = None
    source_file: str = ""
    is_signal: int = 0
    signal_class: int = -1
    signal_class_name: str = ""
    signal_class_key: str = ""
    signal_mass_value: float = float("nan")

class DelphesProcessor:
    def __init__(self, max_constituents: int = 100, max_tracks: int = 50, ghost_dr: float = 0.4):
        self.max_constituents = max_constituents
        self.max_tracks = max_tracks
        self.ghost_dr = ghost_dr  # Delta R threshold for ghost association
        self.jet_features = ["PT", "Eta", "Phi", "Mass", "BTag", "TauTag"]
        self.constituent_features = ["PT", "Eta", "Phi", "Mass", "PID", "Charge"]
        self.track_features = ["PT", "Eta", "Phi", "Charge", "PID", "D0", "DZ", "Mass"]
        self.signal_pids = {36}

    def read_delphes_file(self, filepath: str) -> Tuple[List[JetInfo], Dict]:
        logger.info(f"Processing file: {filepath}")
        try:
            sample_info = self._infer_sample_info(filepath)
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
                tracks_data = self._extract_tracks(tree)
                truth_data = self._extract_truth_info(tree)
                
                # Perform ghost association
                jet_list = self._associate_tracks_to_jets(jets_data, tracks_data, truth_data, sample_info)

                # Annotate with sample metadata
                for jet in jet_list:
                    jet.source_file = sample_info["sample_name"]

                meta = {
                    "filename": os.path.basename(filepath),
                    "n_events": int(tree.num_entries),
                    "n_jets": len(jet_list),
                }
                return jet_list, meta
        except Exception as e:
            logger.error(f"Failed on {filepath}: {e}")
            return [], {}

    def _infer_sample_info(self, filepath: str) -> Dict[str, Any]:
        """Infer per-file metadata to help assign signal classes."""
        base = os.path.basename(filepath)
        lower = base.lower()
        info: Dict[str, Any] = {
            "sample_name": base,
            "is_signal_sample": False,
            "signal_family": None,
            "signal_mass": None,
            "class_key": None,
            "class_name": None,
        }

        # HZA samples with explicit pseudoscalar mass
        if "hza" in lower:
            info["is_signal_sample"] = True
            info["signal_family"] = "HZA"
            mass_match = re.search(r"ma(\d+(?:\.\d+)?)", lower)
            if mass_match:
                mass_val = float(mass_match.group(1))
                info["signal_mass"] = mass_val
                info["class_key"] = f"HZA_mA{mass_match.group(1)}"
                info["class_name"] = f"HZA mA={mass_match.group(1)} GeV"
            else:
                info["class_key"] = "HZA"
                info["class_name"] = "HZA"
        elif "hzjpsi" in lower:
            info["is_signal_sample"] = True
            info["signal_family"] = "HZJpsi"
            info["signal_mass"] = 3.0969
            info["class_key"] = "HZJpsi"
            info["class_name"] = "HZ J/psi"
        elif "hzetac" in lower:
            info["is_signal_sample"] = True
            info["signal_family"] = "HZetac"
            info["signal_mass"] = 2.9839
            info["class_key"] = "HZetac"
            info["class_name"] = "HZ eta_c"
        else:
            info["class_key"] = None
            info["class_name"] = base

        return info

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

            jets_data: List[Dict] = []
            n_events = len(jet_branches["PT"]) if jet_branches["PT"] is not None else 0
            for evt in range(n_events):
                if jet_branches["PT"] is None:
                    continue
                n_j = len(jet_branches["PT"][evt])
                for j in range(n_j):
                    jd: Dict = {"event_idx": evt, "jet_idx": j}
                    for feat in self.jet_features:
                        arr = jet_branches[feat]
                        jd[feat] = float(arr[evt][j]) if arr is not None else 0.0
                    if jet_flavor is not None:
                        jd["Flavor"] = int(jet_flavor[evt][j])
                    jets_data.append(jd)
            return jets_data
        except Exception as e:
            logger.error(f"Error extracting jets: {e}")
            return []

    def _extract_tracks(self, tree) -> Dict[int, List[Dict]]:
        """Extract tracks organized by event index"""
        tracks_by_event: Dict[int, List[Dict]] = {}
        try:
            # Try both Track and EFlowTrack branches
            track_branch_name = "Track"
            if f"{track_branch_name}.PT" not in tree:
                track_branch_name = "EFlowTrack"
            
            if f"{track_branch_name}.PT" not in tree:
                logger.warning("No Track or EFlowTrack branches found")
                return tracks_by_event

            # Extract track features
            track_pt = tree[f"{track_branch_name}.PT"].array(library="np")
            track_eta = tree[f"{track_branch_name}.Eta"].array(library="np")
            track_phi = tree[f"{track_branch_name}.Phi"].array(library="np")
            track_charge = tree[f"{track_branch_name}.Charge"].array(library="np")
            track_pid = tree[f"{track_branch_name}.PID"].array(library="np") if f"{track_branch_name}.PID" in tree else None
            track_d0 = tree[f"{track_branch_name}.D0"].array(library="np") if f"{track_branch_name}.D0" in tree else None
            track_dz = tree[f"{track_branch_name}.DZ"].array(library="np") if f"{track_branch_name}.DZ" in tree else None
            track_mass = tree[f"{track_branch_name}.Mass"].array(library="np") if f"{track_branch_name}.Mass" in tree else None

            for evt in range(len(track_pt)):
                event_tracks = []
                n_tracks = len(track_pt[evt])
                for t in range(n_tracks):
                    track = {
                        "PT": float(track_pt[evt][t]),
                        "Eta": float(track_eta[evt][t]),
                        "Phi": float(track_phi[evt][t]),
                        "Charge": int(track_charge[evt][t]),
                        "PID": int(track_pid[evt][t]) if track_pid is not None else 0,
                        "D0": float(track_d0[evt][t]) if track_d0 is not None else 0.0,
                        "DZ": float(track_dz[evt][t]) if track_dz is not None else 0.0,
                        "Mass": float(track_mass[evt][t]) if track_mass is not None else 0.0,
                    }
                    event_tracks.append(track)
                tracks_by_event[evt] = event_tracks
            
            logger.info(f"Extracted tracks from {len(tracks_by_event)} events")
        except Exception as e:
            logger.warning(f"Track extraction failed: {e}")
        
        return tracks_by_event

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

    def _associate_tracks_to_jets(
        self,
        jets_data: List[Dict],
        tracks_data: Dict[int, List[Dict]],
        truth_data: Dict,
        sample_info: Dict[str, Any]
    ) -> List[JetInfo]:
        """Perform Delta-R matching: match tracks to jets based on Delta R"""
        out: List[JetInfo] = []
        
        for jd in jets_data:
            jet_eta = float(jd.get("Eta", 0.0))
            jet_phi = float(jd.get("Phi", 0.0))
            jet_pt = float(jd.get("PT", 0.0))
            evt = jd.get("event_idx", None)
            
            # Get tracks for this event
            event_tracks = tracks_data.get(evt, [])
            
            # Perform Delta-R matching
            associated_tracks = []
            for track in event_tracks:
                track_eta = track["Eta"]
                track_phi = track["Phi"]
                dr = self._delta_r(jet_eta, jet_phi, track_eta, track_phi)
                
                if dr < self.ghost_dr:
                    associated_tracks.append(TrackInfo(
                        pt=track["PT"],
                        eta=track["Eta"],
                        phi=track["Phi"],
                        charge=track["Charge"],
                        pid=track["PID"],
                        d0=track["D0"],
                        dz=track["DZ"],
                        mass=track["Mass"]
                    ))
            
            # Sort tracks by PT (descending)
            associated_tracks.sort(key=lambda t: t.pt, reverse=True)
            
            # Limit to max_tracks
            associated_tracks = associated_tracks[:self.max_tracks]
            
            # Compute track-based jet substructure variables
            ghost_vars = self._compute_ghost_track_vars(
                associated_tracks, jet_eta, jet_phi, jet_pt
            )
            
            # Match truth
            flavor, tmass = self._match_truth(jd, truth_data)
            is_signal_jet = int(flavor in self.signal_pids and np.isfinite(tmass))
            signal_class_id = -1
            signal_class_name = ""
            signal_class_key = ""
            mass_value = None

            if is_signal_jet:
                mass_value = sample_info.get("signal_mass")
                if mass_value is None or not np.isfinite(mass_value):
                    mass_value = float(tmass) if np.isfinite(tmass) else None

                key_parts = []
                base_key = sample_info.get("class_key")
                if base_key:
                    key_parts.append(base_key)
                key_parts.append(f"PID{int(flavor)}")
                if mass_value is not None and np.isfinite(mass_value):
                    key_parts.append(f"m{mass_value:.3f}")
                signal_class_key = "_".join(key_parts)

                readable = sample_info.get("class_name") or sample_info.get("sample_name") or signal_class_key
                if mass_value is not None and np.isfinite(mass_value):
                    readable = f"{readable} (PID {int(flavor)}, m={mass_value:.3f} GeV)"
                else:
                    readable = f"{readable} (PID {int(flavor)})"

                signal_class_id = deterministic_class_id(signal_class_key)
                signal_class_name = readable
            
            # Placeholder for constituents
            cons: List[JetConstituent] = []
            
            out.append(
                JetInfo(
                    pt=float(jd.get("PT", 0.0)),
                    eta=jet_eta,
                    phi=jet_phi,
                    mass=float(jd.get("Mass", 0.0)),
                    btag=int(jd.get("BTag", 0)),
                    tau_tag=int(jd.get("TauTag", 0)),
                    constituents=cons[:self.max_constituents],
                    tracks=associated_tracks,
                    truth_flavor=int(flavor),
                    truth_mass=float(tmass),
                    ghost_track_vars=ghost_vars,
                    source_file=sample_info.get("sample_name", ""),
                    is_signal=is_signal_jet,
                    signal_class=signal_class_id,
                    signal_class_name=signal_class_name,
                    signal_class_key=signal_class_key,
                    signal_mass_value=mass_value if is_signal_jet else float("nan")
                )
            )
        
        logger.info(f"Associated tracks to {len(out)} jets")
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

        files.sort()

        jets_all: List[JetInfo] = []
        metas: List[Dict] = []
        for fp in tqdm(files, desc="Processing ROOT"):
            jets, meta = self.read_delphes_file(fp)
            jets_all.extend(jets)
            if meta:
                metas.append(meta)
        logger.info(f"Processed {len(jets_all)} jets from {len(files)} file(s)")
        return jets_all, metas

    def _compute_ghost_track_vars(
        self, 
        tracks: List[TrackInfo], 
        jet_eta: float, 
        jet_phi: float,
        jet_pt: float
    ) -> Dict[str, float]:
        """
        Compute track-based jet substructure variables.
        Based on the ATLAS HZX analysis ghost track variables.
        """
        n_tracks = len(tracks)
        
        # Initialize output dictionary
        vars_dict = {
            "nTracks": n_tracks,
            "deltaRLeadTrack": -999999.0,
            "leadTrackPt": -999999.0,
            "angularity_2": -999999.0,
            "U1_0p7": -999999.0,
            "M2_0p3": -999999.0,
            "tau2": -999999.0
        }
        
        if n_tracks == 0:
            return vars_dict
        
        # Lead track variables
        lead_track = max(tracks, key=lambda t: t.pt)
        lead_track_dr = self._delta_r(jet_eta, jet_phi, lead_track.eta, lead_track.phi)
        vars_dict["deltaRLeadTrack"] = lead_track_dr
        vars_dict["leadTrackPt"] = lead_track.pt
        
        # Compute track 4-vectors
        sum_px = sum_py = sum_pz = sum_energy = 0.0
        for track in tracks:
            px = track.pt * np.cos(track.phi)
            py = track.pt * np.sin(track.phi)
            pz = track.pt * np.sinh(track.eta)
            energy = np.sqrt(max(px**2 + py**2 + pz**2 + track.mass**2, 0.0))
            sum_px += px
            sum_py += py
            sum_pz += pz
            sum_energy += energy

        jet_radius = 0.4
        jet_mass = sum_energy**2 - (sum_px**2 + sum_py**2 + sum_pz**2)
        jet_mass = np.sqrt(max(jet_mass, 0.0)) if jet_mass > 0 else 0.0
        
        # Total track PT
        total_pt = sum(track.pt for track in tracks)
        
        # Angularity with beta=2
        if total_pt > 0 and jet_mass > 0:
            angularity = 0.0
            for track in tracks:
                dr = self._delta_r(jet_eta, jet_phi, track.eta, track.phi)
                angle = np.pi * dr / (2.0 * jet_radius)
                sin_term = np.sin(angle)
                cos_term = 1.0 - np.cos(angle)
                # a = 2
                if cos_term <= 0:
                    continue
                term = track.pt * (sin_term ** 2) * (cos_term ** -1)
                angularity += term
            vars_dict["angularity_2"] = angularity / jet_mass
        
        # Energy correlation functions
        u1_0p7 = self._compute_modified_ecf(tracks, n=2, m=1, beta=0.7)
        if u1_0p7 > 0:
            vars_dict["U1_0p7"] = u1_0p7

        ecf2_1_0p3 = self._compute_modified_ecf(tracks, n=2, m=1, beta=0.3)
        ecf3_1_0p3 = self._compute_modified_ecf(tracks, n=3, m=1, beta=0.3)

        if ecf2_1_0p3 > 0:
            vars_dict["M2_0p3"] = ecf3_1_0p3 / ecf2_1_0p3
        
        # N-subjettiness tau2
        tau2 = self._compute_nsubjettiness(tracks, jet_eta, jet_phi, 2, jet_radius, 0.2)
        vars_dict["tau2"] = tau2
        
        return vars_dict
    
    def _compute_ecf(
        self,
        tracks: List[TrackInfo],
        n: int,
        beta: float
    ) -> float:
        """Compute normalized energy correlation function ECF_n^(beta)."""
        if len(tracks) < n or n <= 0:
            return 0.0

        sum_pt = sum(track.pt for track in tracks)
        if sum_pt <= 0:
            return 0.0

        ecf = 0.0
        if n == 1:
            ecf = sum_pt
        elif n == 2:
            for t1, t2 in combinations(tracks, 2):
                dr = self._delta_r(t1.eta, t1.phi, t2.eta, t2.phi)
                ecf += t1.pt * t2.pt * (dr ** beta)
        elif n == 3:
            for t1, t2, t3 in combinations(tracks, 3):
                dr12 = self._delta_r(t1.eta, t1.phi, t2.eta, t2.phi)
                dr13 = self._delta_r(t1.eta, t1.phi, t3.eta, t3.phi)
                dr23 = self._delta_r(t2.eta, t2.phi, t3.eta, t3.phi)
                ecf += t1.pt * t2.pt * t3.pt * (dr12 * dr13 * dr23) ** beta
        else:
            ecf = 0.0

        denominator = sum_pt ** n
        return ecf / denominator if denominator > 0 else 0.0
    
    def _compute_modified_ecf(
        self,
        tracks: List[TrackInfo],
        n: int,
        m: int,
        beta: float
    ) -> float:
        """Compute normalized modified energy correlation function ECF_n^{(m, beta)}."""
        if len(tracks) < n or n <= 0 or m <= 0:
            return 0.0

        sum_pt = sum(track.pt for track in tracks)
        if sum_pt <= 0:
            return 0.0

        mecf = 0.0
        for combo in combinations(tracks, n):
            term = 1.0
            for trk in combo:
                term *= trk.pt

            angles = []
            for t1, t2 in combinations(combo, 2):
                dr = self._delta_r(t1.eta, t1.phi, t2.eta, t2.phi)
                angles.append(dr ** beta)

            if len(angles) < m:
                continue

            angles.sort()
            for idx in range(m):
                term *= angles[idx]

            mecf += term

        denominator = sum_pt ** n
        return mecf / denominator if denominator > 0 else 0.0
    
    def _compute_nsubjettiness(
        self,
        tracks: List[TrackInfo],
        jet_eta: float,
        jet_phi: float,
        n_subjets: int,
        jet_radius: float,
        subjet_radius: float
    ) -> float:
        """Compute N-subjettiness following GhostTracks implementation."""
        if len(tracks) < n_subjets:
            return -999999.0

        subjet_axes = self._find_kt_subjets(tracks, n_subjets, subjet_radius)
        if len(subjet_axes) < n_subjets:
            return -999999.0

        numerator = 0.0
        denominator = 0.0

        for track in tracks:
            min_dr = float('inf')
            for axis in subjet_axes:
                dr = self._delta_r(track.eta, track.phi, axis[0], axis[1])
                if dr < min_dr:
                    min_dr = dr
            numerator += track.pt * min_dr
            denominator += track.pt * jet_radius

        if denominator > 0:
            return numerator / denominator

        return -999999.0
    
    def _find_kt_subjets(
        self,
        tracks: List[TrackInfo],
        n_subjets: int,
        subjet_radius: float
    ) -> List[Tuple[float, float]]:
        """
        Find subjet axes using exclusive kT algorithm.
        Returns list of (eta, phi) tuples for subjet axes.
        """
        # Create pseudo-jets from tracks
        pseudo_jets = [(track.eta, track.phi, track.pt) for track in tracks]
        
        # Iteratively merge until we have n_subjets
        while len(pseudo_jets) > n_subjets:
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            # Find closest pair
            for i in range(len(pseudo_jets)):
                # Distance to beam
                d_iB = pseudo_jets[i][2] ** 2
                
                if d_iB < min_dist:
                    min_dist = d_iB
                    merge_i, merge_j = i, -1
                
                # Distance to other jets
                for j in range(i+1, len(pseudo_jets)):
                    dr = self._delta_r(pseudo_jets[i][0], pseudo_jets[i][1],
                                      pseudo_jets[j][0], pseudo_jets[j][1])
                    d_ij = min(pseudo_jets[i][2], pseudo_jets[j][2]) ** 2 * (dr / subjet_radius) ** 2
                    
                    if d_ij < min_dist:
                        min_dist = d_ij
                        merge_i, merge_j = i, j
            
            # Perform merge
            if merge_j == -1:
                # Remove jet i (merge with beam)
                pseudo_jets.pop(merge_i)
            else:
                # Merge jets i and j
                eta_i, phi_i, pt_i = pseudo_jets[merge_i]
                eta_j, phi_j, pt_j = pseudo_jets[merge_j]
                
                # Combine using PT-weighted average
                pt_tot = pt_i + pt_j
                eta_new = (eta_i * pt_i + eta_j * pt_j) / pt_tot
                
                # Handle phi wraparound
                dphi = phi_j - phi_i
                if dphi > np.pi:
                    dphi -= 2 * np.pi
                elif dphi < -np.pi:
                    dphi += 2 * np.pi
                phi_new = phi_i + dphi * pt_j / pt_tot
                
                # Normalize phi to [-pi, pi]
                while phi_new > np.pi:
                    phi_new -= 2 * np.pi
                while phi_new < -np.pi:
                    phi_new += 2 * np.pi
                
                # Remove old jets and add merged jet
                pseudo_jets.pop(max(merge_i, merge_j))
                pseudo_jets.pop(min(merge_i, merge_j))
                pseudo_jets.append((eta_new, phi_new, pt_tot))
        
        # Return subjet axes (eta, phi)
        return [(pj[0], pj[1]) for pj in pseudo_jets]

class MLDataConverter:
    def __init__(self, max_constituents: int = 100, max_tracks: int = 50):
        self.max_constituents = max_constituents
        self.max_tracks = max_tracks

    def jets_to_arrays(self, jets: List[JetInfo]) -> Dict[str, np.ndarray]:
        n = len(jets)
        jet_features = np.zeros((n, 6), dtype=np.float32)
        constituent_features = np.zeros((n, self.max_constituents, 6), dtype=np.float32)
        constituent_mask = np.zeros((n, self.max_constituents), dtype=bool)
        
        # Track features from Delta-R matching
        track_features = np.zeros((n, self.max_tracks, 8), dtype=np.float32)
        track_mask = np.zeros((n, self.max_tracks), dtype=bool)
        
        # Track-based jet substructure variables (7 variables)
        ghost_track_vars = np.zeros((n, 7), dtype=np.float32)
        
        labels = np.zeros((n,), dtype=np.int32)
        targets = np.full((n,), np.nan, dtype=np.float32)
        is_signal = np.zeros((n,), dtype=np.int8)
        signal_class = np.full((n,), -1, dtype=np.int64)
        signal_mass = np.full((n,), np.nan, dtype=np.float32)

        for i, j in enumerate(jets):
            jet_features[i] = [j.pt, j.eta, j.phi, j.mass, j.btag, j.tau_tag]
            
            # Fill constituent features (if available)
            for c_idx, const in enumerate(j.constituents[:self.max_constituents]):
                constituent_features[i, c_idx] = [
                    const.pt, const.eta, const.phi, const.mass, const.pid, const.charge
                ]
                constituent_mask[i, c_idx] = True
            
            # Fill track features from Delta-R matching
            for t_idx, track in enumerate(j.tracks[:self.max_tracks]):
                track_features[i, t_idx] = [
                    track.pt, track.eta, track.phi, track.charge, 
                    track.pid, track.d0, track.dz, track.mass
                ]
                track_mask[i, t_idx] = True
            
            # Fill ghost track variables
            if j.ghost_track_vars is not None:
                gv = j.ghost_track_vars
                ghost_track_vars[i] = [
                    gv["nTracks"],
                    gv["deltaRLeadTrack"],
                    gv["leadTrackPt"],
                    gv["angularity_2"],
                    gv["U1_0p7"],
                    gv["M2_0p3"],
                    gv["tau2"]
                ]
            else:
                ghost_track_vars[i] = -999999.0
            
            labels[i] = int(j.truth_flavor)
            targets[i] = np.float32(j.truth_mass)
            is_signal[i] = np.int8(j.is_signal)
            signal_class[i] = np.int64(j.signal_class)
            if j.is_signal:
                mass_val = j.signal_mass_value
                if mass_val is None or not np.isfinite(mass_val):
                    mass_val = j.truth_mass
            else:
                mass_val = np.nan

            signal_mass[i] = np.float32(mass_val if mass_val is not None and np.isfinite(mass_val) else np.nan)

        return {
            "jet_features": jet_features,
            "constituent_features": constituent_features,
            "constituent_mask": constituent_mask,
            "track_features": track_features,
            "track_mask": track_mask,
            "ghost_track_vars": ghost_track_vars,
            "labels": labels,
            "targets": targets,
            "is_signal": is_signal,
            "signal_class": signal_class,
            "signal_mass": signal_mass,
        }

    def save_hdf5(self, data: Dict[str, np.ndarray], out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with h5py.File(out_path, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=v, compression="gzip")
                logger.info(f"  Saved dataset '{k}' with shape {v.shape}")
        logger.info(f"Saved HDF5 to {out_path}")
        
        # Log summary statistics
        if "targets" in data:
            targets = data["targets"]
            finite_mask = np.isfinite(targets)
            n_finite = np.sum(finite_mask)
            n_total = len(targets)
            logger.info(f"Target statistics: {n_finite}/{n_total} finite values")
            if n_finite > 0:
                logger.info(f"  Min: {np.min(targets[finite_mask]):.3f}")
                logger.info(f"  Max: {np.max(targets[finite_mask]):.3f}")
                logger.info(f"  Mean: {np.mean(targets[finite_mask]):.3f}")
                logger.info(f"  Std: {np.std(targets[finite_mask]):.3f}")
        
        if "ghost_track_vars" in data:
            gtv = data["ghost_track_vars"]
            logger.info(f"Ghost track vars shape: {gtv.shape}")
            # Check for any invalid values
            n_valid = np.sum(np.all(gtv != -999999.0, axis=1))
            logger.info(f"  Jets with valid ghost track vars: {n_valid}/{len(gtv)}")


def extract_signal_class_metadata(jets: List[JetInfo]) -> Dict[int, Dict[str, Any]]:
    class_meta: Dict[int, Dict[str, Any]] = {}
    for jet in jets:
        if not jet.is_signal or jet.signal_class == -1:
            continue
        if jet.signal_class not in class_meta:
            class_meta[jet.signal_class] = {
                "id": int(jet.signal_class),
                "name": jet.signal_class_name,
                "key": jet.signal_class_key,
                "truth_pid": int(jet.truth_flavor),
                "mass": float(jet.signal_mass_value) if jet.signal_mass_value is not None and np.isfinite(jet.signal_mass_value) else float("nan"),
                "source_file": jet.source_file,
            }
    return class_meta


def append_class_datasets(data: Dict[str, np.ndarray], class_meta: Dict[int, Dict[str, Any]]):
    if not class_meta:
        data["signal_class_ids"] = np.array([], dtype=np.int64)
        data["signal_class_names"] = np.array([], dtype="S80")
        data["signal_class_keys"] = np.array([], dtype="S120")
        data["signal_class_truth_pid"] = np.array([], dtype=np.int32)
        data["signal_class_mass_GeV"] = np.array([], dtype=np.float32)
        data["signal_class_source_file"] = np.array([], dtype="S128")
        return

    sorted_classes = sorted(class_meta.values(), key=lambda entry: entry["id"])
    data["signal_class_ids"] = np.array([entry["id"] for entry in sorted_classes], dtype=np.int64)
    data["signal_class_names"] = np.array(
        [entry["name"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S80"
    )
    data["signal_class_keys"] = np.array(
        [entry["key"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S120"
    )
    data["signal_class_truth_pid"] = np.array([entry["truth_pid"] for entry in sorted_classes], dtype=np.int32)
    data["signal_class_mass_GeV"] = np.array([entry["mass"] for entry in sorted_classes], dtype=np.float32)
    data["signal_class_source_file"] = np.array(
        [entry["source_file"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S128"
    )


def process_root_file(task: Dict[str, Any]) -> Dict[str, Any]:
    filepath = task["filepath"]
    output_dir = task["output_dir"]
    max_const = task["max_constituents"]
    max_tracks = task["max_tracks"]
    ghost_dr = task["ghost_dr"]

    proc = DelphesProcessor(max_constituents=max_const, max_tracks=max_tracks, ghost_dr=ghost_dr)
    jets, meta = proc.read_delphes_file(filepath)

    if not jets:
        logger.warning(f"No jets extracted from {filepath}; skipping HDF5 creation")
        return {
            "input_file": filepath,
            "h5_path": None,
            "meta": meta,
            "n_jets": 0,
            "class_meta": [],
        }

    conv = MLDataConverter(max_constituents=max_const, max_tracks=max_tracks)
    data = conv.jets_to_arrays(jets)
    class_meta = extract_signal_class_metadata(jets)
    append_class_datasets(data, class_meta)

    stem = safe_stem(filepath)
    h5_path = os.path.join(output_dir, f"jet_data_{stem}.h5")
    conv.save_hdf5(data, h5_path)

    # simple verification
    try:
        with h5py.File(h5_path, 'r') as f:
            logger.info(f"Written {h5_path} with datasets: {list(f.keys())}")
    except Exception as err:
        logger.error(f"Verification failed for {h5_path}: {err}")

    return {
        "input_file": filepath,
        "h5_path": h5_path,
        "meta": meta,
        "n_jets": len(jets),
            "class_meta": list(class_meta.values()),
        }


def merge_h5_files(h5_paths: List[str], out_path: str):
    if not h5_paths:
        logger.warning("No HDF5 files provided for merging; skipping combined output")
        return

    required_datasets = [
        "jet_features",
        "constituent_features",
        "constituent_mask",
        "track_features",
        "track_mask",
        "ghost_track_vars",
        "labels",
        "targets",
        "is_signal",
        "signal_class",
        "signal_mass",
    ]

    class_union: Dict[int, Dict[str, Any]] = {}

    with h5py.File(out_path, "w") as fout:
        dataset_handles: Dict[str, h5py.Dataset] = {}

        for path in h5_paths:
            with h5py.File(path, "r") as fin:
                n = fin["jet_features"].shape[0]
                if n == 0:
                    continue

                for name in required_datasets:
                    data = fin[name][:]
                    if name not in dataset_handles:
                        maxshape = (None,) + data.shape[1:]
                        dataset_handles[name] = fout.create_dataset(
                            name,
                            data=data,
                            maxshape=maxshape,
                            chunks=True,
                            compression="gzip",
                        )
                    else:
                        ds = dataset_handles[name]
                        new_size = ds.shape[0] + n
                        ds.resize((new_size,) + ds.shape[1:])
                        ds[-n:] = data

                # merge class metadata
                if "signal_class_ids" in fin:
                    ids = fin["signal_class_ids"][:]
                    names = fin.get("signal_class_names")
                    keys = fin.get("signal_class_keys")
                    pids = fin.get("signal_class_truth_pid")
                    masses = fin.get("signal_class_mass_GeV")
                    sources = fin.get("signal_class_source_file")

                    for idx, cid in enumerate(ids):
                        cid_int = int(cid)
                        if cid_int in class_union:
                            continue
                        entry = {
                            "id": cid_int,
                            "name": names[idx].decode("utf-8") if names is not None else f"signal_{cid_int}",
                            "key": keys[idx].decode("utf-8") if keys is not None else "",
                            "truth_pid": int(pids[idx]) if pids is not None else -1,
                            "mass": float(masses[idx]) if masses is not None else float("nan"),
                            "source_file": sources[idx].decode("utf-8") if sources is not None else "",
                        }
                        class_union[cid_int] = entry

        # store class union datasets
        sorted_classes = sorted(class_union.values(), key=lambda entry: entry["id"])
        fout.create_dataset("signal_class_ids", data=np.array([entry["id"] for entry in sorted_classes], dtype=np.int64))
        fout.create_dataset(
            "signal_class_names",
            data=np.array([entry["name"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S80"),
        )
        fout.create_dataset(
            "signal_class_keys",
            data=np.array([entry["key"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S120"),
        )
        fout.create_dataset(
            "signal_class_truth_pid",
            data=np.array([entry["truth_pid"] for entry in sorted_classes], dtype=np.int32),
        )
        fout.create_dataset(
            "signal_class_mass_GeV",
            data=np.array([entry["mass"] for entry in sorted_classes], dtype=np.float32),
        )
        fout.create_dataset(
            "signal_class_source_file",
            data=np.array([entry["source_file"].encode("utf-8", "ignore") for entry in sorted_classes], dtype="S128"),
        )

    logger.info(f"Merged {len(h5_paths)} files into {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert Delphes ROOT to HDF5 for ML with ghost track association")
    ap.add_argument("--input", "-i", required=True, help="ROOT file or directory")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--max-constituents", type=int, default=100)
    ap.add_argument("--max-tracks", type=int, default=50, help="Max tracks per jet (Delta-R matching)")
    ap.add_argument("--ghost-dr", type=float, default=0.4, help="Delta R threshold for track matching")
    ap.add_argument("--file-pattern", default="*.root")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Number of worker processes (default: cpu_count-1)")
    ap.add_argument("--no-merge", action="store_true", help="Skip creation of combined jet_data.h5")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        root_files = [args.input]
    elif os.path.isdir(args.input):
        root_files = sorted(glob.glob(os.path.join(args.input, args.file_pattern)))
    else:
        raise ValueError(f"Invalid input {args.input}")

    if not root_files:
        logger.error("No ROOT files found to process")
        return
 
    tasks = [
        {
            "filepath": fp,
            "output_dir": args.output,
            "max_constituents": args.max_constituents,
            "max_tracks": args.max_tracks,
            "ghost_dr": args.ghost_dr,
        }
        for fp in root_files
    ]

    logger.info(f"Processing {len(tasks)} ROOT file(s) with {args.n_workers} worker(s)")

    results: List[Dict[str, Any]] = []

    if args.n_workers == 1:
        for task in tqdm(tasks, desc="Converting ROOT"):
            results.append(process_root_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            for res in tqdm(executor.map(process_root_file, tasks), total=len(tasks), desc="Converting ROOT"):
                results.append(res)

    produced = [res for res in results if res.get("h5_path")]
    if not produced:
        logger.error("No HDF5 files were produced")
        return

    produced_paths = [res["h5_path"] for res in produced]

    # Save metadata CSV
    metadata_rows = []
    for res in produced:
        meta = res.get("meta") or {}
        row = {
            "input_file": res["input_file"],
            "h5_file": res["h5_path"],
            "n_jets": res.get("n_jets", 0),
        }
        row.update(meta)
        metadata_rows.append(row)
    pd.DataFrame(metadata_rows).to_csv(os.path.join(args.output, "metadata.csv"), index=False)

    # Save manifest JSON
    manifest = {
        "input": args.input,
        "output_dir": args.output,
        "h5_files": produced_paths,
        "n_workers": args.n_workers,
        "n_total_jets": int(sum(res.get("n_jets", 0) for res in produced)),
    }
    with open(os.path.join(args.output, "conversion_manifest.json"), "w") as jf:
        json.dump(manifest, jf, indent=2)

    if not args.no_merge:
        merged_path = os.path.join(args.output, "jet_data.h5")
        try:
            merge_h5_files(produced_paths, merged_path)
            logger.info(f"Combined dataset written to {merged_path}")
        except Exception as merge_err:
            logger.error(f"Failed to merge per-file HDF5s: {merge_err}")
            logger.info("Falling back to using the first per-file output as jet_data.h5")
            fallback_source = produced_paths[0]
            try:
                import shutil
                shutil.copy2(fallback_source, merged_path)
                logger.info(f"Copied {fallback_source} -> {merged_path}")
            except Exception as copy_err:
                logger.error(f"Fallback copy failed: {copy_err}")
                logger.error("No merged jet_data.h5 available; use per-file outputs or rerun conversion.")
    else:
        logger.info("Skipping merged HDF5 creation (--no-merge set)")

    logger.info("Done")

if __name__ == "__main__":
    main()
