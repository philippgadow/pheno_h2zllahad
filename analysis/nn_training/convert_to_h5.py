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

class DelphesProcessor:
    def __init__(self, max_constituents: int = 100, max_tracks: int = 50, ghost_dr: float = 0.4):
        self.max_constituents = max_constituents
        self.max_tracks = max_tracks
        self.ghost_dr = ghost_dr  # Delta R threshold for ghost association
        self.jet_features = ["PT", "Eta", "Phi", "Mass", "BTag", "TauTag"]
        self.constituent_features = ["PT", "Eta", "Phi", "Mass", "PID", "Charge"]
        self.track_features = ["PT", "Eta", "Phi", "Charge", "PID", "D0", "DZ", "Mass"]

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
                tracks_data = self._extract_tracks(tree)
                truth_data = self._extract_truth_info(tree)
                
                # Perform ghost association
                jet_list = self._associate_tracks_to_jets(jets_data, tracks_data, truth_data)

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
        truth_data: Dict
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
                    ghost_track_vars=ghost_vars
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
            "leadTrackPtRatio": -999999.0,
            "angularity_2": -999999.0,
            "U1_0p7": -999999.0,
            "M2_0p3": -999999.0,
            "tau2": -999999.0
        }
        
        if n_tracks == 0:
            return vars_dict
        
        # Lead track variables
        lead_track = tracks[0]  # Already sorted by PT
        lead_track_dr = self._delta_r(jet_eta, jet_phi, lead_track.eta, lead_track.phi)
        vars_dict["deltaRLeadTrack"] = lead_track_dr
        if jet_pt > 0:
            vars_dict["leadTrackPtRatio"] = lead_track.pt / jet_pt
        
        # Compute track 4-vectors
        track_p4s = []
        for track in tracks:
            px = track.pt * np.cos(track.phi)
            py = track.pt * np.sin(track.phi)
            pz = track.pt * np.sinh(track.eta)
            e = np.sqrt(px**2 + py**2 + pz**2 + track.mass**2)
            track_p4s.append(np.array([e, px, py, pz]))
        
        # Total track PT
        total_pt = sum(track.pt for track in tracks)
        
        # Angularity with beta=2
        if total_pt > 0:
            angularity = 0.0
            for track in tracks:
                dr = self._delta_r(jet_eta, jet_phi, track.eta, track.phi)
                angularity += track.pt * dr**2
            vars_dict["angularity_2"] = angularity / total_pt
        
        # Energy correlation functions
        ecf1_0p7 = self._compute_ecf(tracks, jet_eta, jet_phi, 1, 0.7)
        ecf2_0p7 = self._compute_ecf(tracks, jet_eta, jet_phi, 2, 0.7)
        ecf2_1_0p3 = self._compute_modified_ecf(tracks, jet_eta, jet_phi, 2, 1, 0.3)
        ecf3_1_0p3 = self._compute_modified_ecf(tracks, jet_eta, jet_phi, 3, 1, 0.3)
        
        # U1_0p7 = modified ECF(2,1) with beta=0.7
        vars_dict["U1_0p7"] = ecf2_1_0p3  # Using 0.3 as proxy
        
        # M2_0p3 = ECF3_1 / ECF2_1 with beta=0.3
        if ecf2_1_0p3 > 0:
            vars_dict["M2_0p3"] = ecf3_1_0p3 / ecf2_1_0p3
        
        # N-subjettiness tau2
        tau2 = self._compute_nsubjettiness(tracks, jet_eta, jet_phi, 2, 0.4)
        vars_dict["tau2"] = tau2
        
        return vars_dict
    
    def _compute_ecf(
        self, 
        tracks: List[TrackInfo], 
        jet_eta: float, 
        jet_phi: float, 
        n: int, 
        beta: float
    ) -> float:
        """Compute energy correlation function ECF_n^(beta)"""
        if len(tracks) < n:
            return 0.0
        
        ecf = 0.0
        if n == 1:
            for track in tracks:
                ecf += track.pt
        elif n == 2:
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    dr_ij = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                         tracks[j].eta, tracks[j].phi)
                    ecf += tracks[i].pt * tracks[j].pt * (dr_ij ** beta)
        elif n == 3:
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    for k in range(j+1, len(tracks)):
                        dr_ij = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                             tracks[j].eta, tracks[j].phi)
                        dr_ik = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                             tracks[k].eta, tracks[k].phi)
                        dr_jk = self._delta_r(tracks[j].eta, tracks[j].phi, 
                                             tracks[k].eta, tracks[k].phi)
                        ecf += (tracks[i].pt * tracks[j].pt * tracks[k].pt * 
                               dr_ij * dr_ik * dr_jk) ** beta
        
        return ecf
    
    def _compute_modified_ecf(
        self, 
        tracks: List[TrackInfo], 
        jet_eta: float, 
        jet_phi: float, 
        n: int, 
        m: int, 
        beta: float
    ) -> float:
        """Compute modified energy correlation function ECF_n^(m,beta)"""
        if len(tracks) < n:
            return 0.0
        
        mecf = 0.0
        if n == 2 and m == 1:
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    dr_ij = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                         tracks[j].eta, tracks[j].phi)
                    mecf += tracks[i].pt * tracks[j].pt * (dr_ij ** beta)
        elif n == 3 and m == 1:
            for i in range(len(tracks)):
                for j in range(i+1, len(tracks)):
                    for k in range(j+1, len(tracks)):
                        dr_ij = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                             tracks[j].eta, tracks[j].phi)
                        dr_ik = self._delta_r(tracks[i].eta, tracks[i].phi, 
                                             tracks[k].eta, tracks[k].phi)
                        dr_jk = self._delta_r(tracks[j].eta, tracks[j].phi, 
                                             tracks[k].eta, tracks[k].phi)
                        pt_prod = tracks[i].pt * tracks[j].pt * tracks[k].pt
                        mecf += pt_prod * (dr_ij ** beta) * (dr_ik ** beta) * (dr_jk ** beta)
        
        return mecf
    
    def _compute_nsubjettiness(
        self, 
        tracks: List[TrackInfo], 
        jet_eta: float, 
        jet_phi: float, 
        n_subjets: int, 
        R: float
    ) -> float:
        """
        Compute N-subjettiness using exclusive kT algorithm.
        Simplified implementation for n_subjets=2.
        """
        if len(tracks) < n_subjets:
            return -999999.0
        
        # Use exclusive kT to find subjet axes
        subjet_axes = self._find_kt_subjets(tracks, n_subjets, R)
        
        if len(subjet_axes) < n_subjets:
            return -999999.0
        
        # Compute N-subjettiness
        numerator = 0.0
        denominator = 0.0
        
        for track in tracks:
            min_dr = float('inf')
            for axis in subjet_axes:
                dr = self._delta_r(track.eta, track.phi, axis[0], axis[1])
                min_dr = min(min_dr, dr)
            numerator += track.pt * min_dr
            denominator += track.pt * R
        
        if denominator > 0:
            return numerator / denominator
        
        return -999999.0
    
    def _find_kt_subjets(
        self, 
        tracks: List[TrackInfo], 
        n_subjets: int, 
        R: float
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
                    d_ij = min(pseudo_jets[i][2], pseudo_jets[j][2]) ** 2 * (dr / R) ** 2
                    
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
                    gv["leadTrackPtRatio"],
                    gv["angularity_2"],
                    gv["U1_0p7"],
                    gv["M2_0p3"],
                    gv["tau2"]
                ]
            else:
                ghost_track_vars[i] = -999999.0
            
            labels[i] = int(j.truth_flavor)
            targets[i] = np.float32(j.truth_mass)

        return {
            "jet_features": jet_features,
            "constituent_features": constituent_features,
            "constituent_mask": constituent_mask,
            "track_features": track_features,
            "track_mask": track_mask,
            "ghost_track_vars": ghost_track_vars,
            "labels": labels,
            "targets": targets,
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

def main():
    ap = argparse.ArgumentParser(description="Convert Delphes ROOT to HDF5 for ML with ghost track association")
    ap.add_argument("--input", "-i", required=True, help="ROOT file or directory")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--max-constituents", type=int, default=100)
    ap.add_argument("--max-tracks", type=int, default=50, help="Max tracks per jet (Delta-R matching)")
    ap.add_argument("--ghost-dr", type=float, default=0.4, help="Delta R threshold for track matching")
    ap.add_argument("--file-pattern", default="*.root")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    proc = DelphesProcessor(
        max_constituents=args.max_constituents, 
        max_tracks=args.max_tracks,
        ghost_dr=args.ghost_dr
    )
    jets, metas = proc.process_files(args.input, args.file_pattern)
    if not jets:
        logger.error("No jets found")
        return

    conv = MLDataConverter(max_constituents=args.max_constituents, max_tracks=args.max_tracks)
    data = conv.jets_to_arrays(jets)
    
    # Log what we're about to save
    logger.info("Data arrays to be saved:")
    for key, arr in data.items():
        logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    
    h5_out = os.path.join(args.output, "jet_data.h5")
    conv.save_hdf5(data, h5_out)

    if metas:
        pd.DataFrame(metas).to_csv(os.path.join(args.output, "metadata.csv"), index=False)
    
    # Print summary statistics
    n_jets_with_tracks = np.sum(data["track_mask"].any(axis=1))
    avg_tracks_per_jet = data["track_mask"].sum() / len(jets)
    logger.info(f"Summary: {n_jets_with_tracks}/{len(jets)} jets have associated tracks")
    logger.info(f"Average tracks per jet: {avg_tracks_per_jet:.2f}")
    
    # Verify the file was written correctly
    logger.info(f"\nVerifying written HDF5 file...")
    try:
        with h5py.File(h5_out, 'r') as f:
            logger.info(f"Datasets in file: {list(f.keys())}")
            if 'ghost_track_vars' in f:
                logger.info(f"✓ ghost_track_vars found: shape {f['ghost_track_vars'].shape}")
            else:
                logger.error("✗ ghost_track_vars NOT found in output file!")
            if 'targets' in f:
                logger.info(f"✓ targets found: shape {f['targets'].shape}")
            else:
                logger.error("✗ targets NOT found in output file!")
    except Exception as e:
        logger.error(f"Error verifying output file: {e}")
    
    logger.info("Done")

if __name__ == "__main__":
    main()