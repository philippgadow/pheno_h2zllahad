#!/usr/bin/env python3
"""
Delphes ROOT to ML Pipeline for Jet Tagging

This script reads Delphes ROOT files generated with CMS datacards,
extracts jet information and constituents, and prepares data for 
machine learning training.

Usage:
    python delphes_pipeline.py --input /path/to/delphes_files/ --output /path/to/output/
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
from rich.progress import Progress
from tqdm.rich import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JetConstituent:
    """Data structure for jet constituents"""
    pt: float
    eta: float
    phi: float
    mass: float
    pid: int  # particle ID
    charge: int

@dataclass
class JetInfo:
    """Data structure for jet-level information"""
    pt: float
    eta: float
    phi: float
    mass: float
    btag: int
    tau_tag: int
    constituents: List[JetConstituent]
    truth_flavor: int  # truth-level flavor for training labels

class DelphesProcessor:
    """Main processor class for Delphes ROOT files"""
    
    def __init__(self, max_constituents: int = 100):
        self.max_constituents = max_constituents
        self.jet_features = ['PT', 'Eta', 'Phi', 'Mass', 'BTag', 'TauTag']
        self.constituent_features = ['PT', 'Eta', 'Phi', 'Mass', 'PID', 'Charge']
        
    def read_delphes_file(self, filepath: str) -> Tuple[List[JetInfo], Dict]:
        """
        Read a single Delphes ROOT file and extract jet information
        
        Args:
            filepath: Path to the Delphes ROOT file
            
        Returns:
            Tuple of (jet_list, metadata)
        """
        logger.info(f"Processing file: {filepath}")
        
        try:
            # Open ROOT file
            with uproot.open(filepath) as file:
                # Get the tree (typically called "Delphes")
                tree_name = "Delphes"
                if tree_name not in file:
                    # Try common alternative names
                    available_keys = list(file.keys())
                    tree_candidates = [k for k in available_keys if 'tree' in k.lower() or 'delphes' in k.lower()]
                    if tree_candidates:
                        tree_name = tree_candidates[0]
                    else:
                        raise ValueError(f"Could not find suitable tree in {filepath}")
                
                tree = file[tree_name]
                
                # Get jet and constituent information
                jets_data = self._extract_jets(tree)
                
                # Get truth information if available
                truth_data = self._extract_truth_info(tree)
                
                # Combine jet and truth information
                jet_list = self._combine_jet_truth(jets_data, truth_data)
                
                # Extract metadata
                metadata = {
                    'filename': os.path.basename(filepath),
                    'n_events': len(tree),
                    'n_jets': len(jet_list)
                }
                
                return jet_list, metadata
                
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            return [], {}
    
    def _extract_jets(self, tree) -> List[Dict]:
        """
        Extract jet-level information and basic features from the Delphes tree.

        Returns:
            List of jet dictionaries with kinematic variables, tags, constituents, and optional flavor.
        """
        try:
            # Read jet branches
            jet_branches = {}
            jet_branch_name = "Jet"  # Standard Delphes jet collection name
            
            # Check if jet branch exists
            if f"{jet_branch_name}.PT" not in tree:
                # Try alternative names
                for alt_name in ["Jets", "AntiKtJet", "Jet_"]:
                    if f"{alt_name}.PT" in tree:
                        jet_branch_name = alt_name
                        break
                else:
                    raise ValueError("Could not find jet branch in tree")
            
            # Read jet data
            for feature in self.jet_features:
                branch_name = f"{jet_branch_name}.{feature}"
                if branch_name in tree:
                    jet_branches[feature] = tree[branch_name].array(library="np")
                else:
                    logger.warning(f"Branch {branch_name} not found, using default values")
                    jet_branches[feature] = None
            
            # Read jet constituents (EFlowTrack, EFlowNeutralHadron, EFlowPhoton)
            constituent_data = self._extract_constituents(tree)
            
            # Process jets event by event
            jets_data = []
            n_events = len(jet_branches['PT']) if jet_branches['PT'] is not None else 0
            
            for event_idx in range(n_events):
                event_jets = []
                
                # Get jets for this event
                if jet_branches['PT'] is not None:
                    event_jet_pts = jet_branches['PT'][event_idx]
                    n_jets_in_event = len(event_jet_pts)
                    
                    for jet_idx in range(n_jets_in_event):
                        jet_info = {}
                        
                        # Extract jet-level features
                        for feature in self.jet_features:
                            if jet_branches[feature] is not None:
                                jet_info[feature] = jet_branches[feature][event_idx][jet_idx]
                            else:
                                jet_info[feature] = 0.0 if feature != 'PID' else 0
                        
                        # Add constituent information
                        jet_info['constituents'] = constituent_data.get(event_idx, {}).get(jet_idx, [])

                        # Add flavor information if available
                        if f"{jet_branch_name}.Flavor" in tree:
                            jet_info['Flavor'] = tree[f"{jet_branch_name}.Flavor"].array(library="np")[event_idx][jet_idx]

                        # Add event index for tracking
                        jet_info["event_idx"] = event_idx
                        
                        event_jets.append(jet_info)
                
                jets_data.extend(event_jets)
            
            return jets_data
            
        except Exception as e:
            logger.error(f"Error extracting jets: {e}")
            return []
    
    def _extract_constituents(self, tree) -> Dict:
        """
        Extract jet constituent features from EFlow objects.

        Returns:
            Dictionary mapping event and jet index to lists of constituent feature dicts.
        """
        constituent_data = {}
        
        try:
            # EFlow constituent types in Delphes
            eflow_types = {
                'EFlowTrack': 'charged',
                'EFlowNeutralHadron': 'neutral_hadron', 
                'EFlowPhoton': 'photon'
            }
            
            eflow_data = {}
            for eflow_type in eflow_types:
                eflow_data[eflow_type] = {}
                for feature in ['PT', 'Eta', 'Phi', 'Mass']:
                    branch_name = f"{eflow_type}.{feature}"
                    if branch_name in tree:
                        eflow_data[eflow_type][feature] = tree[branch_name].array(library="np")
                
                # Add PID and Charge information
                if eflow_type == 'EFlowTrack':
                    if f"{eflow_type}.PID" in tree:
                        eflow_data[eflow_type]['PID'] = tree[f"{eflow_type}.PID"].array(library="np")
                    if f"{eflow_type}.Charge" in tree:
                        eflow_data[eflow_type]['Charge'] = tree[f"{eflow_type}.Charge"].array(library="np")
            
            # For simplicity, we'll associate constituents with jets based on proximity
            # In a real implementation, you'd want to use jet clustering information
            # This is a placeholder that creates mock constituent data
            
            return constituent_data
            
        except Exception as e:
            logger.warning(f"Could not extract constituents: {e}")
            return {}
    
    def _extract_truth_info(self, tree) -> Dict[int, List[Dict]]:
        """
        Extract truth-level particles of interest (e.g. Higgs, J/ψ, η_c) from the Delphes tree.

        Returns:
            Dictionary mapping event indices to lists of selected particle dictionaries.
        """
        truth_data = {}

        try:
            if "Particle.PID" not in tree or "Particle.Status" not in tree:
                logger.warning("Missing Particle.PID or Particle.Status in tree")
                return truth_data

            pids = tree["Particle.PID"].array(library="np")
            statuses = tree["Particle.Status"].array(library="np")
            etas = tree["Particle.Eta"].array(library="np")
            phis = tree["Particle.Phi"].array(library="np")

            for i, (pid_arr, status_arr, eta_arr, phi_arr) in enumerate(zip(pids, statuses, etas, phis)):
                event_particles = []
                for pid, status, eta, phi in zip(pid_arr, status_arr, eta_arr, phi_arr):
                    if (status == 1 or status == 2 or status == 22) and (pid == 36 or pid == 35 or pid == 25 or pid == 443 or pid == 441):
                        event_particles.append({
                            "PID": pid,
                            "Status": status,
                            "Eta": eta,
                            "Phi": phi
                        })
                truth_data[i] = event_particles

        except Exception as e:
            logger.warning(f"Could not extract truth info: {e}")

        return truth_data
            
    def _combine_jet_truth(self, jets_data: List[Dict], truth_data: Dict) -> List[JetInfo]:
        """
        Combine jet reconstruction data with truth particle matching.

        Args:
            jets_data: List of raw jet dictionaries.
            truth_data: Dictionary of truth particles indexed by event.

        Returns:
            List of JetInfo dataclass objects with constituents and truth labels.
        """
        jet_list = []
        
        for jet_dict in jets_data:
            # Create constituents list
            constituents = []
            for const_dict in jet_dict.get('constituents', []):
                constituent = JetConstituent(
                    pt=const_dict.get('PT', 0.0),
                    eta=const_dict.get('Eta', 0.0),
                    phi=const_dict.get('Phi', 0.0),
                    mass=const_dict.get('Mass', 0.0),
                    pid=const_dict.get('PID', 0),
                    charge=const_dict.get('Charge', 0)
                )
                constituents.append(constituent)
            
            # Determine truth flavor (placeholder logic)
            truth_flavor = self._determine_truth_flavor(jet_dict, truth_data)
            
            # Create JetInfo object
            jet_info = JetInfo(
                pt=jet_dict.get('PT', 0.0),
                eta=jet_dict.get('Eta', 0.0),
                phi=jet_dict.get('Phi', 0.0),
                mass=jet_dict.get('Mass', 0.0),
                btag=jet_dict.get('BTag', 0),
                tau_tag=jet_dict.get('TauTag', 0),
                constituents=constituents[:self.max_constituents],  # Limit constituents
                truth_flavor=truth_flavor
            )
            
            jet_list.append(jet_info)
        
        return jet_list
    
    def _determine_truth_flavor(self, jet_dict: Dict, truth_data: Dict) -> int:
        """
        Assign a truth-level label to a jet by matching to known signal particles via ΔR.

        Matching priorities:
            - PID 36 → return 36
            - PID 35 or 25 → return 25
            - PID 443 → return 443
            - PID 441 → return 441
            - Fallback: Jet flavor if available
            - Default: 0

        Returns:
            Integer label indicating jet truth category.
        """
        jet_eta = jet_dict.get("Eta", 0.0)
        jet_phi = jet_dict.get("Phi", 0.0)
        event_idx = jet_dict.get("event_idx", None)

        def delta_r(eta1, phi1, eta2, phi2):
            dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
            deta = eta1 - eta2
            return np.sqrt(deta**2 + dphi**2)

        # Try to match to known signal particles (e.g. 36, 35, 25, 443, 441) within ΔR < 0.3
        if event_idx is not None and event_idx in truth_data:
            for particle in truth_data[event_idx]:
                dR = delta_r(jet_eta, jet_phi, particle["Eta"], particle["Phi"])
                if dR < 0.3 and particle["Status"] in [1, 2, 22]:
                    # Hadronically decaying "a" (BSM) Higgs boson
                    if particle["PID"] == 36:
                        return 36
                    # SM Higgs boson
                    # hack in LHE: SM Higgs is PID 35 so that it can decay to BSM Higgs with PDG ID 36
                    elif particle["PID"] == 35:
                        return 25
                    # SM Higgs boson
                    elif particle["PID"] == 25:
                        return 25
                    # J/Psi meson
                    elif particle["PID"] == 443:
                        return 443
                    # eta_c meson
                    elif particle["PID"] == 441:
                        return 441

        # Fall back to Jet Flavor
        if jet_dict.get('Flavor', 0) > 0:
            return jet_dict.get("Flavor", 0.0)
        return 0

    
    def process_files(self, input_path: str, file_pattern: str = "*.root") -> Tuple[List[JetInfo], List[Dict]]:
        """
        Process a ROOT file or all ROOT files in a directory

        Args:
            input_path: Path to a Delphes ROOT file or directory containing ROOT files
            file_pattern: Pattern to match ROOT files if input is a directory

        Returns:
            Tuple of (all_jets, metadata_list)
        """
        if os.path.isfile(input_path):
            # Process a single file
            root_files = [input_path]
        elif os.path.isdir(input_path):
            # Process all matching files in the directory
            file_pattern_path = os.path.join(input_path, file_pattern)
            root_files = glob.glob(file_pattern_path)
            if not root_files:
                raise ValueError(f"No ROOT files found in {input_path} matching {file_pattern}")
        else:
            raise ValueError(f"Input path {input_path} is neither a valid file nor a directory")

        logger.info(f"Found {len(root_files)} ROOT file(s) to process")

        all_jets = []
        metadata_list = []

        for filepath in tqdm(root_files, desc="Processing files"):
            jets, metadata = self.read_delphes_file(filepath)
            all_jets.extend(jets)
            metadata_list.append(metadata)

        logger.info(f"Processed {len(all_jets)} jets from {len(root_files)} file(s)")

        return all_jets, metadata_list


class MLDataConverter:
    """Convert processed jet data to ML-ready formats"""
    
    def __init__(self, max_constituents: int = 100):
        self.max_constituents = max_constituents
    
    def jets_to_arrays(self, jets: List[JetInfo]) -> Dict[str, np.ndarray]:
        """
        Convert structured JetInfo list into numpy arrays for ML training.

        Returns:
            Dictionary with keys:
                - 'jet_features': shape (N, 6)
                - 'constituent_features': shape (N, max_constituents, 6)
                - 'constituent_mask': shape (N, max_constituents), bool
                - 'labels': shape (N,)
        """
        n_jets = len(jets)
        
        # Jet-level features
        jet_features = np.zeros((n_jets, 6))  # pt, eta, phi, mass, btag, tau_tag
        
        # Constituent features
        constituent_features = np.zeros((n_jets, self.max_constituents, 6))  # pt, eta, phi, mass, pid, charge
        constituent_mask = np.zeros((n_jets, self.max_constituents), dtype=bool)
        
        # Labels
        labels = np.zeros(n_jets, dtype=int)
        
        for i, jet in enumerate(jets):
            # Jet-level features
            jet_features[i] = [jet.pt, jet.eta, jet.phi, jet.mass, jet.btag, jet.tau_tag]
            
            # Constituent features
            n_constituents = min(len(jet.constituents), self.max_constituents)
            for j in range(n_constituents):
                const = jet.constituents[j]
                constituent_features[i, j] = [const.pt, const.eta, const.phi, const.mass, const.pid, const.charge]
                constituent_mask[i, j] = True
            
            # Labels
            labels[i] = jet.truth_flavor
        
        return {
            'jet_features': jet_features,
            'constituent_features': constituent_features,
            'constituent_mask': constituent_mask,
            'labels': labels
        }
    
    def save_hdf5(self, data_dict: Dict[str, np.ndarray], output_path: str):
        """
        Save jet and constituent data to an HDF5 file.

        Args:
            data_dict: Dictionary of arrays to save.
            output_path: File path for output .h5 file.
        """
        with h5py.File(output_path, 'w') as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value, compression='gzip')
        
        logger.info(f"Data saved to {output_path}")
    
    def save_numpy(self, data_dict: Dict[str, np.ndarray], output_dir: str):
        """
        Save jet and constituent data as .npy files per feature.

        Args:
            data_dict: Dictionary of arrays to save.
            output_dir: Directory where .npy files will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for key, value in data_dict.items():
            filepath = os.path.join(output_dir, f"{key}.npy")
            np.save(filepath, value)
        
        logger.info(f"Data saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process Delphes ROOT files for jet tagging")
    parser.add_argument("--input", "-i", required=True, help="Input Delphes ROOT file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed data")
    parser.add_argument("--max-constituents", type=int, default=100, help="Maximum number of constituents per jet")
    parser.add_argument("--format", choices=['hdf5', 'numpy'], default='hdf5', help="Output format")
    parser.add_argument("--file-pattern", default="*.root", help="Pattern to match ROOT files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize processor
    processor = DelphesProcessor(max_constituents=args.max_constituents)
    
    # Process files
    jets, metadata = processor.process_files(args.input, args.file_pattern)
    
    if not jets:
        logger.error("No jets found in input files")
        return
    
    # Convert to ML format
    converter = MLDataConverter(max_constituents=args.max_constituents)
    data_dict = converter.jets_to_arrays(jets)
    
    # Save data
    if args.format == 'hdf5':
        output_path = os.path.join(args.output, "jet_data.h5")
        converter.save_hdf5(data_dict, output_path)
    else:
        converter.save_numpy(data_dict, args.output)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(args.output, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    # Print summary
    logger.info(f"Processing complete!")
    logger.info(f"Total jets: {len(jets)}")
    logger.info(f"Jet features shape: {data_dict['jet_features'].shape}")
    logger.info(f"Constituent features shape: {data_dict['constituent_features'].shape}")
    logger.info(f"Label distribution: {np.bincount(data_dict['labels'])}")

if __name__ == "__main__":
    main()