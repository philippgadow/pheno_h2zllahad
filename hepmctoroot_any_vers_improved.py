#!/usr/bin/env python3
"""
Convert HepMC2 or HepMC3 files to ROOT format with eta, phi, pt
Correctly handles HepMC2 format with variable-length particle lines
"""

import numpy as np
import uproot
import awkward as ak
import argparse
import os
import re

def parse_hepmc_file(input_file):
    """Parse HepMC2/3 file - handles line breaks and variable-length formats"""
    
    event_data = {
        'particle_pid': [],
        'particle_pt': [],
        'particle_eta': [],
        'particle_phi': [],
        'particle_e': [],
        'particle_status': []
    }
    
    current_event = {'pids': [], 'pts': [], 'etas': [], 'phis': [], 'energies': [], 'statuses': []}
    in_event = False
    event_num = 0
    
    print(f"Parsing HepMC file: {input_file}")
    
    # Read and reconstruct lines
    with open(input_file, 'r') as f:
        raw_lines = f.readlines()
    
    print("Reconstructing broken lines...")
    lines = []
    keywords = {'E', 'V', 'P', 'HepMC::', 'U', 'W', 'A', 'N', 'C', 'F'}
    
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip('\n\r')
        
        # Check if line starts with a keyword
        starts_with_keyword = any(line.startswith(kw) for kw in keywords)
        
        if starts_with_keyword:
            # Keep merging with next lines until we hit another keyword
            while i + 1 < len(raw_lines):
                next_line = raw_lines[i + 1].rstrip('\n\r')
                next_is_keyword = any(next_line.startswith(kw) for kw in keywords)
                
                if not next_is_keyword and next_line.strip():
                    # Continuation line
                    line = line + next_line
                    i += 1
                else:
                    break
        
        if line.strip():
            lines.append(line)
        i += 1
    
    print(f"Reconstructed {len(lines)} lines from {len(raw_lines)} original lines")
    
    # Parse reconstructed lines
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip header and info lines
        if line.startswith('HepMC::') or line.startswith('U ') or line.startswith('W ') or \
           line.startswith('A ') or line.startswith('N ') or line.startswith('C ') or line.startswith('F '):
            continue
        
        # Event start
        if line.startswith('E '):
            # Save previous event
            if in_event and len(current_event['pids']) > 0:
                event_data['particle_pid'].append(current_event['pids'][:])
                event_data['particle_pt'].append(current_event['pts'][:])
                event_data['particle_eta'].append(current_event['etas'][:])
                event_data['particle_phi'].append(current_event['phis'][:])
                event_data['particle_e'].append(current_event['energies'][:])
                event_data['particle_status'].append(current_event['statuses'][:])
                event_num += 1
                
                if event_num % 100 == 0:
                    print(f"Parsed {event_num} events...")
            
            # Reset for new event
            current_event = {'pids': [], 'pts': [], 'etas': [], 'phis': [], 'energies': [], 'statuses': []}
            in_event = True
        
        # Particle line - this is the key fix!
        elif line.startswith('P ') and in_event:
            try:
                parts = line.split()
                
                # HepMC2 format: P barcode PDG_id px py pz e m status [flow1 flow2 ...]
                # The barcode is the FIRST number after P, not the particle ID!
                # PDG_id is the SECOND number
                
                if len(parts) >= 9:
                    barcode = int(parts[1])      # Barcode (particle number in event)
                    pid = int(parts[2])          # PDG ID (particle type) - THIS WAS THE BUG!
                    px = float(parts[3])
                    py = float(parts[4])
                    pz = float(parts[5])
                    e = float(parts[6])
                    mass = float(parts[7])
                    status = int(parts[8])
                    # parts[9:] are flow/color information, we ignore them
                    
                    # Calculate kinematic variables
                    pt = np.sqrt(px**2 + py**2)
                    p = np.sqrt(px**2 + py**2 + pz**2)
                    
                    # Calculate eta with safeguards
                    if pt > 1e-10:
                        if p > 0 and abs(p - abs(pz)) > 1e-10:
                            arg = (p + pz) / (p - pz)
                            if arg > 0:
                                eta = 0.5 * np.log(arg)
                            else:
                                eta = 10.0 if pz > 0 else -10.0
                        else:
                            eta = 10.0 if pz > 0 else -10.0
                    else:
                        eta = 0.0
                    
                    # Calculate phi
                    phi = np.arctan2(py, px)
                    
                    current_event['pids'].append(pid)
                    current_event['pts'].append(pt)
                    current_event['etas'].append(eta)
                    current_event['phis'].append(phi)
                    current_event['energies'].append(e)
                    current_event['statuses'].append(status)
                else:
                    print(f"Warning: Particle line has only {len(parts)} fields at line {line_num}")
            
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse particle at line {line_num}")
                print(f"  Error: {e}")
                print(f"  Line: {line[:200]}")
                if 'parts' in locals() and len(parts) > 0:
                    print(f"  First few fields: {parts[:min(10, len(parts))]}")
                continue
        
        # Vertex lines - skip
        elif line.startswith('V '):
            continue
    
    # Save last event
    if in_event and len(current_event['pids']) > 0:
        event_data['particle_pid'].append(current_event['pids'])
        event_data['particle_pt'].append(current_event['pts'])
        event_data['particle_eta'].append(current_event['etas'])
        event_data['particle_phi'].append(current_event['phis'])
        event_data['particle_e'].append(current_event['energies'])
        event_data['particle_status'].append(current_event['statuses'])
        event_num += 1
    
    print(f"Successfully parsed {event_num} events")
    return event_data

def write_root_file(event_data, output_file):
    """Write event data to ROOT file using TTree"""
    
    if len(event_data['particle_pid']) == 0:
        raise ValueError("No events to write! All events failed to parse.")
    
    print(f"Creating ROOT file with {len(event_data['particle_pid'])} events...")
    
    data_dict = {
        'pid': ak.Array(event_data['particle_pid']),
        'pt': ak.Array(event_data['particle_pt']),
        'eta': ak.Array(event_data['particle_eta']),
        'phi': ak.Array(event_data['particle_phi']),
        'energy': ak.Array(event_data['particle_e']),
        'status': ak.Array(event_data['particle_status'])
    }
    
    with uproot.recreate(output_file) as file:
        file.mktree("events", data_dict, title="HepMC Events")
    
    print(f"Successfully wrote ROOT file: {output_file}")
    print(f"  Tree: events")
    print(f"  Branches: pid, pt, eta, phi, energy, status")
    print(f"  Events: {len(event_data['particle_pid'])}")

def convert_hepmc_to_root(input_file, output_file):
    """Convert HepMC file to ROOT format"""
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Parse the HepMC file
    event_data = parse_hepmc_file(input_file)
    
    # Write to ROOT
    write_root_file(event_data, output_file)
    
    print("\nConversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HepMC2/3 files to ROOT format with eta, phi, pt"
    )
    parser.add_argument("input", help="Input HepMC file")
    parser.add_argument("output", help="Output ROOT file")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import uproot
        import awkward as ak
    except ImportError:
        print("Error: Missing required packages.")
        print("Install with:")
        print("  conda install -c conda-forge uproot awkward")
        exit(1)
    
    try:
        convert_hepmc_to_root(args.input, args.output)
    except Exception as e:
        print(f"\nConversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
