#!/usr/bin/env python3
"""
Convert HepMC2 or HepMC3 files to ROOT format.

Stores:
  - Raw particle-level branches (pid, pt, eta, phi, energy, status) per event
  - 7 ghost-track proxy / substructure variables (flat, one value per passing event):
      gt_nTracks, gt_leadDR, gt_leadPtRatio, gt_angularity2,
      gt_U1_0p7, gt_M2_0p3, gt_tau2   (proper Nsubjettiness if available, proxy otherwise)
  - Lepton kinematics of the selected Z candidate (lep_pt, lem_pt, mll)
  - Leading jet kinematics (jet1_pt, jet1_eta, jet1_mass, mlljet)

Event selection mirrors the C++ Pythia runcard (VERSION 2 / "reallep" mode):
  - At least one SFOS e+e- or mu+mu- pair
  - Both leptons: pT > 18 GeV, |eta| < 2.47 (e) or 2.70 (mu)
  - Leading lepton: pT > 27 GeV
  - 81 < mll < 101 GeV
  - At least one anti-kT R=0.4 jet with pT > 20 GeV, |eta| < 2.5, mllj < 250 GeV

HepMC version differences handled:
  HepMC2  P line:  P <barcode> <pid> <px> <py> <pz> <e> <m> <status> [...]
  HepMC3  P line:  P <particle_id> <vertex_id> <pid> <px> <py> <pz> <e> <m> <status> [...]
  Detection uses the file header (Asciiv3 or IO_GenEvent).
"""

import numpy as np
import uproot
import awkward as ak
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# FastJet import + capability detection
# ---------------------------------------------------------------------------

try:
    import fastjet
except ImportError:
    print("ERROR: fastjet not found. Install with:  pip install fastjet")
    sys.exit(1)

HAS_NSUBJETTINESS = False
try:
    _test_jet = fastjet.PseudoJet(1.0, 0.0, 0.0, 1.0)
    _nsub = fastjet.contrib.Nsubjettiness(
        2,
        fastjet.contrib.OnePass_KT_Axes(),
        fastjet.contrib.NormalizedMeasure(1.0, 0.4)
    )
    HAS_NSUBJETTINESS = True
    print("fastjet Nsubjettiness contrib: AVAILABLE — will compute proper tau2")
except (AttributeError, Exception):
    pass

if not HAS_NSUBJETTINESS:
    print("fastjet Nsubjettiness contrib: NOT AVAILABLE — will use track-based tau2 proxy")


# ---------------------------------------------------------------------------
# HepMC version detection
# ---------------------------------------------------------------------------

def detect_hepmc_version(input_file):
    """
    Return 2 or 3 based on file header.

    HepMC3 files start with:
      HepMC::Version X.Y.Z
      HepMC::Asciiv3-START_EVENT_LISTING
    HepMC2 files start with:
      HepMC::Version X.Y.Z
      HepMC::IO_GenEvent-START_EVENT_LISTING
    """
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Asciiv3' in line:
                return 3
            if 'IO_GenEvent' in line:
                return 2
            # Fall back on data lines
            if line.startswith('E '):
                return 2
    return 2


# ---------------------------------------------------------------------------
# HepMC2 parsing
# ---------------------------------------------------------------------------

def reconstruct_lines_hepmc2(raw_lines):
    """
    HepMC2 can split long P lines across multiple physical lines.
    Merge continuations (lines not starting with a keyword) back.
    Note: we do NOT include 'A' here because HepMC2 'A' lines are
    attribute lines that should each be kept separate.
    """
    # Only merge true data-record prefixes that can be split
    keywords = {'E ', 'V ', 'P ', 'HepMC::', 'U ', 'W ', 'N ', 'C ', 'F ', 'A '}
    lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip('\n\r')
        starts_kw = any(line.startswith(kw) for kw in keywords)
        if starts_kw:
            # Only merge if the next line does NOT start with a keyword
            # (genuine continuation, not a new record)
            while i + 1 < len(raw_lines):
                nxt = raw_lines[i + 1].rstrip('\n\r')
                nxt_kw = any(nxt.startswith(kw) for kw in keywords)
                if not nxt_kw and nxt.strip():
                    line += nxt
                    i += 1
                else:
                    break
        if line.strip():
            lines.append(line)
        i += 1
    return lines


def _empty_event():
    return {'pids': [], 'px': [], 'py': [], 'pz': [], 'e': [], 'statuses': []}


def _append_particle(ev, pid, px, py, pz, e, status):
    ev['pids'].append(pid)
    ev['px'].append(px)
    ev['py'].append(py)
    ev['pz'].append(pz)
    ev['e'].append(e)
    ev['statuses'].append(status)


def _parse_p_line_hepmc2(parts):
    """
    HepMC2:  P <barcode> <pid> <px> <py> <pz> <e> <m> <status> [...]
    indices:   0    1      2    3    4    5    6   7      8
    """
    if len(parts) < 9:
        return None
    try:
        return (int(parts[2]),                          # pid
                float(parts[3]), float(parts[4]),       # px, py
                float(parts[5]), float(parts[6]),       # pz, e
                int(parts[8]))                          # status
    except (ValueError, IndexError):
        return None


def _parse_p_line_hepmc3(parts):
    """
    HepMC3:  P <particle_id> <vertex_id> <pid> <px> <py> <pz> <e> <m> <status> [...]
    indices:   0      1           2        3    4    5    6    7   8      9
    """
    if len(parts) < 10:
        return None
    try:
        return (int(parts[3]),                          # pid
                float(parts[4]), float(parts[5]),       # px, py
                float(parts[6]), float(parts[7]),       # pz, e
                int(parts[9]))                          # status
    except (ValueError, IndexError):
        return None


def parse_hepmc2(input_file):
    with open(input_file, 'r') as f:
        raw_lines = f.readlines()

    lines = reconstruct_lines_hepmc2(raw_lines)
    print(f"  [HepMC2] Reconstructed {len(lines)} lines from {len(raw_lines)} raw lines")

    events, current = [], None
    skip = ('HepMC::', 'U ', 'W ', 'A ', 'N ', 'C ', 'F ', 'V ')

    for line in lines:
        line = line.strip()
        if any(line.startswith(p) for p in skip):
            continue
        if line.startswith('E '):
            if current is not None and current['pids']:
                events.append(current)
            current = _empty_event()
        elif line.startswith('P ') and current is not None:
            parsed = _parse_p_line_hepmc2(line.split())
            if parsed:
                _append_particle(current, *parsed)

    if current is not None and current['pids']:
        events.append(current)
    return events


def parse_hepmc3(input_file):
    """
    HepMC3 Ascii format — no line continuation issues.
    Parse line by line directly.
    """
    events, current = [], None

    with open(input_file, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith('HepMC::') or not line:
                continue

            if line.startswith('E '):
                if current is not None and current['pids']:
                    events.append(current)
                current = _empty_event()

            elif line.startswith('P ') and current is not None:
                parsed = _parse_p_line_hepmc3(line.split())
                if parsed:
                    _append_particle(current, *parsed)

            # V, U, W, A, N, C, F lines are all skipped (just continue)

    if current is not None and current['pids']:
        events.append(current)
    return events


def parse_hepmc_file(input_file):
    version = detect_hepmc_version(input_file)
    print(f"Detected HepMC version: {version}")
    if version == 3:
        return parse_hepmc3(input_file), version
    else:
        return parse_hepmc2(input_file), version


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def calc_pt_eta_phi_arrays(pxs, pys, pzs):
    pts  = np.sqrt(pxs**2 + pys**2)
    ps   = np.sqrt(pxs**2 + pys**2 + pzs**2)
    phis = np.arctan2(pys, pxs)
    etas = np.zeros(len(pxs))
    for i in range(len(pxs)):
        denom = ps[i] - abs(pzs[i])
        if pts[i] > 1e-10 and denom > 1e-10:
            arg = (ps[i] + pzs[i]) / denom
            etas[i] = 0.5 * np.log(arg) if arg > 0 else (10.0 if pzs[i] > 0 else -10.0)
        elif pts[i] <= 1e-10:
            etas[i] = 0.0
        else:
            etas[i] = 10.0 if pzs[i] > 0 else -10.0
    return pts, etas, phis


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi >  np.pi: dphi -= 2 * np.pi
    while dphi < -np.pi: dphi += 2 * np.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2)**2 + delta_phi(phi1, phi2)**2)


def inv_mass_p4s(p4s):
    arr = np.array(p4s, dtype=float)
    tot = arr.sum(axis=0)
    m2  = tot[3]**2 - tot[0]**2 - tot[1]**2 - tot[2]**2
    return float(np.sqrt(max(m2, 0.0)))


# ---------------------------------------------------------------------------
# Selection constants
# ---------------------------------------------------------------------------

PT_LEP_MIN  = 18.0
PT_LEP_THR  = 27.0
ETA_EL_MAX  = 2.47
ETA_MU_MAX  = 2.70
MLL_MIN     = 81.0
MLL_MAX     = 101.0
ZMASS       = 91.1876

JET_PT_MIN  = 20.0
JET_ETA_MAX = 2.5
MLLJ_MAX    = 250.0
JET_R       = 0.4

TRACK_PT_MIN  = 0.5
TRACK_ETA_MAX = 2.5

INVISIBLE_PIDS = frozenset({12, 14, 16, 18})
NEUTRAL_PIDS   = frozenset({22, 111, 130, 310, 311, 2112, 3122, 3212, 3322,
                             9000111, 9000211, 223, 333, 221, 331, 441})

# HepMC3 from Herwig uses status=1 for final state (same as HepMC2).
# Herwig may also write status=11 for beam/incoming — we only want status=1.
FINAL_STATE_STATUS = 1


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def select_event(pids, pxs, pys, pzs, energies, statuses, pts, etas, phis):
    abs_pid  = np.abs(pids)
    is_final = (statuses == FINAL_STATE_STATUS)

    el_idx = np.where(is_final & (abs_pid == 11) & (pts > PT_LEP_MIN) & (np.abs(etas) < ETA_EL_MAX))[0]
    mu_idx = np.where(is_final & (abs_pid == 13) & (pts > PT_LEP_MIN) & (np.abs(etas) < ETA_MU_MAX))[0]

    best_mll, best_idxP, best_idxM = None, None, None

    def find_best_sfos(indices):
        nonlocal best_mll, best_idxP, best_idxM
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ii, jj = indices[i], indices[j]
                # pid > 0 -> particle (e-, mu-) -> charge -1
                # pid < 0 -> antiparticle (e+, mu+) -> charge +1
                charge_i = -1 if pids[ii] > 0 else +1
                charge_j = -1 if pids[jj] > 0 else +1
                if charge_i * charge_j >= 0:
                    continue
                mll = inv_mass_p4s([(pxs[ii], pys[ii], pzs[ii], energies[ii]),
                                    (pxs[jj], pys[jj], pzs[jj], energies[jj])])
                if best_mll is None or abs(mll - ZMASS) < abs(best_mll - ZMASS):
                    best_mll = mll
                    if charge_i > 0:
                        best_idxP, best_idxM = ii, jj
                    else:
                        best_idxP, best_idxM = jj, ii

    find_best_sfos(mu_idx)
    find_best_sfos(el_idx)

    if best_mll is None:
        return None
    if not (MLL_MIN < best_mll < MLL_MAX):
        return None
    if max(pts[best_idxP], pts[best_idxM]) < PT_LEP_THR:
        return None

    lepP_p4 = (pxs[best_idxP], pys[best_idxP], pzs[best_idxP], energies[best_idxP])
    lepM_p4 = (pxs[best_idxM], pys[best_idxM], pzs[best_idxM], energies[best_idxM])

    # --- Jet finding ---
    jet_input_idx = np.where(is_final & ~np.isin(abs_pid, list(INVISIBLE_PIDS)))[0]
    if len(jet_input_idx) == 0:
        return None

    pseudojets = [
        fastjet.PseudoJet(float(pxs[i]), float(pys[i]), float(pzs[i]), float(energies[i]))
        for i in jet_input_idx
    ]
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_R)
    cs      = fastjet.ClusterSequence(pseudojets, jet_def)
    jets    = fastjet.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN))

    selected_jet, selected_mllj = None, None
    for jet in jets:
        if abs(jet.eta()) > JET_ETA_MAX:
            continue
        jet_p4 = (jet.px(), jet.py(), jet.pz(), jet.e())
        mllj   = inv_mass_p4s([lepP_p4, lepM_p4, jet_p4])
        if mllj > MLLJ_MAX:
            continue
        selected_jet  = jet
        selected_mllj = mllj
        break

    if selected_jet is None:
        return None

    return {
        'lepP_p4': lepP_p4, 'lepM_p4': lepM_p4,
        'jet': selected_jet, 'mllj': selected_mllj, 'mll': best_mll,
        'pids': pids, 'pxs': pxs, 'pys': pys, 'pzs': pzs,
        'energies': energies, 'statuses': statuses,
        'pts': pts, 'etas': etas, 'phis': phis,
        'is_final': is_final, 'abs_pid': abs_pid,
        '_cs': cs,
    }


# ---------------------------------------------------------------------------
# Substructure variables
# ---------------------------------------------------------------------------

def _tau2_proxy(tracks, sum_track_pt):
    if sum_track_pt <= 0 or len(tracks) < 2:
        return 0.0
    ax1, ax2 = tracks[0], tracks[1]
    numer = sum(
        t['pt'] * min(delta_r(t['eta'], t['phi'], ax1['eta'], ax1['phi']),
                      delta_r(t['eta'], t['phi'], ax2['eta'], ax2['phi']))
        for t in tracks
    )
    return numer / (sum_track_pt * JET_R)


def compute_substructure(res):
    jet     = res['jet']
    jet_pt  = jet.pt()
    jet_eta = jet.eta()
    jet_phi = jet.phi()

    pids, pts, etas, phis = res['pids'], res['pts'], res['etas'], res['phis']
    statuses, abs_pid, is_final = res['statuses'], res['abs_pid'], res['is_final']

    tracks = []
    for i in range(len(pids)):
        if not is_final[i] or statuses[i] <= 0:
            continue
        apid = int(abs_pid[i])
        if apid in INVISIBLE_PIDS or apid in NEUTRAL_PIDS:
            continue
        if pts[i] < TRACK_PT_MIN or abs(etas[i]) > TRACK_ETA_MAX:
            continue
        dr = delta_r(float(etas[i]), float(phis[i]), jet_eta, jet_phi)
        if dr > JET_R:
            continue
        tracks.append({'pt': float(pts[i]), 'eta': float(etas[i]),
                       'phi': float(phis[i]), 'dR': dr})

    tracks.sort(key=lambda t: t['pt'], reverse=True)

    n_tracks     = len(tracks)
    sum_pt       = sum(t['pt'] for t in tracks)
    lead_pt      = tracks[0]['pt'] if n_tracks > 0 else 0.0
    lead_dr      = tracks[0]['dR'] if n_tracks > 0 else 0.0
    lead_pt_ratio = (lead_pt / jet_pt) if (jet_pt > 0 and lead_pt > 0) else 0.0

    angularity2 = (sum(t['pt'] * t['dR']**2 for t in tracks) / sum_pt) if sum_pt > 0 else 0.0
    U1_0p7      = (sum(t['pt'] * t['dR']  for t in tracks if t['dR'] <= 0.7) / sum_pt) if sum_pt > 0 else 0.0

    denom_m2 = sum(t['pt'] for t in tracks if t['dR'] <= 0.3)
    M2_0p3   = (sum(t['pt'] * t['dR']**2 for t in tracks if t['dR'] <= 0.3) / denom_m2) if denom_m2 > 0 else 0.0

    if HAS_NSUBJETTINESS:
        try:
            nsub2 = fastjet.contrib.Nsubjettiness(
                2, fastjet.contrib.OnePass_KT_Axes(),
                fastjet.contrib.NormalizedMeasure(1.0, JET_R)
            )
            tau2 = float(nsub2(jet))
        except Exception:
            tau2 = _tau2_proxy(tracks, sum_pt)
    else:
        tau2 = _tau2_proxy(tracks, sum_pt)

    lepP_p4, lepM_p4 = res['lepP_p4'], res['lepM_p4']

    return {
        'gt_nTracks':     float(n_tracks),
        'gt_leadDR':      lead_dr,
        'gt_leadPtRatio': lead_pt_ratio,
        'gt_angularity2': angularity2,
        'gt_U1_0p7':      U1_0p7,
        'gt_M2_0p3':      M2_0p3,
        'gt_tau2':        tau2,
        'jet1_pt':        float(jet_pt),
        'jet1_eta':       float(jet_eta),
        'jet1_mass':      float(jet.m()),
        'mlljet':         float(res['mllj']),
        'mll':            float(res['mll']),
        'lep_pt':         float(np.sqrt(lepP_p4[0]**2 + lepP_p4[1]**2)),
        'lem_pt':         float(np.sqrt(lepM_p4[0]**2 + lepM_p4[1]**2)),
    }


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

FLAT_KEYS = [
    'gt_nTracks', 'gt_leadDR', 'gt_leadPtRatio', 'gt_angularity2',
    'gt_U1_0p7', 'gt_M2_0p3', 'gt_tau2',
    'jet1_pt', 'jet1_eta', 'jet1_mass', 'mlljet', 'mll',
    'lep_pt', 'lem_pt',
]


def convert(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    events, hepmc_version = parse_hepmc_file(input_file)
    print(f"Parsed {len(events)} events from HepMC{hepmc_version} file")

    raw_pid, raw_pt, raw_eta, raw_phi, raw_energy, raw_status = [], [], [], [], [], []
    flat     = {k: [] for k in FLAT_KEYS}
    passed   = []
    n_passed = 0

    for iev, ev in enumerate(events):
        if iev % 500 == 0 and iev > 0:
            print(f"  Processed {iev}/{len(events)} events  ({n_passed} passed selection)")

        pxs      = np.array(ev['px'],       dtype=float)
        pys      = np.array(ev['py'],       dtype=float)
        pzs      = np.array(ev['pz'],       dtype=float)
        energies = np.array(ev['e'],        dtype=float)
        pids     = np.array(ev['pids'],     dtype=int)
        statuses = np.array(ev['statuses'], dtype=int)

        pts, etas, phis = calc_pt_eta_phi_arrays(pxs, pys, pzs)

        raw_pid.append(pids.tolist())
        raw_pt.append(pts.tolist())
        raw_eta.append(etas.tolist())
        raw_phi.append(phis.tolist())
        raw_energy.append(energies.tolist())
        raw_status.append(statuses.tolist())

        result = select_event(pids, pxs, pys, pzs, energies, statuses, pts, etas, phis)

        if result is None:
            passed.append(False)
            for k in FLAT_KEYS:
                flat[k].append(float('nan'))
        else:
            passed.append(True)
            n_passed += 1
            sub = compute_substructure(result)
            for k in FLAT_KEYS:
                flat[k].append(sub[k])

    print(f"\nSelection summary: {n_passed}/{len(events)} events passed")
    print(f"Writing ROOT file: {output_file}")

    data = {
        'pid':    ak.Array(raw_pid),
        'pt':     ak.Array(raw_pt),
        'eta':    ak.Array(raw_eta),
        'phi':    ak.Array(raw_phi),
        'energy': ak.Array(raw_energy),
        'status': ak.Array(raw_status),
        'passed': ak.Array(passed),
    }
    for k in FLAT_KEYS:
        data[k] = ak.Array(flat[k])

    with uproot.recreate(output_file) as f:
        f.mktree("events", data, title="HepMC Events with substructure variables")

    tau2_label = "Nsubjettiness (proper)" if HAS_NSUBJETTINESS else "track-based proxy"
    print(f"\nDone. tau2 computed via: {tau2_label}")
    print(f"Output: {output_file}  (tree: 'events')")
    print(f"  Jagged  : pid, pt, eta, phi, energy, status  [all particles]")
    print(f"  Flat    : passed, mll, mlljet, lep_pt, lem_pt,")
    print(f"            jet1_pt, jet1_eta, jet1_mass")
    print(f"  Substr. : gt_nTracks, gt_leadDR, gt_leadPtRatio,")
    print(f"            gt_angularity2, gt_U1_0p7, gt_M2_0p3, gt_tau2")
    print(f"  (NaN for events that fail selection)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HepMC2/3 → ROOT with full event selection + substructure variables"
    )
    parser.add_argument("input",  help="Input HepMC file (.hepmc / .hepmc2 / .hepmc3)")
    parser.add_argument("output", help="Output ROOT file (.root)")
    args = parser.parse_args()

    try:
        convert(args.input, args.output)
    except Exception as exc:
        import traceback
        print(f"\nConversion failed: {exc}")
        traceback.print_exc()
        sys.exit(1)
