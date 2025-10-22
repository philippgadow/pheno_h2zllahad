// File: standalone_pythia_hepmc_bsm_input.cc
#include "Pythia8/Pythia.h"
// #include "Pythia8Plugins/HepMC2.h"  // Include the HepMC interface
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "CmdLine.hh"

using namespace Pythia8;
using namespace std;

/// Pretty printing of Pythia particle
std::ostream& operator<<(std::ostream& os, const Pythia8::Particle & p) {
  os << "["
     <<  p.name()
     << ", mothers=(" << p.mother1() << "," << p.mother2() << ")"
     << "]";
  return os;
}

/// Pretty printing of Pythia weights
void printInfo(const Pythia8::Info & info) {

  // Print variation groups
  int nGroups = info.nWeightGroups();
  cout << "Variation Groups (" << nGroups << "):" << endl;
  for (int iG = 0; iG < nGroups; ++iG) {
    cout << "  Group " << iG
	 << " Name: "   << info.getGroupName(iG)
	 << " Weight: " << info.getGroupWeight(iG)
	 << endl;
  }

  // Print weights using weightLabel / weight
  int nW = info.nWeights();
  cout << "\nWeights (" << nW << "):" << endl;
  for (int i = 0; i < nW; ++i) {
    cout << "  Weight " << i
	 << " Label: " << info.weightLabel(i)
	 << " Value: " << info.weight(i)
	 << endl;
  }

  // Print weights using numberOfWeights / weightNameByIndex
  int nWI = info.numberOfWeights();
  cout << "\nWeights by Index (" << nWI << "):" << endl;
  for (int i = 0; i < nWI; ++i) {
    cout << "  Weight " << i
	 << " Name: "  << info.weightNameByIndex(i)
	 << " Value: " << info.weightValueByIndex(i)
	 << endl;
  }

  // Print vectors
  vector<string> names  = info.weightNameVector();
  vector<double> values = info.weightValueVector();

  cout << "\nWeight Vectors:" << endl;
  for (size_t i = 0; i < names.size(); ++i) {
    cout << "  [" << i << "] "
	 << names[i] << " = "
	 << values[i]
	 << endl;
  }
}

/// Print histogram to file, replacing existing content
void printHist(const string & fname, const Pythia8::Hist & hh)
{
  // hh.pyplotTable(fname, true, false);
  ofstream outfile(fname);
  outfile << "# " << hh.getTitle() << endl;
  outfile << "# underflow = " << hh.getBinContent(0) << endl;
  outfile << "#  overflow = " << hh.getBinContent(hh.getBinNumber()+1) << endl;
  hh.pyplotTable(outfile, true, false);
  outfile.close();
}


int main(int argc, char* argv[]) {

  // construct the cmdline object
  bool enable_help = true;
  CmdLine cmdline(argc, argv, enable_help);
  cmdline.help("PhPy8_HZA");

  double A_Mass = cmdline.value<double>("-Amass").argname("Amass").help("Mass of A");

  bool truelep = cmdline.present("--truelep").
    help("Use leptons from hard scattering and remove their decay products from clustering input.");

  bool MPIoff = cmdline.present("--MPIoff").help("Turn off multiparton interactions.");
  bool ISRoff = cmdline.present("--ISRoff").help("Turn off initial-state radiation in parton shower.");
  bool FSRoff = cmdline.present("--FSRoff").help("Turn off final-state radiation in parton shower.");
  bool HADoff = cmdline.present("--HADoff").help("Turn off hadronization.");

  cmdline.assert_all_options_used();

  stringstream ssopts;
  ssopts << ( (truelep) ? "_truelep" : "_reallep" );
  ssopts << ( (MPIoff) ? "_MPIoff" : "_MPIon" );
  ssopts << ( (ISRoff) ? "_ISRoff" : "_ISRon" );
  ssopts << ( (FSRoff) ? "_FSRoff" : "_FSRon" );
  ssopts << ( (HADoff) ? "_HADoff" : "_HADon" );
  string stropts = ssopts.str();

  // Get the A mass from command-line argument.
  //double A_Mass = atof(argv[1]);

  double H_Mass   = 125.0;
  double H_Width  = 0.00407;
  double A_Width  = (A_Mass / 100.0) * 0.1; // For a 100 GeV A: 100 MeV width scaled accordingly
  double A_MassMin = A_Mass - 100 * A_Width;
  double A_MassMax = A_Mass + 100 * A_Width;

  // Construct the output file name.
  ostringstream oss;
  oss << "hepmc_output_HZA_mA" << fixed << setprecision(2) << A_Mass << "GeV.hepmc";
  string outputFileName = oss.str();

  // Create an instance of Pythia.
  Pythia pythia;

  // --- Input configuration: LHE file ---
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = input_bsm.lhe");

  // --- BSM Higgs options for H -> Z A with A decaying appropriately ---
  pythia.readString("Higgs:useBSM = on");
  pythia.readString(string("35:m0 = ") + to_string(H_Mass));
  pythia.readString(string("35:mWidth = ") + to_string(H_Width));
  pythia.readString("35:doForceWidth = on");
  pythia.readString("35:onMode = off");
  pythia.readString("35:onIfMatch = 23 36"); // Configure H -> Z A
  pythia.readString("23:onMode = off");
  /// Turning off tau for the moment
  // pythia.readString("23:onIfAny = 11 13 15");
  pythia.readString("23:onIfAny = 11 13");
  pythia.readString("36:onMode = on"); // Allow A to decay
  pythia.readString(string("36:m0 = ") + to_string(A_Mass));
  pythia.readString(string("36:mWidth = ") + to_string(A_Width));
  pythia.readString(string("36:mMin = ") + to_string(A_MassMin));
  pythia.readString(string("36:mMax = ") + to_string(A_MassMax));

  // --- Tune and PDF settings ---
  pythia.readString("Tune:ee = 7");
  pythia.readString("Tune:pp = 14");
  pythia.readString("PDF:pSet = LHAPDF6:NNPDF23_lo_as_0130_qed");
  pythia.readString("SpaceShower:rapidityOrder = on");
  pythia.readString("SigmaProcess:alphaSvalue = 0.140");
  pythia.readString("SpaceShower:pT0Ref = 1.56");
  pythia.readString("SpaceShower:pTmaxFudge = 0.91");
  pythia.readString("SpaceShower:pTdampFudge = 1.05");
  pythia.readString("SpaceShower:alphaSvalue = 0.127");
  pythia.readString("TimeShower:alphaSvalue = 0.127");
  pythia.readString("BeamRemnants:primordialKThard = 1.88");
  pythia.readString("ColourReconnection:range = 1.71");
  pythia.readString("MultipartonInteractions:pT0Ref = 2.09");
  pythia.readString("MultipartonInteractions:alphaSvalue = 0.126");

  // --- POWHEG-specific shower settings ---
  pythia.readString("SpaceShower:pTmaxMatch = 2");
  pythia.readString("TimeShower:pTmaxMatch = 2");
  pythia.readString("Powheg:veto = 1");

  if (MPIoff) pythia.readString("PartonLevel:MPI = off");
  if (ISRoff) pythia.readString("PartonLevel:ISR = off");
  if (FSRoff) pythia.readString("PartonLevel:FSR = off");
  if (HADoff) pythia.readString("HadronLevel:all = off");

  // --- Optional: Enable shower uncertainty variations ---
  // pythia.readString("UncertaintyBands:doVariations = on");
  // pythia.readString("UncertaintyBands:List = {"
  //                    "Var3cUp isr:muRfac=0.549241,"
  //                    "Var3Down isr:muRfac=1.960832,"
  //                    "isr:muRfac=2.0_fsr:muRfac=2.0 isr:muRfac=2.0 fsr:muRfac=2.0,"
  //                    "isr:muRfac=2.0_fsr:muRfac=1.0 isr:muRfac=2.0 fsr:muRfac=1.0,"
  //                    "isr:muRfac=2.0_fsr:muRfac=0.5 isr:muRfac=2.0 fsr:muRfac=0.5,"
  //                    "isr:muRfac=1.0_fsr:muRfac=2.0 isr:muRfac=1.0 fsr:muRfac=2.0,"
  //                    "isr:muRfac=1.0_fsr:muRfac=0.5 isr:muRfac=1.0 fsr:muRfac=0.5,"
  //                    "isr:muRfac=0.5_fsr:muRfac=2.0 isr:muRfac=0.5 fsr:muRfac=2.0,"
  //                    "isr:muRfac=0.5_fsr:muRfac=1.0 isr:muRfac=0.5 fsr:muRfac=1.0,"
  //                    "isr:muRfac=0.5_fsr:muRfac=0.5 isr:muRfac=0.5 fsr:muRfac=0.5,"
  //                    "isr:muRfac=1.75_fsr:muRfac=1.0 isr:muRfac=1.75 fsr:muRfac=1.0,"
  //                    "isr:muRfac=1.5_fsr:muRfac=1.0 isr:muRfac=1.5 fsr:muRfac=1.0,"
  //                    "isr:muRfac=1.25_fsr:muRfac=1.0 isr:muRfac=1.25 fsr:muRfac=1.0,"
  //                    "isr:muRfac=0.625_fsr:muRfac=1.0 isr:muRfac=0.625 fsr:muRfac=1.0,"
  //                    "isr:muRfac=0.75_fsr:muRfac=1.0 isr:muRfac=0.75 fsr:muRfac=1.0,"
  //                    "isr:muRfac=0.875_fsr:muRfac=1.0 isr:muRfac=0.875 fsr:muRfac=1.0,"
  //                    "isr:muRfac=1.0_fsr:muRfac=1.75 isr:muRfac=1.0 fsr:muRfac=1.75,"
  //                    "isr:muRfac=1.0_fsr:muRfac=1.5 isr:muRfac=1.0 fsr:muRfac=1.5,"
  //                    "isr:muRfac=1.0_fsr:muRfac=1.25 isr:muRfac=1.0 fsr:muRfac=1.25,"
  //                    "isr:muRfac=1.0_fsr:muRfac=0.625 isr:muRfac=1.0 fsr:muRfac=0.625,"
  //                    "isr:muRfac=1.0_fsr:muRfac=0.75 isr:muRfac=1.0 fsr:muRfac=0.75,"
  //                    "isr:muRfac=1.0_fsr:muRfac=0.875 isr:muRfac=1.0 fsr:muRfac=0.875"
  //                    "}");




  //pythia.readString("Next:numberShowEvent = 5");

  // --- Initialization ---
  pythia.init();

  // --- Event loop ---
  int nEvents = 10000;

  /// Histograms
  Hist h_jet1pt("hist_jet1pt", 100,  0, 2000);
  Hist h_lep_pt("hist_lep_pt", 100,  0,  500);
  Hist h_lem_pt("hist_lem_pt", 100,  0,  500);
  Hist h_j1mass("hist_j1mass",  40,  0,   20);
  Hist h_z_mass("hist_z_mass",  60, 60,  120);
  Hist h_mlljet("hist_mlljet", 100,  0,  300);
  Hist h_gtproxy_nTracks("hist_proxy_nTracks",  40,   0,   40);
  Hist h_gtproxy_deltaRLead("hist_proxy_deltaRLeadTrack", 40, 0.0, 0.4);
  Hist h_gtproxy_leadPtRatio("hist_proxy_leadTrackPtRatio", 40, 0.0, 1.5);
  Hist h_gtproxy_angularity2("hist_proxy_angularity_2", 40, 0.0, 0.4);
  Hist h_gtproxy_U1("hist_proxy_U1_0p7", 40, 0.0, 0.4);
  Hist h_gtproxy_M2("hist_proxy_M2_0p3", 40, 0.0, 0.4);
  Hist h_gtproxy_tau2("hist_proxy_tau2", 40, 0.0, 1.0);

  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

    if (!pythia.next()) continue; // Skip event if generation failed

    if (iEvent % 1000 == 0 && iEvent != 0) {
      cout << "Processed " << iEvent << " events" << endl;
      /// Saving hists periodically
      for (auto h : {h_jet1pt, h_j1mass, h_mlljet, h_lep_pt, h_lem_pt, h_z_mass,
                     h_gtproxy_nTracks, h_gtproxy_deltaRLead, h_gtproxy_leadPtRatio,
                     h_gtproxy_angularity2, h_gtproxy_U1, h_gtproxy_M2, h_gtproxy_tau2}) {
	stringstream ss;
	ss << "hists_PhPy8_HZA/";
	ss << h.getTitle();
	ss << stropts;
	ss << ".dat";
	Hist htmp(h);
	htmp.normalizeSpectrum(iEvent+1);
	printHist(ss.str(), htmp);
      }
    }

#ifdef MYDEBUG
    cout << "--------------------------------------------------" << endl;
#endif

    /// Can't use the const because the statusNeg method is declared as non-const
    //const Pythia8::Event & ev = pythia.event;
    Pythia8::Event & ev = pythia.event;
    const Pythia8::Info & pyinf = pythia.info;

    double pT_lep_min = 18.0;
    double pT_lep_thr = 27.0;
    double eta_el_max = 2.47;
    double eta_mu_max = 2.70;
    double mll_min = 81.0;
    double mll_max = 101.0;

    Vec4 lepP,lepM;

    if (truelep) {

      ////////////////////////////////////////////////////////////////////////////////
      /// VERSION 1: we use leptons from hard scattering. We find all daughters of
      /// these leptons in the list of final state particles to remove them from
      /// jet clustering input (assigning negative status code)
      ////////////////////////////////////////////////////////////////////////////////

      // int countFinVis = 0;
      /// Find the last Z copy in the event record ...
      int iZ = 0;
      for (int i = 0; i < ev.size(); ++i) {
	if (ev[i].id() == 23) iZ = i;
	// if (ev[i].isFinal() && ev[i].isVisible()) countFinVis += 1;
	// if (ev[i].isFinal()) countFinVis += 1;
      }

      /// ... and save its decay products
      int iLep1 = ev[iZ].daughter1();
      int iLep2 = ev[iZ].daughter2();
      // cout << "iZ = " << iZ << "iLep1 = " << iLep1 << "iLep2 = " << iLep2 << endl;
      // cout << "iLep1: " << ev[iLep1].name() << "  pT = " << ev[iLep1].pT() << " eta = " << ev[iLep1].eta() << endl;
      // cout << "iLep2: " << ev[iLep2].name() << "  pT = " << ev[iLep2].pT() << " eta = " << ev[iLep2].eta() << endl;

      vector<int> iLep1dgs = ev[iLep1].daughterListRecursive();
      vector<int> iLep2dgs = ev[iLep2].daughterListRecursive();
      // int countFinZDec = 0;
      vector<int> iZDecs;
      for (auto idg : iLep1dgs) {
	if (ev[idg].isFinal()) {
#ifdef MYDEBUG
	  cout << "iLep1dgs: " << idg << " " << ev[idg] << endl;
#endif
	  // countFinZDec += 1;
	  ev[idg].statusNeg();
	  iZDecs.push_back(idg);
	}
      }
      for (auto idg : iLep2dgs) {
	if (ev[idg].isFinal()) {
#ifdef MYDEBUG
	  cout << "iLep2dgs: " << idg << " " << ev[idg] << endl;
#endif
	  // countFinZDec += 1;
	  ev[idg].statusNeg();
	  iZDecs.push_back(idg);
	}
      }

      /// Cuts on leptons from Z decay (using leptons from hard scattering)
      if (ev[iLep1].pT() < pT_lep_min || ev[iLep2].pT() < pT_lep_min) continue;
      if (ev[iLep1].pT() < pT_lep_thr && ev[iLep2].pT() < pT_lep_thr) continue;
      bool isel = ( abs(ev[iLep1].id()) == 11 && abs(ev[iLep2].id()) == 11 );
      bool ismu = ( abs(ev[iLep1].id()) == 13 && abs(ev[iLep2].id()) == 13 );
      /// Here, we require both leptons to pass rapidity cuts
      if (isel) {
	if (std::abs(ev[iLep1].eta()) > eta_el_max) continue;
	if (std::abs(ev[iLep2].eta()) > eta_el_max) continue;
      } else if (ismu) {
	if (std::abs(ev[iLep1].eta()) > eta_mu_max) continue;
	if (std::abs(ev[iLep2].eta()) > eta_mu_max) continue;
      } else {
	cerr << "Not electrons nor muons from Z decay?" << endl;
	return 1;
      }

      if (ev[iLep1].charge() > 0) {
	lepP = ev[iLep1].p();
	lepM = ev[iLep2].p();
      } else {
	lepP = ev[iLep2].p();
	lepM = ev[iLep1].p();
      }

    } else {

      ////////////////////////////////////////////////////////////////////////////////
      /// VERSION 2: we implement cut on leptons as done in HZa_2018
      /// In this case we are going to include them also in the clustering input?
      ////////////////////////////////////////////////////////////////////////////////

      vector<int> electrons;
      vector<int> muons;
      for (int i = 0; i < ev.size(); ++i) {
	if (abs(ev[i].id()) == 11) {
	  if (ev[i].pT() > pT_lep_min && std::abs(ev[i].eta()) < eta_el_max)
	    electrons.push_back(i);
	}
	if (abs(ev[i].id()) == 13) {
	  if (ev[i].pT() > pT_lep_min && std::abs(ev[i].eta()) < eta_mu_max)
	    muons.push_back(i);
	}
      }
      // Require at least 2 leptons
      if (!((muons.size() >= 2) || (electrons.size() >= 2))) continue;

      // Select best SFOS lepton pair closest to Z mass
      double Zmass = 91.1876;
      double best_mll = -999;
      size_t idxP = 0, idxM = 1;
      bool foundPairMuons = false;
      bool foundPairElectrons = false;
      for (size_t i = 0; i < muons.size(); ++i) {
	for (size_t j = i+1; j < muons.size(); ++j) {
	  if (ev[muons[i]].charge() * ev[muons[j]].charge() >= 0) continue;
	  double mll = (ev[muons[i]].p() + ev[muons[j]].p()).mCalc();
	  if (!foundPairMuons || fabs(mll - Zmass) < fabs(best_mll - Zmass)) {
	    best_mll = mll;
	    if (ev[muons[i]].charge() > 0) {
	      idxP = i;
	      idxM = j;
	    } else {
	      idxP = j;
	      idxM = i;
	    }
	    foundPairMuons = true;
	  }
	}
      }
      for (size_t i = 0; i < electrons.size(); ++i) {
	for (size_t j = i+1; j < electrons.size(); ++j) {
	  if (ev[electrons[i]].charge() * ev[electrons[j]].charge() >= 0) continue;
	  double mll = (ev[electrons[i]].p() + ev[electrons[j]].p()).mCalc();
	  if (!foundPairElectrons || fabs(mll - Zmass) < fabs(best_mll - Zmass)) {
	    best_mll = mll;
	    if (ev[electrons[i]].charge() > 0) {
	      idxP = i;
	      idxM = j;
	    } else {
	      idxP = j;
	      idxM = i;
	    }
	    foundPairElectrons = true;
	  }
	}
      }

      if (!(foundPairMuons || foundPairElectrons) || best_mll < mll_min || best_mll > mll_max) continue;

      // Merge leptons into one vector
      auto leptons = foundPairElectrons ? electrons : muons;
      if (std::max(ev[leptons[idxP]].pT(), ev[leptons[idxM]].pT()) < pT_lep_thr) continue;

      lepP = ev[leptons[idxP]].p();
      lepM = ev[leptons[idxM]].p();

    }

    ////////////////////////////////////////////////////////////////////////////////

    double mll = (lepP + lepM).mCalc();
    if (mll < mll_min || mll > mll_max) {
#ifdef MYDEBUG
      cout << "Failing mll cut: " << mll << endl;
#endif
      continue;
    }
#ifdef MYDEBUG
    cout << "Passing cuts on leptons" << endl;
#endif

    /// Reconstructing jets
    //int nSel = 1;   // all final-state particles
    int nSel = 2;    // Exclude neutrinos (and other invisible) from study (default)
    int massSet = 2; //  all given their correct masses (default)
    SlowJet slowJet(-1, 0.4, 20.0, 25.0, nSel, massSet);
    slowJet.analyze(ev);

    //// NOT VALID if some invisible particle appears in the Z decay products
    // /// Check consistency of inputs
    // if (countFinVis != (slowJet.sizeOrig()+countFinZDec)) {
    //   cerr << "ERROR: countFinVis != (slowJet.sizeOrig()+countFinZDec)" << endl;
    //   cerr << "countFinVis: " << countFinVis << endl;
    //   cerr << "slowJet.sizeOrig(): " << slowJet.sizeOrig() << endl;
    //   cerr << "countFinZDec: " << countFinZDec << endl;
    //   for (auto idg : iZDecs) {
    // 	cout << "iZdec: " << idg << " " << ev[idg] << endl;
    //   }
    //   return 1;
    // }

    if (slowJet.sizeJet() < 1) continue;
#ifdef MYDEBUG
    slowJet.list();
#endif
    int finjet = -1;
    double mllj;
    for (int ijet = 0; ijet < slowJet.sizeJet(); ++ijet) {
      /// check mllj requirement
      mllj = (lepP + lepM + slowJet.p(ijet)).mCalc();
      if (mllj > 250.0) {
#ifdef MYDEBUG
	cout << "Failing mllj cut: " << mllj << endl;
#endif
	continue;
      }
      /// SlowJet does not have a eta method...
      if (std::abs(slowJet.p(ijet).eta()) > 2.5) continue;
      /// If both requirement are satisfied, we have found a jet
      finjet = ijet;
      break;
    }
#ifdef MYDEBUG
    if (finjet < 0) {
      cout << "Not passing cuts on jets" << endl;
      continue;
    } else {
      cout << "Passing cuts on jets" << endl;
      cout << "finjet = " << finjet << endl;
    }
#endif

#ifdef MYDEBUG
    cout << "lhaStrategy(): " << pyinf.lhaStrategy() << endl;
    cout << "Weight: " << pyinf.weight() << endl;
    //printInfo(pyinf);
#endif

    double wgt = 1.0;
    h_jet1pt.fill(slowJet.pT(finjet), wgt);
    h_j1mass.fill(slowJet.m(finjet),  wgt);
    h_mlljet.fill(mllj, wgt);
    h_lep_pt.fill(lepP.pT(), wgt);
    h_lem_pt.fill(lepM.pT(), wgt);
    h_z_mass.fill(mll, wgt);

    /// ------------------------------------------------------------------
    /// Ghost-track proxies using charged final-state particles
    /// ------------------------------------------------------------------
    const double jetPt  = slowJet.pT(finjet);
    const double jetEta = slowJet.p(finjet).eta(); 
    const double jetPhi = slowJet.phi(finjet);
    const double jetR   = 0.4;
    const double pi     = std::acos(-1.0);
    const double twoPi  = 2.0 * pi;

    auto deltaPhi = [pi, twoPi](double phi1, double phi2) {
      double dphi = phi1 - phi2;
      while (dphi > pi)  dphi -= twoPi;
      while (dphi < -pi) dphi += twoPi;
      return dphi;
    };

    struct TrackSummary {
      double pt;
      double eta;
      double phi;
      double deltaR;
    };
    std::vector<TrackSummary> jetTracks;
    double sumTrackPt = 0.0;

    for (int ip = 0; ip < ev.size(); ++ip) {
      if (!ev[ip].isFinal()) continue;
      if (!ev[ip].isCharged()) continue;
      if (ev[ip].status() <= 0) continue; // remove statuses flipped negative

      double trackPt = ev[ip].pT();
      if (trackPt < 0.5) continue; // avoid extremely soft tracks, track pt cut: > 500 MeV
      if (abs(ev[ip].eta()) > 2.5) continue; // no tracking outside of tracker: |eta| < 2.5
      double trackEta = ev[ip].eta();
      double trackPhi = ev[ip].phi();
      double dEta = trackEta - jetEta;
      double dPhi = deltaPhi(trackPhi, jetPhi);
      double dR   = std::sqrt(dEta * dEta + dPhi * dPhi);
      if (dR > jetR) continue;

      jetTracks.push_back({trackPt, trackEta, trackPhi, dR});
      sumTrackPt += trackPt;
    }

    std::sort(jetTracks.begin(), jetTracks.end(),
              [](const TrackSummary& a, const TrackSummary& b) { return a.pt > b.pt; });

    const std::size_t nTracks = jetTracks.size();
    const double leadTrackPt = nTracks > 0 ? jetTracks.front().pt : 0.0;
    const double leadTrackDR = nTracks > 0 ? jetTracks.front().deltaR : 0.0;
    const double leadTrackPtRatio = (jetPt > 0.0 && leadTrackPt > 0.0) ? leadTrackPt / jetPt : 0.0;

    double angularity2 = 0.0;
    if (sumTrackPt > 0.0) {
      double numer = 0.0;
      for (const auto& trk : jetTracks) numer += trk.pt * trk.deltaR * trk.deltaR;
      angularity2 = numer / sumTrackPt;
    }

    double U1_0p7 = 0.0;
    if (sumTrackPt > 0.0) {
      double numer = 0.0;
      for (const auto& trk : jetTracks) {
        if (trk.deltaR <= 0.7) numer += trk.pt * trk.deltaR;
      }
      U1_0p7 = numer / sumTrackPt;
    }

    double M2_0p3 = 0.0;
    {
      double numer = 0.0;
      double denom = 0.0;
      for (const auto& trk : jetTracks) {
        if (trk.deltaR <= 0.3) {
          numer += trk.pt * trk.deltaR * trk.deltaR;
          denom += trk.pt;
        }
      }
      if (denom > 0.0) M2_0p3 = numer / denom;
    }

    double tau2_proxy = 0.0;
    if (sumTrackPt > 0.0 && nTracks >= 2) {
      const auto& axis1 = jetTracks[0];
      const auto& axis2 = jetTracks[1];
      double tauNumer = 0.0;
      for (const auto& trk : jetTracks) {
        double dEta1 = trk.eta - axis1.eta;
        double dPhi1 = deltaPhi(trk.phi, axis1.phi);
        double dEta2 = trk.eta - axis2.eta;
        double dPhi2 = deltaPhi(trk.phi, axis2.phi);
        double dR1 = std::sqrt(dEta1 * dEta1 + dPhi1 * dPhi1);
        double dR2 = std::sqrt(dEta2 * dEta2 + dPhi2 * dPhi2);
        tauNumer += trk.pt * std::min(dR1, dR2);
      }
      tau2_proxy = tauNumer / (sumTrackPt * jetR);
    }

    h_gtproxy_nTracks.fill(static_cast<double>(nTracks), wgt);
    h_gtproxy_deltaRLead.fill(leadTrackDR, wgt);
    h_gtproxy_leadPtRatio.fill(leadTrackPtRatio, wgt);
    h_gtproxy_angularity2.fill(angularity2, wgt);
    h_gtproxy_U1.fill(U1_0p7, wgt);
    h_gtproxy_M2.fill(M2_0p3, wgt);
    h_gtproxy_tau2.fill(tau2_proxy, wgt);

#ifdef MYDEBUG
    cout << "--------------------------------------------------" << endl;
#endif

  } /// End of event loop

  // --- Statistics ---
  pythia.stat();

  for (auto h : {h_jet1pt, h_j1mass, h_mlljet, h_lep_pt, h_lem_pt, h_z_mass,
                 h_gtproxy_nTracks, h_gtproxy_deltaRLead, h_gtproxy_leadPtRatio,
                 h_gtproxy_angularity2, h_gtproxy_U1, h_gtproxy_M2, h_gtproxy_tau2}) {
    stringstream ss;
    ss << "hists_PhPy8_HZA/";
    ss << h.getTitle();
    ss << stropts;
    ss << ".dat";
    Hist htmp(h);
    htmp.normalizeSpectrum(nEvents);
    printHist(ss.str(), htmp);
  }

  return 0;
}
