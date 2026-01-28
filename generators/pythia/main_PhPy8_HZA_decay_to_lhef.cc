// File: generate_HZA_lhef.cc
// Generate LHEF file with H->ZA process where Z and A are decayed
// but no parton showering or hadronization
#include "Pythia8/Pythia.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string>

using namespace Pythia8;
using namespace std;

int main(int argc, char* argv[]) {

  // Check if the A mass argument is provided.
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <A_mass>" << endl;
    return 1;
  }

  // Get the A mass from command-line argument.
  double A_Mass = atof(argv[1]);
  double H_Mass   = 125.0;
  double H_Width  = 0.00407;
  double A_Width  = (A_Mass / 100.0) * 0.1; // For a 100 GeV A: 100 MeV width scaled accordingly
  double A_MassMin = A_Mass - 100 * A_Width;
  double A_MassMax = A_Mass + 100 * A_Width;

  // Construct the output LHEF file name.
  ostringstream oss;
  oss << "HZA_decayed_mA" << fixed << setprecision(1) << A_Mass << "GeV_for_herwig.lhe";
  string outputFileName = oss.str();

  cout << "==================================================" << endl;
  cout << "Generating LHEF with H->ZA decays" << endl;
  cout << "A mass: " << A_Mass << " GeV" << endl;
  cout << "Output file: " << outputFileName << endl;
  cout << "==================================================" << endl;

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
  
  // Z boson decay settings (leptons only)
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 11 13 15");
  
  // A boson decay settings
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

  // --- CRITICAL: Turn off showering and hadronization, but KEEP resonance decays ---
  // DO NOT use "PartonLevel:all = off" as that disables decays too!
  pythia.readString("PartonLevel:ISR = off");  // No initial-state radiation
  pythia.readString("PartonLevel:FSR = off");  // No final-state radiation
  pythia.readString("PartonLevel:MPI = off");  // No multi-parton interactions
  pythia.readString("HadronLevel:all = off");  // No hadronization
  // ProcessLevel:resonanceDecays is ON by default, which is what we want

  // --- Initialization ---
  cout << "\nInitializing Pythia..." << endl;
  pythia.init();

  // List decay channels of A0 (PID 36) and Z (PID 23)
  cout << "\n=== A boson (PID 36) decay channels ===" << endl;
  pythia.particleData.list(36);
  cout << "\n=== Z boson (PID 23) decay channels ===" << endl;
  pythia.particleData.list(23);

  // Create and open file for LHEF output using Pythia's built-in writer
  // Use pythia.event (not pythia.process) to get the decayed particles
  LHEF3FromPythia8 myLHEF3(&pythia.event, &pythia.info);
  myLHEF3.openLHEF(outputFileName);

  // Write out initialization info on the file.
  myLHEF3.setInit();

  // --- Event loop ---
  int nEvents = 10000;
  cout << "\nGenerating " << nEvents << " events..." << endl;
  
  int nGenerated = 0;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

    // Generate next event.
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) {
        cout << "Reached end of input LHE file" << endl;
        break;
      }
      else continue;
    }

    // Store and write event info.
    myLHEF3.setEvent();
    nGenerated++;

    if (iEvent % 1000 == 0)
      cout << "Processing event " << iEvent << " (generated: " << nGenerated << ")" << endl;
  }

  // --- Statistics ---
  pythia.stat();

  // Write endtag. Overwrite initialization info with new cross sections.
  myLHEF3.closeLHEF(true);

  cout << "\n==================================================" << endl;
  cout << "LHEF file written to: " << outputFileName << endl;
  cout << "Total events generated: " << nGenerated << endl;
  cout << "==================================================" << endl;

  return 0;
}
