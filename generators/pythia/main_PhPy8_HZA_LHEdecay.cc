// File: standalone_pythia_hepmc_bsm_input.cc
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"  // Include the HepMC interface
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
  pythia.readString("23:onIfAny = 11 13 15");
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

  // // --- Optional: Enable shower uncertainty variations ---
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


  /// save only hard process, with H->ZA(->qqb) decay
  pythia.readString("PartonLevel:all = off");

  // --- Initialization ---
  pythia.init();

  // Create and open file for LHEF 3.0 output.
  // LHEF3FromPythia8 myLHEF3(&pythia.event, &pythia.info);
  LHEF3FromPythia8 myLHEF3(&pythia.process, &pythia.info);
  myLHEF3.openLHEF("LHEdecay.lhe");

  // Write out initialization info on the file.
  myLHEF3.setInit();

  // --- Event loop ---
  //int nEvents = 10000;
  int nEvents = 100;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {

    // Generate next event.
    if (!pythia.next()) {
      if( pythia.info.atEndOfFile() ) break;
      else continue;
    }

    // Store and write event info.
    myLHEF3.setEvent();

    if (iEvent % 1000 == 0)
      cout << "Processing event " << iEvent << endl;
  }

  // --- Statistics ---
  pythia.stat();

  // Write endtag. Overwrite initialization info with new cross sections.
  myLHEF3.closeLHEF(true);

  return 0;
}
