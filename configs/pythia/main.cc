// File: standalone_pythia_hepmc.cc
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"  // Include the HepMC interface
#include <iostream>
#include <string>

using namespace Pythia8;
using namespace std;

int main() {

  // Create an instance of Pythia.
  Pythia pythia;

  // --- Input configuration: LHE file ---
  pythia.readString("Beams:frameType = 4");

  // --- Process/decay configuration ---
  pythia.readString("25:onMode = off");
  pythia.readString("25:addChannel = 1 0.002 100 441 23");
  pythia.readString("441:onMode = on");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 11 13 15");

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

  // --- Optional: Enable shower uncertainty variations ---
  pythia.readString("UncertaintyBands:doVariations = on");
  pythia.readString("UncertaintyBands:List = {"
                     "Var3cUp isr:muRfac=0.549241,"
                     "Var3Down isr:muRfac=1.960832,"
                     "isr:muRfac=2.0_fsr:muRfac=2.0 isr:muRfac=2.0 fsr:muRfac=2.0,"
                     "isr:muRfac=2.0_fsr:muRfac=1.0 isr:muRfac=2.0 fsr:muRfac=1.0,"
                     "isr:muRfac=2.0_fsr:muRfac=0.5 isr:muRfac=2.0 fsr:muRfac=0.5,"
                     "isr:muRfac=1.0_fsr:muRfac=2.0 isr:muRfac=1.0 fsr:muRfac=2.0,"
                     "isr:muRfac=1.0_fsr:muRfac=0.5 isr:muRfac=1.0 fsr:muRfac=0.5,"
                     "isr:muRfac=0.5_fsr:muRfac=2.0 isr:muRfac=0.5 fsr:muRfac=2.0,"
                     "isr:muRfac=0.5_fsr:muRfac=1.0 isr:muRfac=0.5 fsr:muRfac=1.0,"
                     "isr:muRfac=0.5_fsr:muRfac=0.5 isr:muRfac=0.5 fsr:muRfac=0.5,"
                     "isr:muRfac=1.75_fsr:muRfac=1.0 isr:muRfac=1.75 fsr:muRfac=1.0,"
                     "isr:muRfac=1.5_fsr:muRfac=1.0 isr:muRfac=1.5 fsr:muRfac=1.0,"
                     "isr:muRfac=1.25_fsr:muRfac=1.0 isr:muRfac=1.25 fsr:muRfac=1.0,"
                     "isr:muRfac=0.625_fsr:muRfac=1.0 isr:muRfac=0.625 fsr:muRfac=1.0,"
                     "isr:muRfac=0.75_fsr:muRfac=1.0 isr:muRfac=0.75 fsr:muRfac=1.0,"
                     "isr:muRfac=0.875_fsr:muRfac=1.0 isr:muRfac=0.875 fsr:muRfac=1.0,"
                     "isr:muRfac=1.0_fsr:muRfac=1.75 isr:muRfac=1.0 fsr:muRfac=1.75,"
                     "isr:muRfac=1.0_fsr:muRfac=1.5 isr:muRfac=1.0 fsr:muRfac=1.5,"
                     "isr:muRfac=1.0_fsr:muRfac=1.25 isr:muRfac=1.0 fsr:muRfac=1.25,"
                     "isr:muRfac=1.0_fsr:muRfac=0.625 isr:muRfac=1.0 fsr:muRfac=0.625,"
                     "isr:muRfac=1.0_fsr:muRfac=0.75 isr:muRfac=1.0 fsr:muRfac=0.75,"
                     "isr:muRfac=1.0_fsr:muRfac=0.875 isr:muRfac=1.0 fsr:muRfac=0.875"
                     "}");

  // --- Initialization ---
  pythia.init();

  // Set up the HepMC output
  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent hepmc_io("hepmc_output.hepmc", std::ios::out);

  // --- Event loop ---
  int nEvents = 10000;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue; // Skip event if generation failed

    // Convert the Pythia event to a HepMC event
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC.fill_next_event(pythia, hepmcevt);

    // Write the event to file
    hepmc_io << hepmcevt;

    // Clean up memory for the current event
    delete hepmcevt;

    if (iEvent % 1000 == 0)
      cout << "Processing event " << iEvent << endl;
  }

  // --- Statistics ---
  pythia.stat();

  return 0;
}
