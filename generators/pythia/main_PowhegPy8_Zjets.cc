#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include <iostream>
#include <string>

using namespace Pythia8;

int main(int argc, char* argv[]) {
  Pythia pythia;

  std::string inputLhe = "input.lhe";
  std::string outputHepmc = "hepmc_output_Zjets.hepmc";
  int maxEvents = -1;

  if (argc > 1) inputLhe = argv[1];
  if (argc > 2) outputHepmc = argv[2];
  if (argc > 3) maxEvents = std::stoi(argv[3]);

  // Input from POWHEG LHEF
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + inputLhe);

  // ATLAS AZNLO-inspired setup (Powheg+Pythia8 with CTEQ6L1)
  pythia.readString("Tune:ee = 7");
  pythia.readString("Tune:pp = 14");
  pythia.readString("SpaceShower:rapidityOrder = on");
  pythia.readString("SigmaProcess:alphaSvalue = 0.140");
  pythia.readString("SpaceShower:pT0Ref = 1.56");
  pythia.readString("SpaceShower:pTmaxFudge = 0.91");
  pythia.readString("SpaceShower:pTdampFudge = 1.05");
  pythia.readString("SpaceShower:alphaSvalue = 0.127");
  pythia.readString("TimeShower:alphaSvalue = 0.127");
  pythia.readString("ColourReconnection:range = 1.71");
  pythia.readString("MultipartonInteractions:pT0Ref = 2.09");
  pythia.readString("MultipartonInteractions:alphaSvalue = 0.126");

  // Needed in ATLAS setup to better reproduce low-pT(V) in newer Pythia versions
  pythia.readString("BeamRemnants:primordialKThard = 1.4");

  pythia.init();

  HepMC::Pythia8ToHepMC toHepMC;
  HepMC::IO_GenEvent hepmcOut(outputHepmc, std::ios::out);

  int writtenEvents = 0;
  for (int iEvent = 0; ; ++iEvent) {
    if (maxEvents > 0 && writtenEvents >= maxEvents) break;

    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      continue;
    }

    auto* hepmcEvt = new HepMC::GenEvent();
    toHepMC.fill_next_event(pythia, hepmcEvt);
    hepmcOut << hepmcEvt;
    delete hepmcEvt;

    ++writtenEvents;
    if (writtenEvents % 1000 == 0) {
      std::cout << "Processed " << writtenEvents << " events" << std::endl;
    }
  }

  std::cout << "Wrote " << writtenEvents << " events to " << outputHepmc << std::endl;
  pythia.stat();
  return 0;
}
