// File: main_PhPy8_HZA_with_opts.cc
// Standalone Pythia HepMC generator with command-line options
// Does NOT require CmdLine library - uses simple argv parsing
// LHE output option. 

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <fstream>

using namespace Pythia8;
using namespace std;

// Simple helper function to check if a flag is present
bool hasFlag(int argc, char* argv[], const string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == flag) return true;
  }
  return false;
}

// Helper function to get value after a flag
double getFlagValue(int argc, char* argv[], const string& flag, double defaultVal = 0.0) {
  for (int i = 1; i < argc - 1; ++i) {
    if (string(argv[i]) == flag) {
      return atof(argv[i + 1]);
    }
  }
  return defaultVal;
}

void printUsage(const char* progName) {
  cout << "Usage: " << progName << " -Amass <value> [options]" << endl;
  cout << "\nRequired:" << endl;
  cout << "  -Amass <value>    Mass of the A particle in GeV" << endl;
  cout << "\nOptional flags:" << endl;
  cout << "  --truelep         Use leptons from hard scattering (analysis flag)" << endl;
  cout << "  --MPIoff          Turn off multiparton interactions" << endl;
  cout << "  --ISRoff          Turn off initial-state radiation" << endl;
  cout << "  --FSRoff          Turn off final-state radiation" << endl;
  cout << "  --HADoff          Turn off hadronization" << endl;
  cout << "  --LHEout          Output LHE file in addition to HepMC" << endl;
  cout << "  --help            Show this help message" << endl;
}

// Write LHE header
void writeLHEHeader(ofstream& lheFile, double A_Mass) {
  lheFile << "<LesHouchesEvents version=\"1.0\">\n";
  lheFile << "<!--\n";
  lheFile << "File generated with Pythia8\n";
  lheFile << "H -> Z A with decays\n";
  lheFile << "A mass = " << A_Mass << " GeV\n";
  lheFile << "-->\n";
  lheFile << "<init>\n";
  lheFile << "  2212  2212  4.0000000e+03  4.0000000e+03 0 0 247000 247000 -4 1\n";
  lheFile << "  1.0000000e+00  0.0000000e+00  1.0000000e+00 1\n";
  lheFile << "</init>\n";
}

// Write LHE event
void writeLHEEvent(ofstream& lheFile, Event& event, int eventNum) {
  // Collect final state particles
  vector<int> toWrite;
  
  for (int i = 0; i < event.size(); ++i) {
    if (event[i].isFinal()) {
      toWrite.push_back(i);
    }
  }
  
  int nPart = toWrite.size();
  
  // Write event header
  lheFile << "<event>\n";
  lheFile << " " << nPart << "  1  1.0000000e+00  1.2500000e+02  7.8125000e-03  1.1803780e-01\n";
  
  // Write particles
  for (int idx : toWrite) {
    Particle& p = event[idx];
    
    int id = p.id();
    int status = 1;
    int mother1 = 1;
    int mother2 = 2;
    int color1 = p.col();
    int color2 = p.acol();
    
    lheFile << " " << setw(8) << id
            << " " << setw(2) << status
            << " " << setw(4) << mother1
            << " " << setw(4) << mother2
            << " " << setw(4) << color1
            << " " << setw(4) << color2
            << " " << scientific << setprecision(10) << setw(18) << p.px()
            << " " << scientific << setprecision(10) << setw(18) << p.py()
            << " " << scientific << setprecision(10) << setw(18) << p.pz()
            << " " << scientific << setprecision(10) << setw(18) << p.e()
            << " " << scientific << setprecision(10) << setw(18) << p.m()
            << " 0. 9.\n";
  }
  
  lheFile << "</event>\n";
}

void writeLHEFooter(ofstream& lheFile) {
  lheFile << "</LesHouchesEvents>\n";
}

int main(int argc, char* argv[]) {
  
  // Check for help flag
  if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
    printUsage(argv[0]);
    return 0;
  }
  
  // Check if A mass is provided
  if (!hasFlag(argc, argv, "-Amass")) {
    cerr << "Error: -Amass argument is required" << endl;
    printUsage(argv[0]);
    return 1;
  }
  
  // Parse A mass from command line
  double A_Mass = getFlagValue(argc, argv, "-Amass");
  
  if (A_Mass <= 0) {
    cerr << "Error: A mass must be positive" << endl;
    return 1;
  }

  // Parse optional physics flags
  bool truelep = hasFlag(argc, argv, "--truelep");
  bool MPIoff = hasFlag(argc, argv, "--MPIoff");
  bool ISRoff = hasFlag(argc, argv, "--ISRoff");
  bool FSRoff = hasFlag(argc, argv, "--FSRoff");
  bool HADoff = hasFlag(argc, argv, "--HADoff");
  bool LHEout = hasFlag(argc, argv, "--LHEout");

  // Build the options string for filename
  stringstream ssopts;
  ssopts << ( (truelep) ? "_truelep" : "_reallep" );
  ssopts << ( (MPIoff) ? "_MPIoff" : "_MPIon" );
  ssopts << ( (ISRoff) ? "_ISRoff" : "_ISRon" );
  ssopts << ( (FSRoff) ? "_FSRoff" : "_FSRon" );
  ssopts << ( (HADoff) ? "_HADoff" : "_HADon" );
  ssopts << "_Amass" << std::fixed << std::setprecision(1) << A_Mass;
  string stropts = ssopts.str();
  
  // Set up mass and width parameters
  double H_Mass   = 125.0;
  double H_Width  = 0.00407;
  double A_Width  = (A_Mass / 100.0) * 0.1; // For a 100 GeV A: 100 MeV width scaled accordingly
  double A_MassMin = A_Mass - 100 * A_Width;
  double A_MassMax = A_Mass + 100 * A_Width;
  
  // Construct the output file names with options encoded
  ostringstream oss_hepmc;
  oss_hepmc << "hepmc_output_HZA_decayed" << stropts << ".hepmc";
  string outputFileName = oss_hepmc.str();
  
  ostringstream oss_lhe;
  oss_lhe << "lhe_output_HZA_decayed" << stropts << ".lhe";
  string lheFileName = oss_lhe.str();
  
  cout << "==================================================" << endl;
  cout << "Pythia HepMC Generator with Options" << endl;
  cout << "==================================================" << endl;
  cout << "A mass: " << A_Mass << " GeV" << endl;
  cout << "Options: " << stropts << endl;
  cout << "HepMC output file: " << outputFileName << endl;
  if (LHEout) {
    cout << "LHE output file: " << lheFileName << endl;
  }
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
  
  // --- Apply optional physics flags ---
  if (MPIoff) {
    pythia.readString("PartonLevel:MPI = off");
    cout << ">>> Multiparton interactions turned OFF" << endl;
  }
  if (ISRoff) {
    pythia.readString("PartonLevel:ISR = off");
    cout << ">>> Initial-state radiation turned OFF" << endl;
  }
  if (FSRoff) {
    pythia.readString("PartonLevel:FSR = off");
    cout << ">>> Final-state radiation turned OFF" << endl;
  }
  if (HADoff) {
    pythia.readString("HadronLevel:all = off");
    cout << ">>> Hadronization turned OFF" << endl;
  }
  
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
  cout << "\nInitializing Pythia..." << endl;
  pythia.init();
  
  // List decay channels of A0 (PID 36)
  pythia.particleData.list(36);
  
  // Set up the HepMC output using the generated file name.
  HepMC::Pythia8ToHepMC ToHepMC;
  HepMC::IO_GenEvent hepmc_io(outputFileName, ios::out);
  
  // Set up LHE output if requested
  ofstream* lheFile = nullptr;
  if (LHEout) {
    lheFile = new ofstream(lheFileName);
    if (!lheFile->is_open()) {
      cerr << "Error: Could not open LHE output file " << lheFileName << endl;
      return 1;
    }
    writeLHEHeader(*lheFile, A_Mass);
  }
  
// --- Event loop ---
int nEvents = 10000;
cout << "Generating " << nEvents << " events..." << endl;

for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    if (!pythia.next()) continue; // Skip event if generation failed

    // Convert the Pythia event to a HepMC event.
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC.fill_next_event(pythia, hepmcevt);
    
    // Write the event to HepMC file.
    hepmc_io << hepmcevt;
    
    // Write to LHE file if requested
    if (LHEout) {
      writeLHEEvent(*lheFile, pythia.event, iEvent);
    }
    
    // Clean up memory for the current event.
    delete hepmcevt;
    
    if (iEvent % 1000 == 0)
      cout << "Processing event " << iEvent << endl;  
}

  // Close LHE file if opened
  if (LHEout) {
    writeLHEFooter(*lheFile);
    lheFile->close();
    delete lheFile;
  }
  
  // --- Statistics ---
  pythia.stat();
  
  cout << "\n==================================================" << endl;
  cout << "HepMC file written to: " << outputFileName << endl;
  if (LHEout) {
    cout << "LHE file written to: " << lheFileName << endl;
  }
  cout << "==================================================" << endl;
  
  return 0;
}
