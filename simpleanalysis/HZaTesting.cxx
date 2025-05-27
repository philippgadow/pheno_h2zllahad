#include "SimpleAnalysisFramework/AnalysisClass.h"

DefineAnalysis(HZaTesting)

void HZaTesting::Init() {
   // add regions
   //addRegions({"Preselection", "SR_test"});

   // Book 1/2D histograms
   addHistogram("h_cutflow",20,0,20);
   addHistogram("h_SumOfWeights",1,-1E5,1E5);

}

void HZaTesting::ProcessEvent(AnalysisEvent *event) {

   fill("h_SumOfWeights", event->getMCWeights()[0]);
   
   
   //====================================================================================================
   // get baseline objects
   // cfr. STconfig: https://gitlab.cern.ch/atlas-phys-susy-wg/AnalysisSUSYToolsConfigurations/
   auto baselineElectrons  = filterCrack(event->getElectrons(4.5, 2.47, ELooseBLLH|EZ05mm));
   auto baselineMuons      = event->getMuons(3.0, 2.5, MuMedium|MuZ05mm);
   auto baselineJets       = event->getJets(20., 4.5);
   auto metVec             = event->getMET();
   double met              = metVec.Et();

   if(countObjects(baselineJets, 20, 4.5, NOT(LooseBadJet))!=0) return; //Bad jet veto
   if (countObjects(baselineMuons, 20, 2.7,NOT(MuQoPSignificance))!=0) return; //Bad muon is before OR, before signal

   //====================================================================================================
   auto baselineleptons = baselineElectrons + baselineMuons;
   int nbaselineleptons = baselineleptons.size();

   // start cutflow counter
   int cutflowcounter = 0;
   fill("h_cutflow", cutflowcounter++);
   bool DEBUG = false;

   //====================================================================================================
   // overlap removal
   auto radiusCalcLepton = [] (const AnalysisObject& lepton, const AnalysisObject&) { return std::min(0.4, 0.04 + 10./lepton.Pt()); };
   baselineElectrons = overlapRemoval(baselineElectrons, baselineElectrons, 0.05);
   baselineMuons = overlapRemoval(baselineMuons, baselineElectrons, 0.01);
   baselineJets = overlapRemoval(baselineJets, baselineElectrons, 0.2, NOT(BTag85DL1r)); // DL1dv01 not supported
   baselineElectrons = overlapRemoval(baselineElectrons, baselineJets, radiusCalcLepton);
   baselineJets = overlapRemoval(baselineJets, baselineElectrons, 0.2, NOT(BTag85DL1r)); // DL1dv01 not supported
   baselineMuons = overlapRemoval(baselineMuons, baselineJets, radiusCalcLepton);
   //====================================================================================================
   baselineleptons = baselineElectrons + baselineMuons;
   nbaselineleptons = baselineleptons.size();
   
   //====================================================================================================
   // signal objects
   auto electrons = filterObjects(baselineElectrons, 18, 2.47, EMediumLH|ED0Sigma5|EIsoFCLoose); 
   auto muons     = filterObjects(baselineMuons, 18, 2.5, MuD0Sigma3|MuIsoPflowLoose_VarRad);
   auto jets      = filterObjects(baselineJets, 20, 2.8, JVT59Jet);
   auto bjets     = filterObjects(jets, 20., 2.5, BTag85DL1r); // was btag not used?
   auto leptons   = electrons + muons;

   sortObjectsByPt(leptons);
   sortObjectsByPt(bjets);
   sortObjectsByPt(jets);
   sortObjectsByPt(electrons);
   sortObjectsByPt(muons);   
   
   //====================================================================================================
   int nleptons   = leptons.size();
   int nelectrons = electrons.size();
   int nmuons     = muons.size();
   int njets      = jets.size();
   int nbjets     = bjets.size();

   //====================================================================================================
   // Preselection
   if(nbaselineleptons != 2) return;
   if(nleptons != 2) return;
   if(njets == 0) return;
   if(leptons[0].type() != leptons[1].type()) return;
   if(leptons[0].charge() == leptons[1].charge()) return;
   if(leptons[0].Pt()<27.) return;

   if (DEBUG) {
      std::cout << "nleptons: " << nleptons << std::endl;
      for (auto lep : leptons) { std::cout << "lep: (type:" << lep.type() << "; id: (" << lep.id() << ") --> "; lep.Print(); }
      std::cout << "njets: " << njets << std::endl;
      for (auto jet : jets) { std::cout << "jet: " << jet.Pt(); jet.Print();  }
      std::cout << "met: "; metVec.Print();
   }

   double mll  = (leptons[0]+leptons[1]).M(); 
   double mllj = (leptons[0]+leptons[1]+jets[0]).M(); 

   float l1flav, l2flav;
   if ( leptons[0].type()== AnalysisObjectType::ELECTRON ) l1flav = 11.0;
   else l1flav = 13.0;
   if ( leptons[1].type()== AnalysisObjectType::ELECTRON ) l2flav = 11.0;
   else l2flav = 13.0;

   double jet2pT = 0.0, jet2phi = 0.0, jet2eta = 0.0, jet2E = 0.0;
   if (jets.size()>1) {
     jet2pT = jets[1].Pt();
     jet2phi = jets[1].Phi();
     jet2eta = jets[1].Eta();
     jet2E = jets[1].E();
   }
  
   double jet3pT = 0.0, jet3phi = 0.0, jet3eta = 0.0, jet3E = 0.0;
   if (jets.size()>2) {
     jet3pT = jets[2].Pt();
     jet3phi = jets[2].Phi();
     jet3eta = jets[2].Eta();
     jet3E = jets[2].E();
   }
   
   //====================================================================================================
   // Signal regions
   // TODO, to add:
   // 81 GeV < mll < 101 GeV
   // mllj < 250 GeV

   //====================================================================================================
   // Fill optional ntuple
   //
   ntupVar("mcChannel", event->getMCNumber());
   //ntupVar("xsec", -1.0);
   ntupVar("mcWeight", event->getMCWeights()[0]);
   ntupVar("lep1Flav",l1flav);
   ntupVar("lep2Flav",l2flav);
   ntupVar("lep1Pt",leptons[0].Pt());
   ntupVar("lep1Eta",leptons[0].Eta());
   ntupVar("lep1Phi",leptons[0].Phi());
   ntupVar("lep1Energy",leptons[0].E());
   //ntupVar("lep1Mass",leptons[0].M());
   ntupVar("lep2Pt",leptons[1].Pt());
   ntupVar("lep2Eta",leptons[1].Eta());
   ntupVar("lep2Phi",leptons[1].Phi());
   ntupVar("lep2Energy",leptons[1].E());
   //ntupVar("lep2Mass",leptons[1].M());
   ntupVar("jet1Pt",jets[0].Pt());
   ntupVar("jet1Eta",jets[0].Eta());
   ntupVar("jet1Phi",jets[0].Phi());
   ntupVar("jet1Energy",jets[0].E());
   ntupVar("jet2Pt",jet2pT);
   ntupVar("jet2Eta",jet2phi);
   ntupVar("jet2Phi",jet2eta);
   ntupVar("jet2Energy",jet2E);
   ntupVar("jet3Pt",jet3pT);
   ntupVar("jet3Eta",jet3phi);
   ntupVar("jet3Phi",jet3eta);
   ntupVar("jet3Energy",jet3E);      
   ntupVar("MET", met);
   ntupVar("METPhi", metVec.Phi());
   ntupVar("mll", mll);
   ntupVar("mllj", mllj);
   //
   ntupVar("nleptons",     nleptons);
   ntupVar("nelectrons",   nelectrons);
   ntupVar("nmuons",       nmuons);
   ntupVar("njets",        njets);

   // ntupVar("SusyProcess",  event->getSUSYChannel());
   // ntupVar("DSID",         event->getMCNumber());


   //====================================================================================================
   return;
}

