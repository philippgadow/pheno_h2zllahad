#include "SimpleAnalysisFramework/AnalysisClass.h"

#include <TFile.h>
#include <TH2.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>

#include <algorithm>

// SimpleAnalysis implementation for HZ(ll)a(had) analysis (HDBS-2021-09)

DefineAnalysis(HZa_2018)

void HZa2018::Init() {
    addRegions({"Inclusive"});

    // Jet Histograms
    addHistogram("hist_jet1pt", 100, 0, 2000);
    addHistogram("hist_jet2pt", 100, 0, 2000);
    addHistogram("hist_jet3pt", 100, 0, 2000);
    addHistogram("hist_jet4pt", 100, 0, 2000);

    // Electron and Muon Histograms
    addHistogram("hist_electron_pt", 100, 0, 500);
    addHistogram("hist_muon_pt", 100, 0, 500);

    // Z Candidate Mass
    addHistogram("hist_z_mass", 60, 60, 120);

    // Three-body Mass
    addHistogram("hist_mllj", 100, 0, 300);
}

void HZa2018::ProcessEvent(AnalysisEvent *event) { 

    // Jets with baseline kinematics
    auto jets = event->getJets(20.0, 2.5);
    auto muons = event->getMuons(18.0, 2.7, MuMedium);
    auto electrons = event->getElectrons(18.0, 2.47, EMediumBLLH);

    // Exclude electrons in the transition region
    electrons.erase(std::remove_if(electrons.begin(), electrons.end(), [](const AnalysisObject& el) {
        return (fabs(el.Eta()) > 1.37 && fabs(el.Eta()) < 1.52);
    }), electrons.end());

    // Fill muon and electron pT histograms
    for (const auto& mu : muons) fill("hist_muon_pt", mu.Pt());
    for (const auto& el : electrons) fill("hist_electron_pt", el.Pt());

    // Merge leptons into one vector
    auto leptons = electrons;
    leptons.insert(leptons.end(), muons.begin(), muons.end());

    // Require at least 2 leptons
    if (leptons.size() < 2) return;

    // Select best SFOS lepton pair closest to Z mass
    double best_mll = -999;
    size_t idx1 = 0, idx2 = 1;
    bool foundPair = false;
    for (size_t i = 0; i < leptons.size(); ++i) {
        for (size_t j = i+1; j < leptons.size(); ++j) {
            if (leptons[i].Charge() * leptons[j].Charge() >= 0) continue;
            if (abs(leptons[i].PdgId()) != abs(leptons[j].PdgId())) continue;
            double mll = (leptons[i].P4() + leptons[j].P4()).M();
            if (!foundPair || fabs(mll - 91.1876) < fabs(best_mll - 91.1876)) {
                best_mll = mll;
                idx1 = i;
                idx2 = j;
                foundPair = true;
            }
        }
    }

    if (!foundPair || best_mll < 81.0 || best_mll > 101.0) return;
    if (std::max(leptons[idx1].Pt(), leptons[idx2].Pt()) < 27.0) return;

    // Fill Z candidate mass histogram
    fill("hist_z_mass", best_mll);

    // Jet cleaning and JVT cut
    jets.erase(std::remove_if(jets.begin(), jets.end(), [](const AnalysisObject& jet) {
        if (!jet.pass(JetCleaning)) return true;
        if (jet.Pt() < 60.0 && fabs(jet.Eta()) < 2.4 && jet.JVT() <= 0.59) return true;
        return false;
    }), jets.end());

    if (jets.empty()) return;

    // Fill jet pT histograms
    if (jets.size() >= 1) fill("hist_jet1pt", jets[0].Pt());
    if (jets.size() >= 2) fill("hist_jet2pt", jets[1].Pt());
    if (jets.size() >= 3) fill("hist_jet3pt", jets[2].Pt());
    if (jets.size() >= 4) fill("hist_jet4pt", jets[3].Pt());

    // Form three-body system mass
    auto dilepton_p4 = leptons[idx1].P4() + leptons[idx2].P4();
    auto jet_p4 = jets[0].P4();
    double mllj = (dilepton_p4 + jet_p4).M();

    if (mllj < 50.0 || mllj > 250.0) return;

    // Fill three-body mass histogram
    fill("hist_mllj", mllj);


    // NN classifier placeholder
    // double nn_output = computeNNOutput(jets[0]); // Replace with actual NN call
    // if (nn_output <= 0.93) return;

    // // Accept event into Inclusive region
    // accept("Inclusive");
}
