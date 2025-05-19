#include "SimpleAnalysisFramework/AnalysisClass.h"

#include <TFile.h>
#include <TH2.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>

#include <algorithm>

// SimpleAnalysis implementation for HZ(ll)a(had) analysis (HDBS-2021-09)

DefineAnalysis(HZa2018)

void HZa2018::Init() {
    addRegions({"Inclusive"});
    addRegions({"SR"});

    // Jet Histograms
    addHistogram("hist_jet1pt", 100, 0, 2000);
    addHistogram("hist_jet2pt", 100, 0, 2000);
    addHistogram("hist_jet3pt", 100, 0, 2000);
    addHistogram("hist_jet4pt", 100, 0, 2000);

    // Electron and Muon Histograms
    addHistogram("hist_electron_pt", 100, 0, 500);
    addHistogram("hist_muon_pt", 100, 0, 500);

    // Leading Jet Mass
    addHistogram("hist_jet1mass", 40, 0, 20);

    // Z Candidate Mass
    addHistogram("hist_z_mass", 60, 60, 120);

    // Three-body Mass
    addHistogram("hist_mllj", 100, 0, 300);
}

void HZa2018::ProcessEvent(AnalysisEvent *event) { 

    // Accept event into Inclusive region
    accept("Inclusive");

    // Jets with baseline kinematics
    auto jets = event->getJets(20.0, 2.5);
    auto muons = event->getMuons(18.0, 2.7, MuMedium);
    auto electrons = event->getElectrons(18.0, 2.47, EMediumLH);

    // Exclude electrons in the transition region
    // electrons.erase(std::remove_if(electrons.begin(), electrons.end(), [](const AnalysisObject& el) {
    //     return (fabs(el.Eta()) > 1.37 && fabs(el.Eta()) < 1.52);
    // }), electrons.end());

    // Fill muon and electron pT histograms
    for (const auto& mu : muons) fill("hist_muon_pt", mu.Pt());
    for (const auto& el : electrons) fill("hist_electron_pt", el.Pt());

    // Require at least 2 leptons
    if (!((muons.size() >= 2) || (electrons.size() >= 2))) return;

    // Select best SFOS lepton pair closest to Z mass
    double best_mll = -999;
    size_t idx1 = 0, idx2 = 1;
    bool foundPairMuons = false;
    bool foundPairElectrons = false;
    for (size_t i = 0; i < muons.size(); ++i) {
        for (size_t j = i+1; j < muons.size(); ++j) {
            if (muons[i].charge() * muons[j].charge() >= 0) continue;
            double mll = (muons[i] + muons[j]).M();
            if (!foundPairMuons || fabs(mll - 91.1876) < fabs(best_mll - 91.1876)) {
                best_mll = mll;
                idx1 = i;
                idx2 = j;
                foundPairMuons = true;
            }
        }
    }
    for (size_t i = 0; i < electrons.size(); ++i) {
        for (size_t j = i+1; j < electrons.size(); ++j) {
            if (electrons[i].charge() * electrons[j].charge() >= 0) continue;
            double mll = (electrons[i] + electrons[j]).M();
            if (!foundPairElectrons || fabs(mll - 91.1876) < fabs(best_mll - 91.1876)) {
                best_mll = mll;
                idx1 = i;
                idx2 = j;
                foundPairElectrons = true;
            }
        }
    }

    if (!(foundPairMuons || foundPairElectrons) || best_mll < 81.0 || best_mll > 101.0) return;

    // Merge leptons into one vector
    auto leptons = foundPairElectrons ? electrons : muons;
    if (std::max(leptons[idx1].Pt(), leptons[idx2].Pt()) < 27.0) return;

    // Fill Z candidate mass histogram
    fill("hist_z_mass", best_mll);

    // Jet cleaning and JVT cut
    // jets.erase(std::remove_if(jets.begin(), jets.end(), [](const AnalysisObject& jet) {
    //     // if (!jet.pass(JetCleaning)) return true;
    //     if (jet.Pt() < 60.0 && fabs(jet.Eta()) < 2.4 && jet.JVT() <= 0.59) return true;
    //     return false;
    // }), jets.end());

    if (jets.empty()) return;

    // Fill jet pT histograms
    if (jets.size() >= 1) fill("hist_jet1pt", jets[0].Pt());
    if (jets.size() >= 2) fill("hist_jet2pt", jets[1].Pt());
    if (jets.size() >= 3) fill("hist_jet3pt", jets[2].Pt());
    if (jets.size() >= 4) fill("hist_jet4pt", jets[3].Pt());

    // Form three-body system mass
    auto dilepton_p4 = leptons[idx1] + leptons[idx2];
    auto jet_p4 = jets[0];
    double mllj = (dilepton_p4 + jet_p4).M();

    if (mllj < 50.0 || mllj > 250.0) return;

    // Fill three-body mass histogram
    fill("hist_mllj", mllj);

    // Fill leading jet mass histogram
    if (jets.size() >= 1) fill("hist_jet1mass", jets[0].M());


    // NN classifier placeholder
    // double nn_output = computeNNOutput(jets[0]); // Replace with actual NN call
    // if (nn_output <= 0.93) return;

    // Accept event into signal region (SR)
    accept("SR");
}
