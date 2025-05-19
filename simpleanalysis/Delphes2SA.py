#!/usr/bin/env python

# Example for converting Delphes output to SimpleAnalysis slimmed format
# lepton id/isolation and flavour tagging levels are not implemented
#
# Assumes Delphes and ROOT is setup already 
#

from array import array
import sys

import ROOT

if len(sys.argv) < 3:
  print(" Usage: Delphes2SA.py <input Delphes file> <Output SA file>")
  sys.exit(1)

ROOT.gSystem.Load("libDelphes")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass

class NtupleVar:
  def __init__(self, name, tree, floatVar=False):
    if floatVar:
      rootVar = array('f',[0])
      tree.Branch(name,rootVar, name+'/F')
    else:
      rootVar = array('i',[0])
      tree.Branch(name,rootVar, name+'/I')
    self.var = rootVar
  
  def Set(self,value):
    self.var[0] = value

outVectors=[]

class NtupleVector:
  def __init__(self, name, tree, floatVar=False):
    global outVectors
    if floatVar:
      rootVar = ROOT.std.vector('float')()
    else:
      rootVar = ROOT.std.vector('int')()
    tree.Branch(name, rootVar)
    self.var = rootVar  
    outVectors.append(rootVar)

  def Add(self,value):
    self.var.push_back(value)

class ObjectVector:
  def __init__(self, name, tree, storeMass=False):
    self.pt = NtupleVector(name+"_pt", tree, True)
    self.eta = NtupleVector(name+"_eta", tree, True)
    self.phi = NtupleVector(name+"_phi", tree, True)
    self.charge = NtupleVector(name+"_charge", tree)
    self.objID = NtupleVector(name+"_id", tree)
    self.motherID = NtupleVector(name+"_motherID", tree)
    self.storeMass = storeMass
    if storeMass:
      self.mass = NtupleVector(name+"_m", tree, True)
    
  def Add(self,obj,objID=0x7FFFFFFF,charge=999):
    self.pt.Add(obj.PT)
    self.eta.Add(obj.Eta)
    self.phi.Add(obj.Phi)
    if charge==999:
      self.charge.Add(obj.Charge)
    else:
      self.charge.Add(charge)
    self.objID.Add(objID)
    self.motherID.Add(0) #FIXME truth pointing not implemented
    if self.storeMass:
      self.mass.Add(obj.Mass)

inputFile = sys.argv[1]
outputFile = sys.argv[2]

# Create chain of root trees
chain = ROOT.TChain("Delphes")
chain.Add(inputFile)

# Create object of class ExRootTreeReader
treeReader = ROOT.ExRootTreeReader(chain)
numberOfEntries = treeReader.GetEntries()

# Get pointers to branches used in this analysis
branchEvent = treeReader.UseBranch("Event")
branchMET = treeReader.UseBranch("MissingET")
branchHT = treeReader.UseBranch("ScalarHT")

branchPhoton = treeReader.UseBranch("Photon")
branchElectron = treeReader.UseBranch("Electron")
branchMuon = treeReader.UseBranch("Muon")
branchJet = treeReader.UseBranch("Jet")
branchFatJet = treeReader.UseBranch("FatJet")

# Output file and tree
outFH = ROOT.TFile(outputFile, "RECREATE")
outTree = ROOT.TTree("ntuple", "Simple Analysis slim format from delphes")
outTree.SetDirectory(outFH)

EventNumber = NtupleVar("Event", outTree)
mcChannel = NtupleVar("mcChannel", outTree)
mcVetoCode = NtupleVar("mcVetoCode", outTree)
susyChannel = NtupleVar("susyChannel", outTree)
mcWeights = NtupleVector("mcWeights", outTree, True)
genMET = NtupleVar("genMET", outTree, True)
genHT = NtupleVar("genHT", outTree, True)
pdf_id1 = NtupleVar("pdf_id1", outTree)
pdf_x1 = NtupleVar("pdf_x1", outTree, True)
pdf_pdf1 = NtupleVar("pdf_pdf1", outTree, True)
pdf_id2 = NtupleVar("pdf_id2", outTree)
pdf_x2 = NtupleVar("pdf_x2", outTree, True)
pdf_pdf2 = NtupleVar("pdf_pdf2", outTree, True)
pdf_scale = NtupleVar("pdf_scale", outTree, True)

sumet = NtupleVar("sumet", outTree, True)
met_pt = NtupleVar("met_pt", outTree, True)
met_phi = NtupleVar("met_phi", outTree, True)

electrons = ObjectVector("el", outTree)
muons = ObjectVector("mu", outTree)
taus = ObjectVector("tau", outTree)
photons = ObjectVector("ph", outTree)
jets =  ObjectVector("jet", outTree, True)
fatjets =  ObjectVector("fatjet", outTree, True)
# not supporting hard-scatter truth record for now

# Loop over all events
for entry in range(0, numberOfEntries):
  for vec in outVectors:
    vec.clear()

  # Load selected branches with data from specified event
  treeReader.ReadEntry(entry)

  # Fill in event info (some is left at default 0)
  EventNumber.Set(branchEvent.At(0).Number)
  mcChannel.Set(branchEvent.At(0).ProcessID)
  mcWeights.Add(branchEvent.At(0).Weight)
  # FIXME: add PDF info etc. if available

  # MET/HT
  sumet.Set(branchHT.At(0).HT)
  met_pt.Set(branchMET.At(0).MET)
  met_phi.Set(branchMET.At(0).Phi)

  # object id not implemented for now 
  for idx in range(branchElectron.GetEntries()):
    electrons.Add(branchElectron.At(idx))

  for idx in range(branchMuon.GetEntries()):
    muons.Add(branchMuon.At(idx))

  for idx in range(branchPhoton.GetEntries()):
    photons.Add(branchPhoton.At(idx),charge=0)

  # If event contains at least 1 jet
  for idx in range(branchJet.GetEntries()):
    jet=branchJet.At(idx)
    jetID=0x000FDF00  # flag as good jet
    if jet.BTag:
      jetID|=0x00F000FF # flags as bjet
    jets.Add(jet,jetID)
    if jet.TauTag:
      tauID=0xFF
      if jet.NCharged==1:
        tauID|=1<<10
      if jet.NCharged==3:
        tauID|=1<<11
      taus.Add(jet,tauID)

  if branchFatJet:
    for idx in range(branchFatJet.GetEntries()):
      jet=branchFatJet.At(idx)
      jetID=0x000FDF00  # flag as good jet
      if jet.BTag:
        jetID|=0x00F000FF # flags as bjet
      fatjets.Add(jet,jetID)

  outTree.Fill()
outFH.Write()
outFH.Close()
