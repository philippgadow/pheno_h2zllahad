#!/usr/bin/env python
#from dbm.ndbm import library
import os
import math
import argparse
#import tracemalloc
import gc
from re import M

import tensorflow as tf
#from tensorflow import keras
#tf.__version__ = '2.6.2'
import uproot
from keras.layers import Input, Dense, Reshape, BatchNormalization, Dropout, Activation, ReLU, Masking
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras import initializers
from keras.initializers import RandomNormal, RandomUniform, Constant
from keras.constraints import max_norm
from keras.regularizers import l1, l2
import keras.backend as K
import matplotlib
matplotlib.use("Agg") # stops matplotlib crashing when remote disconneted for a while
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler, StandardScaler
import json
import random
import ROOT
from ROOT import TH1F, TCanvas, TFile, TPad
from root_numpy import array2root
import hyperopt
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
from pickle import dump, load


def MakeNN (HPSpace):

  #print(tracemalloc.get_traced_memory()) 
  
  global I_tr, Chi2 , ActThr, Bkg_Signal_DF_Tr, Bkg_Signal_DF_Val
  I_tr=I_tr+1 #---

  N  = HPSpace['N'] #100#
  L  = HPSpace['L'] #5#
  R  = HPSpace['R']
  BS = HPSpace['BS'] #100# 
  
  Layer = []
  Layer.append(Input(shape=(8,))) #8
  
  for i in range(1,int(L+1)):       #11
   #Layer.append(Dense(int(N), activation='relu')(Layer[i-1]))
   Layer.append(Dense(int(N), kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.RandomNormal(), activation='relu')(Layer[i-1]))
  Layer.append(Dense(1, activation='sigmoid')(Layer[int(L)])) #13
  mlp = Model(inputs=Layer[0], outputs=Layer[int(L+1)])
  
  mlp.compile(loss='binary_crossentropy', optimizer="Adam")
 
  
  StopTraining = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.00001, patience=15, verbose=1, restore_best_weights=True)
  hist=mlp.fit(Bkg_Signal_DF_Tr, Bkg_Signal_lab_Tr, sample_weight=weights_Tr, validation_data=(Bkg_Signal_DF_Val, Bkg_Signal_lab_Val, weights_Val), epochs=50, batch_size=int(BS), callbacks=[StopTraining],verbose=1) 
  
  mlp.summary()
  mlp.save("NN_"+str(I_tr)+".h5")
  
  #---------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  fig= plt.figure(figsize=(8,8)) 
  plt.plot(hist.history['loss'],label='Train')
  plt.plot(hist.history['val_loss'], label='Test')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(loc='upper left')
  fig.savefig("Training_curve_"+str(I_tr)+".png")
  plt.close()
  
  weights = mlp.get_weights()
  weightsFlat = np.zeros((0))
  for layer in weights:
   weightsFlat = np.append(weightsFlat, layer.flatten(), axis=0)
  weightsFlat = np.absolute(weightsFlat)
  weightsFlat = np.sort(weightsFlat)
  #print(weightsFlat)
  #print((weightsFlat[np.absolute(weightsFlat)>1.e-4]).shape[0])
  fig = plt.figure(figsize=(8,8))
  plt.hist(weightsFlat, bins=10**(np.arange(-24,7)/2.), range=(1.e-12, 1.e3), density=False, histtype="step", linewidth=1, linestyle="solid", edgecolor="black")
  plt.xscale("log")
  plt.yscale("log")
  plt.xlim(1.e-12, 1.e3)
  # plt.ylim(0.0, 1.0)
  plt.xlabel("Value")
  plt.ylabel("nWeights")
  fig.savefig("Weights_"+str(I_tr)+".png")
  plt.close()  
  
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
  # print ("---- a -----")
  
  # Bkg_DF_Eval = Bkg_Signal_DF_Eval[Bkg_Signal_lab_Eval==0]
  # Bkg_weights_Eval = weights_Eval[Bkg_Signal_lab_Eval==0]
  
  # Signal_DF_Eval = Bkg_Signal_DF_Eval[Bkg_Signal_lab_Eval==1]
  # Signal_weights_Eval = weights_Eval[Bkg_Signal_lab_Eval==1]
  
  # Bkg_Out = mlp.predict(Bkg_DF_Eval, verbose=1)
  # Sig_Out = mlp.predict(Signal_DF_Eval, verbose=1)
  
  
  # HistBkg_NNout=TH1F("","",100,0,1)    
  # HistSig_NNout=TH1F("","",100,0,1)
  
  # for i in range (0,Bkg_Out.size):
      # HistBkg_NNout.Fill(Bkg_Out[i],Bkg_weights_Eval[i])
            
  # for i in range (0,Sig_Out.size):
      # HistSig_NNout.Fill(Sig_Out[i],Signal_weights_Eval[i])
   
  # Canv = TCanvas()
  # HistBkg_NNout.Draw("hist")
  # HistBkg_NNout.SetLineColor(2)
  # HistSig_NNout.Draw("hist same")
  # HistSig_NNout.SetLineColor(3)
  # Canv.SaveAs("NN_out"+str(I_tr)+".root")
  
  
  
  a,b,c = CalculateSignificance("NN_"+str(I_tr)+".h5","Scales_SB.pkl")  
  
  f = open("HypPars2.txt", "a")
  f.write(str(I_tr)+", N: "+str(N)+", L: "+str(L)+", Reg: "+str(R)+", BS: "+str(BS)+", MaxSig:"+str(a)+", C:"+str(b)+", C2:"+str(c)+"\n")
  f.close()
  
  # if C<Chi2:
    # Chi2=c

  #print(tracemalloc.get_traced_memory()) 

  return 10 
   
#---------------------------------------------------------------------------------------------------------------------------------------------------------------   


def CalculateSignificance(NN, Scales):

  global Bkg_Signal_DF_Eval, Data_m, Bkg_m, Bkg_Evs, Signal_Evs

  mlp=load_model(NN)
  scaler = load(open(Scales, 'rb'))

  HistBkg_NN  = TH1F("","",100,0,1) 
  HistData_NN = TH1F("","",100,0,1) 
  HistS05_NN  = TH1F("","",100,0,1) 
  HistS2_NN   = TH1F("","",100,0,1) 
  HistS4_NN   = TH1F("","",100,0,1) 
  
  HistBkg_m  = TH1F("","",80,100,180) 
  HistData_m = TH1F("","",80,100,180) 
  HistS05_m  = TH1F("","",80,100,180) 
  HistS2_m   = TH1F("","",80,100,180) 
  HistS4_m   = TH1F("","",80,100,180) 


  ######Bkg_Signal_DF_Tr, Bkg_Signal_lab_Tr, weights_Tr, Bkg_Signal_DF_Val, Bkg_Signal_lab_Val, weights_Val 
  
  RWN=3.55072e+07/3.37115e+07
  
  Bkg_DF_Eval = Bkg_Signal_DF_Eval[Bkg_Signal_lab_Eval==0]
  Bkg_weights_Eval = weights_Eval[Bkg_Signal_lab_Eval==0]
  Bkg_weights_Eval = Bkg_weights_Eval*(Bkg_Evs/Signal_Evs)*RWN
  
  Signal_DF_Eval = Bkg_Signal_DF_Eval[Bkg_Signal_lab_Eval==1]
  Signal_weights_Eval = weights_Eval[Bkg_Signal_lab_Eval==1]

  
  Bkg_NN = mlp.predict(Bkg_DF_Eval, verbose=1)
  Sig_NN = mlp.predict(Signal_DF_Eval, verbose=1)
  Data_NN = mlp.predict(Data_DF, verbose=1)
  
  Signal_DF_Eval = scaler.inverse_transform(Signal_DF_Eval)


  print(type(Signal_DF_ma),type(Sig_NN))
  
  Sig05_NN = Sig_NN[Signal_DF_ma==0.5]
  Sig05_w = Signal_weights_Eval[Signal_DF_ma==0.5]
  Sig05_m = Signal_m[Signal_DF_ma==0.5]
  Sig2_NN = Sig_NN[Signal_DF_ma==2]
  Sig2_w = Signal_weights_Eval[Signal_DF_ma==2]
  Sig2_m = Signal_m[Signal_DF_ma==2]
  Sig4_NN = Sig_NN[Signal_DF_ma==4]
  Sig4_w = Signal_weights_Eval[Signal_DF_ma==4]
  Sig4_m = Signal_m[Signal_DF_ma==4]
  
  # m05=np.random.choice(SigMasses, Sig05_NN.shape[0]) 
  # m2=np.random.choice(SigMasses, Sig05_NN.shape[0]) 
  # m4=np.random.choice(SigMasses, Sig05_NN.shape[0]) 
  
  for i in range (0,Data_NN.shape[0]):
    if Data_NN[i]>0.9 : 
      continue
    HistData_NN.Fill(Data_NN[i],10)

  for i in range (0,Bkg_NN.shape[0]):
    HistBkg_NN.Fill(Bkg_NN[i], 10.*Bkg_weights_Eval[i])
    
  for i in range (0,Sig05_NN.shape[0]):
    HistS05_NN.Fill(Sig05_NN[i], Sig05_w[i])
    
  for i in range (0,Sig2_NN.shape[0]):
    HistS2_NN.Fill(Sig2_NN[i], Sig2_w[i])
    
  for i in range (0,Sig4_NN.shape[0]):
    HistS4_NN.Fill(Sig4_NN[i], Sig4_w[i])
    
     
  Canvb = TCanvas()
  pad1b = TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
  pad1b.SetTopMargin(0.08)
  pad1b.SetBottomMargin(0.02)
  pad1b.Draw()
  pad1b.cd()
  
  
  HistBkg_NN.Draw("hist")
  HistBkg_NN.SetLineColor(1)
  HistS05_NN.Draw("hist same")
  HistS05_NN.SetLineColor(2)
  HistS2_NN.Draw("hist same")
  HistS2_NN.SetLineColor(3)
  HistS4_NN.Draw("hist same")
  HistS4_NN.SetLineColor(4)
  HistData_NN.Draw("EP same")
  HistData_NN.SetMarkerStyle(20)
  HistData_NN.SetMarkerColor(1)

  pad2b = TPad("pad2","pad2", 0, 0, 1, 0.3)
  pad2b.SetBottomMargin(0.4)
  pad2b.SetGridy()
  pad2b.Draw()
  pad2b.cd()
  
  HistRatiob=HistData_NN.Clone()
  HistRatiob.Divide(HistBkg_NN)
  HistRatiob.Draw("PE same")

  Canvb.SaveAs("NN_out_"+str(I_tr)+".root")
    
  C = HistData_NN.Chi2Test(HistBkg_NN,"UW, CHI2")
       
  Sig = np.empty((0,101), float)
    
  for i in range (0,102):
    #B=HistBkg_NN.Integral(i,102)
    if HistBkg_NN.Integral(i,102)/HistBkg_NN.Integral()<0.01 : 
     MaxSigCut=i
     MaxSig = HistS2_NN.Integral(i,102) 
     break
    # S=HistS2_NN.Integral(i,102)
    # if math.sqrt(B) == 0 :
      # Sig = np.insert(Sig,  i, 0)
    # else :  
      # Sig   = np.insert(Sig,  i, S/math.sqrt(B))
    #print Sig


  
  # MaxSigCut = np.where(Sig == np.amax(Sig))
  # MaxSigCut=np.argmax(Sig)
  # MaxSig = np.amax(Sig)
  
  # print (Sig, type(Sig))
  print (MaxSigCut, MaxSig)
  # print (type(MaxSigCut), type(MaxSig))
  print ("---------------------before-----------------------------") 
  print (Data_m.shape, Data_NN.shape)
  print (Bkg_m.shape, Bkg_NN.shape)
  # print (type(Data_m), type(Data_NN), type(Bkg_m), type(Bkg_NN))
    
  Data_mCut = Data_m[Data_NN[:,0]>MaxSigCut/100.] 
  Bkg_mCut = Bkg_m[Bkg_NN[:,0]>MaxSigCut/100.] 
  Bkg_weights_EvalCut = Bkg_weights_Eval[Bkg_NN[:,0]>MaxSigCut/100.] 

  print ("-----------------------after---------------------------") 
  print (Data_m.shape, Data_NN.shape)
  print (Bkg_m.shape, Bkg_NN.shape)
    
    
  for i in range (0,Data_mCut.shape[0]):
    if Data_mCut[i]>120 and Data_mCut[i]<140 :
      continue
    HistData_m.Fill(Data_mCut[i],10)

  for i in range (0,Bkg_mCut.shape[0]):
   HistBkg_m.Fill(Bkg_mCut[i], 10.*Bkg_weights_EvalCut[i])
   
  for i in range (0,Sig05_m.shape[0]):
   HistS05_m.Fill(Sig05_m[i], Sig05_w[i])

  for i in range (0,Sig2_m.shape[0]):
   HistS2_m.Fill(Sig2_m[i], Sig2_w[i])

  for i in range (0,Sig4_m.shape[0]):
   HistS4_m.Fill(Sig4_m[i], Sig4_w[i])
   
  #print(np.sum(Bkg_weights_Eval), Data_m.shape[0])
    
    
  Canv = TCanvas()
  pad1 = TPad("pad1", "pad1", 0, 0.3, 1, 1.0)
  pad1.SetTopMargin(0.08)
  pad1.SetBottomMargin(0.02)
  pad1.Draw()
  pad1.cd()
  
  
  HistBkg_m.Draw("hist")
  HistBkg_m.SetLineColor(2)
  HistData_m.Draw("EP same")
  HistData_m.SetMarkerStyle(20)
  HistData_m.SetMarkerColor(2)
  HistS05_m.Draw("hist same")
  HistS05_m.SetLineColor(2)
  HistS2_m.Draw("hist same")
  HistS2_m.SetLineColor(3)
  HistS4_m.Draw("hist same")
  HistS4_m.SetLineColor(4)
  
  #Canv.cd();
  pad2 = TPad("pad2","pad2", 0, 0, 1, 0.3)
  pad2.SetBottomMargin(0.4)
  pad2.SetGridy()
  pad2.Draw()
  pad2.cd()
  
  HistRatio=HistData_m.Clone()
  HistRatio.Divide(HistBkg_m)
  HistRatio.Draw("PE same")
  
  #Canv.cd();


  Canv.SaveAs("m_out_"+str(I_tr)+".root")


  C2 = HistData_m.Chi2Test(HistBkg_m,"UW, CHI2")
    
  return MaxSig, C, C2
    


  
if __name__ == "__main__":
   
    varNames = ["GhostTrackVars_deltaRLeadTrack","GhostTrackVars_leadTrackPtRatio","GhostTrackVars_angularity_2","GhostTrackVars_U1_0p7","GhostTrackVars_M2_0p3","GhostTrackVars_tau2","Reg_102","GhostTrackVars_nTracks"]
    TreeName = "treeHZX"
    # path= "/disk/moose/atlas/Panagiotis/NtuplesHDBS3/"
    path= "/disk/moose/atlas/cxw/Regression/RegTest/"
    FullBkg = "TotalBkg2.txt"
    Signal = "Signal.txt"
    Data = "Data2.txt"    
    version = "Sl_Skb_NN"
    Bkg_df = []
    Signal_df = []
    Data_df = []
    
#----------------------------------------------------------------------------------------------------------------------------------------------------
    print ("---------0")  
    SigMasses=[0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    with open(Data) as f3:
     Data_l = [line.rstrip() for line in f3] 
     
    for i in range (0,3): #len(Data_l)
     DataFile = uproot.open(path+version+Data_l[i]) 
     DataTree = DataFile[TreeName]   
     Data_df.append(DataTree.pandas.df())    
     print (Data_l[i], Data_df[i].shape) 
    
    with open(FullBkg) as f:
     Bkg_l = [line.rstrip() for line in f]
    
    for i in range (0,2*len(Bkg_l)/3):  #len(Bkg_l)
     BkgFile = uproot.open(path+version+Bkg_l[i]) 
     BkgTree = BkgFile[TreeName] 
     Bkg_df.append(BkgTree.pandas.df())
     print (Bkg_l[i], Bkg_df[i].shape) 
     
     
    with open(Signal) as f2:
     Signal_l = [line.rstrip() for line in f2] 
     
    for i in range (0,20): #len(Signal_l)
     SignalFile = uproot.open(path+version+Signal_l[i]) 
     SignalTree = SignalFile[TreeName]   
     Signal_df.append(SignalTree.pandas.df())
     Signal_df[i]['mass'] = SigMasses[i]
     
     print (Signal_l[i], Signal_df[i].shape)
     Signal_DF = pd.concat(Signal_df,sort=False)     
     
#----------------------------------------------------------------------------------------------------------------------------------------------------
    global Bkg_Signal_DF_Tr, Bkg_Signal_lab_Tr, weights_Tr, Bkg_Signal_DF_Val, Bkg_Signal_lab_Val, weights_Val, scaler, Bkg_Signal_DF, Bkg_Signal_DF_Eval, Data_m, Bkg_m, Bkg_Evs, Signal_Evs
 
    Signal_DF = pd.concat(Signal_df,sort=False)
    
    Bkg_DF    = pd.concat(Bkg_df, sort=False)
    
    Data_DF   = pd.concat(Data_df,sort=False)
    
    Bkg_DF=Bkg_DF[0::10]
    Data_DF=Data_DF[0::10]

    del Bkg_df
    del Signal_df
    gc.collect()
    print ("---------1") 
    print (Bkg_DF.shape, Signal_DF.shape)
    print ("---------2")
    #print(tracemalloc.get_traced_memory())

    Bkg_DF = Bkg_DF[Bkg_DF["H_m"]<180]
    Signal_DF = Signal_DF[Signal_DF["H_m"]<140]
    Signal_DF = Signal_DF[Signal_DF["H_m"]>120]
    Data_DF = Data_DF[Data_DF["H_m"]<180]

    XS = (4.858E+01+3.782E+00+1.373E+00+8.839E-01+5.071E-01+4.880E-01+7.425E-02+2.879E-03+1.517E-02)*(0.033632+0.033662+0.033696)/1000.

    weightsBackground   =Bkg_DF['Total_Weight'].values
    weightsSignal       =Signal_DF['Total_Weight'].values
    Bkg_RW        = Bkg_DF['RW_3022'].values
    
    Signal_DF_ma=Signal_DF["mass"]
    
    weightsBackground=np.multiply(weightsBackground, np.exp(Bkg_RW))
    #weightsSignal       =np.multiply(weightsSignal, Signal_DF_ma)
    
    
    Bkg_m = Bkg_DF['H_m'].values 
    Data_m = Data_DF['H_m'].values 
    Signal_m = Signal_DF['H_m'].values 

    

    print(type(Signal_DF))
    # Signal_DF_ma_0p5=Signal_DF_ma[Signal_DF_ma["mass"]==0.5]
    # Signal_DF_ma_2=Signal_DF_ma[Signal_DF_ma["mass"]==2]
    # Signal_DF_ma_4=Signal_DF_ma[Signal_DF_ma["mass"]==4]
   
    Signal_DF = Signal_DF[varNames].values
    Bkg_DF    = Bkg_DF[varNames].values
    Data_DF = Data_DF[varNames].values 
       
    
    # for i in range (0,Bkg_DF.shape[0]):
    #    Bkg_DF[i][6] = SigMasses[random.randint(0, 28)]
    
    # Bkg_DF=Bkg_DF[0::10]
    # weightsBackground=weightsBackground[0::10]
    
    Bkg_Signal_DF = np.append(Signal_DF, Bkg_DF, axis=0)

    print ("---------3")
    ##print (Bkg_DF.shape, Signal_DF_2.shape)
    #print(tracemalloc.get_traced_memory())
            
    labelsSignal = np.ones((Signal_DF.shape[0]),dtype='f')
    labelsBackground = np.zeros((Bkg_DF.shape[0]),dtype='f')
    Bkg_Signal_lab = np.append(labelsSignal, labelsBackground, axis=0)
    
    del Signal_DF
    del Bkg_DF
    gc.collect()
    
    #print(tracemalloc.get_traced_memory())

    Bkg_Evs  = np.sum(weightsBackground)
    Signal_Evs = np.sum(weightsSignal) 
    
    print ("--------- Initial Events -------------")
    print (Bkg_Evs, Signal_Evs, Data_DF.shape[0])    
        
    weights = np.append(weightsSignal, weightsBackground*Signal_Evs/Bkg_Evs, axis=0)
    
    
    del weightsSignal
    del weightsBackground
    gc.collect()
    
    
       
    print ("---------")
    print ("---------4")
    #print(tracemalloc.get_traced_memory())
          
    scaler = RobustScaler()
    Bkg_Signal_DF = scaler.fit_transform(Bkg_Signal_DF)
    Data_DF = scaler.transform(Data_DF)
      
    outputs = {"input_vars": varNames, "center": list([float(center) for center in scaler.center_]), "scale": list(scaler.scale_) }
    
    print ("---------5")
    #print(tracemalloc.get_traced_memory())
    
    #with open("Scales.json", "w") as outf: json.dump(outputs, outf, indent=4)
    
    dump(scaler, open("Scales_SB.pkl", 'wb'))
    print (scaler.center_)
    print (scaler.scale_)

    Bkg_Signal_DF = np.float16(Bkg_Signal_DF)
    weights = np.float16(weights)
    
    Bkg_Signal_DF_Tr   = Bkg_Signal_DF[1::2]
    Bkg_Signal_lab_Tr  = Bkg_Signal_lab[1::2]
    weights_Tr       = weights[1::2]
    
    Bkg_Signal_DF_Val  = Bkg_Signal_DF[0::2]
    Bkg_Signal_lab_Val = Bkg_Signal_lab[0::2]
    weights_Val      = weights[0::2] 
    
    Bkg_Signal_DF_Eval  =  Bkg_Signal_DF  # Bkg_Signal_DF[0::100]
    Bkg_Signal_lab_Eval =  Bkg_Signal_lab #Bkg_Signal_lab[0::100]
    weights_Eval      =  weights      # weights[0::100]
    
    del Bkg_Signal_DF
    del Bkg_Signal_lab
    del weights
    
    gc.collect()
    #print(tracemalloc.get_traced_memory())

    Bkg_Signal_DF_Tr, Bkg_Signal_lab_Tr, weights_Tr = shuffle(Bkg_Signal_DF_Tr, Bkg_Signal_lab_Tr, weights_Tr)
    
    print ("---------6")
    #print(tracemalloc.get_traced_memory())
    
#----------------------------------------------------------------------------------------------------------------------------------------------------
 
    #MakeNN(0)
    # CalculateSignificance("NN_"+str(I_tr)+".h5","Scales_SB.pkl")
    # a=b
    
#----------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------
    I_tr=250
    Chi2=0
    ev=30

    
    HPSpace = {
     'N'  :hp.quniform('N', 10, 20, 5),
     'L'  :hp.quniform('L', 3, 6, 1),
     'R'  :hp.choice('R', [1e-8, 1e-9, 1e-10]),
     'BS' :hp.choice('BS', [50, 100, 200, 500])}
       
    Al_trials = Trials()
    best_param = fmin(fn=MakeNN, space=HPSpace, algo=tpe.suggest, max_evals=ev, trials=Al_trials, verbose=0)

    print ("----------------------- Done ----------------------------------------")

    #tracemalloc.stop()
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
