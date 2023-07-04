import awkward as ak
import numpy as np
import vector
import matplotlib.pyplot as plt
import mplhep as hep
vector.register_awkward()
import json,glob,argparse,os,time
import Util as utl
import tracemalloc

def exportCleanedEvents(inputFileName,oFname=None):
    
    print("Processing the input file ",inputFileName)
    inputFSize=os.path.getsize(inputFileName)   / 1e6
    print("           the input file size is : ",inputFSize)
    
    all_added_fields=[]
    
    photon_fields=["Diphoton","LeadPhoton","SubleadPhoton"]
    all_added_fields+=photon_fields
    
    events=ak.from_parquet(inputFileName)

    photon_objects=utl.getMomentumArrayFromObjects(events,photon_fields,extraColums=["dR","mvaID","genPartFlav","pixelSeed"])

    jetVarList=[]
    for ky in events.fields:
        if ky.startswith('jet_1'):
            jetVarList.append(ky.replace('jet_1_',''))
    print("\tJet Vars = ",len(jetVarList))
    fatJetVarList=[]
    for ky in events.fields:
        if 'fatjet_1' in ky:
    #         print(ky)
            fatJetVarList.append(ky.replace('fatjet_1_',''))
    print("\tFat Jet Vars = ",len(fatJetVarList))
    
    print("\tLoading Fat Jet vars !")
    fatJets=utl.getObjectAsArray(events,"fatjet",colums=fatJetVarList)
    fatJets_cleaned=fatJets[fatJets.pt >0]
    all_added_fields+=["fatjet"]
    
    print("\tLoading Jet vars !")
    ak4Jets=utl.getObjectAsArray(events,"jet",colums=jetVarList)
    ak4Jets_cleaned=ak4Jets[ak4Jets.pt > 0 ]
    all_added_fields+=["jet"]

    hasGen=False
    for fld in events.fields:
        if fld.startswith("GenHggHiggs"):
            hasGen=True
    if hasGen:
        gen_fields=["GenHggHiggs","GenHbbHiggs_1","GenHbbHiggs_2",
                            "GenHggLeadPhoton","GenHggSubleadPhoton",
                            "GenHbbLeadB_1","GenHbbLeadB_2",
                            "GenHbbSubleadB_1","GenHbbSubleadB_2"]
        gen_objects=utl.getMomentumArrayFromObjects(events,gen_fields,extraColums=["dR"])
        all_added_fields+=gen_fields
   

    print("\tSetting the cleaned events !")
    events_cleaned= ak.Array({"event" : events.event})
    events_cleaned["Jet"]=ak4Jets_cleaned
    events_cleaned["FatJet"]=fatJets_cleaned

    for fld in photon_objects.fields:
        events_cleaned[fld]=photon_objects[fld]
    
    if hasGen:
        for fld in gen_objects.fields:
            events_cleaned[fld]=gen_objects[fld]
    
    for fld in events.fields:
        isCopied=False
        for ky in all_added_fields:
            if fld.startswith(ky):
                isCopied=True
        if  isCopied:
            continue
        events_cleaned[fld]=events[fld]
    
    if oFname:
        print(f"\tExporting to {oFname}")
        ak.to_parquet(events_cleaned,oFname)
        time.sleep(1)
        outputFSize=os.path.getsize(oFname)   / 1e6
        print(f"\t\t\t[{inputFSize:.3f} -> {outputFSize:.3f} ,{ (inputFSize-outputFSize)*100.0 / inputFSize:2f} % reduction]")
    else:
        return events_cleaned


def main():
    tracemalloc.start()
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--version", help="Base trigger set",default='v1')
    parser.add_argument(     "--cfg"    , help="Input configurateion")
    
    args = parser.parse_args()
    
    config={}
    with open(args.cfg) as f:
        config=json.load(f)
    if len(config["inputFiles"])!=len(config["destination"]):
        print("Number of input tragets and their destionation are not same ! exiting")
        exit(40001)

    for inFileName, oDir in zip(config["inputFiles"],config["destination"]):
        flist=glob.glob(inFileName)
        os.system("mkdir -p "+oDir)
        for fl in flist:
            fnme=fl.split('/')[-1]
            fout=oDir+'/cleaned_'+fnme
            if os.path.exists(fout):
                print(f"{fout} already exists ! skipping the cleaning :) ")
                continue
            mem=tracemalloc.get_traced_memory() ; print(f"\t*** Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
            exportCleanedEvents(fl,fout)    
            mem=tracemalloc.get_traced_memory() ; print(f"\t*** Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
    
    tracemalloc.stop()

if __name__=='__main__':
    main()
