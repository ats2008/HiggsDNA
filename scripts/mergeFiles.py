import tracemalloc

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


import awkward
import json, argparse,copy
import sys,os,glob
import numpy as np
from higgs_dna.constants import NOMINAL_TAG, CENTRAL_WEIGHT, BRANCHES
from higgs_dna.utils import awkward_utils

lumiMap={'2018':58.0,'2017':41.5,'2016':36.3 , '2016PreVFP':19.5,'2016PostVFP':16.8,'run2':137.61}

DEFAULT_DICT={  
        "parameters" : {
                "ofileSizeMax" : 150 ,
                "tagsToProcess" : []
            },
        "is_data" : False,
        "year" : "2018",
        "output_dir"  : "./",
        "output_search_str" : "",
        "outputs"  : [],
        "summary_base" : "/eos/user/a/athachay/HiggsDNA/v7/ggHHH_M125_2018/",
        "summaries" : [ ]
}

def merge_outputs(configs):
    """
    Merge output files from all completed jobs into a single file per systematic with independent collection.
    If MC, we also scale the central weight by scale1fb and luminosity, where scale1fb is calculated
    from the sum of weights of completed jobs. We also add a branch CENTRAL_WEIGHT + "_no_lumi" which has just
    scale1fb applied and is not scaled by luminosity.
    """
    tracemalloc.start()
    
    merged_outputs = {}
    logger.info("Adding  a log")
    ofileSizeMax=150
    idx=0
    if 'ofileSizeMax' in configs["parameters"]:
        ofileSizeMax=configs["parameters"]['ofileSizeMax']
        print("Setting the ofileSizeMax as ",ofileSizeMax," MB" )
    for syst_tag in configs["parameters"]["tagsToProcess"]:
        idx+=1
        lumi=1.0
        print()
        print("== "*10,f" Processing {syst_tag} [ {idx} / {len(configs['parameters']['tagsToProcess'])}]"," =="*10)
        config=copy.deepcopy(DEFAULT_DICT)
        config.update(configs[syst_tag])
        outputs=config['outputs']
        output_search_str=config['output_search_str']
        if len(output_search_str) > 0:
            print(f"Making out flileList from the base string {output_search_str}")
            outputs=glob.glob(output_search_str)
            print(f"\tObtained {len(outputs)} files to merge")
            #print(f"Output files identified as : {outputs}")
        summaries=config['summaries']
        summaries=[]
        if not outputs:
            continue
        MissedFiles=[]
        outputs_toMerge=[]
        for output in  outputs:
            jobFolder=output.split('/')[-2]
            #print(f"checking {config['summary_base']+'/'+jobFolder+'/*summary*.json'}")
            summarySearchString=config["summary_base"]+'/'+jobFolder+'/*summary*.json'
            summary_name=glob.glob(summarySearchString)
            #print(f"Sumary for job {output} found at \n\t{summary_name}")
            if len(summary_name) < 1:
                print(f" Summary not found for  : {output} looking at {summarySearchString}")
                MissedFiles.append(output)
                continue
            outputs_toMerge.append(output)
            summaries.append(summary_name[0])
        print(f"\tObtained {len(outputs)} summary files . skipping {len(MissedFiles)} files")
        merged_outputs[syst_tag] = []
        filedata=[[]]
        totFsize=0
        oFsize=0
        sumWeights=1e-12
        for output,summary in zip(outputs_toMerge,summaries):
            fSize=os.path.getsize(output)*1.0 / 1e6
            jobData={}
            with open(summary,'r') as f:
                jobData=json.load(f)
            #print(json.dumps(jobData,indent=4))
            filedata[-1].append(
                {   'file' : output,'size' : fSize,'sumWeights': jobData['sum_weights'] }
            )
            sumWeights+=jobData['sum_weights']
            print(f"DEBUG | ",f"{sumWeights=}  [ {jobData['sum_weights']=}] ")
            oFsize+=fSize
            totFsize+=oFsize
            if oFsize>ofileSizeMax:
                filedata.append([])
                oFsize=0
        print(f"DEBUG  | ",f"{lumi=}")
        if not config["is_data"]:
            with open(summaries[0],'r') as f:
                jobData=json.load(f)
            #print(json.dumps(jobData,indent=4))
            scale1fb = jobData["config"]['sample']['bf']*jobData["config"]['sample']['xs']*1000/sumWeights
            lumi = jobData["config"]['sample']['lumi']
            print(f"\tSetting {lumi=} , br : {jobData['config']['sample']['bf']} , xs : {jobData['config']['sample']['xs']},-> {scale1fb = } [  sum weights = {sumWeights} ] ")
        
        print(f"Set values :  {lumi=} , br : {jobData['config']['sample']['bf']} , {scale1fb = }")

        print(f"Merging a total of  {totFsize} MB of data")
        # FIXME : merging could be improved so that we avoid merging huge numbers of events into a single file and instead split them across multiple files
        nFiles=len(filedata)
        if filedata[-1]==[]:
            nFiles-=1
        print("[Task : merge_outputs] Task '%s' : merging %d outputs into  '%s' files." % ( syst_tag, len(outputs), nFiles ) )
        logger.debug("[Task : merge_outputs] Task '%s' : merging %d outputs into  '%s' files." % ( syst_tag, len(outputs), nFiles ) )
        mem=tracemalloc.get_traced_memory() ; print(f"\t Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
        k=0
        if not os.path.exists(config['output_dir']):
            print(f"Making directory {config['output_dir']}")
            os.system('mkdir -p '+config['output_dir'])
        for ofileData in filedata:
            k+=1
            merged_output = config['output_dir'] + "/merged_%s_part%s.parquet" % (syst_tag,k) 
            if os.path.exists(merged_output):
                print(f"\t Skipping {merged_output} since file already exists ! ")
                #continue
            mem=tracemalloc.get_traced_memory() ; print(f"\t Current [MB]  : {mem[0]/1e6} , Peak : {mem[1]/1e6} ")
            merge_outputs=[]
            fSize=0
            for fData in ofileData:
                fSize+=fData['size']
            if fSize==0:
                continue
            print(f"  [{k}] Merging {len(ofileData)} , totaling {fSize} MB")
            
            merged_events = []

            #################################
            
            #for fData in ofileData:
            #    merged_events.append(awkward.from_parquet(fData['file']))
            
            for fData in ofileData:
                merged_events.append(awkward.from_parquet(fData['file'] , columns=[CENTRAL_WEIGHT]))
                print(f"DEBUG | ","for a file  central_weight[:10]" , merged_events[-1][CENTRAL_WEIGHT][:10])

            merged_events = awkward.concatenate(merged_events)
            print(f"DEBUG  | ","{scale1fb=} , {lumi=} , {sumWeights =} ","sum of event weights : ",np.sum(merged_events[CENTRAL_WEIGHT]))
            central_weight = merged_events[CENTRAL_WEIGHT] * scale1fb * lumi
            print(f"DEBUG  | ","pre c_weight = ",merged_events[CENTRAL_WEIGHT][:10] )
            print(f"DEBUG  | ","Scale*lumi*c_weight = ",merged_events[CENTRAL_WEIGHT][:10] * scale1fb * lumi)
            print(f"DEBUG  | ","c_weight = ",central_weight[:10] )
        
            break
            continue
            
            #################################

            mem=tracemalloc.get_traced_memory() ; print(f"\tPost file Open Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
            print("\t Concatinating ! ")
            merged_events = awkward.concatenate(merged_events)
            mem=tracemalloc.get_traced_memory() ; print(f"\tPost Concat Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
            
            if not config["is_data"]:
                logger.debug("[Task : merge_outputs] Task '%s' : Applying scale1fb and lumi. Scaling central weight branch '%s' in output file '%s' by scale1fb (%.9f) times lumi (%.2f). Adding branch '%s' in output file which has no lumi scaling applied." % (syst_tag, CENTRAL_WEIGHT, merged_output, scale1fb, lumi, CENTRAL_WEIGHT + "_no_lumi"))
                #logger.debug(f"[Task : sumWeight is {self.phys_summary['sum_weights']}]")
                #logger.debug(f"[Task : typical central weight is {merged_events['weight_central']}]")

                central_weight = merged_events[CENTRAL_WEIGHT] * scale1fb * lumi
                central_weight_no_lumi = merged_events[CENTRAL_WEIGHT] * scale1fb

                awkward_utils.add_field(
                        events = merged_events,
                        name = CENTRAL_WEIGHT,
                        data = central_weight,
                        overwrite = True
                )
                awkward_utils.add_field(
                        events = merged_events,
                        name = CENTRAL_WEIGHT + "_no_lumi",
                        data = central_weight_no_lumi,
                        overwrite = False # merging and scale1fb application should always be done from unmerged files, which should not have this branch already. If they somehow do have the branch, that is a bad sign something is going wrong...
                )

            mem=tracemalloc.get_traced_memory() ; print(f"\tPost FiledAdd Current [MB]  : {mem[0]*1.0/1e6} , Peak : {mem[1]*1.0/1e6} ")
            awkward.to_parquet(merged_events, merged_output)
            print("\t File saved to : ",merged_output)
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--version", help="Version of the specific derivation ",default='')
    parser.add_argument('-i',"--inputFile", help="Input File")
    args = parser.parse_args()
    config={}
    with open(args.inputFile,'r')  as f:
        config=json.load(f)
    #print(json.dumps(config,indent=4))
    merge_outputs(config)

if __name__=='__main__':
    main()
