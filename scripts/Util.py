import awkward as ak
import numpy as np
import vector
vector.register_awkward()


def getMomentumArrayFromObjects(raw_data,obj_tagList=["Photon"],extraColums=[]):
    allColums=['pt','eta','phi','mass']+extraColums
    objects = ak.zip({
        obj_tag: ak.zip({
             ky : raw_data[f"{obj_tag}_{ky}"]  for ky in allColums if f"{obj_tag}_{ky}"  in raw_data.fields
        }, with_name="Momentum4D")  for obj_tag in obj_tagList
      },
    depth_limit=1)
    return objects

def getObjectAsArray(events,object_tag="Photon",colums=["pt","eta","phi","mass"],nameTag="Momentum4D"):
    objectArr=[ 
                ak.zip({var : events[f"{object_tag}_{i}_{var}"]  for var in colums},with_name=nameTag)
                 for i in range(1,3+1)
            ]
    objectArr=[ ak.unflatten(objArr,counts=1,axis=-1) for objArr in objectArr]
    objectArr= ak.concatenate(objectArr,axis=1)
    return objectArr    
