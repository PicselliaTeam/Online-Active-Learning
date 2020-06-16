import json
import os 


def get_diff(path_old, path_new):

    with open(path_old, "r") as f0:
        dic_old = json.load(f0)
    with open(path_new, "r") as f1:
        dic_new = json.load(f1)


    assert dic_old.keys() == dic_new.keys(), "dict are not the same :("

    diff = {}
    diff["categories"] = dic_new["categories"]
    diff["infos"]=dic_new["infos"]
    diff["images"]=dic_new["images"]
    diff["annotations"]= []
    for ann in dic_new["annotations"]:
        if ann in dic_old["annotations"]:
            continue
        else:
            diff["annotations"].append(ann)
    
    return diff 