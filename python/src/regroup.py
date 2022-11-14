
import numpy as np
from glob import glob
import pandas as pd

files = glob("*.npz")
dic = {
    "dataset" : [],
    "method" : [],
    "chunks" : [],
    "IoU" : [],
    "IoU_mc": []
}


for f in files:
    npz = np.load(f)
    OT = "OT" in f
    if OT:
        name = f.split("_")[0]
        method = "OT"
        IoU_str_name = "IoU"
    else:
        method, name = f.split("_")[:2]
        IoU_str_name = "IoU_bin"


    if "Chunks" in f:
        name = name[6:]
        for i in range(4):
            try:
                chunks = int(name[:i])
            except:
                break
        dataname = name[i:]
    else:
        if "tmp" in name:
            dataname = name[3:]
        chunks = 0
    
    
    IoU_bin = float(npz["IoU_str_name"])

    if OT:
        IoU_mc = 0.0
    else:
        IoU_mc = float(npz["IoU_mc"])


    dic["chunks"].append(chunks)
    dic["dataset"].append(name)
    dic["method"].append(method)
    dic["IoU"].append(IoU_bin)
    dic["IoU_mc"].append(IoU_mc)

table = pd.DataFrame(dic)
table.to_csv("before_mean_results.csv")
table = table.groupby(["dataset", "method"]).mean()
table = table.drop("chunks", axis=1)
table.to_csv("final_results.csv")
