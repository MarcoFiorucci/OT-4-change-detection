
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
    method, name = f.split("_")[:2]
    if "chunks" in f:
        import pdb; pdb.set_trace()
    else:
        if "tmp" in name:
            name = name[3:]
        chunks = 0

    IoU_bin = float(npz["IoU_bin"])
    IoU_mc = float(npz["IoU_mc"])


    dic["chunks"].append(chunks)
    dic["dataset"].append(name)
    dic["method"].append(method)
    dic["IoU"].append(IoU_bin)
    dic["IoU_mc"].append(IoU_mc)

table = pd.DataFrame(dic)
table = table.groupby(["dataset", "method"]).mean()
table = table.drop("chunks", axis=1)
table.to_csv("final_results.csv")
