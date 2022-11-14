
import numpy as np
from glob import glob
import pandas as pd

files = glob("*.npz")
dic = {
    "dataset" : [],
    "method" : [],
    "chunks" : [],
    "IoU" : [],
    "IoU_mc": [],
    "n0" : [],
    "n1" : [],
    "label1": [],
    "label2": [],
}


for f in files:
    npz = np.load(f)
    n0 = npz["z0_n"]
    n1 = npz["z1_n"]
    label1 = npz["labels_1_n"]
    label2 = npz["labels_2_n"]

    OT = "OT" in f
    if OT:
        name = f.split("_")[0]
        method = "OT"
        IoU_str_name = "score"
    else:
        method, name = f.split("_")[:2]
        IoU_str_name = "IoU_bin"


    if "Chunks" in f:
        name = name[6:]
        for i in range(1, 6):
            try:
                chunks = int(name[:i])
                j = i
            except:
                break
        dataname = name[j:]
    else:
        if "tmp" in name:
            dataname = name[3:]
        chunks = 0
    
    
    IoU_bin = float(npz[IoU_str_name])

    if OT:
        IoU_mc = 0.0
    else:
        IoU_mc = float(npz["IoU_mc"])



    dic["chunks"].append(chunks)
    dic["dataset"].append(dataname)
    dic["method"].append(method)
    dic["IoU"].append(IoU_bin)
    dic["IoU_mc"].append(IoU_mc)
    dic["n0"].append(n0)
    dic["n1"].append(n1)
    dic["label1"].append(label1)
    dic["label2"].append(label2)

table = pd.DataFrame(dic)
table.to_csv("before_mean_results.csv")
# drop those with 0 label change
table = table.loc[(table["label1"] != 0) & table["label2"] != 0]

table_std = table.groupby(["dataset", "method"]).std()
table_std = table_std.drop("chunks", axis=1)
table_std.columns = ["IoU_bin_std", "IoU_mc_std"]
table = table.groupby(["dataset", "method"]).mean()
table = table.join(table_std)
table = table.drop("chunks", axis=1)
table.to_csv("final_results.csv")
