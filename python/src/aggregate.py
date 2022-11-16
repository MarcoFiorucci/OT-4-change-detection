
import numpy as np
from glob import glob
import sys
from utils_diff import compute_iou
import pandas as pd

def open_npz_compute(f, OT=False):
    file = np.load(f)
    gt = file["labels_on1"]
    th = file["thresh_bin"]
    if OT:
        z = file["changes_intesity_y"]
    else:
        z = file["z1_on1"] - file["z0_on1"]
    pred = np.zeros_like(z).astype(int)
    pred[z > th] = 1
    pred[z < -th] = 1
    return z, gt, pred

files = glob("*.npz")

dataname = sys.argv[1]
method = sys.argv[2]
is_OT = method == "OT"
diff_z = []
prediction = []
gt = []

for f in files:
    z_f, gt_f, yhat = open_npz_compute(f, is_OT)
    diff_z.append(z_f)
    prediction.append(yhat)
    gt.append(gt_f)

diff_z = np.concatenate(diff_z, axis=0)
prediction = np.concatenate(prediction, axis=0)
gt = np.concatenate(gt, axis=0)

table = pd.DataFrame()

output = compute_iou(diff_z, gt, mc=not is_OT)
if is_OT:
    iou, _, _ = output
else:
    bin_, mc = output
    iou, iou_mc = bin_[0], mc[0]
    table.loc[dataname, "iou_mc"] = iou_mc

table.loc[dataname, "iou"] = iou
table.loc[dataname, "method"] = method
table.to_csv("{}_{}.csv".format(dataname, method))



