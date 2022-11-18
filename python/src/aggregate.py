
import numpy as np
from glob import glob
import sys
from utils import compute_iou
import pandas as pd

def open_npz_compute(f, OT=False):
    file = np.load(f)
    gt = file["labels_on1"]
    th = file["thresh_bin"]
    if OT:
        z = file["changes"]
    else:
        z = file["z1_on1"] - file["z0_on1"]
    pred = np.zeros_like(z).astype(int)
    pred[z > th] = 1
    pred[z < -th] = 1
    iou = file["IoU_bin"]
    return z, gt, pred, iou

files = glob("*.npz")

dataname = sys.argv[1]
method = sys.argv[2]
is_OT = method == "OT"

diff_z, prediction, gt = [], [], []
iou_chunks, chunk_id, size = [], [], []
max_changes, min_changes, labels = [], [], []

for f in files:
    z_f, gt_f, yhat, iou = open_npz_compute(f, is_OT)
    diff_z.append(z_f)
    prediction.append(yhat)
    gt.append(gt_f)
    iou_chunks.append(iou)
    chunk_id.append(f.split("-")[0])
    max_changes.append(z_f.max())
    min_changes.append(z_f.min())
    size.append(z_f.shape[0])
    labels.append((gt_f != 0).sum())


tmp_table = pd.DataFrame(
    {"chunk_id": chunk_id,
    "iou": iou_chunks,
    "max_changes": max_changes,
    "min_changes": min_changes,
    "nlabels": labels,
    "size": size
    }
)

tmp_table.to_csv("{}_{}_chunkinfo.csv".format(dataname, method), index=False)


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



