
import numpy as np
from glob import glob
import sys

def open_npz_compute(f, filetype="std"):
    file = np.load(f)
    gt = file["y_on1"]
    th = file["thresh_bin"]
    if filetype == "OT":
        z = file["changes_intesity_y"]
    else:
        z = file["z1_on1"] - file["z1_on0"]
    pred = np.zeros_like(z).astype(int)
    pred[z > th] = 1
    pred[z < -th] = 1
    return z, gt, pred
files = glob("*.npz")

diff_z = []
prediction = []
gt = []

for f in files:
    z_f, gt_f, yhat = open_npz_compute(f, sys.argv[1])
    diff_z.append(z_f)
    prediction.append(yhat)
    gt.append(gt_f)

diff_z = np.concatenate(diff_z, axis=0)
prediction = np.concatenate(prediction, axis=0)
gt = np.concatenate(gt, axis=0)
