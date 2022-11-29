
import numpy as np
from sklearn.metrics import jaccard_score

def compute_iou(diffz, y, mc=True, threshold=None):
    if threshold:
        y_pred = np.zeros_like(y)
        y_pred[diffz > threshold] = 1
        y_pred[diffz < -threshold] = 2
        if mc:
            score = jaccard_score(y, y_pred, average=None)
            score = np.mean(score)#[1:])
        final_pred_bin = y_pred > 0
        score_bin = jaccard_score(y > 0, final_pred_bin)
        out_bin = score_bin, threshold, final_pred_bin
        if mc:
            out_mc = score,  threshold, y_pred
    else:
        best_score_bin = 0
        best_thresh_bin = 0
        final_pred_bin = np.zeros_like(y)
        if mc:
            best_score = 0
            best_thresh = 0
            final_pred = np.zeros_like(y)
        std = np.std(diffz)
        for thresh in np.arange(0, diffz.max(), step=std / 100):
            y_pred = np.zeros_like(y)
            y_pred[diffz > thresh] = 1
            y_pred[diffz < -thresh] = 2
            if mc:
                score = jaccard_score(y, y_pred, average=None)
                score = np.mean(score)#[1:])
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
                    final_pred = y_pred
            score_bin = jaccard_score(y > 0, y_pred > 0)
            if score_bin > best_score_bin:
                best_score_bin = score_bin
                best_thresh_bin = thresh
                final_pred_bin = y_pred > 0
        out_bin = best_score_bin, best_thresh_bin, final_pred_bin
    if mc:
        out_mc = best_score,  best_thresh, final_pred
        return (*out_bin, *out_mc)
    else:
        return out_bin
