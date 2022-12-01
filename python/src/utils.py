
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
            best_score_mc = score
            best_thresh_mc = threshold
            final_pred_mc = y_pred
        final_pred_bin = y_pred > 0
        score_bin = jaccard_score(y > 0, final_pred_bin)
        out_bin = score_bin, threshold, final_pred_bin
    else:
        best_score_bin = 0
        best_thresh_bin = 0
        final_pred_bin = np.zeros_like(y)
        if mc:
            best_score_mc = 0
            best_thresh_mc = 0
            final_pred_mc = np.zeros_like(y)
        std = np.std(diffz)
        for thresh in np.arange(0, diffz.max(), step=std / 100):
            y_pred = np.zeros_like(y)
            y_pred[diffz > thresh] = 1
            y_pred[diffz < -thresh] = 2
            if mc:
                score = jaccard_score(y, y_pred, average=None)
                score = np.mean(score)#[1:])
                if score > best_score_mc:
                    best_score_mc = score
                    best_thresh_mc = thresh
                    final_pred_mc = y_pred
            score_bin = jaccard_score(y > 0, y_pred > 0)
            if score_bin > best_score_bin:
                best_score_bin = score_bin
                best_thresh_bin = thresh
                final_pred_bin = y_pred > 0
        out_bin = best_score_bin, best_thresh_bin, final_pred_bin
    if mc:
        out_mc = best_score_mc,  best_thresh_mc, final_pred_mc
        return (*out_bin, *out_mc)
    else:
        return out_bin
