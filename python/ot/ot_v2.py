import numpy as np
import time

import jax
import jax.numpy as jnp

from ott.tools import transport

import ot
import ot.plot
import argparse

def parser_f():

    parser = argparse.ArgumentParser(
        description="TO arguments",
    )
    parser.add_argument(
        "--csv0",
        type=str,
    )
    parser.add_argument(
        "--csv1",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    return args

#Flag variables
opt = parser_f()
dataname = opt.output

print('----------------------------------')
print('| Data Loader                    |')
print('----------------------------------')

pc0_file = opt.csv0 
pc1_file = opt.csv1

pc0 = np.loadtxt(pc0_file, skiprows=1, delimiter= ',')
pc1 = np.loadtxt(pc1_file, skiprows=1, delimiter= ',')
z0_n = pc0.shape[0]
z1_n = pc1.shape[1]
labels_1_n = (pc1[:,-1].astype(int) == 1).sum()
labels_2_n = (pc1[:,-1].astype(int) == 2).sum()
print('pc0 shape', pc0.shape)
print('pc1 shape', pc1.shape)

X = jnp.asarray(pc0[:,0:3])
Y = jnp.asarray(pc1[:,0:3])


#a is now a uniform probability distribution
a_shape = pc0[:,2].shape
a = np.full((a_shape), 1/a_shape[0]) 
b_shape = pc1[:,2].shape
#b is now a uniform probability distribution
b = np.full((b_shape), 1/b_shape[0])


print('------------------------------------------------')
print('| Compute the transportation plan with JAX OTT |')
print('------------------------------------------------')

start = time.time()
ot = transport.solve(X, Y, a=a, b=b, epsilon=1e-2)
P = ot.matrix
end = time.time()

print('Computation time for transportation plan: ',end - start)
jnp.save('{}_P'.format(dataname),P)

print('----------------------------------')
print('| Displacement Interpolation     |')
print('----------------------------------')
one_n1 = np.ones(len(a))
one_n2 = np.ones(len(b))

Yt_hat = np.matmul(np.linalg.inv(np.diag(np.matmul(np.transpose(P),one_n1))) ,
                     np.matmul(np.transpose(P), X))

print('----------------------------------')
print('| Quantity of changes              |')
print('----------------------------------')

#changes in the latitue-longitude plane |Y-Yt_hat|^2
changes_intensity_y = jnp.sum(jnp.square(Y-Yt_hat), axis=1)
# changes_intesity_y/= jnp.max(changes_intesity_y)

print('----------------------------------')
print('| Evaluation Metrics              |')
print('----------------------------------')
#iou for a given change type (class)
def iou(gt, prediction):
     intersection = np.logical_and(gt, prediction)
     union = np.logical_or(gt, prediction)
     iou_score = np.sum(intersection) / np.sum(union)
     return iou_score

gt = pc1[:,-1]

#change gt: 0 no change, 1 changes (both positive and negative changes)
idxs = np.where(gt == 2)
gt[idxs] = 1
print('-------------------------------------------------------------')
print('shape of change_intensity', changes_intensity_y.shape)

#quantize the change_intensity_y into 0 and 1 w.r.t of the threshold th
thresholds_y = np.arange(np.min(changes_intensity_y), np.max(changes_intensity_y), np.std(changes_intensity_y)/10)
#print(thresholds_y)
iou_th_y = []
iou_i = 0
for th in thresholds_y:
     predicted_change_y = np.where(changes_intensity_y >= th, 1, 0)
     iou_i = iou(gt,predicted_change_y)
     iou_th_y.append(iou_i)

print('max iou of changes on y:' +  str(max(iou_th_y)))
best_threshold = thresholds_y[np.array(iou_th_y).argmax()]

np.savez(dataname + ".npz", IoU_bin=max(iou_th_y),
    thresh_bin=best_threshold, changes=changes_intensity_y,
    z0_n=z0_n, z1_n=z1_n, labels_1_n=labels_1_n,
    labels_2_n=labels_2_n)
