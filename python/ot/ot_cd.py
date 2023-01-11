import numpy as np
import time

import jax.numpy as jnp

from ott.tools import transport

import ot
import argparse
from utils import compute_iou

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
    parser.add_argument(
        "--method",
        default="vanilla",
        type=str,
    )
    parser.add_argument(
        "--epsilon",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--epsilon_unbalanced",
        default=None,
        type=float,
    )
    args = parser.parse_args()
    return args

def init_ot_variables(csv0, csv1):
    print('----------------------------------')
    print('| Data Loader                    |')
    print('----------------------------------')


    pc0 = np.loadtxt(csv0, skiprows=1, delimiter= ',')
    pc1 = np.loadtxt(csv1, skiprows=1, delimiter= ',')
    GT = pc1[:,-1].astype(int)
    print('pc0 shape', pc0.shape)
    print('pc1 shape', pc1.shape)

    X = jnp.asarray(pc0[:,0:3])
    Y = jnp.asarray(pc1[:,0:3])

    #a is now a uniform probability distribution
    a_shape = pc0.shape[0]
    b_shape = pc1.shape[0]
    a = jnp.full((a_shape), 1/a_shape) 
    #b is now a uniform probability distribution
    b = jnp.full((b_shape), 1/b_shape)
    return X, Y, GT, a, b, a_shape, b_shape

#Flag variables
opt = parser_f()
dataname = opt.output

X, Y, gt, a, b, labels_1_n, labels_2_n = init_ot_variables(opt.csv0, opt.csv1)

print('------------------------------------------------')
print('| Compute the transportation plan with JAX OTT |')
print('------------------------------------------------')
eps = opt.epsilon
start = time.time()

if opt.method == "vanilla":
    done = False
    times = 0
    while not done and times < 10:
        try:
            ot = transport.solve(X, Y, a=a, b=b, epsilon=eps)
            done = True
        except:
            eps *= 10
            times += 1

    P = ot.matrix
    
elif opt.method == "unbalanced":
    reg_kl = opt.epsilon_unbalanced
    M = ot.dist(X, Y)
    M /= M.max()
    P = round(ot.sinkhorn_unbalanced(a, b, M, eps, reg_kl), 9)
else:
    print("unknown option method {}".format(opt.method))

end = time.time()
print('Computation time for transportation plan: ', end - start)

print('----------------------------------')
print('| Displacement Interpolation     |')
print('----------------------------------')
one_n1 = np.ones(len(a))
one_n2 = np.ones(len(b))

atmp = np.matmul(P.T, one_n1)
atmp2 = np.diag(np.where(atmp == 0, 0, 1 / atmp)) #matrix inversion
btmp = np.matmul(P.T, X)

Yt_hat = np.matmul(atmp2, btmp)

print('----------------------------------')
print('| Quantity of changes            |')
print('----------------------------------')

#changes in the latitue-longitude plane |Y-Yt_hat|^2
diff_Y = jnp.sum(jnp.square(Y-Yt_hat), axis=1) * np.sign(Y[:,2]-Yt_hat[:,2])

print('----------------------------------')
print('| Evaluation Metrics             |')
print('----------------------------------')

#iou for a given change type (class)
#change gt: 0 no change, 1 changes (both positive and negative changes)
labels_1_n = (gt == 1).sum()
labels_2_n = (gt == 2).sum()

iou_bin, thresh_bin, pred_bin, iou_mc, thresh_mc, pred_mc = compute_iou(np.array(diff_Y), gt)


print('-------------------------------------------------------------')
print('shape of change_intensity', diff_Y.shape)
print('max iou of changes on y:' +  str(iou_bin))
print('max iou mc of changes on y:' +  str(iou_mc))

np.savez(dataname + ".npz", IoU_bin=iou_bin,
    IoU_mc=iou_mc, thresh_mc=thresh_mc, 
    thresh_bin=thresh_bin, changes=diff_Y,
    z0_n=a.shape[0], z1_n=a.shape[0], 
    labels_1_n=labels_1_n,
    labels_2_n=labels_2_n, labels_on1=gt)

