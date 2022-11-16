#from tkinter import X
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#from sklearn.preprocessing import OneHotEncoder
import time

import jax
import jax.numpy as jnp
#from jax import device_put

#import ott
#from ott.geometry import pointcloud
#from ott.core import sinkhorn
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


#import pdb; pdb.set_trace()

#Flag variables
synthetic = False
visualization = False
opt = parser_f()
dataname = opt.output
#auxiliary plot function
myplot = lambda x,y,ms,col: plt.scatter(x,y, s=ms*20, edgecolors="k", c=col, linewidths=2)

if synthetic == False:
    print('----------------------------------')
    print('| Data Loader                    |')
    print('----------------------------------')

    folder_path_pc0 ='/home/marco/IEEE_Dataset_V1/2-Lidar10/Train/LyonN10/'
    folder_path_pc1 ='/home/marco/IEEE_Dataset_V1/2-Lidar10/Train/LyonN10/'

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

    xy_s = pc0[:,0:3]
    #xy_s/= np.sum(xy_s)

    #print('shape of x', xy_s.shape, 'mean of x:', np.average(xy_s), 'median: of x', np.median(xy_s) ,' max of x:', np.max(xy_s), 'min of x:', np.min(xy_s))
    
    xy_t = pc1[:,0:3]
    #xy_t/= np.sum(xy_t)
    
    #print('shape of y', xy_t.shape, 'mean of y:', np.average(xy_t), 'median: of y', np.median(xy_t) ,' max of y:', np.max(xy_t), 'min of y:', np.min(xy_t))
    
    x = jnp.asarray(xy_s)
    y = jnp.asarray(xy_t)

   
    #a is now a uniform probability distribution
    a_shape = pc0[:,2].shape
    a = np.full((a_shape), 1/a_shape[0]) 
    #print('shape of a',a.shape, 'median: of a:', np.median(a), ' mean: of a:', np.mean(a), ' max of a:', np.max(a), 'min of a:', np.min(a))
    b_shape = pc1[:,2].shape
    #b is now a uniform probability distribution
    b = np.full((b_shape), 1/b_shape[0])
    #print('shape of b', b.shape, 'median: of b', np.median(b), ' mean: of b', np.mean(b), ' max of b:', np.max(b), 'min of b:', np.min(b))


else:
    #Syntethic data
    def create_points(rng, n, m, d):
        rngs = jax.random.split(rng, 3)
        x = jax.random.normal(rngs[0], (n, d)) 
        y = jax.random.uniform(rngs[1], (m, d)) +1 
        a = jnp.ones((n,)) / n
        b = jnp.ones((m,)) / m
        return x, y, a, b

    rng = jax.random.PRNGKey(0)
    n, m, d = 5, 7, 2
    x, y, a, b = create_points(rng, n=n, m=m, d=d)
    


print('------------------------------------------------')
print('| Compute the transportation plan with JAX OTT |')
print('------------------------------------------------')

start = time.time()
ot = transport.solve(x, y, a=a, b=b, epsilon=1e-2)
P = ot.matrix
end = time.time()

print('Computation time for transportation plan: ',end - start)
#print('Descriptive statistics of the Transportation Plan')
#print('mean:of P', jnp.average(P), 'std of P:', np.std(P), 'median of P:', jnp.median(P) ,' max of P:', jnp.max(P), 'min of P:', jnp.min(P))
jnp.save('{}_P'.format(dataname),P)

print('----------------------------------')
print('| Displacement Interpolation     |')
print('----------------------------------')
one_n1 = np.ones(len(a))
one_n2 = np.ones(len(b))
#print('shape of P', P.shape)
#print('shape of P Transpose', jnp.transpose(P).shape)
#print('shape of y', y.shape)
#print('shape of x', x.shape)

#compute the diplacement interpolation of the first cloud to the second
# xs_hat = np.matmul(np.linalg.inv(np.diag(np.matmul(P,one_n2))) ,
#                      np.matmul(P, y))
# print('Descriptive statistics of x')
# print('mean:', jnp.average(x), 'std:', np.std(x), 'median:', jnp.median(x),' max:', jnp.max(x), 'min:', jnp.min(x))
# print('Descriptive statistics of xs_hat')
# print('mean:', jnp.average(xs_hat), 'std:', np.std(xs_hat), 'median:', jnp.median(xs_hat),' max:', jnp.max(xs_hat), 'min:', jnp.min(xs_hat))
# print('shape of xs_hat', xs_hat.shape)
#compute the diplacement interpolation of the second cloud to the first
yt_hat = np.matmul(np.linalg.inv(np.diag(np.matmul(np.transpose(P),one_n1))) ,
                     np.matmul(np.transpose(P), x))
np.linalg.inv(np.diag(np.matmul(jnp.transpose(P),one_n1)))
#print('Descriptive statistics of y')
#print('mean:', jnp.average(y), 'std:', np.std(y), 'median:', jnp.median(y),' max:', jnp.max(y), 'min:', jnp.min(y))
#print('Descriptive statistics of xs_hat')
#print('mean:', jnp.average(yt_hat), 'std:', np.std(yt_hat), 'median:', jnp.median(yt_hat),' max:', jnp.max(yt_hat), 'min:', jnp.min(yt_hat))
#print('shape of yt_hat', yt_hat.shape)

print('----------------------------------')
print('| Quantity of changes              |')
print('----------------------------------')
# # changes in the latitue-longitude plane |X-Xs_hat|^2
# changes_intesity_x = jnp.sum(jnp.square(x-xs_hat), axis=1)
# changes_intesity_x/= jnp.max(changes_intesity_x)
# print('shape of change_intensity', changes_intesity_x.shape)
# print('Descriptive statistics of |X-Xs_hat|^2')
# print('mean:', jnp.average(changes_intesity_x), 'std:', jnp.std(changes_intesity_x), 'median:', jnp.median(changes_intesity_x),' max:', jnp.max(changes_intesity_x), 'min:', jnp.min(changes_intesity_x))


#change as ||z_y -z_yhat ||^2
#y_z  = y[:,2].reshape(y[:,2].shape[0],1)
#yt_hat_z = yt_hat[:,2].reshape(yt_hat[:,2].shape[0],1)
#changes_intesity_y = jnp.square(y_z - yt_hat_z)
#print('y_z.shape', y_z.shape)
#print('yt_hat_z.shape' ,yt_hat_z.shape)

#changes in the latitue-longitude plane |Y-Yt_hat|^2
changes_intesity_y = jnp.sum(jnp.square(y-yt_hat), axis=1)
changes_intesity_y/= jnp.max(changes_intesity_y)
#print('shape of change_intensity', changes_intesity_y.shape)
#print('Descriptive statistics of |Y-Ys_hat|^2')
#print('mean:', jnp.average(changes_intesity_y), 'std:', jnp.std(changes_intesity_y), 'median:', jnp.median(changes_intesity_y),' max:', jnp.max(changes_intesity_y), 'min:', jnp.min(changes_intesity_y))

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
#print('shape of g:', gt.shape)
#print('# of positive changed pixels:', np.count_nonzero(gt == 1)) 
#print('# of unchanged pixels:', np.count_nonzero(gt == 0))
#print('sum of pixels:', np.sum( np.count_nonzero(gt == 1) + np.count_nonzero(gt == 0)))

print('-------------------------------------------------------------')
print('shape of change_intensity', changes_intesity_y.shape)

#quantize the change_intensity_y into 0 and 1 w.r.t of the threshold th
thresholds_y = np.arange(np.min(changes_intesity_y), np.max(changes_intesity_y),np.std(changes_intesity_y)/10)
#print(thresholds_y)
iou_th_y = []
iou_i = 0
for th in thresholds_y:
     predicted_change_y = np.where(changes_intesity_y >= th, 1, 0)
     iou_i = iou(gt,predicted_change_y)
    # print('threshold:' + str(th) + 'iou_i: ' + str(iou_i))
    # print('shape of predicted ', predicted_change_y.shape)
    # print('shape of gt ', gt.shape)
     iou_th_y.append(iou_i)
print('max iou of changes on y:' +  str(max(iou_th_y)))
best_threshold = thresholds_y[np.array(iou_th_y).argmax()]

np.savez(dataname + ".npz", IoU_bin=max(iou_th_y),
    thresh_bin=best_threshold, changes=changes_intesity_y,
    z0_n=z0_n, z1_n=z1_n, labels_1_n=labels_1_n,
    labels_2_n=labels_2_n, y_on1=pc1[:,-1])

if(visualization == True):
    print('----------------------------------')
    print('| Visualization                  |')
    print('----------------------------------')

    #myplot = lambda x,y,ms,col: plt.scatter(x,y, s=ms*20, edgecolors="k", c=col, linewidths=2)
    
    # plt.figure(1, figsize = (10,7))
    # plt.axis("off")
    # for i in range(len(a)):
    #     myplot(x[i,0], x[i,1], a[i]*len(a)*10, 'b')
    # for j in range(len(b)):
    #     myplot(y[j,0], y[j,1], b[j]*len(b)*10, 'r')
    # plt.savefig('/home/marco/experiments/x_y_plot.png')

    # plt.figure(2)
    # plt.imshow(P, cmap='Purples')
    # plt.colorbar();
    # plt.savefig('/home/marco/experiments/transportation_matrix_OTT.png')

    # plt.figure(3, figsize = (10,7))
    # plt.axis("off")
    # for i in range(len(a)):
    #      myplot(x[i,0], x[i,1], a[i]*len(a)*10, 'b')
    # for j in range(len(b)):
    #      myplot(y[j,0], y[j,1], b[j]*len(b)*10, 'r')
    # for k in range(len(a)):
    #      myplot(xs_hat[k,0], xs_hat[k,1], a[k]*len(a)*10, 'g')
    # plt.savefig('/home/marco/experiments/displacement_interpolation.png')

    # fig =plt.figure(4)
    # ax = plt.axes(projection='3d')
    # colours = changes_intesity/jnp.max(changes_intesity)
    # print(colours.shape)
    # plot = ax.scatter(x[:,0], x[:,1], a[:], s=0.1, c=colours, cmap='rainbow' )
    # fig.colorbar(plot)
    # plt.savefig('/home/marco/experiments/detected_changes.png')

    # fig =plt.figure(4)
    # ax = plt.axes(projection='3d')
    # colours = changes_intesity_x
    # plot = ax.scatter(x[:,0], x[:,1], pc0[:,2], s=0.1, c=colours, cmap='rainbow' )
    # fig.colorbar(plot)
    # plt.savefig('/home/marco/experiments/detected_changes_x.png')


    fig =plt.figure(5)
    ax = plt.axes(projection='3d')
    colours = changes_intesity_y
    plot = ax.scatter(y[:,0], y[:,1], pc1[:,2], s=0.1, c=colours, cmap='rainbow' )
    fig.colorbar(plot)
    plt.savefig('{}_detected_changes_y.png'.format(dataname))

    fig =plt.figure(6)
    ax = plt.axes(projection='3d')
    #colours = a
    #plot = ax.scatter(x[:,0], x[:,1], pc0[:,2], s=0.1, c=colours, cmap='rainbow' )
    plot = ax.scatter(x[:,0], x[:,1], x[:,2], s=0.1)
    #fig.colorbar(plot)
    plt.savefig('{}_first_cloud.png'.format(dataname))

    fig =plt.figure(7)
    ax = plt.axes(projection='3d')
    colours = gt
    plot = ax.scatter(y[:,0], y[:,1], pc1[:,2], s=0.1, c=colours, cmap='rainbow' )
    #plot = ax.scatter(y[:,0], y[:,1], y[:,2], s=0.1, c='red')
    #fig.colorbar(plot)
    plt.savefig('{}_second_cloud.png'.format(dataname))
