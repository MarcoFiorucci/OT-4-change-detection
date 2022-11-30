import numpy as np
import argparse
from scipy.spatial import Delaunay

from sklearn.neighbors import NearestNeighbors
from utils import compute_iou

def parser_f():

    parser = argparse.ArgumentParser(
        description="TO arguments",
    )
    parser.add_argument(
        "--npz",
        type=str,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="voronoi",
    )
    args = parser.parse_args()
    return args

def project_2d(x, y, z):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xrange, yrange = int(xmax - xmin) + 1, int(ymax - ymin) + 1
    scale = 1
    x, y = (x - xmin) * scale, (y - ymin) * scale
    res = np.zeros((xrange*scale, yrange*scale))
    count = np.zeros((xrange*scale, yrange*scale))
    mat = np.vstack([x, y, z])
    def g(xyz):
        xc, yc, zc = xyz
        xc, yc = int(xc), int(yc)
        res[xc, yc] += zc
        count[xc, yc] += 1
    
    func = lambda xyz: g(xyz)

    np.apply_along_axis(func, 0, mat)
    count[count == 0] = 1

    return res / count

# opt = parser_f()

# npz = np.load(opt.npz)

# tte = npz["changes"]
# pointcloud = npz["Yhat"]
# x = pointcloud[:,0]
# y = pointcloud[:,1]
# z = pointcloud[:,2]
# z_2d = project_2d(x, y, z)

# z_2d_changes = project_2d(x, y, tte)

# gt = npz["labels_on1"]
# z_2d_gt = project_2d(x, y, gt)

# pointcloud = npz["Y"]
# x = pointcloud[:,0]
# y = pointcloud[:,1]
# z = pointcloud[:,2]
# z_true = project_2d(x, y, z)
# z_true_gt = project_2d(x, y, gt)

# z_true_changes = project_2d(x, y, tte)

# import pdb; pdb.set_trace()


def voronoi_graph(x, y):
    X = np.vstack([x, y]).T
    indptr_neigh, neighbours = Delaunay(X, qhull_options="QJ").vertex_neighbor_vertices
    neighbour_list = []
    distances = []
    for i in range(X.shape[0]):
        i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
        neighbour_list.append(i_neigh)
        i_dist = np.sum(np.power(X[i_neigh] - X[i], 2), axis=1)
        distances.append(i_dist)
    return distances, neighbour_list


def nearestneighboor_graph(x, y, k):
    knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=5, p=2,
            radius=1.0)
    X = np.vstack([x, y]).T
    knn.fit(X)    
    return knn.kneighbors(X, return_distance=True)

def erode_graph(z, n_z):
    return min(z, min(n_z))

def dilate_graph(z, n_z):
    return max(z, max(n_z))

def erosion(z, neighboors):
    res = np.zeros_like(z)
    try:
        for i in range(len(z)):
            res[i] = erode_graph(z[i], z[neighboors[i]])
    except:
        import pdb; pdb.set_trace()
    return res

def dilation(z, neighboors):
    res = np.zeros_like(z)
    for i in range(len(z)):
        res[i] = dilate_graph(z[i], z[neighboors[i]])
    return res

def softmax_erosion(z, neighboors, ):
    pass

def post_processing(x, y, z, method="voronoi", return_both=False):
    if method == "knn":
        distance, neighboors = nearestneighboor_graph(x, y, 5)
    elif method == "voronoi":
        distance, neighboors = voronoi_graph(x, y)
    eroded_once = erosion(z, neighboors)
    dilated_once = dilation(eroded_once, neighboors)
    if return_both:
        return dilated_once, eroded_once
    else:
        return dilated_once

if __name__ == "__main__":

    opt = parser_f()

    npz = np.load(opt.npz)
    pointcloud = npz["Y"]
    gt = npz["labels_on1"]
    x = pointcloud[:,0]
    y = pointcloud[:,1]
    z = npz["changes"]
    dilated_once, eroded_once = post_processing(x, y, z, opt.method, return_both=True)
    bin_score, thresh, _, mc_score, thresh2, _ = compute_iou(z, gt, mc=True)
    print("Normal, score:", (bin_score, thresh))
    print("Eroded, score:", compute_iou(eroded_once, gt, mc=False))
    print("Eroded + dilated, score:", compute_iou(dilated_once, gt, mc=False))

    print("threshold is fixed")
    print("Normal, score:", compute_iou(z, gt, mc=False, threshold=5))
    print("Eroded, score:", compute_iou(eroded_once, gt, mc=False, threshold=5))
    print("Eroded + dilated, score:", compute_iou(dilated_once, gt, mc=False, threshold=5))
