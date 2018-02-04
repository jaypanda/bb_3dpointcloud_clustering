import csv
import numpy as np
import argparse

DEBUG=True

def visualize_3d(pt, labels=None, mode='labels'):
    '''Visualizes the points on a 3d graph, labeled with cluster indices

    Args:
        pt(list): list of (x,y,z) tuples Nx3
        labels(list): list of cluster indices corresponding to each point in pt
    Returns:
        none
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt         ### To avoid the initialization unless necessary
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if mode == 'points':
        ax.scatter([float(x[0]) for x in pt],
                   [float(z[2]) for z in pt],
                   [float(y[1]) for y in pt],
                   c=[r[3] for r in pt], marker='o')

    elif mode == 'labels':
        unique_labels = set(labels)
        if DEBUG:
            print(len(unique_labels))
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for il,l in enumerate(unique_labels):
            class_member_mask = (labels==l)
            ptmasked = np.array(pt,dtype=np.float32)[class_member_mask]

            x_mean,z_mean = np.mean(ptmasked[:,:2],axis=0)
            print('Position for a likely person: %d %d' % (x_mean,z_mean))
            ax.scatter([float(x[0]) for x in ptmasked],
                   [float(z[2]) for z in ptmasked],
                   [float(y[1]) for y in ptmasked],
                   c=colors[il], marker='o')

    plt.show()

def cluster_XYZRGB(points):
    '''Clusters the 3d points

    Args:
        points(list): list of (x,y,z,r,g,b) point tuples Nx6
    Returns:
        labels(list): Nx1 for each point's cluster index
    '''
    from sklearn.cluster import DBSCAN
#    points = [x + (y,) for x,y in zip(pt,rgb_pt)]
    db = DBSCAN(eps=0.1).fit(points)
    return db.labels_

def normalized_xyz(pt):
    '''Normalizes the 3d points

    Args:
        pt(list): (x,y,z) list of tuples Nx3
    Returns:
        pt_3dn(list): normalized numpy array Nx3 values in range [0,1]
    '''
    pt_3d = np.array(pt,dtype=np.float32)
    maxCoord = np.max(pt_3d, axis=0)
    minCoord = np.min(pt_3d, axis=0)
    pt_3dn = (pt_3d - minCoord) / (maxCoord - minCoord)
    return pt_3dn

def normalized_rgb(rgb):
    '''Normalizes 0-255 rgb values

    Args:
        rgb(list): list of (r,g,b) tuples Nx3
    Returns:
        normalized numpy array Nx3 values in range [0,1]
    '''
    return np.array(rgb,dtype=np.float32)/255


def concat_XYZRGB(pt,rgb):
    '''Concatenates pt Nx3 and rgb Nx3 matrices to Nx6

    Args:
        pt(list): Nx3 x,y,z matrices
        rgb: Nx3 r,g,b matrices
    Returns:
        Nx6 concatenated matrix
    '''
    return np.concatenate((pt,rgb), axis=1)

def load_data(point_cloud_file):
    '''

    Args:
        point_cloud_file: path to the point cloud data file
    Returns:
        pt(list): list of (x,y,z) tuples for point coordinates in 3d space
        rgb(list): list of (r,g,b) tuples for color of corresponding 3d points
        rgb_pt(list): list of rgb values as (r<<16|g<<8|b)
    '''
    pt = []
    rgb = []
    rgb_pt = []

    with open(point_cloud_file) as fp:
        reader = csv.reader(fp, delimiter=' ')
        for row in reader:
            pt.append((row[0], row[1], row[2]))
            rgb.append((row[3], row[4], row[5]))
            rgb_pt.append(np.uint8(row[3]) << 16 | np.uint8(row[4]) << 8 | np.uint8(row[5]))

    return pt, rgb, rgb_pt

if __name__ == "__main__":
    parser = argparse.ArgumentParser("3D point cloud clustering to "
                                     "identify basketball players and referees")
    parser.add_argument('--data_file','-d', required=False,
                        default='data/point_cloud_data.txt')
    args = parser.parse_args()

    pt, rgb, rgb_pt = load_data(args.data_file)
    ptn = normalized_xyz(pt)
    rgbn = normalized_rgb(rgb)
    points = concat_XYZRGB(ptn,rgbn)
    labels = cluster_XYZRGB(points)
    visualize_3d(pt, labels)