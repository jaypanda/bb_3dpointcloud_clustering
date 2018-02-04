-----------------
DESCRIPTION
-----------------
To cluster and group players/referees in 3d point clouds of a basketball game,
in order to enable optical tracking, etc

I. bb_supervoxel_segmentor.cpp
    The complete implemented approach in "ProcessPointCloud" is broadly performing the following steps:

1) "SupervoxelClustering": This step is inspired from PCL tutorial on the Supervoxel Clustering
    approach (http://pointclouds.org/documentation/tutorials/supervoxel_clustering.php), based
    on the work by Papon et. al. in CVPR 2013 (Voxel Cloud Connectivity Segmentation - Supervoxels
    for Point Clouds). It groups the 3D point cloud into supervoxels (volumetric
    oversegmentations) based on a region-growing variant of k-means clustering to generate a point
    labeling directly within a voxel octree structure.
2) "ColorHistClustering": Processes object point clouds after recovering adjacent supervoxel
    centers (from step 1, and treating all point clouds as a single object), to identify
    3 groups of people based on their outfit color information on the RGB pixel values. It
    computes normalized color histogram features for each identified person point cloud, followed
    by a k-means clustering step to identify "3" clusters of people in the point cloud.
3) In order to visualize the objects identified into 3 groups of people - teamA, teamB and
    referees, the visualization via PCLVisualizer, projects the original 3D point cloud on
    a white background and marks the centroid points of each person with a unique color voxel -
    Red, Green or Blue.

II. py_cluster3d.py #Incomplete Script - initial trial
    This was an attempt to use python based libraries and approach this problem. Eventually, due to
lack of full support for PCL via python, and/or other supporting libraries, the complete approach
was implemented with C++.
   This script uses DBSCAN from scikit-learn, to perform the clustering of point clouds. It doesn't
seem to perform very well, particularly with point clouds of people standing close to each other.

-----------------
DEPENDENCIES
-----------------
i) PCL C++ library
ii) OpenCV

-----------------
INSTALL
-----------------
$ mkdir build && cd build
$ cmake ..
$ make

-----------------
USAGE
-----------------
#Default params
cpp/build/bb_3dpointcloud_clustering data/point_cloud_data.txt \
                                     -v 0.8 -s 10.0 -z 0.5 -c 0.1 -n 0.5 -b 10 -D 0 -V 1

