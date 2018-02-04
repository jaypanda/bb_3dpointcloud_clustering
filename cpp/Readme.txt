-----------------
DESCRIPTION
-----------------
To cluster and group players/referees in 3d point clouds of a basketball game, in order to enable optical tracking, etc

-----------------
DEPENDENCIES
-----------------
i) PCL C++ library
ii) OpenCV >=3.0.0

-----------------
INSTALL
-----------------
$ mkdir build && cd build
$ cmake ..
$ make

-----------------
USAGE
-----------------
./supervoxel_segmentor ../../data/point_cloud_data.txt -v 0.8 -s 10.0 -z 0.5 -c 0.1 -n 0.5 -b 10
