-----------------
DESCRIPTION
-----------------
To cluster and group players/referees in 3d point clouds of a basketball game, in order to enable optical tracking, etc

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

