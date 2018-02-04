cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(supervoxel_clustering)

find_package(PCL 1.8 REQUIRED)
find_package( OpenCV )


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-inconsistent-missing-override -Wno-max-unsigned-zero")
include_directories(${PCL_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (supervoxel_segmentor supervoxel_segmentor.cpp)
target_link_libraries (supervoxel_segmentor ${PCL_LIBRARIES} ${OpenCV_LIBS} )