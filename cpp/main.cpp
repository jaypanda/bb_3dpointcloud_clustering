#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/grsd.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include<pcl/visualization/pcl_plotter.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

//OpenCV for histogram comparisions
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<string>
#include<fstream>
#include<sstream>
#include<vector>

#include "bb_supervoxel_segmentor.hpp"

// Types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

int main (int argc, char ** argv){
    if (argc < 2){
        pcl::console::print_error ("Syntax is: %s <point-cloud-data-file> \n "
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n"
                                "-b <num color histogram bins per channel", argv[0]);
        return (1);
    }
    float voxel_resolution = 0.8f;
    bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
    if (voxel_res_specified)
        pcl::console::parse (argc, argv, "-v", voxel_resolution);

    float seed_resolution = 10.0f;
    bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
    if (seed_res_specified)
        pcl::console::parse (argc, argv, "-s", seed_resolution);

    float color_importance = 0.1f;
    if (pcl::console::find_switch (argc, argv, "-c"))
        pcl::console::parse (argc, argv, "-c", color_importance);

    float spatial_importance = 0.5f;
    if (pcl::console::find_switch (argc, argv, "-z"))
        pcl::console::parse (argc, argv, "-z", spatial_importance);

    float normal_importance = 0.5f;
    if (pcl::console::find_switch (argc, argv, "-n"))
        pcl::console::parse (argc, argv, "-n", normal_importance);

    int rgb_histbins = 10;
    if (pcl::console::find_switch(argc, argv, "-b"))
        pcl::console::parse(argc, argv, "-b", rgb_histbins);

    bool DEBUG = false;
    if (pcl::console::find_switch(argc, argv, "-D"))
        pcl::console::parse(argc, argv, "-D", DEBUG);

    bool VISUALIZE = true;
    if (pcl::console::find_switch(argc, argv, "-V"))
        pcl::console::parse(argc, argv, "-V", VISUALIZE);


    BBSupervoxelSegmentor bbss(argv[1]);
    bbss.setParameters(voxel_resolution, seed_resolution, color_importance, spatial_importance,
                        normal_importance, rgb_histbins, DEBUG, VISUALIZE);
    bbss.process_pointcloud();
}
