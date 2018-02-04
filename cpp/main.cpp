/*
    @author Jay (jaypanda16@gmail.com)
    @file   bb_supervoxel_segmentor.cpp
    @date   2018/02/03
    @version 0.1

    @brief  Main program to run clustering of 3D point cloud data of people on a Basketball/NBA court
*/
#include <pcl/console/parse.h>

#include "bb_supervoxel_segmentor.hpp"

int main (int argc, char ** argv) {
    if (argc < 2) {
        pcl::console::print_error ("Syntax is: %s <point-cloud-data-file> \n "
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n"
                                "-b <num color histogram bins per channel\n"
                                "-D DEBUG_FLAG 0/1\n"
                                "-V VISUALIZE_FLAG 0/1\n", argv[0]);
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
    if (pcl::console::find_switch (argc, argv, "-b"))
        pcl::console::parse(argc, argv, "-b", rgb_histbins);

    bool DEBUG = false;
    if (pcl::console::find_switch (argc, argv, "-D"))
        pcl::console::parse(argc, argv, "-D", DEBUG);

    bool VISUALIZE = true;
    if (pcl::console::find_switch (argc, argv, "-V"))
        pcl::console::parse(argc, argv, "-V", VISUALIZE);

    BBSupervoxelSegmentor bbss(argv[1]);
    bbss.SetParameters(voxel_resolution, seed_resolution, color_importance, spatial_importance,
                        normal_importance, rgb_histbins, DEBUG, VISUALIZE);
    bbss.ProcessPointCloud();

    return (0);
}
