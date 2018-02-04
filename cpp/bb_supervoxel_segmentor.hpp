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

// Types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

class BBSupervoxelSegmentor{

    //Point cloud data structure to store the input data points
    PointCloudT::Ptr cloud;

    //Point cloud data structure to store the final object centers;
    PointCloudT::Ptr obj_centroids;

    // viewer to visualize point cloud and results
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

    //config parameters
    float voxel_resolution = 0.8f;
    float seed_resolution = 10.0f;
    float color_importance = 0.1f;
    float spatial_importance = 0.5f;
    float normal_importance = 0.5f;
    int rgb_histbins = 10;
    bool DEBUG = false;
    bool VISUALIZE = true;


    // to load the point cloud data file into PointCloudT data structure
    bool loadPointCloud (char *point_cloud_data_file);

    // work on processed object point clouds after recovering adjacent
    // supervoxel centers treating as a single object
    void colorHistClusters(std::vector<PointCloudT::Ptr> final_objclouds,
                                   std::vector<int>& labelvecs, int histSize=10);

    // Use supervoxel clustering method from PCL library to cluster and identify people point clouds
    void supervoxelClustering(std::vector<PointCloudT::Ptr>& final_objclouds,
                                                 std::vector<PointT>& final_centroids);

    public:
        BBSupervoxelSegmentor(char *point_cloud_data_file);

        // set config parameters before process_pointcloud call
        void setParameters(float voxel_resolution, float seed_resolution, float color_importance,
                            float spatial_importance, float normal_importance, int rgb_histbins,
                            bool DEBUG, bool VISUALIZE);

        // runs the approach pipeline
        bool process_pointcloud();
};