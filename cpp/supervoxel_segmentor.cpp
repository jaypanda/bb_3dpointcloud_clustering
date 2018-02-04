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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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


void addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                       PointCloudT &adjacent_supervoxel_centers,
                                       std::string supervoxel_name,
                                       boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer);

// to load the point cloud data file into PointCloudT data structure
bool loadPointCloud (char *point_cloud_data_file, PointCloudT::Ptr cloud);

// work on processed object point clouds after recovering adjacent
// supervoxel centers treating as a single object
void color_hist_clusters(std::vector<PointCloudT::Ptr> final_objclouds,
                                   std::vector<int>& labelvecs, int histSize=10);


bool DEBUG = false;
bool VISUALIZE = true;

int
main (int argc, char ** argv){
  if (argc < 2){
    pcl::console::print_error ("Syntax is: %s <point-cloud-data-file> \n "
                                "--NT Dsables the single cloud transform \n"
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n", argv[0]);
    return (1);
  }


  PointCloudT::Ptr cloud = boost::shared_ptr <PointCloudT> (new PointCloudT ());
  pcl::console::print_highlight ("Loading point cloud...\n");
/*  if (pcl::io::loadPCDFile<PointT> (argv[1], *cloud)){
    pcl::console::print_error ("Error loading cloud file!\n");
    return (1);
  }
*/
  if (!loadPointCloud(argv[1], cloud)){
    pcl::console::print_error ("Error loading cloud file!\n");
    return (1);
  }


  //bool disable_transform = pcl::console::find_switch (argc, argv, "--NT");

  float voxel_resolution = 0.008f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);

  float seed_resolution = 0.1f;
  bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
  if (seed_res_specified)
    pcl::console::parse (argc, argv, "-s", seed_resolution);

  float color_importance = 0.2f;
  if (pcl::console::find_switch (argc, argv, "-c"))
    pcl::console::parse (argc, argv, "-c", color_importance);

  float spatial_importance = 0.4f;
  if (pcl::console::find_switch (argc, argv, "-z"))
    pcl::console::parse (argc, argv, "-z", spatial_importance);

  float normal_importance = 1.0f;
  if (pcl::console::find_switch (argc, argv, "-n"))
    pcl::console::parse (argc, argv, "-n", normal_importance);

  int rgb_histbins = 10;
  if (pcl::console::find_switch(argc, argv, "-b"))
    pcl::console::parse(argc, argv, "-b", rgb_histbins);

  //////////////////////////////  //////////////////////////////
  ////// This is how to use supervoxels
  //////////////////////////////  //////////////////////////////

  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
  super.setUseSingleCameraTransform (false);
  super.setInputCloud (cloud);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);

  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

  pcl::console::print_highlight ("Extracting supervoxels!\n");
  super.extract (supervoxel_clusters);
  pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);

  if (DEBUG){
    PointCloudT::Ptr voxel_centroid_cloud = super.getVoxelCentroidCloud ();
    viewer->addPointCloud (voxel_centroid_cloud, "voxel centroids");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2.0, "voxel centroids");

    PointLCloudT::Ptr labeled_voxel_cloud = super.getLabeledVoxelCloud ();
  //PointCloudT::Ptr labeled_voxel_cloud = super.getColoredVoxelCloud ();
    viewer->addPointCloud (labeled_voxel_cloud, "labeled voxels");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, "labeled voxels");

    PointNCloudT::Ptr sv_normal_cloud = super.makeSupervoxelNormalCloud (supervoxel_clusters);
    //We have this disabled so graph is easy to see, uncomment to see supervoxel normals
    //viewer->addPointCloudNormals<PointNT> (sv_normal_cloud,1,0.05f, "supervoxel_normals");
  }

  // Person clouds and their centroid points as lists: [PointCloudT] & [PointXYZRGBA]
  std::vector<PointCloudT::Ptr> final_objclouds;
  std::vector<PointT> final_centroids;
  PointCloudT::Ptr obj_centroids = boost::shared_ptr <PointCloudT> (new PointCloudT ());


  pcl::console::print_highlight ("Getting supervoxel adjacency\n");
  std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
  super.getSupervoxelAdjacency (supervoxel_adjacency);
  //To make a graph of the supervoxel adjacency, we need to iterate through the supervoxel adjacency multimap
  std::multimap<uint32_t,uint32_t>::iterator label_itr = supervoxel_adjacency.begin ();
  std::map<uint32_t,bool> processed_clusters;

  for ( ; label_itr != supervoxel_adjacency.end (); )
  {
    //First get the label
    uint32_t supervoxel_label = label_itr->first;
    if( processed_clusters.find(supervoxel_label) != processed_clusters.end()){
      label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);
      continue;
    }
    processed_clusters[supervoxel_label] = true;
    //Now get the supervoxel corresponding to the label
    pcl::Supervoxel<PointT>::Ptr supervoxel = supervoxel_clusters.at (supervoxel_label);
    if (DEBUG)
      pcl::console::print_info("Supervoxel cluster: %d\n", supervoxel_label);

    //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
    PointCloudT adjacent_supervoxel_centers;
    PointCloudT::Ptr currObj = supervoxel->voxels_;
    std::multimap<uint32_t,uint32_t>::iterator adjacent_itr = supervoxel_adjacency.equal_range (supervoxel_label).first;
    for ( ; adjacent_itr!=supervoxel_adjacency.equal_range (supervoxel_label).second; ++adjacent_itr)
    {
      processed_clusters[adjacent_itr->second] = true;
      if (DEBUG)
        pcl::console::print_info("merging with cluster: %d\n", adjacent_itr->second);
      pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = supervoxel_clusters.at (adjacent_itr->second);
      adjacent_supervoxel_centers.push_back (neighbor_supervoxel->centroid_);
      PointCloudT::Ptr tmpcl = neighbor_supervoxel->voxels_;

      currObj->insert(currObj->end(),tmpcl->begin(), tmpcl->end());
    }

    //Now we make a name for this polygon
    std::stringstream ss;
    ss << "supervoxel_" << supervoxel_label;
    //This function below generates a "star" polygon mesh from the points given
    addSupervoxelConnectionsToViewer (supervoxel->centroid_, adjacent_supervoxel_centers, ss.str (), viewer);

    //Add this adjacent_supervoxel_centers point cloud - considered as a single object in the scene
    final_objclouds.push_back(currObj);
    final_centroids.push_back(supervoxel->centroid_);

    //Move iterator forward to next label
    label_itr = supervoxel_adjacency.upper_bound (supervoxel_label);

  }

  //Add individual object point clouds to final_objclouds, (those left out in the adjacency graph)
  std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr >::iterator cluster_itr = supervoxel_clusters.begin();
  for( ; cluster_itr != supervoxel_clusters.end() ; ++cluster_itr ){
    if ( processed_clusters.find(cluster_itr->first) == processed_clusters.end() ){

      // Naive thresholding based on size of point cloud to eliminate small/noisy objects/points
      if( cluster_itr->second->voxels_->size() < 25 )
        continue;
      //Add this point cloud - considered as a single object in the scene
      final_objclouds.push_back(cluster_itr->second->voxels_);
      final_centroids.push_back(cluster_itr->second->centroid_);
    }
  }

  // Cluster final object clouds based on their rgb histograms
  std::vector<int> labels;
  std::map<int,int> labelCount;
  std::vector<std::string> output(3,"[");
  color_hist_clusters(final_objclouds,labels, rgb_histbins);
  uint32_t uniqcolors[] = {255 << 16 | 0 << 8 | 0,
                           0 << 16 | 255 << 8 | 0,
                           0 << 16 | 0 << 8 | 255};

  //visualize and process the final obj clouds for RGB based person classification
  for( int i = 0 ; i < final_objclouds.size() ; i++ ){

    uint32_t rgbD = (uint32_t)final_centroids[i].rgba;
    uint16_t rD = (rgbD >> 16) & 0x0000ff;
    uint16_t gD = (rgbD >> 8) & 0x0000ff;
    uint16_t bD = (rgbD) & 0x0000ff;
    if (DEBUG){
      pcl::console::print_info("RGB: (%d)", (int)final_centroids[i].rgba);
      pcl::console::print_info("RGB: (%d, %d, %d)", rD, gD, bD);
      pcl::console::print_highlight("Person %d at X,Y = (%.2f, %.2f)\n",
                                    i+1,
                                    final_centroids[i].x,
                                    final_centroids[i].y);
    }

    // color the object centroid with unique color for teamA/teamB/referree
    PointT centroid = final_centroids[i];
    centroid.rgba = uniqcolors[labels[i]];
    obj_centroids -> push_back(centroid);

    // output for corresponding teamA/teamB/referee and track label occurrence
    std::stringstream curr_output;
    curr_output << std::setprecision(2);
    curr_output << " (" << final_centroids[i].x << "," << final_centroids[i].y << "), ";
    output[labels[i]] += curr_output.str();
    if (labelCount.find(labels[i]) == labelCount.end()){
      labelCount[labels[i]] = 1;
    }else{
      labelCount[labels[i]] += 1;
    }

    // Visualize  RGB colored point cloud for each identified object i.e. individual cluster
    if (VISUALIZE){
      std::stringstream objid;
      objid << "P_" << i+1 ;
      pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color(final_objclouds[i],rD, gD, bD);
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgba(final_objclouds[i]);
      viewer->setBackgroundColor(255,255,255);
      viewer->addPointCloud<PointT>(final_objclouds[i], rgba, objid.str());
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,0.8, objid.str());
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, objid.str());
    }
  }

  //Display final output as required:
  // TeamA: [[x1,y1],[x2,y2]...]
  // TeamB: [[x1,y1],[x2,y2]...]
  // Referees: [[x1,y1],[x2,y2]...]
  int teamAFlag = true;
  for( std::map<int,int>::iterator l_itr = labelCount.begin() ; l_itr != labelCount.end() ; ++l_itr){
    std::string val = output[l_itr->first];
    val[val.size() - 2] = ' ';
    val[val.size() - 1] = ']';
    if( l_itr->second < 4 ){
      pcl::console::print_highlight("Referees: %s\n", val.c_str());
    }else if(teamAFlag){
      pcl::console::print_highlight("TeamA: %s\n", val.c_str());
      teamAFlag = false;
    }else{
      pcl::console::print_highlight("TeamB: %s\n", val.c_str());
    }
  }
  if (VISUALIZE){

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> uniqcol(obj_centroids);
    viewer->addPointCloud<PointT>(obj_centroids, uniqcol, "people");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY,1.0, "people");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "people");
  }



  while (!viewer->wasStopped ()){
    viewer->spinOnce (100);
  }
  return (0);
}

void
addSupervoxelConnectionsToViewer (PointT &supervoxel_center,
                                  PointCloudT &adjacent_supervoxel_centers,
                                  std::string supervoxel_name,
                                  boost::shared_ptr<pcl::visualization::PCLVisualizer> & viewer)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();

  //Iterate through all adjacent points, and add a center point to adjacent point pair
  PointCloudT::iterator adjacent_itr = adjacent_supervoxel_centers.begin ();
  for ( ; adjacent_itr != adjacent_supervoxel_centers.end (); ++adjacent_itr)
  {
    points->InsertNextPoint (supervoxel_center.data);
    points->InsertNextPoint (adjacent_itr->data);
  }
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
  // Add the points to the dataset
  polyData->SetPoints (points);
  polyLine->GetPointIds  ()->SetNumberOfIds(points->GetNumberOfPoints ());
  for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
    polyLine->GetPointIds ()->SetId (i,i);
  cells->InsertNextCell (polyLine);
  // Add the lines to the dataset
  polyData->SetLines (cells);
  viewer->addModelFromPolyData (polyData,supervoxel_name);
}

bool loadPointCloud(char *point_cloud_data_file, PointCloudT::Ptr cloud){
  ifstream fp(point_cloud_data_file);
  PointT currPoint;

  int count = 0;
  uint32_t r, g, b, a = 255;
  while( fp >> currPoint.x ){
    try{
      fp >> currPoint.z >> currPoint.y;
      fp >> r >> g >> b;
    }catch(int e){
      pcl::console::print_error ("Error %d occurred\n", e);
      return false;
    }
    uint32_t rgba = (static_cast<uint32_t>(r) << 16 |
              static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b)); //  | static_cast<uint32_t>(a));
    currPoint.rgba = *reinterpret_cast<uint32_t*>(&rgba);
    //pcl::console::print_info("%u, %u, %u, %u\n", r, g, b, currPoint.rgba);
    cloud -> points.push_back(currPoint);
    count++;

  }
  return true;
}

void color_hist_clusters(std::vector<PointCloudT::Ptr> final_objclouds, std::vector<int>& labelvecs, int histSize){
  float range[] = { 0, 255 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  cv::Mat histfeatures = cv::Mat::zeros(final_objclouds.size(), histSize*3, CV_32FC1);
  for( int c = 0 ; c < final_objclouds.size() ; c++ ){

      PointCloudT::Ptr cloud = final_objclouds[c];
      cv::Mat b_hist, g_hist, r_hist;

      cv::Mat rvals = cv::Mat::zeros(cloud->size(),1, CV_32FC1);
      cv::Mat gvals = cv::Mat::zeros(cloud->size(),1, CV_32FC1);
      cv::Mat bvals = cv::Mat::zeros(cloud->size(),1, CV_32FC1);

      int i = 0;
      for( PointCloudT::iterator pitr = cloud->begin()  ; pitr != cloud->end() ; ++pitr, ++i ){
        uint32_t rgbD = pitr->rgba;
        uint16_t rD = (rgbD >> 16) & 0x0000ff;
        uint16_t gD = (rgbD >> 8) & 0x0000ff;
        uint16_t bD = (rgbD) & 0x0000ff;
        rvals.at<float>(i,0) = (float)rD;
        gvals.at<float>(i,0) = (float)gD;
        bvals.at<float>(i,0) = (float)bD;
        //if (DEBUG)
        //  pcl::console::print_info("(%u, %u, %u) ", rD, gvals.at<uchar>(i,0), bD);
      }
      cv::calcHist( &rvals, 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
      cv::calcHist( &gvals, 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
      cv::calcHist( &bvals, 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );


      /// Normalize the result to [ 0, 20]
      cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
      cv::normalize(g_hist, g_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
      cv::normalize(r_hist, r_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );


      for( int j = 0 ; j < r_hist.rows ; j++ ){
        histfeatures.at<float>(c,j) = r_hist.at<float>(j,0);
        if(DEBUG)
          pcl::console::print_info("%.4f ", r_hist.at<float>(j,0));
      }
      for( int j = 0 ; j < g_hist.rows ; j++ ){
        histfeatures.at<float>(c,j+histSize) = g_hist.at<float>(j,0);
        if(DEBUG)
          pcl::console::print_info("%.4f ", g_hist.at<float>(j,0));
      }
      for( int j = 0 ; j < b_hist.rows ; j++ ){
        histfeatures.at<float>(c,j+2*histSize) = b_hist.at<float>(j,0);
        if(DEBUG)
          pcl::console::print_info("%.4f ", b_hist.at<float>(j,0));
      }
  }
  cv::Mat labels;
  int attempts = 5;
  cv::Mat centers;
  cv::kmeans(histfeatures, 3, labels,
        cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
        attempts, cv::KMEANS_PP_CENTERS, centers );
  for(int i = 0 ; i < labels.rows ; i++ ){
    labelvecs.push_back(labels.at<int>(i,0));
    //pcl::console::print_info("%d\n", labels.at<int>(i,0));
  }

}
