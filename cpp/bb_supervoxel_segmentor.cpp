#include "bb_supervoxel_segmentor.hpp"

BBSupervoxelSegmentor::BBSupervoxelSegmentor(char *point_cloud_data_file){
    cloud = boost::shared_ptr <PointCloudT> (new PointCloudT ());

    pcl::console::print_highlight ("Loading point cloud...\n");
    if (!loadPointCloud(point_cloud_data_file)){
        pcl::console::print_error ("Error loading cloud file!\n");
    }
    if( DEBUG )
        pcl::console::print_highlight("%d cloud size loaded", cloud -> size());
}

void BBSupervoxelSegmentor::setParameters(float voxel_resolution, float seed_resolution, float color_importance,
                            float spatial_importance, float normal_importance, int rgb_histbins,
                            bool DEBUG, bool VISUALIZE){
    pcl::console::print_highlight("Setting config parameters...\n");

    voxel_resolution = voxel_resolution;
    seed_resolution = seed_resolution;
    color_importance = color_importance;
    spatial_importance = spatial_importance;
    normal_importance = normal_importance;
    rgb_histbins = rgb_histbins;
    DEBUG = DEBUG;
    VISUALIZE = VISUALIZE;
}

bool BBSupervoxelSegmentor::loadPointCloud(char* point_cloud_data_file){
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

void BBSupervoxelSegmentor::supervoxelClustering(std::vector<PointCloudT::Ptr>& final_objclouds,
                                                 std::vector<PointT>& final_centroids){
    pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
    super.setUseSingleCameraTransform (false);
    super.setInputCloud (cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);

    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

    pcl::console::print_highlight ("Extracting supervoxels!\n");
    super.extract (supervoxel_clusters);
    if (DEBUG)
        pcl::console::print_info ("Found %d supervoxels\n", supervoxel_clusters.size ());

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
}

void BBSupervoxelSegmentor::colorHistClusters(std::vector<PointCloudT::Ptr> final_objclouds,
                                                    std::vector<int>& labelvecs, int histSize){
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

    //Cluster with k-means for 3 groups (teamA, teamB and Referees)
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

bool BBSupervoxelSegmentor::process_pointcloud(){
    if (cloud -> size() == 0 ){
    // point cloud data not loaded
        return false;
    }
    pcl::console::print_highlight("Processing starts...\n");

    // Person clouds and their centroid points as lists: [PointCloudT] & [PointXYZRGBA]
    std::vector<PointCloudT::Ptr> final_objclouds;
    std::vector<PointT> final_centroids;
    obj_centroids = boost::shared_ptr <PointCloudT> (new PointCloudT ());
    viewer =  boost::shared_ptr <pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    // call to cluster into supervoxels and get final obj clouds and centroids
    supervoxelClustering(final_objclouds, final_centroids);

    // Cluster final object clouds based on their rgb histograms
    std::vector<int> labels;
    std::map<int,int> labelCount;
    colorHistClusters(final_objclouds,labels, rgb_histbins);

    // Output result and visualize the same
    std::vector<std::string> output(3,"[");
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



    while (VISUALIZE && !viewer->wasStopped ()){
        viewer->spinOnce (100);
    }
    return true;
}