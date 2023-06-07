#include <unistd.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>


#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>

// GridMapRosConverter includes cv_bridge which includes OpenCV4 which uses _Atomic
// We want to ignore this warning entirely.
#if defined(__clang__)
# pragma clang diagnostic push
#endif

#if defined(__clang__) && defined(__has_warning)
# if __has_warning( "-Wc11-extensions" )
#  pragma clang diagnostic ignored "-Wc11-extensions"
# endif
#endif

#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

// tf
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

#include "plane_seg/BlockFitter.hpp"

// #define WITH_TIMING

#ifdef WITH_TIMING
#include <chrono>
#endif


// convenience methods
auto vecToStr = [](const Eigen::Vector3f& iVec) {
  std::ostringstream oss;
  oss << iVec[0] << ", " << iVec[1] << ", " << iVec[2];
  return oss.str();
};
auto rotToStr = [](const Eigen::Matrix3f& iRot) {
  std::ostringstream oss;
  Eigen::Quaternionf q(iRot);
  oss << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z();
  return oss.str();
};


class Pass{
  public:
    Pass(ros::NodeHandle node_);
    ~Pass() = default;

    void elevationMapCallback(const grid_map_msgs::GridMap& msg);
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void processCloud(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir);
    void processFromFile(int test_example);

    void publishHullsAsCloud(const std::string& cloud_frame, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_ptrs, int secs, int nsecs);

    void publishHullsAsMarkers(const std::string& cloud_frame, std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs, int secs, int nsecs);
    void publishHullsAsMarkerArray(const std::string& cloud_frame, std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs, int secs, int nsecs);
    void printResultAsJson();
    void publishResult(const std::string& cloud_frame);

  private:
    ros::NodeHandle node_;
    std::vector<double> colors_;

    ros::Subscriber point_cloud_sub_, grid_map_sub_, pose_sub_;
    ros::Publisher received_cloud_pub_, filtted_cloud_pub_, hull_cloud_pub_, hull_markers_pub_, look_pose_pub_, hull_marker_array_pub_;

    std::string fixed_frame_ = "odom";  // Frame in which all results are published. "odom" for backwards-compatibility. Likely should be "map".

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    planeseg::BlockFitter::Result result_;
};

Pass::Pass(ros::NodeHandle node_):
    node_(node_),
    tfBuffer_(ros::Duration(5.0)),
    tfListener_(tfBuffer_) {
  grid_map_sub_ = node_.subscribe("/os_sensor", 100,
                                    &Pass::elevationMapCallback, this);
  point_cloud_sub_ = node_.subscribe("/ouster/points2", 100,
                                    &Pass::pointCloudCallback, this);
  // grid_map_sub_ = node_.subscribe("/elevation_mapping/elevation_map", 100,
  //                                   &Pass::elevationMapCallback, this);
  // point_cloud_sub_ = node_.subscribe("/plane_seg/point_cloud_in", 100,
  //                                   &Pass::pointCloudCallback, this);

  received_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg/received_cloud", 10);
  filtted_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg/filtted_cloud", 10);
  hull_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg/hull_cloud", 10);
  hull_markers_pub_ = node_.advertise<visualization_msgs::Marker>("/plane_seg/hull_markers", 10);
  hull_marker_array_pub_ = node_.advertise<visualization_msgs::MarkerArray>("/plane_seg/hull_marker_array", 10);
  look_pose_pub_ = node_.advertise<geometry_msgs::PoseStamped>("/plane_seg/look_pose", 10);

  colors_ = {
       51/255.0, 160/255.0, 44/255.0,  //0
       166/255.0, 206/255.0, 227/255.0,
       178/255.0, 223/255.0, 138/255.0,//6
       31/255.0, 120/255.0, 180/255.0,
       251/255.0, 154/255.0, 153/255.0,// 12
       227/255.0, 26/255.0, 28/255.0,
       253/255.0, 191/255.0, 111/255.0,// 18
       106/255.0, 61/255.0, 154/255.0,
       255/255.0, 127/255.0, 0/255.0, // 24
       202/255.0, 178/255.0, 214/255.0,
       1.0, 0.0, 0.0, // red // 30
       0.0, 1.0, 0.0, // green
       0.0, 0.0, 1.0, // blue// 36
       1.0, 1.0, 0.0,
       1.0, 0.0, 1.0, // 42
       0.0, 1.0, 1.0,
       0.5, 1.0, 0.0,
       1.0, 0.5, 0.0,
       0.5, 0.0, 1.0,
       1.0, 0.0, 0.5,
       0.0, 0.5, 1.0,
       0.0, 1.0, 0.5,
       1.0, 0.5, 0.5,
       0.5, 1.0, 0.5,
       0.5, 0.5, 1.0,
       0.5, 0.5, 1.0,
       0.5, 1.0, 0.5,
       0.5, 0.5, 1.0};

}

// from one frame to another frame: 
// R^(3*3)
// Quaternion: qx, qy, qz, qw
// roll, pitch and yaw
// convert Quaterniond to roll, pitch and yaw
void quat_to_euler(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw) {
  const double q0 = q.w();
  const double q1 = q.x();
  const double q2 = q.y();
  const double q3 = q.z();
  roll = atan2(2.0*(q0*q1+q2*q3), 1.0-2.0*(q1*q1+q2*q2));
  pitch = asin(2.0*(q0*q2-q3*q1));
  yaw = atan2(2.0*(q0*q3+q1*q2), 1.0-2.0*(q2*q2+q3*q3));
}

// convert robot rotation pose (Quaternion) to roll, pitch and yaw
// (xDir, yDir, zDir) is the projection of a point in original frame to the new frame
Eigen::Vector3f convertRobotPoseToSensorLookDir(Eigen::Isometry3d robot_pose) {
  Eigen::Quaterniond quat = Eigen::Quaterniond( robot_pose.rotation() );
  double r,p,y;
  quat_to_euler(quat, r, p, y);
  //std::cout << r*180/M_PI << ", " << p*180/M_PI << ", " << y*180/M_PI << " rpy in Degrees\n";

  double yaw = y;
  double pitch = -p;
  double xDir = cos(yaw)*cos(pitch);
  double yDir = sin(yaw)*cos(pitch);
  double zDir = sin(pitch);
  return Eigen::Vector3f(xDir, yDir, zDir);
}

// elevation map call back function
void Pass::elevationMapCallback(const grid_map_msgs::GridMap& msg){
  //std::cout << "got grid map / ev map\n";

  // convert message to GridMap, to PointCloud to LabeledCloud
  grid_map::GridMap map;
  grid_map::GridMapRosConverter::fromMessage(msg, map);

  sensor_msgs::PointCloud2 pointCloud;
  grid_map::GridMapRosConverter::toPointCloud(map, "elevation", pointCloud);

  planeseg::LabeledCloud::Ptr inCloud(new planeseg::LabeledCloud());
  pcl::fromROSMsg(pointCloud,*inCloud);

  // Expressed in is first argument
  if (tfBuffer_.canTransform(fixed_frame_, map.getFrameId(), msg.info.header.stamp, ros::Duration(0.02)))
  {
    // this function is to find transformation from fixed_frame_ to the captured map
    // Look up transform of elevation map frame when it was captured
    geometry_msgs::TransformStamped fixed_frame_to_elevation_map_frame_tf;
    Eigen::Isometry3d map_T_elevation_map;

    fixed_frame_to_elevation_map_frame_tf = tfBuffer_.lookupTransform(fixed_frame_, map.getFrameId(), msg.info.header.stamp, ros::Duration(0.02));
    map_T_elevation_map = tf2::transformToEigen(fixed_frame_to_elevation_map_frame_tf);
  
    Eigen::Vector3f origin, lookDir;
    origin << map_T_elevation_map.translation().cast<float>();
    lookDir = convertRobotPoseToSensorLookDir(map_T_elevation_map);

    // ROS_INFO_STREAM("Received new elevation map & looked up transform -- processing.");
    processCloud(map.getFrameId(), inCloud, origin, lookDir);
  }
  else
  {
    ROS_WARN_STREAM("Cannot look up transform from '" << map.getFrameId() << "' to fixed frame ('" << fixed_frame_ <<"'). Skipping elevation map.");
  }
}


// process a point cloud 
// This method is mostly for testing
// To transmit a static point cloud:
// rosrun pcl_ros pcd_to_pointcloud 06.pcd   _frame_id:=/odom /cloud_pcd:=/plane_seg/point_cloud_in
void Pass::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
  planeseg::LabeledCloud::Ptr inCloud(new planeseg::LabeledCloud());
  pcl::fromROSMsg(*msg,*inCloud);

  // Look up transform from fixed frame to point cloud frame (?)
  // "msg->header.frame_id" is the frame where the data originated 
  // I think the process should tarnsform the point cloud data to a fixed frame
  geometry_msgs::TransformStamped fixed_frame_to_cloud_frame_tf;
  Eigen::Isometry3d map_T_pointcloud = Eigen::Isometry3d::Identity();

  if (tfBuffer_.canTransform(fixed_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.0)))
  {
    fixed_frame_to_cloud_frame_tf = tfBuffer_.lookupTransform(fixed_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.0));
    map_T_pointcloud = tf2::transformToEigen(fixed_frame_to_cloud_frame_tf);
  }
  else
  {
    ROS_WARN_STREAM("Cannot look up transform from '" << msg->header.frame_id << "' to fixed frame ('" << fixed_frame_ <<"')");
  }

  Eigen::Vector3f origin, lookDir;
  origin << map_T_pointcloud.translation().cast<float>();
  lookDir = convertRobotPoseToSensorLookDir(map_T_pointcloud);

  processCloud(msg->header.frame_id, inCloud, origin, lookDir);
}


void Pass::processFromFile(int test_example){

  // to allow ros connections to register
  sleep(2);

  std::string inFile;
  std::string home_dir = ros::package::getPath("plane_seg_ros");
  Eigen::Vector3f origin, lookDir;
  if (test_example == 0){ // LIDAR example from Atlas during DRC
    inFile = home_dir + "/data/terrain/tilted-steps.pcd";
    origin <<0.248091, 0.012443, 1.806473;
    lookDir <<0.837001, 0.019831, -0.546842;
    // origin << 100, 100, 100;
    // lookDir << 100, 100, 100;
  }else if (test_example == 1){ // LIDAR example from Atlas during DRC
    inFile = home_dir + "/data/terrain/terrain_med.pcd";
    origin << -0.028862, -0.007466, 0.087855;
    lookDir << 0.999890, -0.005120, -0.013947;
    // origin << 100, 100, 100;
    // lookDir << 100, 100, 100;
  }else if (test_example == 2){ // LIDAR example from Atlas during DRC
    inFile = home_dir + "/data/terrain/terrain_close_rect.pcd";
    origin << -0.028775, -0.005776, 0.087898;
    lookDir << 0.999956, -0.005003, 0.007958;
    // origin << 100, 100, 100;
    // lookDir << 100, 100, 100;
  }else if (test_example == 3){ // RGBD (Realsense D435) example from ANYmal
    inFile = home_dir + "/data/terrain/anymal/ori_entrance_stair_climb/06.pcd";
    origin << -0.028775, -0.005776, 0.987898;
    lookDir << 0.999956, -0.005003, 0.007958;
    // origin << 100, 100, 100;
    // lookDir << 100, 100, 100;
  }else if (test_example == 4){ // Leica map
    inFile = home_dir + "/data/leica/race_arenas/RACE_crossplaneramps_sub1cm_cropped_meshlab_icp.ply";
    origin << -0.028775, -0.005776, 0.987898;
    lookDir << 0.999956, -0.005003, 0.007958;
  }else if (test_example == 5){ // Leica map
    inFile = home_dir + "/data/leica/race_arenas/RACE_stepfield_sub1cm_cropped_meshlab_icp.ply";
    origin << -0.028775, -0.005776, 0.987898;
    lookDir << 0.999956, -0.005003, 0.007958;
  }

  std::cout << "\nProcessing test example " << test_example << "\n";
  std::cout << inFile << "\n";

  std::size_t found_ply = inFile.find(".ply");
  std::size_t found_pcd = inFile.find(".pcd");

  planeseg::LabeledCloud::Ptr inCloud(new planeseg::LabeledCloud());
  if (found_ply!=std::string::npos){
    std::cout << "readply\n";
    pcl::io::loadPLYFile(inFile, *inCloud);
  }else if (found_pcd!=std::string::npos){
    std::cout << "readpcd\n";
    pcl::io::loadPCDFile(inFile, *inCloud);
  }else{
    std::cout << "extension not understood\n";
    return;
  }

  processCloud(fixed_frame_, inCloud, origin, lookDir);
}


void Pass::processCloud(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir){
#ifdef WITH_TIMING
  auto tic = std::chrono::high_resolution_clock::now();
#endif

  planeseg::BlockFitter fitter;
  fitter.setSensorPose(origin, lookDir);
  fitter.setCloud(inCloud);
  fitter.setDebug(true); // MFALLON modification
  fitter.setRemoveGround(false); // MFALLON modification from default

  // this was 5 for LIDAR. changing to 10 really improved elevation map segmentation
  // I think its because the RGB-D map can be curved
  fitter.setMaxAngleOfPlaneSegmenter(10);

  result_ = fitter.go();

#ifdef WITH_TIMING
  auto toc_1 = std::chrono::high_resolution_clock::now();
#endif

  // publish a geometry_msgs::PoseStamped message
  // std_msgs/Header header
  // geometry_msgs/Pose pose
  if (look_pose_pub_.getNumSubscribers() > 0) {
    Eigen::Vector3f rz = lookDir;
    Eigen::Vector3f rx = rz.cross(Eigen::Vector3f::UnitZ());
    Eigen::Vector3f ry = rz.cross(rx);
    Eigen::Matrix3f rotation;
    rotation.col(0) = rx.normalized();
    rotation.col(1) = ry.normalized();
    rotation.col(2) = rz.normalized();
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = rotation;
    pose.translation() = origin;
    Eigen::Isometry3d pose_d = pose.cast<double>();

    geometry_msgs::PoseStamped msg;
    msg.header.stamp = ros::Time(0, 0);
    msg.header.frame_id = cloudFrame;
    tf::poseEigenToMsg(pose_d, msg.pose);
    look_pose_pub_.publish(msg);
  }

  // publish the sensor data
  if (received_cloud_pub_.getNumSubscribers() > 0) {
    // This message is published correctly
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*inCloud, output);
    output.header.stamp = ros::Time(0, 0);
    output.header.frame_id = cloudFrame;
    received_cloud_pub_.publish(output);
  }

  // publish the sensor data
  if (filtted_cloud_pub_.getNumSubscribers() > 0) {
    // This message is published correctly
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*result_.filttedCloud, output);
    output.header.stamp = ros::Time(0, 0);
    output.header.frame_id = cloudFrame;
    filtted_cloud_pub_.publish(output);
  }

  // publish fitting result in cloudFrame
  publishResult(cloudFrame);

#ifdef WITH_TIMING
  auto toc_2 = std::chrono::high_resolution_clock::now();

  std::cout << "[BlockFitter] took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(toc_1 - tic).count() << "ms" << std::endl;
  // std::cout << "[Publishing] took " << 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(toc_2 - toc_1).count() << "ms" << std::endl;
#endif
}


void Pass::printResultAsJson(){
  std::string json;

  for (int i = 0; i < (int)result_.mBlocks.size(); ++i) {
    const auto& block = result_.mBlocks[i];
    std::string dimensionString = vecToStr(block.mSize);
    std::string positionString = vecToStr(block.mPose.translation());
    std::string quaternionString = rotToStr(block.mPose.rotation());
    Eigen::Vector3f color(0.5, 0.4, 0.5);
    std::string colorString = vecToStr(color);
    float alpha = 1.0;
    std::string uuid = "0_" + std::to_string(i+1);
    
    json += "    \"" + uuid + "\": {\n";
    json += "      \"classname\": \"BoxAffordanceItem\",\n";
    json += "      \"pose\": [[" + positionString + "], [" +
      quaternionString + "]],\n";
    json += "      \"uuid\": \"" + uuid + "\",\n";
    json += "      \"Dimensions\": [" + dimensionString + "],\n";
    json += "      \"Color\": [" + colorString + "],\n";
    json += "      \"Alpha\": " + std::to_string(alpha) + ",\n";
    json += "      \"Name\": \" mNamePrefix " +
      std::to_string(i) + "\"\n";
    json += "    },\n";

  }

  std::cout << json << "\n";
}


void Pass::publishResult(const std::string& cloud_frame){
  // convert result to a vector of point clouds
  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs;

  for (size_t i=0; i<result_.mBlocks.size(); ++i){
    pcl::PointCloud<pcl::PointXYZ> cloud;
    const auto& block = result_.mBlocks[i];
    cloud.points.reserve(block.mHull.size());
    // std::cout << "block results: " << result_.mBlocks.size() << ", " << block.mHull.size() << std::endl;

    if (block.mHull.size() == 0) continue;
    for (size_t j =0; j < block.mHull.size(); ++j){
      pcl::PointXYZ pt;
      pt.x =block.mHull[j](0);
      pt.y =block.mHull[j](1);
      pt.z =block.mHull[j](2);
      cloud.points.push_back(pt);
    }
    cloud.height = cloud.points.size();
    cloud.width = 1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
    cloud_ptr = cloud.makeShared();
    cloud_ptrs.push_back(cloud_ptr);
  }

  if (hull_cloud_pub_.getNumSubscribers() > 0) publishHullsAsCloud(cloud_frame, cloud_ptrs, 0, 0);
  if (hull_markers_pub_.getNumSubscribers() > 0) publishHullsAsMarkers(cloud_frame, cloud_ptrs, 0, 0);
  if (hull_marker_array_pub_.getNumSubscribers() > 0) publishHullsAsMarkerArray(cloud_frame, cloud_ptrs, 0, 0);

  //pcl::PCDWriter pcd_writer_;
  //pcd_writer_.write<pcl::PointXYZ> ("/home/mfallon/out.pcd", cloud, false);
  //std::cout << "blocks: " << result_.mBlocks.size() << " blocks\n";
  //std::cout << "cloud: " << cloud.points.size() << " pts\n";
}


// combine the individual clouds into one, with a different each
void Pass::publishHullsAsCloud(const std::string& cloud_frame,
                               std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                               int secs, int nsecs){
  pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;

  for (size_t i=0; i<cloud_ptrs.size(); ++i){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_ptrs[i], *cloud_rgb);

    int nColor = i % (colors_.size()/3);
    double r = colors_[nColor*3]*255.0;
    double g = colors_[nColor*3+1]*255.0;
    double b = colors_[nColor*3+2]*255.0;
    
    for (size_t j = 0; j < cloud_rgb->points.size (); j++){
        cloud_rgb->points[j].r = r;
        cloud_rgb->points[j].g = g;
        cloud_rgb->points[j].b = b;
    }
    combined_cloud += *cloud_rgb;
  }

  std::cout << "size of combined cloud: " << combined_cloud.size() << std::endl; // new
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(combined_cloud, output);

  output.header.stamp = ros::Time(secs, nsecs);
  output.header.frame_id = cloud_frame;
  hull_cloud_pub_.publish(output);
}


void Pass::publishHullsAsMarkers(const std::string& cloud_frame,
                                 std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                                 int secs, int nsecs){
  geometry_msgs::Point point;
  std_msgs::ColorRGBA point_color;
  visualization_msgs::Marker marker;

  // define markers
  marker.header.frame_id = cloud_frame;
  marker.header.stamp = ros::Time(secs, nsecs);
  marker.ns = "hull lines";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST; //visualization_msgs::Marker::POINTS;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.03;
  marker.scale.y = 0.03;
  marker.scale.z = 0.03;
  marker.color.a = 1.0;

  for (size_t i = 0; i < cloud_ptrs.size (); i++){
    int nColor = i % (colors_.size()/3);
    double r = colors_[nColor*3]*255.0;
    double g = colors_[nColor*3+1]*255.0;
    double b = colors_[nColor*3+2]*255.0;

    for (size_t j = 1; j < cloud_ptrs[i]->points.size (); j++){
      point.x = cloud_ptrs[i]->points[j-1].x;
      point.y = cloud_ptrs[i]->points[j-1].y;
      point.z = cloud_ptrs[i]->points[j-1].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);

      //
      point.x = cloud_ptrs[i]->points[j].x;
      point.y = cloud_ptrs[i]->points[j].y;
      point.z = cloud_ptrs[i]->points[j].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);
    }

    // start to end line:
    point.x = cloud_ptrs[i]->points[0].x;
    point.y = cloud_ptrs[i]->points[0].y;
    point.z = cloud_ptrs[i]->points[0].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    point.x = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].x;
    point.y = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].y;
    point.z = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);
  }
  marker.frame_locked = true;
  hull_markers_pub_.publish(marker);
}

void Pass::publishHullsAsMarkerArray(const std::string& cloud_frame,
                                     std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                                     int secs, int nsecs){
  geometry_msgs::Point point;
  std_msgs::ColorRGBA point_color;
  visualization_msgs::MarkerArray ma;

  for (size_t i = 0; i < cloud_ptrs.size (); i++){
    visualization_msgs::Marker marker;
    marker.header.frame_id = cloud_frame;
    marker.header.stamp = ros::Time(secs, nsecs);
    marker.ns = "hull_" + std::to_string(i);
    marker.id = i;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;
    marker.color.a = 1.0;

    const int nColor = i % (colors_.size()/3);
    const double r = colors_[nColor*3]*255.0;
    const double g = colors_[nColor*3+1]*255.0;
    const double b = colors_[nColor*3+2]*255.0;

    marker.points.reserve(cloud_ptrs[i]->points.size());
    marker.colors.reserve(cloud_ptrs[i]->points.size());
    for (size_t j = 1; j < cloud_ptrs[i]->points.size(); j++){
      point.x = cloud_ptrs[i]->points[j-1].x;
      point.y = cloud_ptrs[i]->points[j-1].y;
      point.z = cloud_ptrs[i]->points[j-1].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);

      point.x = cloud_ptrs[i]->points[j].x;
      point.y = cloud_ptrs[i]->points[j].y;
      point.z = cloud_ptrs[i]->points[j].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);
    }

    // start to end line:
    point.x = cloud_ptrs[i]->points[0].x;
    point.y = cloud_ptrs[i]->points[0].y;
    point.z = cloud_ptrs[i]->points[0].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    point.x = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].x;
    point.y = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].y;
    point.z = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    marker.frame_locked = true;
    ma.markers.push_back(marker);
  }
  hull_marker_array_pub_.publish(ma);
}

int main( int argc, char** argv ){
  // Turn off warning message about labels
  // TODO: look into how labels are used
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);


  ros::init(argc, argv, "plane_seg");
  ros::NodeHandle nh("~"); // new
  Pass pass_pt (nh); // new
  std::unique_ptr<Pass> app = std::make_unique<Pass>(nh);

  ROS_INFO_STREAM("plane_seg ros ready");
  ROS_INFO_STREAM("=============================");

  bool run_test_program = false;
  nh.param("/plane_seg/run_test_program", run_test_program, false); 
  std::cout << "run_test_program: " << run_test_program << "\n";


  // Enable this to run the test programs
  if (run_test_program){
    std::cout << "Running test examples\n";
    app->processFromFile(0);
    app->processFromFile(1);
    app->processFromFile(2);
    app->processFromFile(3);
    // RACE examples don't work well
    //app->processFromFile(4);
    //app->processFromFile(5);

    std::cout << "Finished!\n";
    exit(-1);
  }

  ROS_INFO_STREAM("Waiting for ROS messages");
  // ros::Subscriber grid_map_sub_ = nh.subscribe("/os_sensor", 100,
  //                                   &Pass::elevationMapCallback, &pass_pt);
  ros::Subscriber point_cloud_sub_ = nh.subscribe ("/ouster/points", 100,
                                    &Pass::pointCloudCallback, &pass_pt); // new
  ros::spin();
  // new
  // ros::Rate loop_rate (30);
  // while (ros::ok ())
  // {
  //   ros::spinOnce ();
  //   loop_rate.sleep ();
  // }

  return 1;
}
