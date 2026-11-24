// Derived from Diff-planner-indoor indoor_navigation cloud registration.
// GPL-3.0; see LICENSE and PROVENANCE.md. Physical ray returns only.
#include <Eigen/Geometry>
#include <geometry_msgs/Point32.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <vla_sim/indoor_sampling.h>
#include <vla_sim/acquisition_sync.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

class VlaCloudBridge
{
public:
  VlaCloudBridge()
  {
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    pnh.param<std::string>("input_topic", input_topic_,
                           "/vla/lidar/points_raw");
    pnh.param<std::string>("output_topic", output_topic_,
                           "/cloud_registered");
    pnh.param<std::string>("display_output_topic", display_output_topic_,
                           "/vla/lidar/returns_world");
    pnh.param<std::string>("odom_topic", odom_topic_,
                           "/odom");
    pnh.param<std::string>("world_frame", world_frame_, "world");
    pnh.param("lidar_x", lidar_translation_.x(), 0.0);
    pnh.param("lidar_y", lidar_translation_.y(), 0.0);
    pnh.param("lidar_z", lidar_translation_.z(), 0.16);
    pnh.param("lidar_roll", lidar_rpy_.x(), 0.0);
    pnh.param("lidar_pitch", lidar_rpy_.y(), 0.0);
    pnh.param("lidar_yaw", lidar_rpy_.z(), 0.0);
    pnh.param("range_min", range_min_, 0.10);
    pnh.param("range_max", range_max_, 12.0);
    pnh.param("horizontal_samples", horizontal_samples_, 240);
    pnh.param("vertical_samples", vertical_samples_, 32);
    pnh.param("vertical_min", vertical_min_, -0.12217304764);
    pnh.param("vertical_max", vertical_max_, 0.90757121104);
    pnh.param("drop_max_range_shell", drop_max_range_shell_, true);
    pnh.param("display_drop_max_range_shell",
              display_drop_max_range_shell_, true);
    pnh.param("max_range_shell_margin", max_range_shell_margin_, 0.08);
    pnh.param("min_world_z", min_world_z_, -0.05);
    pnh.param("max_world_z", max_world_z_, 3.35);
    pnh.param("self_filter_radius", self_filter_radius_, 0.30);
    pnh.param("self_filter_half_height", self_filter_half_height_, 0.24);
    pnh.param("point_stride", point_stride_, 1);
    pnh.param("display_point_stride", display_point_stride_, 2);
    pnh.param("odom_history_seconds", odom_history_seconds_, 2.0);
    pnh.param("max_sync_gap", max_sync_gap_, 0.16);
    pnh.param("pending_scan_capacity", pending_scan_capacity_, 4);
    pnh.param("pending_scan_timeout_wall_seconds", pending_timeout_wall_, 0.5);
    pnh.param<std::string>("sensor_profile", sensor_profile_, "fast");
    int sampling_seed = 360;
    pnh.param("sampling_seed", sampling_seed, 360);
    if (sampling_seed < 0 || (sensor_profile_ != "fast" &&
        sensor_profile_ != "dense" && sensor_profile_ != "indoor"))
      throw std::runtime_error("Invalid LiDAR profile or negative sampling seed");
    sampling_.enabled = sensor_profile_ == "indoor";
    sampling_.seed = static_cast<uint64_t>(sampling_seed);

    range_min_ = std::max(0.0, range_min_);
    range_max_ = std::max(range_min_ + 0.1, range_max_);
    sampling_.range_min = range_min_;
    sampling_.range_max = range_max_;
    max_range_shell_margin_ =
        std::max(0.0, std::min(max_range_shell_margin_, range_max_ - range_min_));
    point_stride_ = std::max(1, point_stride_);
    display_point_stride_ = std::max(1, display_point_stride_);
    odom_history_seconds_ = std::max(0.5, odom_history_seconds_);
    max_sync_gap_ = std::max(0.01, max_sync_gap_);
    pending_scan_capacity_ = std::max(1, std::min(8, pending_scan_capacity_));
    if (!std::isfinite(pending_timeout_wall_) || pending_timeout_wall_ <= 0.0)
      throw std::runtime_error("Invalid pending LiDAR scan timeout");
    pending_timeout_wall_ = std::min(2.0, pending_timeout_wall_);
    acquisition_sync_.reset(new ScanSync(
        ros::Duration(odom_history_seconds_).toNSec(),
        ros::Duration(max_sync_gap_).toNSec(), pending_scan_capacity_, pending_timeout_wall_));

    const Eigen::AngleAxisd roll(lidar_rpy_.x(), Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch(lidar_rpy_.y(), Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw(lidar_rpy_.z(), Eigen::Vector3d::UnitZ());
    lidar_rotation_ = (yaw * pitch * roll).toRotationMatrix();

    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_, 2);
    rays_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/vla/lidar/rays_world", 2);
    lidar_odom_pub_ = nh.advertise<nav_msgs::Odometry>("/vla/lidar/odom", 2);
    display_cloud_pub_ =
        nh.advertise<sensor_msgs::PointCloud2>(display_output_topic_, 2);
    status_pub_ = nh.advertise<std_msgs::String>("/vla/lidar/status", 2, true);
    odom_sub_ = nh.subscribe(odom_topic_, 200,
                             &VlaCloudBridge::odomCallback, this,
                             ros::TransportHints().tcpNoDelay());
    cloud_sub_ = nh.subscribe(input_topic_, 2,
                              &VlaCloudBridge::cloudCallback, this,
                              ros::TransportHints().tcpNoDelay());
    pending_timer_ = nh.createWallTimer(ros::WallDuration(0.02),
                                       &VlaCloudBridge::pendingTimer, this);

    ROS_INFO("[VLA LiDAR] register %s -> map=%s, display=%s using %s, "
             "range %.1f m",
             input_topic_.c_str(), output_topic_.c_str(),
             display_output_topic_.c_str(), odom_topic_.c_str(), range_max_);
  }

private:
  using ScanSync = vla_sim::AcquisitionSync<sensor_msgs::PointCloud::ConstPtr>;
  struct ScanCounts {
    size_t input_hits{0}, input_misses{0}, retained_hits{0}, retained_misses{0};
    size_t dropped_sampling{0}, dropped_burst{0}, dropped_range{0};
    size_t dropped_invalid{0}, dropped_self{0};
  } counts_;

  void dropScan(const ScanSync::Pending &scan, vla_sim::SyncDropReason reason)
  {
    ++dropped_sync_frames_;
    counts_ = ScanCounts{};
    last_sync_drop_reason_ = vla_sim::syncDropName(reason);
    ROS_WARN_THROTTLE(1.0, "[VLA LiDAR] drop acquisition: %s; waiting for bracketing odometry",
                      last_sync_drop_reason_.c_str());
    publishStatus(scan.payload->points.size(), 0, 0, 0.0, true);
  }

  void drainPending()
  {
    acquisition_sync_->drain(ros::SteadyTime::now().toSec(),
        [this](const ScanSync::Pending &scan, const vla_sim::AcquisitionPose &pose,
               double error) { registerCloud(scan.payload, pose, error); },
        [this](const ScanSync::Pending &scan, vla_sim::SyncDropReason reason) {
          dropScan(scan, reason);
        });
  }

  void pendingTimer(const ros::WallTimerEvent &) { drainPending(); }

  void odomCallback(const nav_msgs::Odometry::ConstPtr &message)
  {
    vla_sim::AcquisitionPose pose;
    // Preserve acquisition time, including a legitimate simulated t=0. Never
    // substitute callback time for a sensor or odometry timestamp.
    pose.stamp_ns = message->header.stamp.toNSec();
    pose.position = Eigen::Vector3d(message->pose.pose.position.x,
                                    message->pose.pose.position.y,
                                    message->pose.pose.position.z);
    pose.orientation = Eigen::Quaterniond(message->pose.pose.orientation.w,
                                          message->pose.pose.orientation.x,
                                          message->pose.pose.orientation.y,
                                          message->pose.pose.orientation.z);
    acquisition_sync_->addPose(pose,
        [this](const ScanSync::Pending &scan, vla_sim::SyncDropReason reason) {
          dropScan(scan, reason);
        });
    drainPending();
  }

  void cloudCallback(const sensor_msgs::PointCloud::ConstPtr &message)
  {
    acquisition_sync_->enqueue(message->header.stamp.toNSec(),
        ros::SteadyTime::now().toSec(), message,
        [this](const ScanSync::Pending &scan, vla_sim::SyncDropReason reason) {
          dropScan(scan, reason);
        });
    drainPending();
  }

  void registerCloud(const sensor_msgs::PointCloud::ConstPtr &message,
                     const vla_sim::AcquisitionPose &base_pose, double sync_error)
  {
    counts_ = ScanCounts{};
    const ros::Time stamp = message->header.stamp;

    // gazebo_ros_block_laser (Noetic) reports the midpoint angle of adjacent
    // samples while resolution=1 measures the lower-index ray. Reconstruct the
    // actual sensor ray directions; retain the measured range unchanged.
    // Reject an unexpected packet layout instead of guessing its beam angles.
    if (horizontal_samples_<2 || vertical_samples_<2 ||
        message->points.size()!=size_t(horizontal_samples_*vertical_samples_)) {
      ROS_ERROR_THROTTLE(1,"Unexpected block-laser layout; scan rejected");
      publishStatus(message->points.size(), 0, 0, sync_error, true); return;
    }
    const Eigen::Matrix3d world_from_base = base_pose.orientation.toRotationMatrix();
    std::vector<Eigen::Vector3f> output_points, ray_points;
    std::vector<uint8_t> ray_hits;
    std::vector<Eigen::Vector3f> display_points;
    const bool publish_display = display_cloud_pub_.getNumSubscribers() > 0;
    output_points.reserve(message->points.size() / static_cast<size_t>(point_stride_) + 1);
    if (publish_display)
      display_points.reserve(output_points.capacity());
    const double usable_max = range_max_ + 0.05;

    for (size_t index = 0; index < message->points.size();
         index += static_cast<size_t>(point_stride_))
    {
      const geometry_msgs::Point32 &raw = message->points[index];
      Eigen::Vector3d lidar_point(raw.x, raw.y, raw.z);
      if (!lidar_point.allFinite()) {
        ++counts_.dropped_invalid;
        continue;
      }
      const double range = lidar_point.norm();
      if (range < range_min_ || range > usable_max) { ++counts_.dropped_invalid; continue; }
      const double horizontal=-M_PI+2*M_PI*(index%horizontal_samples_)/(horizontal_samples_-1);
      const double vertical=vertical_min_+(vertical_max_-vertical_min_)*(index/horizontal_samples_)/(vertical_samples_-1);
      lidar_point=range*Eigen::Vector3d(std::cos(vertical)*std::cos(horizontal),
        std::cos(vertical)*std::sin(horizontal),std::sin(vertical));

      // Freeze the physical return type before any sensor degradation. The
      // indoor profile removes measurements; it never synthesizes free rays,
      // shifts a physical beam direction, or replays an older acquisition.
      const bool max_range_miss = std::abs(range-range_max_)<=2e-6;
      if (max_range_miss) ++counts_.input_misses; else ++counts_.input_hits;
      const auto decision = vla_sim::retainIndoorBeam(sampling_, stamp.toNSec(),
          index, horizontal, range, !max_range_miss);
      if (decision != vla_sim::SampleDecision::Keep) {
        if (decision == vla_sim::SampleDecision::Burst) ++counts_.dropped_burst;
        else if (decision == vla_sim::SampleDecision::RangeAmbiguous) ++counts_.dropped_range;
        else if (decision == vla_sim::SampleDecision::Invalid) ++counts_.dropped_invalid;
        else ++counts_.dropped_sampling;
        continue;
      }

      const Eigen::Vector3d base_point = lidar_rotation_ * lidar_point + lidar_translation_;
      if (std::hypot(base_point.x(), base_point.y()) < self_filter_radius_ &&
          std::abs(base_point.z()) < self_filter_half_height_) {
        ++counts_.dropped_self;
        continue;
      }

      const Eigen::Vector3d world_point = world_from_base * base_point + base_pose.position;
      if (!world_point.allFinite()) { ++counts_.dropped_invalid; continue; }
      // The upstream plugin omits xyz noise only within 1e-6 of maxRange.
      // Float32 reconstruction needs 2e-6 tolerance. A real near-limit hit is
      // retained; this is not the former 8 cm shell heuristic.
      ray_points.push_back(world_point.cast<float>());
      ray_hits.push_back(max_range_miss ? 0 : 1);
      if (max_range_miss) ++counts_.retained_misses; else ++counts_.retained_hits;
      // Registered observations are real surface returns only. Max-range rays
      // are carried separately and can only prove free space along the beam.
      if (!max_range_miss) output_points.push_back(world_point.cast<float>());
      const size_t mapping_point_index =
          index / static_cast<size_t>(point_stride_);
      if (publish_display &&
          (!display_drop_max_range_shell_ || !max_range_miss) &&
          mapping_point_index % static_cast<size_t>(display_point_stride_) == 0)
        display_points.push_back(world_point.cast<float>());
    }

    const Eigen::Vector3d origin = base_pose.position + world_from_base * lidar_translation_;
    sensor_msgs::PointCloud2 rays;
    rays.header.stamp = stamp; rays.header.frame_id = world_frame_;
    sensor_msgs::PointCloud2Modifier rm(rays);
    rm.setPointCloud2Fields(14, "x",1,sensor_msgs::PointField::FLOAT32,
      "y",1,sensor_msgs::PointField::FLOAT32,"z",1,sensor_msgs::PointField::FLOAT32,
      "hit",1,sensor_msgs::PointField::UINT8,
      "origin_x",1,sensor_msgs::PointField::FLOAT32,"origin_y",1,sensor_msgs::PointField::FLOAT32,
      "origin_z",1,sensor_msgs::PointField::FLOAT32,
      "body_x",1,sensor_msgs::PointField::FLOAT32,"body_y",1,sensor_msgs::PointField::FLOAT32,
      "body_z",1,sensor_msgs::PointField::FLOAT32,
      "body_qx",1,sensor_msgs::PointField::FLOAT32,"body_qy",1,sensor_msgs::PointField::FLOAT32,
      "body_qz",1,sensor_msgs::PointField::FLOAT32,"body_qw",1,sensor_msgs::PointField::FLOAT32);
    rm.resize(ray_points.size());
    sensor_msgs::PointCloud2Iterator<float> rx(rays,"x"), ry(rays,"y"), rz(rays,"z"),
      ox(rays,"origin_x"), oy(rays,"origin_y"), oz(rays,"origin_z"),
      bx(rays,"body_x"), by(rays,"body_y"), bz(rays,"body_z"),
      qx(rays,"body_qx"),qy(rays,"body_qy"),qz(rays,"body_qz"),qw(rays,"body_qw");
    sensor_msgs::PointCloud2Iterator<uint8_t> hit(rays,"hit");
    for (size_t i=0;i<ray_points.size();++i,++rx,++ry,++rz,++ox,++oy,++oz,++bx,++by,++bz,++qx,++qy,++qz,++qw,++hit) {
      *rx=ray_points[i].x(); *ry=ray_points[i].y(); *rz=ray_points[i].z(); *hit=ray_hits[i];
      *ox=origin.x(); *oy=origin.y(); *oz=origin.z();
      *bx=base_pose.position.x(); *by=base_pose.position.y(); *bz=base_pose.position.z();
      *qx=base_pose.orientation.x();*qy=base_pose.orientation.y();*qz=base_pose.orientation.z();*qw=base_pose.orientation.w();
    }
    rays_pub_.publish(rays);
    nav_msgs::Odometry lidar_odom;
    lidar_odom.header=rays.header; lidar_odom.child_frame_id="mid360_link";
    lidar_odom.pose.pose.position.x=origin.x(); lidar_odom.pose.pose.position.y=origin.y();
    lidar_odom.pose.pose.position.z=origin.z();
    const Eigen::Quaterniond lidar_q(world_from_base*lidar_rotation_);
    lidar_odom.pose.pose.orientation.x=lidar_q.x(); lidar_odom.pose.pose.orientation.y=lidar_q.y();
    lidar_odom.pose.pose.orientation.z=lidar_q.z(); lidar_odom.pose.pose.orientation.w=lidar_q.w();
    lidar_odom_pub_.publish(lidar_odom);
    sensor_msgs::PointCloud2 output = makeCloud(output_points, stamp);
    cloud_pub_.publish(output);
    if (publish_display)
      display_cloud_pub_.publish(makeCloud(display_points, stamp));
    ++published_frames_;
    publishStatus(message->points.size(), output_points.size(),
                  display_points.size(), sync_error, false);
  }

  sensor_msgs::PointCloud2 makeCloud(
      const std::vector<Eigen::Vector3f> &points,
      const ros::Time &stamp) const
  {
    sensor_msgs::PointCloud2 output;
    output.header.stamp = stamp;
    output.header.frame_id = world_frame_;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> z(output, "z");
    for (const Eigen::Vector3f &point : points)
    {
      *x = point.x();
      *y = point.y();
      *z = point.z();
      ++x;
      ++y;
      ++z;
    }
    return output;
  }

  void publishStatus(size_t input_points, size_t output_points,
                     size_t display_points, double sync_error, bool dropped)
  {
    std_msgs::String message;
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4)
           << "{\"input_points\":" << input_points
           << ",\"output_points\":" << output_points
           << ",\"display_points\":" << display_points
           << ",\"sync_error\":" << sync_error
           << ",\"dropped\":" << (dropped ? "true" : "false")
           << ",\"profile\":\"" << sensor_profile_ << "\""
           << ",\"sampling_seed\":" << sampling_.seed
           << ",\"input_hits\":" << counts_.input_hits
           << ",\"input_misses\":" << counts_.input_misses
           << ",\"retained_hits\":" << counts_.retained_hits
           << ",\"retained_misses\":" << counts_.retained_misses
           << ",\"dropped_sampling\":" << counts_.dropped_sampling
           << ",\"dropped_burst\":" << counts_.dropped_burst
           << ",\"dropped_range_ambiguous\":" << counts_.dropped_range
           << ",\"dropped_invalid\":" << counts_.dropped_invalid
           << ",\"dropped_self\":" << counts_.dropped_self
           << ",\"additional_range_noise\":false,\"historical_sectors\":0"
           << ",\"published_frames\":" << published_frames_
           << ",\"dropped_sync_frames\":" << dropped_sync_frames_
           << ",\"sync_policy\":\"bracketed_acquisition\""
           << ",\"pending_scans\":" << acquisition_sync_->pendingSize()
           << ",\"last_sync_drop_reason\":\"" << last_sync_drop_reason_ << "\"}";
    message.data = stream.str();
    status_pub_.publish(message);
  }

  ros::Subscriber cloud_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher cloud_pub_, rays_pub_, lidar_odom_pub_;
  ros::Publisher display_cloud_pub_;
  ros::Publisher status_pub_;
  std::unique_ptr<ScanSync> acquisition_sync_;
  ros::WallTimer pending_timer_;
  int pending_scan_capacity_{4};
  double pending_timeout_wall_{0.5};
  std::string last_sync_drop_reason_;

  std::string input_topic_;
  std::string output_topic_;
  std::string display_output_topic_;
  std::string odom_topic_;
  std::string world_frame_;
  std::string sensor_profile_;
  vla_sim::IndoorSampling sampling_;
  Eigen::Vector3d lidar_translation_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d lidar_rpy_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d lidar_rotation_{Eigen::Matrix3d::Identity()};
  double range_min_{0.10};
  double range_max_{12.0};
  bool drop_max_range_shell_{true};
  bool display_drop_max_range_shell_{true};
  double max_range_shell_margin_{0.08};
  double min_world_z_{-0.05};
  double max_world_z_{3.35};
  double self_filter_radius_{0.30};
  double self_filter_half_height_{0.24};
  int horizontal_samples_{240}, vertical_samples_{32};
  double vertical_min_{-0.12217304764},vertical_max_{0.90757121104};
  int point_stride_{1};
  int display_point_stride_{2};
  double odom_history_seconds_{2.0};
  double max_sync_gap_{0.16};
  uint64_t published_frames_{0};
  uint64_t dropped_sync_frames_{0};
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vla_cloud_bridge");
  VlaCloudBridge bridge;
  ros::spin();
  return 0;
}
