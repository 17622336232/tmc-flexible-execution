/*
 * Copyright 2013 Open Source Robotics Foundation (legacy encoding behavior)
 * Copyright 2026 Diff-VLA contributors (completed-scan ROS publisher)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/MultiRayShape.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/tf.h>
#include <vla_sim/acquisition_lidar_encoding.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gazebo {

// RayPlugin's NewLaserScans callback runs INSIDE RaySensor::UpdateImpl,
// before lastMeasurementTime/lastUpdateTime advance. Use SensorPlugin instead.
class VlaAcquisitionLidar : public SensorPlugin {
public:
  ~VlaAcquisitionLidar() override {
    updated_.reset();
    std::lock_guard<std::mutex> lock(mutex_);
    publisher_.shutdown();
    if (node_) node_->shutdown();
  }

  void Load(sensors::SensorPtr parent, sdf::ElementPtr sdf) override {
    sensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(parent);
    if (!sensor_ || !ros::isInitialized())
      gzthrow("VlaAcquisitionLidar requires a ray sensor and initialized ROS");
    const std::string ns = sdf->HasElement("robotNamespace") ? sdf->Get<std::string>("robotNamespace") : "";
    const std::string topic = sdf->HasElement("topicName") ? sdf->Get<std::string>("topicName") : "";
    frame_ = sdf->HasElement("frameName") ? sdf->Get<std::string>("frameName") : "";
    noise_sigma_ = sdf->HasElement("gaussianNoise") ? sdf->Get<double>("gaussianNoise") : 0.;
    if (topic.empty() || frame_.empty() || !std::isfinite(noise_sigma_) || noise_sigma_ < 0)
      gzthrow("VlaAcquisitionLidar needs explicit topic/frame and valid noise");
    if (sensor_->RayCount() != sensor_->RangeCount() ||
        sensor_->VerticalRayCount() != sensor_->VerticalRangeCount())
      gzthrow("VlaAcquisitionLidar requires physical ray resolution=1");
    grid_ = {static_cast<size_t>(sensor_->RayCount()), static_cast<size_t>(sensor_->VerticalRayCount()),
             sensor_->AngleMin().Radian(), sensor_->AngleMax().Radian(),
             sensor_->VerticalAngleMin().Radian(), sensor_->VerticalAngleMax().Radian(),
             sensor_->RangeMin(), sensor_->RangeMax()};
    vla_sim::validateBlockLaserGrid(grid_);
    node_.reset(new ros::NodeHandle(ns));
    std::string prefix;
    node_->getParam("tf_prefix", prefix);
    frame_ = tf::resolve(prefix, frame_);
    publisher_ = node_->advertise<sensor_msgs::PointCloud>(topic, 1);
    // This event is emitted AFTER UpdateImpl succeeds. LastMeasurementTime
    // is the timestamp used by native Gazebo LaserScanStamped. No callback
    // arrival clock or nominal scan-period correction is used.
    updated_ = sensor_->ConnectUpdated(std::bind(&VlaAcquisitionLidar::CompletedScan, this));
    sensor_->SetActive(true);
    ROS_INFO("[VLA LiDAR] completed-scan measurement clock -> %s (%zu x %zu)",
             topic.c_str(), grid_.horizontal, grid_.vertical);
  }

private:
  double Gaussian() const {
    if (noise_sigma_ == 0.) return 0.;
    // Preserve independent xyz Gaussian noise. RaySensor::Ranges() is not
    // used: it already applies a different SDF range-noise/miss policy.
    const double u = (double(std::rand())+1.)/(double(RAND_MAX)+1.);
    const double v = double(std::rand())/double(RAND_MAX);
    return noise_sigma_*std::sqrt(-2.*std::log(u))*std::cos(2.*M_PI*v);
  }

  void CompletedScan() {
    std::lock_guard<std::mutex> lock(mutex_);
    const common::Time measurement = sensor_->LastMeasurementTime();
    if (measurement.sec < 0 || measurement.nsec < 0 || measurement.nsec >= 1000000000)
      return;
    const uint64_t stamp_ns = uint64_t(measurement.sec)*1000000000ULL+uint64_t(measurement.nsec);
    if (!clock_.accept(stamp_ns)) {
      if (clock_.resetDetected())
        ROS_WARN("[VLA LiDAR] measurement clock reset; discard transition scan");
      return;
    }
    if (publisher_.getNumSubscribers() == 0) return;

    // Synchronous copy in this sensor's update callback. Do not toggle active
    // state as a lock or hand live shape access to a ROS/worker thread.
    const auto shape = sensor_->LaserShape();
    const size_t count = grid_.horizontal*grid_.vertical;
    std::vector<double> ranges(count), retro(count);
    for (size_t i=0; i<count; ++i) {
      ranges[i] = shape->GetRange(i);
      retro[i] = shape->GetRetro(i);
    }
    sensor_msgs::PointCloud cloud;
    cloud.header.frame_id = frame_;
    cloud.header.stamp = ros::Time(measurement.sec, measurement.nsec);
    cloud.points.resize(count);
    cloud.channels.resize(1);
    cloud.channels[0].values.resize(count);
    for (size_t v=0; v<grid_.vertical; ++v) {
      for (size_t h=0; h<grid_.horizontal; ++h) {
        const auto point = vla_sim::encodeBlockReturn(grid_, h, v,
            [&ranges](size_t i) { return ranges[i]; },
            [&retro](size_t i) { return retro[i]; }, [this]() { return Gaussian(); });
        const size_t index = h+v*grid_.horizontal;
        cloud.points[index].x = point.x;
        cloud.points[index].y = point.y;
        cloud.points[index].z = point.z;
        cloud.channels[0].values[index] = point.intensity;
      }
    }
    publisher_.publish(cloud);
  }

  sensors::RaySensorPtr sensor_;
  event::ConnectionPtr updated_;
  std::unique_ptr<ros::NodeHandle> node_;
  ros::Publisher publisher_;
  std::mutex mutex_;
  std::string frame_;
  double noise_sigma_{0.};
  vla_sim::BlockLaserGrid grid_{};
  vla_sim::CompletedScanStampGate clock_;
};

GZ_REGISTER_SENSOR_PLUGIN(VlaAcquisitionLidar)
}  // namespace gazebo
