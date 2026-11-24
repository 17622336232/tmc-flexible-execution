// Gazebo Classic rate-limits ~/pose/info, but clears the changed-pose list
// even when the publication was skipped. A static object's one-off move can
// therefore reach rendering sensors through ~/pose/local/info and miss gzclient.
// Copy actual static-object poses to the GUI topic, with a short replay window.
// This process never publishes to physics, ROS, cameras or task interfaces.
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
struct Pending {
  gazebo::msgs::Pose pose;
  Clock::time_point until;
};
std::atomic<bool> running{true};
std::set<std::string> staticModels;
std::map<unsigned int, Pending> pending;
gazebo::msgs::Time latestTime;
std::mutex mutex;

void stop(int) { running = false; }

void received(ConstPosesStampedPtr &message) {
  std::lock_guard<std::mutex> guard(mutex);
  latestTime.CopyFrom(message->time());
  const auto until = Clock::now() + std::chrono::seconds(1);
  for (int index = 0; index < message->pose_size(); ++index) {
    const auto &pose = message->pose(index);
    const auto model = pose.name().substr(0, pose.name().find("::"));
    if (!pose.has_id() || !staticModels.count(model)) continue;
    pending[pose.id()] = Pending{pose, until};
    if (model == pose.name()) {
      std::cout << "{\"event\":\"actual_static_pose_received\",\"model\":\""
        << model << "\",\"id\":" << pose.id()
        << ",\"x\":" << pose.position().x()
        << ",\"y\":" << pose.position().y()
        << ",\"z\":" << pose.position().z() << "}" << std::endl;
    }
  }
}
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: gui_pose_sync WORLD STATIC_MODEL_NAMES_FILE\n";
    return 2;
  }
  std::ifstream names(argv[2]);
  if (!names) return 2;
  std::string name;
  while (std::getline(names, name)) if (!name.empty()) staticModels.insert(name);
  if (staticModels.empty()) return 2;
  gazebo::client::setup();
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init(argv[1]);
  // No frequency cap on this independent GUI publisher.
  auto publisher = node->Advertise<gazebo::msgs::PosesStamped>("~/pose/info", 5);
  auto subscriber = node->Subscribe("~/pose/local/info", received);
  while (running) {
    gazebo::msgs::PosesStamped output;
    {
      std::lock_guard<std::mutex> guard(mutex);
      output.mutable_time()->CopyFrom(latestTime);
      for (auto item = pending.begin(); item != pending.end();) {
        if (Clock::now() > item->second.until) item = pending.erase(item);
        else {
          output.add_pose()->CopyFrom(item->second.pose);
          ++item;
        }
      }
    }
    if (output.pose_size() && publisher->HasConnections()) publisher->Publish(output);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  subscriber.reset();
  publisher.reset();
  node->Fini();
  gazebo::client::shutdown();
  return 0;
}
