// Diff-VLA trajectory adapter. Polynomial conversion follows Diff-Planner's
// plan_manage/src/traj_server.cpp (GPL-3.0). Original source is untouched.
// Fixes: endpoint hold, finite/shape validation, measured yaw handover and yaw rate.
#include <ros/ros.h>
#include <traj_utils/PolyTraj.h>
#include <optimizer/poly_traj_utils.hpp>
#include <quadrotor_msgs/PositionCommand.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Empty.h>
#include <cmath>
#include <memory>
#include <algorithm>
#include <vla_sim/trajectory_yaw_state.h>

class Server {
  ros::NodeHandle nh_{"~"};
  ros::Publisher pub_, execution_pub_;
  ros::Subscriber trajectory_sub_, heartbeat_sub_, yaw_sub_, odom_sub_;
  ros::Timer timer_;
  std::unique_ptr<poly_traj::Trajectory> trajectory_;
  ros::Time start_, heartbeat_, yaw_stamp_, last_;
  uint32_t id_{0};
  double duration_{0}, requested_yaw_{0}, lookahead_{0.5};
  vla_sim::TrajectoryYawState yaw_state_;
  bool have_odom_{false}, healthy_{false};
  Eigen::Vector3d hold_{Eigen::Vector3d::Zero()};
public:
  Server() {
    pub_=nh_.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 10);
    execution_pub_=nh_.advertise<traj_utils::PolyTraj>("/vla/execution_trajectory", 4);
    trajectory_sub_=nh_.subscribe("planning/trajectory", 2, &Server::trajectory, this);
    heartbeat_sub_=nh_.subscribe("heartbeat", 2, &Server::heartbeat, this);
    yaw_sub_=nh_.subscribe("/planning/yaw", 2, &Server::yaw, this);
    odom_sub_=nh_.subscribe("/odom", 1, &Server::odom, this);
    nh_.param("traj_server/time_forward", lookahead_, 0.5);
    timer_=nh_.createTimer(ros::Duration(0.01), &Server::tick, this);
  }
  void heartbeat(const std_msgs::Empty::ConstPtr&) { heartbeat_=ros::Time::now(); }
  void yaw(const quadrotor_msgs::PositionCommand::ConstPtr& m) {
    if (std::isfinite(m->yaw) && m->yaw > -100) {requested_yaw_=m->yaw; yaw_stamp_=ros::Time::now();}
  }
  void odom(const nav_msgs::Odometry::ConstPtr& m) {
    const auto &q=m->pose.pose.orientation;
    const auto &p=m->pose.pose.position;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !yaw_state_.observe(q.x,q.y,q.z,q.w,m->header.stamp.toSec(),ros::Time::now().toSec()))
      return;
    if (!have_odom_) hold_=Eigen::Vector3d(p.x,p.y,p.z);
    have_odom_=true;
  }
  void trajectory(const traj_utils::PolyTraj::ConstPtr& m) {
    const size_t n=m->duration.size();
    if (m->order!=5 || n==0 || m->coef_x.size()!=6*n || m->coef_y.size()!=6*n || m->coef_z.size()!=6*n) {
      ROS_ERROR("Reject malformed trajectory"); return;
    }
    for(double d:m->duration) if(!std::isfinite(d)||d<=0) return;
    for(const auto* v:{&m->coef_x,&m->coef_y,&m->coef_z}) for(double x:*v) if(!std::isfinite(x)) return;
    std::vector<poly_traj::CoefficientMat> matrices(n);
    for(size_t i=0;i<n;++i) for(size_t j=0;j<6;++j) {
      matrices[i](0,j)=m->coef_x[6*i+j];
      matrices[i](1,j)=m->coef_y[6*i+j];
      matrices[i](2,j)=m->coef_z[6*i+j];
    }
    std::vector<double> durations(m->duration.begin(),m->duration.end());
    if (!yaw_state_.beginTrajectory(ros::Time::now().toSec())) {
      ROS_WARN_THROTTLE(1.0,"Reject trajectory handover without fresh valid measured yaw");
      return;
    }
    trajectory_.reset(new poly_traj::Trajectory(durations,matrices));
    duration_=trajectory_->getTotalDuration(); start_=m->start_time; ++id_; healthy_=true;
    // Carry the exact execution generation and coefficients to the controller
    // before streaming commands. Photo missions require a full-curve proof.
    traj_utils::PolyTraj execution=*m;
    execution.traj_id=static_cast<int32_t>(id_);
    execution_pub_.publish(execution);
    // Upstream resets traj_id to 1 for each goal. This execution generation
    // is monotonic across goals so holds cannot be revived by old streams.
  }
  void tick(const ros::TimerEvent&) {
    const auto now=ros::Time::now();
    const double dt=last_.isZero()?0.01:std::max(0.001,std::min(0.05,(now-last_).toSec())); last_=now;
    if(!trajectory_ || !have_odom_) return;
    if(heartbeat_.isZero() || (now-heartbeat_).toSec()>0.6) {
      healthy_=false;
      ROS_WARN_THROTTLE(1.0,"Planner heartbeat missing: stop streaming; controller timeout will hold");
      return;
    }
    if(!healthy_) return;
    const double t=(now-start_).toSec();
    if(t<0) return;
    const double sample=std::min(t,duration_);
    Eigen::Vector3d p=trajectory_->getPos(sample),v=Eigen::Vector3d::Zero(),a=v,j=v;
    if(t<duration_) {v=trajectory_->getVel(t);a=trajectory_->getAcc(t);j=trajectory_->getJer(t);}
    double target=yaw_state_.commandedYaw();
    const auto dir=(trajectory_->getPos(std::min(sample+lookahead_,duration_))-p).eval();
    if(dir.head<2>().norm()>.1) target=std::atan2(dir.y(),dir.x());
    if(!yaw_stamp_.isZero() && (now-yaw_stamp_).toSec()<.6) target=requested_yaw_;
    const auto yaw_command=yaw_state_.advance(target,dt);
    quadrotor_msgs::PositionCommand cmd;
    cmd.header.stamp=now;cmd.header.frame_id="world";
    cmd.trajectory_flag=quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;cmd.trajectory_id=id_;
    cmd.position.x=p.x();cmd.position.y=p.y();cmd.position.z=p.z();
    cmd.velocity.x=v.x();cmd.velocity.y=v.y();cmd.velocity.z=v.z();
    cmd.acceleration.x=a.x();cmd.acceleration.y=a.y();cmd.acceleration.z=a.z();
    cmd.jerk.x=j.x();cmd.jerk.y=j.y();cmd.jerk.z=j.z();
    cmd.yaw=yaw_command.yaw;cmd.yaw_dot=yaw_command.rate;
    pub_.publish(cmd);hold_=p;
  }
};
int main(int argc,char**argv){ros::init(argc,argv,"vla_traj_server");Server server;ros::spin();}
