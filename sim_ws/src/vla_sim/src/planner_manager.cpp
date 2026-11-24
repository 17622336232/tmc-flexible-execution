// GPL-3.0 simulation overlay of Diff-Planner plan_manage/src/planner_manager.cpp.
// Local endpoint selection plus endpoint/time validity and rest-goal guards.
// Original source remains untouched.
// #include <fstream>
#include <plan_manage/planner_manager.h>
#include <thread>
#include <vla_sim/planner_trajectory_safety.h>
#include <vla_sim/local_photo_route.h>
#include "visualization_msgs/Marker.h" // zx-todo

namespace diff_planner
{
  namespace {
    bool short_rest_recovery_enabled = false;
    double short_rest_max_jerk = 1.0;
    std::vector<Eigen::Vector3d> photo_route;
    Eigen::Vector4d photo_bounds(-8.,8.,-5.,5.);
  }

  // SECTION interfaces for setup and query

  DiffPlannerManager::DiffPlannerManager() {}

  DiffPlannerManager::~DiffPlannerManager() { std::cout << "des manager" << std::endl; }

  void DiffPlannerManager::initPlanModules(ros::NodeHandle &nh, PlanningVisualization::Ptr vis)
  {
    /* read algorithm parameters */

    nh.param("manager/max_vel", pp_.max_vel_, -1.0);
    nh.param("manager/max_acc", pp_.max_acc_, -1.0);
    nh.param("manager/feasibility_tolerance", pp_.feasibility_tolerance_, 0.0);
    nh.param("manager/polyTraj_piece_length", pp_.polyTraj_piece_length, -1.0);
    nh.param("manager/planning_horizon", pp_.planning_horizen_, 5.0);
    nh.param("manager/use_multitopology_trajs", pp_.use_multitopology_trajs, false);
    nh.param("manager/drone_id", pp_.drone_id, -1);
    nh.param("manager/short_rest_recovery", short_rest_recovery_enabled, false);
    nh.param("optimization/max_jer", short_rest_max_jerk, 1.0);
    nh.param("manager/photo_min_x",photo_bounds[0],-8.);
    nh.param("manager/photo_max_x",photo_bounds[1],8.);
    nh.param("manager/photo_min_y",photo_bounds[2],-5.);
    nh.param("manager/photo_max_y",photo_bounds[3],5.);

    grid_map_.reset(new GridMap);
    grid_map_->initMap(nh);

    ploy_traj_opt_.reset(new PolyTrajOptimizer);
    ploy_traj_opt_->setParam(nh);
    ploy_traj_opt_->setEnvironment(grid_map_);

    visualization_ = vis;

    ploy_traj_opt_->setSwarmTrajs(&traj_.swarm_traj);
    ploy_traj_opt_->setDroneId(pp_.drone_id);
  }

  bool DiffPlannerManager::reboundReplan(
      const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_vel,
      const Eigen::Vector3d &start_acc, const Eigen::Vector3d &local_target_pt,
      const Eigen::Vector3d &local_target_vel, const bool flag_polyInit,
      const bool flag_randomPolyTraj, const bool touch_goal)
  {
    if (!start_pt.allFinite() || !start_vel.allFinite() || !start_acc.allFinite() ||
        !local_target_pt.allFinite() || !local_target_vel.allFinite() ||
        !std::isfinite(pp_.max_vel_) || pp_.max_vel_ <= 0.0 ||
        !std::isfinite(pp_.max_acc_) || pp_.max_acc_ <= 0.0 ||
        !std::isfinite(pp_.polyTraj_piece_length) || pp_.polyTraj_piece_length <= 0.0 ||
        getCpsNumPrePiece() < 1)
      return false;
    if (grid_map_->photoRoutingActive()) {
      if (!grid_map_->photoGuardCurrent() || photo_route.size()<2 ||
          (photo_route.front()-start_pt).norm()>.01 ||
          (photo_route.back()-local_target_pt).norm()>.001) return false;
      poly_traj::MinJerkOpt navigation;
      const auto free=[&](const Eigen::Vector3d &p) {
        return p.x()>=photo_bounds[0] && p.x()<=photo_bounds[1] &&
          p.y()>=photo_bounds[2] && p.y()<=photo_bounds[3] &&
          grid_map_->isKnownFree(p) && grid_map_->clearsPhotoPeople(p,.025);
      };
      if (!vla_sim::makePhotoRouteTrajectory(photo_route,start_vel,start_acc,
            pp_.max_vel_,pp_.max_acc_,short_rest_max_jerk,free,navigation) ||
          !ploy_traj_opt_->checkDynamicFeasibility(navigation) ||
          !setLocalTrajFromOpt(navigation,touch_goal)) {
        ROS_WARN_THROTTLE(1.,"[VLA] Local photo route has no dynamically safe observed curve");
        return false;
      }
      visualization_->displayOptimalList(navigation.getInitConstraintPoints(getCpsNumPrePiece()),0);
      ROS_INFO("[VLA] Continuous photo route: %zu internal supports, %.2fs, final=%d",photo_route.size(),
          navigation.getTraj().getTotalDuration(),touch_goal);
      return true;
    }
    // Do not treat the upstream map's outside-buffer return value (zero)
    // as free space when local endpoint recovery could not find a valid point.
    if (!grid_map_->containsLocalMap(local_target_pt) ||
        grid_map_->getInflateOccupancy(local_target_pt) != 0)
    {
      ROS_WARN_THROTTLE(1.0, "[VLA] Reject occupied or out-of-map local target");
      return false;
    }
    ros::Time t_start = ros::Time::now();
    ros::Duration t_init, t_opt;

    static int count = 0;
    cout << "\033[47;30m\n[" << t_start << "] Drone " << pp_.drone_id << " Replan " << count++ << "\033[0m" << endl;
    // cout.precision(3);
    // cout << "start: " << start_pt.transpose() << ", " << start_vel.transpose() << "\ngoal:" << local_target_pt.transpose() << ", " << local_target_vel.transpose()
    //      << endl;
    // if ((start_pt - local_target_pt).norm() < 0.2)
    //   cout << "Close to goal" << endl;

    /*** STEP 1: INIT ***/
    ploy_traj_opt_->setIfTouchGoal(touch_goal);
    double ts = pp_.polyTraj_piece_length / pp_.max_vel_;

    poly_traj::MinJerkOpt initMJO;
    if (!computeInitState(start_pt, start_vel, start_acc, local_target_pt, local_target_vel,
                          flag_polyInit, flag_randomPolyTraj, ts, initMJO))
    {
      return false;
    }

    if (!vla_sim::finiteTrajectory(initMJO.getTraj())) return false;
    Eigen::MatrixXd cstr_pts = initMJO.getInitConstraintPoints(ploy_traj_opt_->get_cps_num_prePiece_());

    // L-BFGS time optimization has no translational task for a yaw-only goal.
    // Keep its valid rest polynomial, with the same collision samples, dynamic
    // checks, trajectory ID and downstream whole-trajectory proof as any goal.
    if (vla_sim::nearZeroGoal(start_pt, local_target_pt))
    {
      for (int i = 0; i < cstr_pts.cols(); ++i)
        if (!grid_map_->containsLocalMap(cstr_pts.col(i)) ||
            grid_map_->getInflateOccupancy(cstr_pts.col(i)) != 0)
          return false;
      if (!ploy_traj_opt_->checkDynamicFeasibility(initMJO) ||
          !setLocalTrajFromOpt(initMJO, touch_goal))
        return false;
      visualization_->displayOptimalList(cstr_pts, 0);
      continous_failures_count_ = 0;
      return true;
    }
    vector<std::pair<int, int>> segments;
    if (ploy_traj_opt_->finelyCheckAndSetConstraintPoints(segments, initMJO, true) == PolyTrajOptimizer::CHK_RET::ERR)
    {
      return false;
    }

    t_init = ros::Time::now() - t_start;

    std::vector<Eigen::Vector3d> point_set;
    for (int i = 0; i < cstr_pts.cols(); ++i)
      point_set.push_back(cstr_pts.col(i));
    visualization_->displayInitPathList(point_set, 0.2, 0);

    t_start = ros::Time::now();

    /*** STEP 2: OPTIMIZE ***/
    bool flag_success = false;
    vector<vector<Eigen::Vector3d>> vis_trajs;
    poly_traj::MinJerkOpt best_MJO;

    // ROS_ERROR("BBBB");

    if (pp_.use_multitopology_trajs)
    {
      std::vector<ConstraintPoints> trajs = ploy_traj_opt_->distinctiveTrajs(segments);
      Eigen::VectorXi success = Eigen::VectorXi::Zero(trajs.size());
      poly_traj::Trajectory initTraj = initMJO.getTraj();
      int PN = initTraj.getPieceNum();
      Eigen::MatrixXd all_pos = initTraj.getPositions();
      Eigen::MatrixXd innerPts = all_pos.block(0, 1, 3, PN - 1);
      Eigen::Matrix<double, 3, 3> headState, tailState;
      headState << initTraj.getJuncPos(0), initTraj.getJuncVel(0), initTraj.getJuncAcc(0);
      tailState << initTraj.getJuncPos(PN), initTraj.getJuncVel(PN), initTraj.getJuncAcc(PN);
      double final_cost, min_cost = 999999.0;

      for (int i = trajs.size() - 1; i >= 0; i--)
      {
        ploy_traj_opt_->setConstraintPoints(trajs[i]);
        ploy_traj_opt_->setUseMultitopologyTrajs(true);
        if (ploy_traj_opt_->optimizeTrajectory(headState, tailState,
                                               innerPts, initTraj.getDurations(), final_cost))
        {
          success[i] = true;

          if (final_cost < min_cost)
          {
            min_cost = final_cost;
            best_MJO = ploy_traj_opt_->getMinJerkOpt();
            flag_success = true;
          }

          // visualization
          Eigen::MatrixXd ctrl_pts_temp = ploy_traj_opt_->getMinJerkOpt().getInitConstraintPoints(ploy_traj_opt_->get_cps_num_prePiece_());
          std::vector<Eigen::Vector3d> point_set;
          for (int j = 0; j < ctrl_pts_temp.cols(); j++)
          {
            point_set.push_back(ctrl_pts_temp.col(j));
          }
          vis_trajs.push_back(point_set);
        }
      }

      t_opt = ros::Time::now() - t_start;

      if (trajs.size() > 1)
      {
        cout << "\033[1;33m"
             << "multi-trajs=" << trajs.size() << ",\033[1;0m"
             << " Success:fail=" << success.sum() << ":" << success.size() - success.sum() << endl;
      }

      visualization_->displayMultiOptimalPathList(vis_trajs, 0.1); // This visuallization will take up several milliseconds.
    }
    else
    {
      poly_traj::Trajectory initTraj = initMJO.getTraj();
      int PN = initTraj.getPieceNum();
      Eigen::MatrixXd all_pos = initTraj.getPositions();
      Eigen::MatrixXd innerPts = all_pos.block(0, 1, 3, PN - 1);
      Eigen::Matrix<double, 3, 3> headState, tailState;
      headState << initTraj.getJuncPos(0), initTraj.getJuncVel(0), initTraj.getJuncAcc(0);
      tailState << initTraj.getJuncPos(PN), initTraj.getJuncVel(PN), initTraj.getJuncAcc(PN);
      double final_cost;
      flag_success = ploy_traj_opt_->optimizeTrajectory(headState, tailState,
                                                        innerPts, initTraj.getDurations(), final_cost);
      best_MJO = ploy_traj_opt_->getMinJerkOpt();

      t_opt = ros::Time::now() - t_start;
    }

    // Some short, free-space rest goals fail the upstream time optimizer.
    // Recover only the identical endpoint, using a slower exact quintic whose
    // entire path retains known-free BODY-envelope checks. Obstructed/unknown
    // routes and moving boundaries cannot enter this branch. The downstream
    // exact polynomial/person guard is unchanged.
    if (!flag_success && short_rest_recovery_enabled)
    {
      Eigen::Matrix3d head, tail;
      head << start_pt, start_vel, start_acc;
      tail << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();
      poly_traj::MinJerkOpt recovery;
      if (vla_sim::makeShortRestRecovery(head, tail, pp_.max_vel_, pp_.max_acc_,
                                       short_rest_max_jerk, recovery) &&
          vla_sim::shortRecoveryKnownFree(recovery.getTraj(),
              [&](const Eigen::Vector3d &point) { return grid_map_->isKnownFree(point); }) &&
          ploy_traj_opt_->checkDynamicFeasibility(recovery))
      {
        std::vector<std::pair<int,int>> recovery_segments;
        if (ploy_traj_opt_->finelyCheckAndSetConstraintPoints(recovery_segments, recovery, false) ==
            PolyTrajOptimizer::CHK_RET::OBS_FREE)
        {
          best_MJO = recovery;
          flag_success = true;
          ROS_INFO("[VLA] Short rest recovery: identical goal, observed body envelope, speed/acceleration/jerk verified");
        }
      }
    }

    /*** STEP 3: Store and display results ***/
    cout << "Success=" << (flag_success ? "yes" : "no") << endl;
    if (flag_success)
    {
      static double sum_time = 0;
      static int count_success = 0;
      sum_time += (t_init + t_opt).toSec();
      count_success++;
      printf("Time:\033[42m%.3fms,\033[0m init:%.3fms, optimize:%.3fms, avg=%.3fms\n",
             (t_init + t_opt).toSec() * 1000, t_init.toSec() * 1000, t_opt.toSec() * 1000, sum_time / count_success * 1000);
      // cout << "total time:\033[42m" << (t_init + t_opt).toSec()
      //      << "\033[0m,init:" << t_init.toSec()
      //      << ",optimize:" << t_opt.toSec()
      //      << ",avg_time=" << sum_time / count_success << endl;

      if (!setLocalTrajFromOpt(best_MJO, touch_goal)) return false;
      cstr_pts = best_MJO.getInitConstraintPoints(ploy_traj_opt_->get_cps_num_prePiece_());
      visualization_->displayOptimalList(cstr_pts, 0);

      continous_failures_count_ = 0;
    }
    else
    {
      cstr_pts = ploy_traj_opt_->getMinJerkOpt().getInitConstraintPoints(ploy_traj_opt_->get_cps_num_prePiece_());
      visualization_->displayFailedList(cstr_pts, 0);

      continous_failures_count_++;
    }

    return flag_success;
  }

  bool DiffPlannerManager::computeInitState(
      const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
      const Eigen::Vector3d &local_target_pt, const Eigen::Vector3d &local_target_vel,
      const bool flag_polyInit, const bool flag_randomPolyTraj, const double &ts,
      poly_traj::MinJerkOpt &initMJO)
  {
    static bool flag_first_call = true;
    if (!start_pt.allFinite() || !start_vel.allFinite() || !start_acc.allFinite() ||
        !local_target_pt.allFinite() || !local_target_vel.allFinite() ||
        !std::isfinite(ts) || ts <= 0.0 ||
        !std::isfinite(pp_.polyTraj_piece_length) || pp_.polyTraj_piece_length <= 0.0)
      return false;
    if (vla_sim::nearZeroGoal(start_pt, local_target_pt))
    {
      Eigen::Matrix3d head, tail;
      head << start_pt, start_vel, start_acc;
      tail << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();
      return vla_sim::makeRestGoalTrajectory(head, tail, initMJO);
    }

    if (flag_first_call || flag_polyInit) /*** case 1: polynomial initialization ***/
    {
      flag_first_call = false;

      /* basic params */
      Eigen::Matrix3d headState, tailState;
      Eigen::MatrixXd innerPs;
      Eigen::VectorXd piece_dur_vec;
      int piece_nums;
      constexpr double init_of_init_totaldur = 2.0;
      headState << start_pt, start_vel, start_acc;
      tailState << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();

      /* determined or random inner point */
      if (!flag_randomPolyTraj)
      {
        if (innerPs.cols() != 0)
        {
          ROS_ERROR("innerPs.cols() != 0");
        }

        piece_nums = 1;
        piece_dur_vec.resize(1);
        piece_dur_vec(0) = init_of_init_totaldur;
      }
      else
      {
        const Eigen::Vector3d direction = start_pt - local_target_pt;
        Eigen::Vector3d horizen_dir = direction.cross(Eigen::Vector3d::UnitZ());
        if (horizen_dir.norm() < 1e-9) horizen_dir = direction.unitOrthogonal();
        else horizen_dir.normalize();
        Eigen::Vector3d vertical_dir = direction.cross(horizen_dir).normalized();
        innerPs.resize(3, 1);
        innerPs = (start_pt + local_target_pt) / 2 +
                  (((double)rand()) / RAND_MAX - 0.5) *
                      (start_pt - local_target_pt).norm() *
                      horizen_dir * 0.8 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989) +
                  (((double)rand()) / RAND_MAX - 0.5) *
                      (start_pt - local_target_pt).norm() *
                      vertical_dir * 0.4 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989);

        piece_nums = 2;
        piece_dur_vec.resize(2);
        piece_dur_vec = Eigen::Vector2d(init_of_init_totaldur / 2, init_of_init_totaldur / 2);
      }

      /* generate the init of init trajectory */
      initMJO.reset(headState, tailState, piece_nums);
      initMJO.generate(innerPs, piece_dur_vec);
      poly_traj::Trajectory initTraj = initMJO.getTraj();

      /* generate the real init trajectory */
      piece_nums = round((headState.col(0) - tailState.col(0)).norm() / pp_.polyTraj_piece_length);
      if (piece_nums < 2)
        piece_nums = 2;
      double piece_dur = init_of_init_totaldur / (double)piece_nums;
      piece_dur_vec.resize(piece_nums);
      piece_dur_vec = Eigen::VectorXd::Constant(piece_nums, ts);
      innerPs.resize(3, piece_nums - 1);
      int id = 0;
      double t_s = piece_dur, t_e = init_of_init_totaldur - piece_dur / 2;
      for (double t = t_s; t < t_e; t += piece_dur)
      {
        innerPs.col(id++) = initTraj.getPos(t);
      }
      if (id != piece_nums - 1)
      {
        ROS_ERROR("Should not happen! x_x");
        return false;
      }
      initMJO.reset(headState, tailState, piece_nums);
      initMJO.generate(innerPs, piece_dur_vec);
    }
    else /*** case 2: initialize from previous optimal trajectory ***/
    {
      if (traj_.global_traj.last_glb_t_of_lc_tgt < 0.0)
      {
        ROS_ERROR("You are initialzing a trajectory from a previous optimal trajectory, but no previous trajectories up to now.");
        return false;
      }

      /* the trajectory time system is a little bit complicated... */
      double passed_t_on_lctraj = ros::Time::now().toSec() - traj_.local_traj.start_time;
      double t_to_lc_end = traj_.local_traj.duration - passed_t_on_lctraj;
      if (t_to_lc_end < 0)
      {
        ROS_INFO("t_to_lc_end < 0, exit and wait for another call.");
        return false;
      }
      double t_to_lc_tgt = t_to_lc_end +
                           (traj_.global_traj.glb_t_of_lc_tgt - traj_.global_traj.last_glb_t_of_lc_tgt);
      // A shorter valid prefix can move the rolling target backward in time.
      // Let the existing FSM retry with polynomial initialization instead of
      // constructing zero/negative-duration pieces from the previous path.
      if (!std::isfinite(t_to_lc_tgt) || t_to_lc_tgt <= 1e-3)
        return false;
      int piece_nums = ceil((start_pt - local_target_pt).norm() / pp_.polyTraj_piece_length);
      if (piece_nums < 2)
        piece_nums = 2;

      Eigen::Matrix3d headState, tailState;
      Eigen::MatrixXd innerPs(3, piece_nums - 1);
      Eigen::VectorXd piece_dur_vec = Eigen::VectorXd::Constant(piece_nums, t_to_lc_tgt / piece_nums);
      headState << start_pt, start_vel, start_acc;
      tailState << local_target_pt, local_target_vel, Eigen::Vector3d::Zero();

      double t = piece_dur_vec(0);
      for (int i = 0; i < piece_nums - 1; ++i)
      {
        if (t < t_to_lc_end)
        {
          innerPs.col(i) = traj_.local_traj.traj.getPos(t + passed_t_on_lctraj);
        }
        else if (t <= t_to_lc_tgt)
        {
          double glb_t = t - t_to_lc_end + traj_.global_traj.last_glb_t_of_lc_tgt - traj_.global_traj.global_start_time;
          innerPs.col(i) = traj_.global_traj.traj.getPos(glb_t);
        }
        else
        {
          ROS_ERROR("Should not happen! x_x 0x88 t=%.2f, t_to_lc_end=%.2f, t_to_lc_tgt=%.2f", t, t_to_lc_end, t_to_lc_tgt);
        }

        t += piece_dur_vec(i + 1);
      }

      initMJO.reset(headState, tailState, piece_nums);
      initMJO.generate(innerPs, piece_dur_vec);
    }

    return vla_sim::finiteTrajectory(initMJO.getTraj());
  }

  void DiffPlannerManager::getLocalTarget(
      const double planning_horizen, const Eigen::Vector3d &start_pt,
      const Eigen::Vector3d &global_end_pt, Eigen::Vector3d &local_target_pos,
      Eigen::Vector3d &local_target_vel, bool &touch_goal)
  {
    if (grid_map_->photoRoutingActive()) {
      const auto free=[&](const Eigen::Vector3d &p) {
        const double cushion=.025+.135*std::min({1.,(p-start_pt).norm()/.5,(p-global_end_pt).norm()/.5});
        return p.x()>=photo_bounds[0] && p.x()<=photo_bounds[1] &&
          p.y()>=photo_bounds[2] && p.y()<=photo_bounds[3] &&
          grid_map_->isKnownFree(p) && grid_map_->clearsPhotoPeople(p,cushion);
      };
      const auto reference=[&](const Eigen::Vector3d &p){return grid_map_->photoReferenceTraversable(p);};
      const auto route=vla_sim::localPhotoRoute(start_pt,global_end_pt,photo_bounds,free,reference);
      photo_route=vla_sim::routePrefix(route,std::max(4.,planning_horizen));
      local_target_pos=photo_route.size()<2?start_pt:photo_route.back();
      ROS_INFO_THROTTLE(1.,"[VLA] Photo local target %.2f %.2f, intention %.2f %.2f, supports %zu",local_target_pos.x(),
          local_target_pos.y(),global_end_pt.x(),global_end_pt.y(),photo_route.size());
      local_target_vel=Eigen::Vector3d::Zero();
      touch_goal=(local_target_pos-global_end_pt).norm()<1e-4;
      // This time is a bookkeeping reference only. Route progress is measured
      // along the actual local curve, not the original straight global chord.
      auto &g=traj_.global_traj;
      g.last_glb_t_of_lc_tgt=g.glb_t_of_lc_tgt;
      return;
    }
    // Preserve upstream horizon selection, but never use an occupied rolling
    // endpoint. A straight global reference is NOT an obstacle-free route:
    // truncating it at a pedestal makes the optimizer reject every seed.
    auto &global = traj_.global_traj;
    const double end = global.global_start_time + global.duration;
    const double previous = global.glb_t_of_lc_tgt;
    global.last_glb_t_of_lc_tgt = previous;
    const double coarse_step = planning_horizen / 20.0 / pp_.max_vel_;
    double chosen = end;
    for (double t = previous; t < end; t += coarse_step)
    {
      const Eigen::Vector3d candidate = global.traj.getPos(t - global.global_start_time);
      if ((candidate - start_pt).norm() >= planning_horizen) { chosen = t; break; }
    }
    touch_goal = chosen >= end - 1e-5;
    local_target_pos = touch_goal ? global_end_pt : global.traj.getPos(chosen - global.global_start_time);
    bool stop_at_proxy = false;

    // A requested final goal is never silently moved. The existing FSM and
    // optimizer reject an occupied final pose (mondify_final_goal=false).
    if (!touch_goal && (!grid_map_->containsLocalMap(local_target_pos) ||
                         grid_map_->getInflateOccupancy(local_target_pos) != 0))
    {
      const double nominal = chosen;
      const double step = std::min(0.10, grid_map_->getResolution() / (2.0 * pp_.max_vel_));
      const double max_distance = planning_horizen + 2.0;
      bool found = false;
      auto valid = [&](const Eigen::Vector3d &point) {
        return grid_map_->containsLocalMap(point) && grid_map_->getInflateOccupancy(point) == 0 &&
               (point - start_pt).norm() <= max_distance;
      };
      // First look just beyond the inflated obstacle, still in the actual
      // sensor-derived local map. The trajectory optimizer chooses the detour.
      for (double t = nominal + step; t <= end + step; t += step)
      {
        const double sample = std::min(t, end);
        const Eigen::Vector3d point = global.traj.getPos(sample - global.global_start_time);
        if (!grid_map_->containsLocalMap(point) || (point-start_pt).norm() > max_distance) break;
        if (valid(point))
        {
          chosen = sample; local_target_pos = point; found = true;
          touch_goal = sample >= end - 1e-5;
          break;
        }
      }
      // If the far side is outside the map, take the furthest free prefix
      // with positive mission progress. Stop there to acquire a fresh view.
      if (!found)
      {
        const Eigen::Vector3d direction = (global_end_pt-start_pt).normalized();
        for (double t = nominal-step; t >= global.global_start_time; t -= step)
        {
          const Eigen::Vector3d point = global.traj.getPos(t-global.global_start_time);
          if ((point-start_pt).dot(direction) < 0.5) break;
          if (valid(point))
          {
            chosen=t; local_target_pos=point; found=true; stop_at_proxy=true; break;
          }
        }
      }
      if (found)
        ROS_INFO_THROTTLE(1.0, "[VLA] Selected a free rolling endpoint from the observed local map");
      else
        ROS_WARN_THROTTLE(1.0, "[VLA] No free rolling endpoint with forward progress; planner will hold");
    }

    global.glb_t_of_lc_tgt = chosen;
    if (stop_at_proxy || touch_goal || (global_end_pt-local_target_pos).norm() <
        pp_.max_vel_*pp_.max_vel_/(2*pp_.max_acc_))
      local_target_vel = Eigen::Vector3d::Zero();
    else
      local_target_vel = global.traj.getVel(chosen-global.global_start_time);
  }

  bool DiffPlannerManager::setLocalTrajFromOpt(const poly_traj::MinJerkOpt &opt, const bool touch_goal)
  {
    poly_traj::Trajectory traj = opt.getTraj();
    if (!vla_sim::finiteTrajectory(traj) || getCpsNumPrePiece() < 1) return false;
    Eigen::MatrixXd cps = opt.getInitConstraintPoints(getCpsNumPrePiece());
    PtsChk_t pts_to_check;
    bool ret = ploy_traj_opt_->computePointsToCheck(traj, ConstraintPoints::two_thirds_id(cps, touch_goal), pts_to_check);
    if (ret && pts_to_check.size() >= 1 && pts_to_check.back().size() >= 1)
    {
      traj_.setLocalTraj(traj, pts_to_check, ros::Time::now().toSec());
    }

    return ret && !pts_to_check.empty() && !pts_to_check.back().empty();
  }

  bool DiffPlannerManager::EmergencyStop(Eigen::Vector3d stop_pos)
  {
    auto ZERO = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 3> headState, tailState;
    headState << stop_pos, ZERO, ZERO;
    tailState = headState;
    poly_traj::MinJerkOpt stopMJO;
    stopMJO.reset(headState, tailState, 2);
    stopMJO.generate(stop_pos, Eigen::Vector2d(1.0, 1.0));

    setLocalTrajFromOpt(stopMJO, false);

    return true;
  }

  bool DiffPlannerManager::checkCollision(int drone_id)
  {
    if (traj_.local_traj.start_time < 1e9) // It means my first planning has not started
      return false;
    if (traj_.swarm_traj[drone_id].drone_id != drone_id) // The trajectory is invalid
      return false;

    double my_traj_start_time = traj_.local_traj.start_time;
    double other_traj_start_time = traj_.swarm_traj[drone_id].start_time;

    double t_start = max(my_traj_start_time, other_traj_start_time);
    double t_end = min(my_traj_start_time + traj_.local_traj.duration * 2 / 3,
                       other_traj_start_time + traj_.swarm_traj[drone_id].duration);

    for (double t = t_start; t < t_end; t += 0.03)
    {
      if ((traj_.local_traj.traj.getPos(t - my_traj_start_time) -
           traj_.swarm_traj[drone_id].traj.getPos(t - other_traj_start_time))
              .norm() < (getSwarmClearance() + traj_.swarm_traj[drone_id].des_clearance) )
      {
        return true;
      }
    }

    return false;
  }

  bool DiffPlannerManager::planGlobalTrajWaypoints(
      const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel,
      const Eigen::Vector3d &start_acc, const std::vector<Eigen::Vector3d> &waypoints,
      const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {
    if (waypoints.empty()) return false;
    poly_traj::MinJerkOpt globalMJO;
    Eigen::Matrix<double, 3, 3> headState, tailState;
    headState << start_pos, start_vel, start_acc;
    tailState << waypoints.back(), end_vel, end_acc;
    if (!vla_sim::makeGlobalReference(headState, tailState, waypoints, pp_.max_vel_, globalMJO))
    {
      ROS_WARN_THROTTLE(1.0, "[VLA] Reject invalid global reference or non-rest near-zero goal");
      return false;
    }

    auto time_now = ros::Time::now();
    traj_.setGlobalTraj(globalMJO.getTraj(), time_now.toSec());

    return true;
  }

} // namespace diff_planner
