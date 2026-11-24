#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <optimizer/poly_traj_utils.hpp>

namespace vla_sim
{
// These thresholds distinguish numerical residue from a translating vehicle.
// They do not move either endpoint or overwrite its velocity/acceleration.
constexpr double kNearZeroGoalDistance = 1e-4;
constexpr double kRestBoundaryDerivative = 1e-6;

inline bool nearZeroGoal(const Eigen::Vector3d &start, const Eigen::Vector3d &end)
{
  return start.allFinite() && end.allFinite() &&
         (end - start).norm() <= kNearZeroGoalDistance;
}

inline bool finiteTrajectory(const poly_traj::Trajectory &trajectory)
{
  if (trajectory.getPieceNum() < 1 || !std::isfinite(trajectory.getTotalDuration()))
    return false;
  for (const auto &piece : trajectory)
    if (!std::isfinite(piece.getDuration()) || piece.getDuration() <= 0.0 ||
        !piece.getCoeffMat().allFinite())
      return false;
  return true;
}

// A yaw-only goal still needs a real, positive-duration PolyTraj. Preserve all
// six endpoint derivatives in the solution, including numerical odometry noise.
// An appreciably moving vehicle must first brake through its normal controller;
// this path deliberately refuses to turn a translating state into a fixed hold.
inline bool makeRestGoalTrajectory(const Eigen::Matrix3d &head,
                                   const Eigen::Matrix3d &tail,
                                   poly_traj::MinJerkOpt &result)
{
  if (!head.allFinite() || !tail.allFinite() ||
      !nearZeroGoal(head.col(0), tail.col(0)) ||
      head.col(1).norm() > kRestBoundaryDerivative ||
      head.col(2).norm() > kRestBoundaryDerivative ||
      tail.col(1).norm() > kRestBoundaryDerivative ||
      tail.col(2).norm() > kRestBoundaryDerivative)
    return false;
  Eigen::Matrix<double, 3, 1> midpoint = (head.col(0) + tail.col(0)) * 0.5;
  result.reset(head, tail, 2);
  result.generate(midpoint, Eigen::Vector2d(1.0, 1.0));
  return finiteTrajectory(result.getTraj());
}

// A bounded rest-to-rest segment for an optimizer failure in observed free
// space. This is only a candidate: the manager must still approve the complete
// path against its body-envelope map and the normal trajectory checks. It
// preserves all boundary derivatives and refuses substantial moving states.
inline bool makeShortRestRecovery(const Eigen::Matrix3d &head,
                                  const Eigen::Matrix3d &tail,
                                  double max_velocity, double max_acceleration,
                                  double max_jerk, poly_traj::MinJerkOpt &result)
{
  if (!head.allFinite() || !tail.allFinite() ||
      !std::isfinite(max_velocity) || max_velocity <= 0.0 ||
      !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
      !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
      head.col(1).norm() > .05 || head.col(2).norm() > .05 ||
      tail.rightCols<2>().norm() != 0.0)
    return false;
  const double distance = (tail.col(0)-head.col(0)).norm();
  if (distance <= kNearZeroGoalDistance || distance > 1.2+1e-6 ||
      std::abs(tail(2,0)-head(2,0)) > .05)
    return false;
  // Exact extrema of the zero-derivative quintic, with a 10% time reserve.
  double duration = 1.1*std::max({1.875*distance/max_velocity,
      std::sqrt(5.7735026919*distance/max_acceleration), std::cbrt(120.*distance/max_jerk), 1.0});
  for (int attempt=0; attempt<4; ++attempt, duration*=1.3)
  {
    if (!std::isfinite(duration) || duration > 30.) return false;
    result.reset(head, tail, 1);
    result.generate(Eigen::MatrixXd(3,0), Eigen::VectorXd::Constant(1,duration));
    const auto trajectory = result.getTraj();
    if (!finiteTrajectory(trajectory)) return false;
    // The quadratic jerk curve lies in the convex hull of these three
    // Bernstein controls. This bounds the whole curve, not just its ends.
    const auto c = trajectory[0].getCoeffMat();
    const Eigen::Vector3d j0 = 6.*c.col(2);
    const Eigen::Vector3d j1 = j0+12.*duration*c.col(1);
    const Eigen::Vector3d j2 = j0+24.*duration*c.col(1)+60.*duration*duration*c.col(0);
    if (trajectory.getMaxVelRate() <= max_velocity && trajectory.getMaxAccRate() <= max_acceleration &&
        std::max({j0.norm(),j1.norm(),j2.norm()}) <= max_jerk)
      return true;
  }
  return false;
}

template<class KnownFree>
inline bool shortRecoveryKnownFree(const poly_traj::Trajectory &trajectory, KnownFree known)
{
  if (!finiteTrajectory(trajectory)) return false;
  const double duration = trajectory.getTotalDuration();
  const int count = std::max(2, static_cast<int>(std::ceil(duration/.025)));
  for (int i=0; i<=count; ++i)
    if (!known(trajectory.getPos(duration*i/count))) return false;
  return true;
}

// The global path is a reference only; normal local optimization and map/flight
// guards still approve the executable trajectory. Repeated waypoints must never
// produce zero-duration pieces before those checks are reached.
inline bool makeGlobalReference(const Eigen::Matrix3d &head,
                                const Eigen::Matrix3d &tail,
                                const std::vector<Eigen::Vector3d> &waypoints,
                                double max_velocity,
                                poly_traj::MinJerkOpt &result)
{
  if (!head.allFinite() || !tail.allFinite() || waypoints.empty() ||
      !std::isfinite(max_velocity) || max_velocity <= 0.0)
    return false;
  for (const auto &waypoint : waypoints)
    if (!waypoint.allFinite()) return false;
  if ((tail.col(0) - waypoints.back()).norm() != 0.0) return false;
  if (waypoints.size() == 1 && nearZeroGoal(head.col(0), tail.col(0)))
    return makeRestGoalTrajectory(head, tail, result);

  Eigen::MatrixXd inner(3, waypoints.size() - 1);
  for (size_t i = 0; i + 1 < waypoints.size(); ++i) inner.col(i) = waypoints[i];
  result.reset(head, tail, waypoints.size());
  Eigen::VectorXd durations(waypoints.size());
  double reference_velocity = max_velocity / 1.5;
  for (int attempt = 0; attempt < 2; ++attempt)
  {
    for (size_t i = 0; i < waypoints.size(); ++i)
    {
      const double distance = (waypoints[i] - (i == 0 ? head.col(0).eval() : waypoints[i - 1])).norm();
      durations(i) = std::max(0.1, distance / reference_velocity);
    }
    if (!durations.allFinite()) return false;
    result.generate(inner, durations);
    const auto trajectory = result.getTraj();
    if (!finiteTrajectory(trajectory)) return false;
    const double peak_velocity = trajectory.getMaxVelRate();
    if (!std::isfinite(peak_velocity)) return false;
    if (peak_velocity < max_velocity || head.col(1).norm() > max_velocity ||
        tail.col(1).norm() > max_velocity)
      break;
    reference_velocity /= 1.5;
  }
  return true;
}
} // namespace vla_sim
