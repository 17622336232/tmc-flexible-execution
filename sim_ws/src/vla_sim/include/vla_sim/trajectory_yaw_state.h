#pragma once

#include <algorithm>
#include <cmath>

namespace vla_sim {

struct YawCommand { double yaw, rate; };

// Search/hold can rotate the vehicle without this trajectory server emitting
// commands. Each new trajectory therefore takes control from measured yaw,
// rather than the last commanded yaw (which may date back to startup).
class TrajectoryYawState {
public:
  static constexpr double kOdometryMaxAge = .25;

  bool observe(double x, double y, double z, double w,
               double measurement_stamp, double now) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(w) || !validTime(measurement_stamp, now) ||
        (have_measurement_ && measurement_stamp < measured_stamp_))
      return false;
    const double norm = std::hypot(std::hypot(x, y), std::hypot(z, w));
    if (!std::isfinite(norm) || norm < 1e-12) return false;
    x /= norm; y /= norm; z /= norm; w /= norm;
    const double measured = std::atan2(2. * (w*z+x*y), 1. - 2. * (y*y+z*z));
    if (!std::isfinite(measured)) return false;
    measured_yaw_ = measured;
    measured_stamp_ = measurement_stamp;
    if (!have_measurement_) commanded_yaw_ = measured;
    have_measurement_ = true;
    return true;
  }

  bool beginTrajectory(double now) {
    if (!have_measurement_ || !validTime(measured_stamp_, now)) return false;
    commanded_yaw_ = measured_yaw_;
    return true;
  }

  double commandedYaw() const { return commanded_yaw_; }

  YawCommand advance(double target, double dt, double max_rate = 1.2) {
    if (!std::isfinite(target) || !std::isfinite(dt) || dt <= 0. ||
        !std::isfinite(max_rate) || max_rate <= 0.)
      return {commanded_yaw_, 0.};
    const double error = std::atan2(std::sin(target-commanded_yaw_),
                                    std::cos(target-commanded_yaw_));
    const double rate = std::max(-max_rate, std::min(max_rate, error/dt));
    commanded_yaw_ += rate*dt;
    return {commanded_yaw_, rate};
  }

private:
  static bool validTime(double stamp, double now) {
    return std::isfinite(stamp) && std::isfinite(now) && stamp >= 0. &&
           now >= stamp && now-stamp <= kOdometryMaxAge;
  }
  bool have_measurement_{false};
  double measured_yaw_{0.}, measured_stamp_{0.}, commanded_yaw_{0.};
};

} // namespace vla_sim
