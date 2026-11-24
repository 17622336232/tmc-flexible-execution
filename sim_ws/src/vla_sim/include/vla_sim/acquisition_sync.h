#pragma once

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>

namespace vla_sim {

// A sensor acquisition can only use an exact pose or two poses bracketing its
// original timestamp. Callback arrival order must never become extrapolation.
struct AcquisitionPose {
  uint64_t stamp_ns{0};
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

enum class SyncDropReason { MissingBefore, GapTooWide, Timeout, Overflow, ClockReset };

inline const char *syncDropName(SyncDropReason reason) {
  switch (reason) {
    case SyncDropReason::MissingBefore: return "missing_before";
    case SyncDropReason::GapTooWide: return "gap_too_wide";
    case SyncDropReason::Timeout: return "wait_timeout";
    case SyncDropReason::Overflow: return "queue_overflow";
    case SyncDropReason::ClockReset: return "odom_clock_reset";
  }
  return "unknown";
}

template<class Payload> class AcquisitionSync {
public:
  struct Pending {
    uint64_t stamp_ns;
    double received_wall_seconds;
    Payload payload;
  };
  AcquisitionSync(uint64_t history_ns, uint64_t max_gap_ns,
                  size_t capacity, double timeout_wall_seconds)
    : history_ns_(history_ns), max_gap_ns_(max_gap_ns),
      capacity_(std::max(size_t(1), capacity)), timeout_(timeout_wall_seconds) {}

  template<class Drop> void addPose(AcquisitionPose pose, Drop drop) {
    if (!pose.position.allFinite() || !pose.orientation.coeffs().allFinite() ||
        pose.orientation.norm() < 1e-6) return;
    pose.orientation.normalize();
    if (!history_.empty() && pose.stamp_ns < history_.back().stamp_ns) {
      // Never register an acquisition from before a simulation clock reset
      // against the new run, even if their numeric timestamps happen to match.
      while (!pending_.empty()) {
        const Pending scan = pending_.front();
        pending_.pop_front();
        drop(scan, SyncDropReason::ClockReset);
      }
      history_.clear();
    }
    if (!history_.empty() && pose.stamp_ns == history_.back().stamp_ns)
      history_.back() = pose;
    else history_.push_back(pose);
    const uint64_t oldest = pose.stamp_ns > history_ns_ ? pose.stamp_ns - history_ns_ : 0;
    while (history_.size() > 2 && history_.front().stamp_ns < oldest)
      history_.pop_front();
  }

  template<class Drop> void enqueue(uint64_t stamp_ns, double received_wall_seconds,
                                    const Payload &payload, Drop drop) {
    if (pending_.size() >= capacity_) {
      const Pending scan = pending_.front();
      pending_.pop_front();
      drop(scan, SyncDropReason::Overflow);
    }
    pending_.push_back({stamp_ns, received_wall_seconds, payload});
  }

  template<class Ready, class Drop> void drain(double wall_seconds, Ready ready, Drop drop) {
    for (auto scan = pending_.begin(); scan != pending_.end();) {
      if (wall_seconds - scan->received_wall_seconds >= timeout_) {
        const Pending event = *scan;
        scan = pending_.erase(scan);
        drop(event, SyncDropReason::Timeout);
        continue;
      }
      if (history_.empty()) { ++scan; continue; }
      const auto after = std::lower_bound(history_.begin(), history_.end(), scan->stamp_ns,
          [](const AcquisitionPose &pose, uint64_t stamp) { return pose.stamp_ns < stamp; });
      if (after != history_.end() && after->stamp_ns == scan->stamp_ns) {
        const Pending event = *scan;
        scan = pending_.erase(scan);
        ready(event, *after, 0.0);
        continue;
      }
      if (after == history_.begin()) {
        const Pending event = *scan;
        scan = pending_.erase(scan);
        drop(event, SyncDropReason::MissingBefore);
        continue;
      }
      if (after == history_.end()) { ++scan; continue; }
      const auto &before = *(after - 1);
      const uint64_t before_gap = scan->stamp_ns - before.stamp_ns;
      const uint64_t after_gap = after->stamp_ns - scan->stamp_ns;
      if (before_gap > max_gap_ns_ || after_gap > max_gap_ns_) {
        const Pending event = *scan;
        scan = pending_.erase(scan);
        drop(event, SyncDropReason::GapTooWide);
        continue;
      }
      const double alpha = double(before_gap) / double(after->stamp_ns - before.stamp_ns);
      AcquisitionPose pose;
      pose.stamp_ns = scan->stamp_ns;
      pose.position = before.position + alpha * (after->position - before.position);
      pose.orientation = before.orientation.slerp(alpha, after->orientation).normalized();
      const Pending event = *scan;
      scan = pending_.erase(scan);
      ready(event, pose, double(std::max(before_gap, after_gap)) * 1e-9);
    }
  }

  size_t pendingSize() const { return pending_.size(); }

private:
  uint64_t history_ns_, max_gap_ns_;
  size_t capacity_;
  double timeout_;
  std::deque<AcquisitionPose> history_;
  std::deque<Pending> pending_;
};

}  // namespace vla_sim
