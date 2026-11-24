#pragma once

// Encoding compatibility with the Apache-2.0 gazebo_ros_block_laser plugin.
// Timing is deliberately separate: callers supply a completed measurement.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vla_sim {

struct BlockLaserGrid {
  size_t horizontal, vertical;
  double h_min, h_max, v_min, v_max, range_min, range_max;
};

inline void validateBlockLaserGrid(const BlockLaserGrid &grid) {
  if (grid.horizontal < 2 || grid.vertical < 2 ||
      grid.horizontal > 4096 || grid.vertical > 512 ||
      !std::isfinite(grid.h_min) || !std::isfinite(grid.h_max) ||
      !std::isfinite(grid.v_min) || !std::isfinite(grid.v_max) ||
      !std::isfinite(grid.range_min) || !std::isfinite(grid.range_max) ||
      grid.h_max <= grid.h_min || grid.v_max <= grid.v_min ||
      grid.range_min < 0 || grid.range_max <= grid.range_min)
    throw std::invalid_argument("Invalid physical block-laser grid");
}

struct EncodedBlockReturn { float x, y, z, intensity; };

template<class Range, class Retro, class Noise>
EncodedBlockReturn encodeBlockReturn(const BlockLaserGrid &grid,
                                     size_t h, size_t v,
                                     Range range, Retro retro, Noise noise) {
  if (h >= grid.horizontal || v >= grid.vertical)
    throw std::out_of_range("Physical block-laser beam index");
  const size_t next_h = h+1 < grid.horizontal ? h+1 : h;
  const size_t next_v = v+1 < grid.vertical ? v+1 : v;
  const size_t index = h+v*grid.horizontal;
  double measured = range(index);
  if (measured < grid.range_min) measured = -std::numeric_limits<double>::infinity();
  else if (measured > grid.range_max) measured = std::numeric_limits<double>::infinity();
  const double horizontal = .5*double(h+next_h)*(grid.h_max-grid.h_min)/double(grid.horizontal-1)+grid.h_min;
  const double vertical = .5*double(v+next_v)*(grid.v_max-grid.v_min)/double(grid.vertical-1)+grid.v_min;
  // Keep legacy midpoint xyz layout. cloud_bridge reconstructs actual ray
  // direction from row-major index and retains this measured norm.
  EncodedBlockReturn result;
  result.x = measured*std::cos(vertical)*std::cos(horizontal);
  result.y = measured*std::cos(vertical)*std::sin(horizontal);
  result.z = measured*std::sin(vertical);
  if (std::abs(grid.range_max-measured) > 1e-6) {
    result.x += noise(); result.y += noise(); result.z += noise();
  }
  result.intensity = .25*(retro(index)+retro(next_h+v*grid.horizontal)+
                           retro(h+next_v*grid.horizontal)+retro(next_h+next_v*grid.horizontal))+noise();
  return result;
}

class CompletedScanStampGate {
public:
  bool accept(uint64_t measurement_stamp_ns) {
    reset_ = has_last_ && measurement_stamp_ns < last_;
    const bool accepted = !has_last_ || measurement_stamp_ns > last_;
    last_ = measurement_stamp_ns;
    has_last_ = true;
    return accepted;
  }
  bool resetDetected() const { return reset_; }
private:
  bool has_last_{false}, reset_{false};
  uint64_t last_{0};
};

}  // namespace vla_sim
