// GPL-3.0. Sampling equations adapted from Diff-planner-indoor's
// mid360_realism_filter.py. This is a measurement-subset model, not Livox's
// physical non-repetitive scan pattern. No coordinates or return types change.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vla_sim {

struct IndoorSampling {
  bool enabled{false};
  uint64_t seed{360};
  double range_min{0.10}, range_max{8.0};
  double base_dropout{0.025}, range_dropout{0.13}, pattern_dropout{0.10};
  double azimuth_cycles{7.0}, phase_per_scan{0.71}, scan_rate{8.0};
  uint64_t burst_period{19};
  int sectors{24};
};

enum class SampleDecision { Keep, Dropout, Burst, RangeAmbiguous, Invalid };

// A counter-based generator makes a beam's fate independent of callback timing,
// processing order, earlier rejected beams and display subscribers. The same
// seed, acquisition timestamp and original beam index reproduce the same scan.
inline uint64_t sampleHash(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

inline SampleDecision retainIndoorBeam(const IndoorSampling &p,
                                      uint64_t stamp_ns, uint64_t beam_index,
                                      double azimuth, double range, bool hit) {
  if (!p.enabled) return SampleDecision::Keep;  // fast/dense legacy behavior
  if (!std::isfinite(range) || !std::isfinite(azimuth) || range < p.range_min)
    return SampleDecision::Invalid;
  // The caller freezes the physical hit/miss type before this sampling stage.
  // A noisy near-limit hit is never clamped into a no-return/free-space beam.
  constexpr double shell_tolerance = 2e-6;
  if ((hit && range >= p.range_max - 2 * shell_tolerance) ||
      (!hit && std::abs(range - p.range_max) > shell_tolerance))
    return SampleDecision::RangeAmbiguous;

  constexpr double pi = 3.14159265358979323846;
  const double seconds = static_cast<double>(stamp_ns) * 1e-9;
  const double scan_position = seconds * p.scan_rate;
  const uint64_t scan = static_cast<uint64_t>(std::floor(scan_position));
  const int sector = std::min(p.sectors - 1, std::max(0,
      static_cast<int>(std::floor((azimuth + pi) * p.sectors / (2 * pi)))));
  if (p.burst_period && scan > 0 && scan % p.burst_period == 0 &&
      sector == static_cast<int>((scan * 3) % p.sectors))
    return SampleDecision::Burst;

  const double normalized_range = std::min(1.0, std::max(0.0, range / p.range_max));
  const double pattern = 0.5 + 0.5 * std::sin(
      p.azimuth_cycles * azimuth + p.phase_per_scan * scan_position);
  const double probability = std::min(0.92, std::max(0.0,
      p.base_dropout + p.range_dropout * normalized_range * normalized_range +
      p.pattern_dropout * pattern));
  const uint64_t key = sampleHash(p.seed) ^ sampleHash(stamp_ns) ^ sampleHash(beam_index);
  const double uniform = static_cast<double>(sampleHash(key) >> 11) *
                         (1.0 / 9007199254740992.0);
  return uniform < probability ? SampleDecision::Dropout : SampleDecision::Keep;
}

}  // namespace vla_sim
