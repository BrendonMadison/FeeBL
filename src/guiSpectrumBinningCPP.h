#ifndef GUI_SPECTRUM_BINNING_CPP_H
#define GUI_SPECTRUM_BINNING_CPP_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace feebl { namespace gui_spectrum {

// End is exclusive. Every native index belongs to exactly one output range.
struct BinRange {
  std::size_t first, end;
  BinRange(std::size_t a, std::size_t b) : first(a), end(b) {}
};

inline bool finite_value(double value)
{
  // Keep input guards effective in production builds using -ffast-math.
  static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                std::numeric_limits<double>::is_iec559,
                "GUI spectrum partition requires IEEE binary64");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

inline std::size_t ceil_div(std::size_t n, std::size_t d)
{ return n / d + (n % d != 0); }

inline void append_ranges(std::vector<BinRange>& ranges,
                          std::size_t first, std::size_t end,
                          std::size_t group)
{
  while (first < end) {
    const std::size_t next = first + std::min(group, end - first);
    ranges.push_back(BinRange(first, next));
    first = next;
  }
}

// A deterministic geometry-only partition of the ACTUAL uniform native bins.
// For n <= 2048 preserve every bin. Otherwise reserve tail capacity using
// groups >= ceil(n/1024), then spend the remaining 2048-bin budget on the peak.
// Peak bins are those intersecting [peak_low, peak_high); no bin is split.
// If the peak alone is too large, only adjacent peak bins are grouped.
// The native accumulation, native file, flow counters and RNG are untouched.
inline std::vector<BinRange> partition(std::size_t n, double low, double high,
                                       double peak_low, double peak_high)
{
  const std::size_t limit = 2048;
  if (n == 0 || !finite_value(low) || !finite_value(high) ||
      !(high > low) || !finite_value(high - low) ||
      !finite_value(peak_low) || !finite_value(peak_high) ||
      peak_high < peak_low)
    throw std::invalid_argument("invalid GUI luminosity bin geometry");
  const double width = (high - low) / static_cast<double>(n);
  if (!(width > 0.0) || !finite_value(width))
    throw std::invalid_argument("invalid GUI luminosity native bin width");
  std::vector<BinRange> ranges;
  ranges.reserve(limit);
  if (n <= limit) {
    append_ranges(ranges, 0, n, 1);
    return ranges;
  }

  std::size_t first = 0, end = 0;
  if (peak_high > peak_low && peak_high > low && peak_low < high) {
    // Binary searches use the identical edge expression as SpectrumW. This
    // avoids rounding an almost-integral (cut-low)/width to the wrong index.
    std::size_t left = 0, right = n;
    while (left < right) {
      const std::size_t middle = left + (right - left) / 2;
      const double edge = low + width * static_cast<double>(middle + 1);
      if (edge <= peak_low) left = middle + 1;
      else right = middle;
    }
    first = left;
    left = first; right = n;
    while (left < right) {
      const std::size_t middle = left + (right - left) / 2;
      const double edge = low + width * static_cast<double>(middle);
      if (edge < peak_high) left = middle + 1;
      else right = middle;
    }
    end = left;
  }
  const std::size_t outside_group = ceil_div(n, std::size_t(1024));
  const std::size_t outside_count = ceil_div(first, outside_group) +
                                    ceil_div(n - end, outside_group);
  // At most 1025 outside groups: two independently rounded tail segments.
  if (outside_count >= limit)
    throw std::logic_error("GUI luminosity partition exhausted peak budget");
  const std::size_t peak_group = std::max(std::size_t(1),
      ceil_div(end - first, limit - outside_count));
  append_ranges(ranges, 0, first, outside_group);
  append_ranges(ranges, first, end, peak_group);
  append_ranges(ranges, end, n, outside_group);
  if (ranges.empty() || ranges.size() > limit)
    throw std::logic_error("GUI luminosity partition exceeded bin bound");
  return ranges;
}

} } // namespace feebl::gui_spectrum
#endif
