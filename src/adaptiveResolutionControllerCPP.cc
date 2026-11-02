#include "adaptiveResolutionControllerCPP.h"
#include "fftwPlannerMutexCPP.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>

#if defined(USE_FFTW3)
#include <fftw3.h>
#endif

namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* label) {
  if (a != 0U && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::overflow_error(std::string(label) + " product overflow");
  return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b,
                        const char* label) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error(std::string(label) + " sum overflow");
  return a + b;
}

bool power_of_two(std::size_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

std::uint64_t target_identity_hash(
    std::size_t target_count,
    const std::vector<unsigned char>* accepted_targets,
    const std::vector<std::uint64_t>* target_ids = NULL) {
  if (accepted_targets && accepted_targets->size() != target_count)
    throw std::invalid_argument(
        "accepted-target identity mask has the wrong size");
  if (target_ids && target_ids->size() != target_count)
    throw std::invalid_argument(
        "accepted-target stable-ID vector has the wrong size");
  std::uint64_t hash = UINT64_C(1469598103934665603);
  const auto mix_byte = [&hash](unsigned char value) {
    hash ^= static_cast<std::uint64_t>(value);
    hash *= UINT64_C(1099511628211);
  };
  const std::uint64_t count = static_cast<std::uint64_t>(target_count);
  for (unsigned int byte = 0U; byte < 8U; ++byte)
    mix_byte(static_cast<unsigned char>((count >> (8U * byte)) & 0xffU));
  for (std::size_t index = 0U; index < target_count; ++index) {
    mix_byte(accepted_targets ? (*accepted_targets)[index] != 0U : 1U);
    if (target_ids) {
      const std::uint64_t id = (*target_ids)[index];
      for (unsigned int byte = 0U; byte < 8U; ++byte)
        mix_byte(static_cast<unsigned char>((id >> (8U * byte)) & 0xffU));
    }
  }
  // Reserve zero as the explicit "identity not bound" sentinel used by
  // manually assembled evidence structures.
  return hash == 0U ? UINT64_C(1) : hash;
}

std::string trim(const std::string& value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return std::string();
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

std::size_t parse_positive_size(const std::string& token,
                                const char* label) {
  const std::string clean = trim(token);
  if (clean.empty() || clean[0] == '-' || clean[0] == '+')
    throw std::invalid_argument(std::string(label) +
                                " contains a malformed integer");
  errno = 0;
  char* end = NULL;
  const unsigned long long parsed = std::strtoull(clean.c_str(), &end, 10);
  if (errno == ERANGE || end == clean.c_str() || *end != '\0' ||
      parsed == 0ULL || parsed > static_cast<unsigned long long>(
                                     std::numeric_limits<std::size_t>::max()))
    throw std::invalid_argument(std::string(label) +
                                " contains an invalid positive integer");
  return static_cast<std::size_t>(parsed);
}

double parse_positive_double(const std::string& token,
                             const char* label) {
  const std::string clean = trim(token);
  errno = 0;
  char* end = NULL;
  const double parsed = std::strtod(clean.c_str(), &end);
  if (errno == ERANGE || end == clean.c_str() || *end != '\0' ||
      !std::isfinite(parsed) || !(parsed > 0.0))
    throw std::invalid_argument(std::string(label) +
                                " contains an invalid positive value");
  return parsed;
}

std::size_t even_extent(std::size_t begin, std::size_t end,
                        std::size_t limit) {
  if (end <= begin || end > limit)
    throw std::invalid_argument("invalid adaptive level extent");
  if (((end - begin) & 1U) == 0U) return end;
  if (end < limit) return end + 1U;
  if (begin > 0U) return end;  // The caller moves begin down below.
  throw std::invalid_argument(
      "adaptive level cannot form an even transform extent");
}

std::vector<double> smooth_binomial(const std::vector<double>& values,
                                    std::size_t nx, std::size_t ny,
                                    unsigned int passes) {
  std::vector<double> current(values);
  std::vector<double> next(values.size(), 0.0);
  for (unsigned int pass = 0; pass < passes; ++pass) {
    for (std::size_t x = 0; x < nx; ++x) {
      for (std::size_t y = 0; y < ny; ++y) {
        double sum = 0.0;
        double norm = 0.0;
        for (int dx = -1; dx <= 1; ++dx) {
          const long xx = static_cast<long>(x) + dx;
          if (xx < 0 || xx >= static_cast<long>(nx)) continue;
          const double wx = dx == 0 ? 2.0 : 1.0;
          for (int dy = -1; dy <= 1; ++dy) {
            const long yy = static_cast<long>(y) + dy;
            if (yy < 0 || yy >= static_cast<long>(ny)) continue;
            const double wy = dy == 0 ? 2.0 : 1.0;
            const double weight = wx * wy;
            sum += weight * current[static_cast<std::size_t>(xx) * ny +
                                    static_cast<std::size_t>(yy)];
            norm += weight;
          }
        }
        next[x * ny + y] = sum / norm;
      }
    }
    current.swap(next);
  }
  return current;
}

}  // namespace

namespace feebl {
namespace adaptive_grid {

FieldOperator parse_field_operator(const std::string& value) {
  const std::string clean = trim(value);
  if (clean == "legacy_cic12_v1") return FieldOperator::LegacyCic12V1;
  if (clean == "fd4_lagrange4_v1") return FieldOperator::Fd4Lagrange4V1;
  throw std::invalid_argument(
      "adaptive_grid_field_operator must be legacy_cic12_v1 or "
      "fd4_lagrange4_v1");
}

const char* field_operator_name(FieldOperator value) {
  switch (value) {
    case FieldOperator::LegacyCic12V1:
      return "legacy_cic12_v1";
    case FieldOperator::Fd4Lagrange4V1:
      return "fd4_lagrange4_v1";
  }
  throw std::invalid_argument("unknown adaptive field operator");
}

std::size_t field_operator_additional_window_guard_cells(
    FieldOperator value) {
  switch (value) {
    case FieldOperator::LegacyCic12V1:
      return 0U;
    case FieldOperator::Fd4Lagrange4V1:
      // Relative to the canonical legacy footprint, the fused FD4/cubic
      // footprint reaches three additional potential nodes on either face.
      return 3U;
  }
  throw std::invalid_argument("unknown adaptive field operator");
}

std::size_t field_operator_transition_guard_cells(FieldOperator value) {
  switch (value) {
    case FieldOperator::LegacyCic12V1:
      return 2U;
    case FieldOperator::Fd4Lagrange4V1:
      return 4U;
  }
  throw std::invalid_argument("unknown adaptive field operator");
}

std::vector<std::size_t> parse_refinement_factors(
    const std::string& value) {
  std::vector<std::size_t> result;
  std::size_t begin = 0U;
  while (begin <= value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::string token = value.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin);
    if (trim(token).empty())
      throw std::invalid_argument(
          "adaptive_grid_field_spacing_classes contains an empty entry");
    const std::size_t factor = parse_positive_size(
        token, "adaptive_grid_field_spacing_classes");
    if (!power_of_two(factor))
      throw std::invalid_argument(
          "adaptive_grid_field_spacing_classes entries must be powers of two");
    if (!result.empty() && factor <= result.back())
      throw std::invalid_argument(
          "adaptive_grid_field_spacing_classes must be strictly increasing");
    result.push_back(factor);
    if (comma == std::string::npos) break;
    begin = comma + 1U;
  }
  if (result.size() < 2U || result.front() != 1U)
    throw std::invalid_argument(
        "adaptive_grid_field_spacing_classes must begin with 1 and contain "
        "at least one finer power-of-two class");
  return result;
}

EncounterTailBudget make_encounter_tail_budget(
    const std::vector<std::size_t>& beam1_slice_counts,
    const std::vector<std::size_t>& beam2_slice_counts,
    double fractional_budget) {
  if (beam1_slice_counts.empty() || beam2_slice_counts.empty())
    throw std::invalid_argument(
        "longitudinal tail-budget selection requires two non-empty beams");
  if (!std::isfinite(fractional_budget) || fractional_budget < 0.0 ||
      !(fractional_budget < 1.0))
    throw std::invalid_argument(
        "longitudinal tail fraction budget must be finite and in [0,1)");

  long double beam1_total = 0.0L;
  long double beam2_total = 0.0L;
  for (std::size_t count : beam1_slice_counts)
    beam1_total += static_cast<long double>(count);
  for (std::size_t count : beam2_slice_counts)
    beam2_total += static_cast<long double>(count);
  if (!std::isfinite(beam1_total) || !std::isfinite(beam2_total) ||
      !(beam1_total > 0.0L) || !(beam2_total > 0.0L))
    throw std::invalid_argument(
        "longitudinal tail-budget selection requires positive beam totals");

  struct WeightedPair {
    long double weight;
    std::size_t beam1_slice;
    std::size_t beam2_slice;
  };
  const std::size_t pair_count = checked_product(
      beam1_slice_counts.size(), beam2_slice_counts.size(),
      "longitudinal tail-budget mask");
  std::vector<WeightedPair> pairs;
  pairs.reserve(pair_count);
  for (std::size_t i = 0U; i < beam1_slice_counts.size(); ++i) {
    const long double beam1_fraction =
        static_cast<long double>(beam1_slice_counts[i]) / beam1_total;
    for (std::size_t j = 0U; j < beam2_slice_counts.size(); ++j) {
      const long double beam2_fraction =
          static_cast<long double>(beam2_slice_counts[j]) / beam2_total;
      pairs.push_back(WeightedPair{
          beam1_fraction * beam2_fraction, i, j});
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const WeightedPair& left, const WeightedPair& right) {
              if (left.weight != right.weight)
                return left.weight < right.weight;
              if (left.beam1_slice != right.beam1_slice)
                return left.beam1_slice < right.beam1_slice;
              return left.beam2_slice < right.beam2_slice;
            });

  const long double fraction_budget =
      static_cast<long double>(fractional_budget);
  EncounterTailBudget result;
  result.exemption_mask.assign(pair_count, 0U);
  for (const WeightedPair& pair : pairs) {
    const long double candidate = result.selected_fraction + pair.weight;
    if (candidate > fraction_budget) break;
    result.exemption_mask[
        pair.beam1_slice * beam2_slice_counts.size() + pair.beam2_slice] = 1U;
    ++result.selected_pairs;
    result.selected_fraction = candidate;
  }
  return result;
}

double richardson_order_for_pair(const std::string& table,
                                 std::size_t coarse_factor,
                                 std::size_t fine_factor) {
  const std::string clean_table = trim(table);
  if (clean_table.empty())
    throw std::invalid_argument(
        "adaptive_grid_richardson_order_table must not be empty");
  if (clean_table.find('=') == std::string::npos &&
      clean_table.find(':') == std::string::npos)
    return parse_positive_double(clean_table,
                                 "adaptive_grid_richardson_order_table");

  // The card contract uses factor:order entries.  The order attached to the
  // coarse member of a compared pair controls that Richardson estimate.
  if (clean_table.find('=') == std::string::npos) {
    std::size_t begin = 0U;
    while (begin <= clean_table.size()) {
      const std::size_t comma = clean_table.find(',', begin);
      const std::string entry = trim(clean_table.substr(
          begin, comma == std::string::npos ? std::string::npos
                                             : comma - begin));
      const std::size_t colon = entry.find(':');
      if (colon == std::string::npos ||
          entry.find(':', colon + 1U) != std::string::npos)
        throw std::invalid_argument(
            "adaptive_grid_richardson_order_table has a malformed entry");
      const std::size_t factor = parse_positive_size(
          entry.substr(0U, colon),
          "adaptive_grid_richardson_order_table");
      const double order = parse_positive_double(
          entry.substr(colon + 1U),
          "adaptive_grid_richardson_order_table");
      if (factor == coarse_factor) return order;
      if (comma == std::string::npos) break;
      begin = comma + 1U;
    }
    throw std::invalid_argument(
        "adaptive_grid_richardson_order_table has no entry for the coarse "
        "spacing class");
  }

  std::size_t begin = 0U;
  while (begin <= clean_table.size()) {
    const std::size_t comma = clean_table.find(',', begin);
    const std::string entry = trim(clean_table.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin));
    const std::size_t equal = entry.find('=');
    const std::size_t colon = entry.find(':');
    if (equal == std::string::npos || colon == std::string::npos ||
        colon > equal || entry.find('=', equal + 1U) != std::string::npos)
      throw std::invalid_argument(
          "adaptive_grid_richardson_order_table has a malformed entry");
    const std::size_t coarse = parse_positive_size(
        entry.substr(0U, colon), "adaptive_grid_richardson_order_table");
    const std::size_t fine = parse_positive_size(
        entry.substr(colon + 1U, equal - colon - 1U),
        "adaptive_grid_richardson_order_table");
    const double order = parse_positive_double(
        entry.substr(equal + 1U), "adaptive_grid_richardson_order_table");
    if (coarse == coarse_factor && fine == fine_factor) return order;
    if (comma == std::string::npos) break;
    begin = comma + 1U;
  }
  throw std::invalid_argument(
      "adaptive_grid_richardson_order_table has no entry for the requested "
      "spacing-class pair");
}

std::size_t FieldLevelGeometry::cells() const {
  return checked_product(n_x, n_y, "adaptive field level");
}

bool FieldLevelGeometry::valid() const noexcept {
  if (refinement_factor == 0U || global_n_x < 4U || global_n_y < 4U ||
      origin_x < 0 || origin_y < 0 || n_x < 4U || n_y < 4U ||
      global_n_x > static_cast<std::size_t>(
                         std::numeric_limits<long>::max()) ||
      global_n_y > static_cast<std::size_t>(
                         std::numeric_limits<long>::max()) ||
      n_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      n_y > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !std::isfinite(delta_x) || !std::isfinite(delta_y) ||
      !std::isfinite(inverse_delta_x) ||
      !std::isfinite(inverse_delta_y) || !std::isfinite(offset_x) ||
      !std::isfinite(offset_y) || !(delta_x > 0.0f) ||
      !(delta_y > 0.0f) || !(inverse_delta_x > 0.0f) ||
      !(inverse_delta_y > 0.0f))
    return false;
  const std::size_t ox = static_cast<std::size_t>(origin_x);
  const std::size_t oy = static_cast<std::size_t>(origin_y);
  return ox <= global_n_x && oy <= global_n_y &&
      n_x <= global_n_x - ox && n_y <= global_n_y - oy;
}

void integrate_first_order_bspline_cells(
    double coordinate,
    double half_width,
    FieldAxis axis,
    const FieldLevelGeometry& geometry,
    std::vector<BSplineCellIntegral>& result) {
  if (!geometry.valid() || !std::isfinite(coordinate) ||
      !std::isfinite(half_width) || !(half_width > 0.0))
    throw std::invalid_argument(
        "invalid fixed-width B-spline integration request");

  std::size_t global_cells = 0U;
  std::size_t local_cells = 0U;
  CanonicalIndex signed_origin = 0;
  float delta = 0.0f;
  float offset = 0.0f;
  switch (axis) {
    case FieldAxis::X:
      global_cells = geometry.global_n_x;
      local_cells = geometry.n_x;
      signed_origin = geometry.origin_x;
      delta = geometry.delta_x;
      offset = geometry.offset_x;
      break;
    case FieldAxis::Y:
      global_cells = geometry.global_n_y;
      local_cells = geometry.n_y;
      signed_origin = geometry.origin_y;
      delta = geometry.delta_y;
      offset = geometry.offset_y;
      break;
    default:
      throw std::invalid_argument("invalid fixed-width B-spline axis");
  }

  // Work in dimensionless global cell-edge coordinates.  This avoids the
  // cancellation and overflow of forming coordinate +/- half_width in the
  // caller's physical unit system.  Long double also makes the endpoint CDF
  // subtraction robust when a support edge lies very close to a cell face.
  const long double centre =
      static_cast<long double>(coordinate) /
          static_cast<long double>(delta) +
      static_cast<long double>(offset);
  const long double radius =
      static_cast<long double>(half_width) /
      static_cast<long double>(delta);
  if (!std::isfinite(centre) || !std::isfinite(radius) ||
      !(radius > 0.0L))
    throw std::invalid_argument(
        "non-finite fixed-width B-spline lattice coordinate");
  const long double support_begin = centre - radius;
  const long double support_end = centre + radius;
  if (!std::isfinite(support_begin) || !std::isfinite(support_end))
    throw std::invalid_argument(
        "non-finite fixed-width B-spline support");

  const std::size_t origin = static_cast<std::size_t>(signed_origin);
  const std::size_t local_end = checked_add(
      origin, local_cells, "fixed-width B-spline local end");
  if (support_begin < 0.0L ||
      support_end > static_cast<long double>(global_cells))
    throw std::out_of_range(
        "fixed-width B-spline support escapes global geometry");
  if (support_begin < static_cast<long double>(origin) ||
      support_end > static_cast<long double>(local_end))
    throw std::out_of_range(
        "fixed-width B-spline support escapes local geometry");

  const long double first_floor = std::floor(support_begin);
  const long double end_ceil = std::ceil(support_end);
  if (first_floor < static_cast<long double>(origin) ||
      end_ceil > static_cast<long double>(local_end) ||
      !(end_ceil > first_floor))
    throw std::logic_error(
        "fixed-width B-spline support produced invalid cell bounds");
  const std::size_t first_global = static_cast<std::size_t>(first_floor);
  const std::size_t end_global = static_cast<std::size_t>(end_ceil);

  const auto normalized_cdf = [](long double normalized_displacement) {
    if (normalized_displacement <= -1.0L) return 0.0L;
    if (normalized_displacement < 0.0L) {
      const long double t = normalized_displacement + 1.0L;
      return 0.5L * t * t;
    }
    if (normalized_displacement < 1.0L) {
      const long double t = 1.0L - normalized_displacement;
      return 1.0L - 0.5L * t * t;
    }
    return 1.0L;
  };

  result.clear();
  result.reserve(end_global - first_global);
  long double sum = 0.0L;
  for (std::size_t global_index = first_global;
       global_index < end_global; ++global_index) {
    const long double left =
        (static_cast<long double>(global_index) - centre) / radius;
    const long double right =
        (static_cast<long double>(global_index + 1U) - centre) / radius;
    long double integral = normalized_cdf(right) - normalized_cdf(left);
    const long double roundoff =
        32.0L * std::numeric_limits<long double>::epsilon();
    if (integral < 0.0L && integral >= -roundoff) integral = 0.0L;
    if (integral < 0.0L || !std::isfinite(integral))
      throw std::logic_error(
          "fixed-width B-spline produced an invalid cell integral");
    if (integral == 0.0L) continue;
    BSplineCellIntegral cell;
    cell.local_index = global_index - origin;
    cell.weight = static_cast<double>(integral);
    if (!std::isfinite(cell.weight) || !(cell.weight > 0.0))
      throw std::logic_error(
          "fixed-width B-spline cell integral is not representable");
    result.push_back(cell);
    sum += static_cast<long double>(cell.weight);
  }
  if (result.empty() || std::fabs(sum - 1.0L) > 1.0e-12L)
    throw std::logic_error(
        "fixed-width B-spline cell integrals do not close to unity");
}

std::vector<BSplineCellIntegral> integrate_first_order_bspline_cells(
    double coordinate,
    double half_width,
    FieldAxis axis,
    const FieldLevelGeometry& geometry) {
  std::vector<BSplineCellIntegral> result;
  integrate_first_order_bspline_cells(
      coordinate, half_width, axis, geometry, result);
  return result;
}

FieldLevelGeometry make_refined_level_geometry(
    const CanonicalLatticeSnapshot& lattice,
    const UniformWindowPlan& window,
    std::size_t factor,
    std::size_t extra_fine_guard_cells,
    double anchor_x,
    double anchor_y,
    FieldOperator field_operator) {
  if (window.empty() || factor == 0U || !power_of_two(factor))
    throw std::invalid_argument("invalid adaptive refined-level request");
  if (!std::isfinite(anchor_x) || !std::isfinite(anchor_y))
    throw std::invalid_argument("adaptive field anchor must be finite");
  FieldLevelGeometry result;
  result.refinement_factor = factor;
  result.global_n_x = checked_product(lattice.n_x(), factor,
                                      "refined global n_x");
  result.global_n_y = checked_product(lattice.n_y(), factor,
                                      "refined global n_y");
  const std::size_t scaled_origin_x = checked_product(
      static_cast<std::size_t>(window.origin_x()), factor,
      "refined origin x");
  const std::size_t scaled_origin_y = checked_product(
      static_cast<std::size_t>(window.origin_y()), factor,
      "refined origin y");
  const std::size_t scaled_end_x = checked_product(
      checked_add(static_cast<std::size_t>(window.origin_x()),
                  window.local_n_x(), "window end x"),
      factor, "refined end x");
  const std::size_t scaled_end_y = checked_product(
      checked_add(static_cast<std::size_t>(window.origin_y()),
                  window.local_n_y(), "window end y"),
      factor, "refined end y");
  const std::size_t operator_guard =
      field_operator_additional_window_guard_cells(field_operator);
  const std::size_t explicit_guard = checked_add(
      extra_fine_guard_cells, operator_guard,
      "refined explicit/operator guard");
  // One refinement ratio plus the explicit guard is enough to contain both
  // the fine source cloud and the selected gather stencil across phase
  // shifts.  The legacy path retains exactly its former guard arithmetic;
  // FD4/cubic adds the three nodes by which its footprint exceeds the
  // canonical twelve-read gather used to build the window.
  const std::size_t guard = factor == 1U
      ? explicit_guard
      : checked_add(std::max<std::size_t>(2U, factor),
                    explicit_guard, "refined guard");
  std::size_t begin_x = scaled_origin_x > guard
      ? scaled_origin_x - guard : 0U;
  std::size_t begin_y = scaled_origin_y > guard
      ? scaled_origin_y - guard : 0U;
  std::size_t end_x = std::min(
      result.global_n_x, checked_add(scaled_end_x, guard,
                                     "refined guarded end x"));
  std::size_t end_y = std::min(
      result.global_n_y, checked_add(scaled_end_y, guard,
                                     "refined guarded end y"));
  const std::size_t adjusted_end_x = even_extent(
      begin_x, end_x, result.global_n_x);
  if (adjusted_end_x == end_x && ((end_x - begin_x) & 1U) != 0U) --begin_x;
  else end_x = adjusted_end_x;
  const std::size_t adjusted_end_y = even_extent(
      begin_y, end_y, result.global_n_y);
  if (adjusted_end_y == end_y && ((end_y - begin_y) & 1U) != 0U) --begin_y;
  else end_y = adjusted_end_y;
  result.origin_x = static_cast<CanonicalIndex>(begin_x);
  result.origin_y = static_cast<CanonicalIndex>(begin_y);
  result.n_x = end_x - begin_x;
  result.n_y = end_y - begin_y;
  result.delta_x = lattice.delta_x() / static_cast<float>(factor);
  result.delta_y = lattice.delta_y() / static_cast<float>(factor);
  result.inverse_delta_x = lattice.inverse_delta_x() *
                           static_cast<float>(factor);
  result.inverse_delta_y = lattice.inverse_delta_y() *
                           static_cast<float>(factor);
  result.offset_x = lattice.offset_x() * static_cast<float>(factor) -
                    static_cast<float>(anchor_x) * result.inverse_delta_x;
  result.offset_y = lattice.offset_y() * static_cast<float>(factor) -
                    static_cast<float>(anchor_y) * result.inverse_delta_y;
  if (!result.valid())
    throw std::invalid_argument("constructed adaptive field level is invalid");
  return result;
}

LevelDepositDescriptor map_deposit_to_level(
    float x, float y, const FieldLevelGeometry& geometry) {
  if (!geometry.valid() || !std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("invalid coordinate or adaptive level");
  LevelDepositDescriptor result;
  float coordinate_x = x * geometry.inverse_delta_x + geometry.offset_x -
                       0.5f;
  float coordinate_y = y * geometry.inverse_delta_y + geometry.offset_y -
                       0.5f;
  if (!std::isfinite(coordinate_x) || !std::isfinite(coordinate_y))
    throw std::invalid_argument(
        "non-finite coordinate in adaptive deposit mapping");
  if (coordinate_x < 0.0f || coordinate_y < 0.0f ||
      static_cast<double>(coordinate_x) >=
          static_cast<double>(geometry.global_n_x) ||
      static_cast<double>(coordinate_y) >=
          static_cast<double>(geometry.global_n_y))
    return result;
  const double floored_x = std::floor(static_cast<double>(coordinate_x));
  const double floored_y = std::floor(static_cast<double>(coordinate_y));
  // The half-open global check above makes these conversions representable.
  // Perform containment and subtraction unsigned so a far-away local window
  // cannot trigger signed overflow for an outside source coordinate.
  const std::size_t global_x = static_cast<std::size_t>(floored_x);
  const std::size_t global_y = static_cast<std::size_t>(floored_y);
  const std::size_t origin_x = static_cast<std::size_t>(geometry.origin_x);
  const std::size_t origin_y = static_cast<std::size_t>(geometry.origin_y);
  if (global_x >= geometry.global_n_x - 1U ||
      global_y >= geometry.global_n_y - 1U ||
      global_x < origin_x || global_y < origin_y)
    return result;
  const std::size_t local_x = global_x - origin_x;
  const std::size_t local_y = global_y - origin_y;
  if (local_x >= geometry.n_x - 1U || local_y >= geometry.n_y - 1U)
    return result;
  result.weight_x = static_cast<float>(
      static_cast<double>(coordinate_x) - floored_x);
  result.weight_y = static_cast<float>(
      static_cast<double>(coordinate_y) - floored_y);
  result.accepted = true;
  result.base_x = static_cast<int>(local_x);
  result.base_y = static_cast<int>(local_y);
  return result;
}

LevelGatherDescriptor map_gather_to_level(
    float x, float y, const FieldLevelGeometry& geometry) {
  if (!geometry.valid() || !std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("invalid coordinate or adaptive level");
  LevelGatherDescriptor result;
  const double absolute_x = static_cast<double>(
      x * geometry.inverse_delta_x + geometry.offset_x);
  const double absolute_y = static_cast<double>(
      y * geometry.inverse_delta_y + geometry.offset_y);
  if (!std::isfinite(absolute_x) || !std::isfinite(absolute_y))
    throw std::invalid_argument(
        "non-finite coordinate in adaptive gather mapping");
  // Every accepted twelve-node stencil is wholly inside the global lattice.
  // Reject its former halo-only range before integer conversion; this also
  // excludes the rounded LONG_MAX+1 boundary on very large public geometries.
  if (absolute_x < 0.0 || absolute_y < 0.0 ||
      absolute_x >= static_cast<double>(geometry.global_n_x) ||
      absolute_y >= static_cast<double>(geometry.global_n_y))
    return result;
  double y_shift = absolute_y - 0.5;
  double x_shift = absolute_x - 0.5;
  const double y_base_x_floor = std::floor(absolute_x);
  const double y_base_y_floor = std::floor(y_shift);
  const double x_base_y_floor = std::floor(absolute_y);
  const double x_base_x_floor = std::floor(x_shift);
  if (y_base_x_floor < 0.0 || y_base_y_floor < 0.0 ||
      x_base_y_floor < 0.0 || x_base_x_floor < 0.0)
    return result;
  const std::size_t y_global_base_x =
      static_cast<std::size_t>(y_base_x_floor);
  const std::size_t y_global_base_y =
      static_cast<std::size_t>(y_base_y_floor);
  const std::size_t x_global_base_y =
      static_cast<std::size_t>(x_base_y_floor);
  const std::size_t x_global_base_x =
      static_cast<std::size_t>(x_base_x_floor);
  const std::size_t origin_x = static_cast<std::size_t>(geometry.origin_x);
  const std::size_t origin_y = static_cast<std::size_t>(geometry.origin_y);
  if (y_global_base_x < origin_x || y_global_base_y < origin_y ||
      x_global_base_x < origin_x || x_global_base_y < origin_y)
    return result;
  const std::size_t y_base_x = y_global_base_x - origin_x;
  const std::size_t y_base_y = y_global_base_y - origin_y;
  const std::size_t x_base_x = x_global_base_x - origin_x;
  const std::size_t x_base_y = x_global_base_y - origin_y;
  // These four inequalities are exactly the old twelve inside() checks,
  // expressed without forming base +/- 1 in a signed global index.
  if (y_base_x == 0U || y_base_x >= geometry.n_x - 1U ||
      y_base_y >= geometry.n_y - 1U ||
      x_base_x >= geometry.n_x - 1U ||
      x_base_y == 0U || x_base_y >= geometry.n_y - 1U)
    return result;
  result.accepted = true;
  result.y_base_x = static_cast<int>(y_base_x);
  result.y_base_y = static_cast<int>(y_base_y);
  result.y_weight = static_cast<float>(y_shift - y_base_y_floor);
  result.x_base_x = static_cast<int>(x_base_x);
  result.x_base_y = static_cast<int>(x_base_y);
  result.x_weight = static_cast<float>(x_shift - x_base_x_floor);
  result.fractional_x = absolute_x - y_base_x_floor;
  result.fractional_y = absolute_y - x_base_y_floor;
  return result;
}

LevelGatherDescriptor map_gather_to_level(
    float x, float y, const FieldLevelGeometry& geometry,
    FieldOperator field_operator) {
  if (field_operator == FieldOperator::LegacyCic12V1)
    return map_gather_to_level(x, y, geometry);
  if (field_operator != FieldOperator::Fd4Lagrange4V1)
    throw std::invalid_argument("unknown adaptive field operator");
  if (!geometry.valid() || !std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("invalid coordinate or adaptive level");

  LevelGatherDescriptor result;
  result.field_operator = FieldOperator::Fd4Lagrange4V1;
  if (geometry.global_n_x < 5U || geometry.global_n_y < 5U ||
      geometry.n_x < 5U || geometry.n_y < 5U)
    return result;

  // Potential samples are cell centred, hence the -0.5.  Unlike the legacy
  // compatibility mapper, all new affine arithmetic remains binary64.
  const double global_x =
      static_cast<double>(x) * geometry.inverse_delta_x +
      geometry.offset_x - 0.5;
  const double global_y =
      static_cast<double>(y) * geometry.inverse_delta_y +
      geometry.offset_y - 0.5;
  if (!std::isfinite(global_x) || !std::isfinite(global_y))
    throw std::invalid_argument(
        "non-finite coordinate in fourth-order adaptive gather mapping");
  const double absolute_x = global_x + 0.5;
  const double absolute_y = global_y + 0.5;
  if (absolute_x < 0.0 || absolute_y < 0.0 ||
      absolute_x >= static_cast<double>(geometry.global_n_x) ||
      absolute_y >= static_cast<double>(geometry.global_n_y))
    return result;

  const auto cubic_lagrange_weights = [](double t) {
    if (!std::isfinite(t) || t < 0.0 || !(t < 1.0))
      throw std::invalid_argument(
          "invalid phase in fourth-order adaptive gather mapping");
    return std::array<double, 4>{{
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0}};
  };

  // Keep the complete pre-closure mapper path for every stencil that was
  // already interior-valid.  In particular, subtracting a nonzero local
  // origin before floor/fraction extraction preserves its former binary64
  // rounding and therefore the established central result bit pattern.
  const double local_x = global_x -
      static_cast<double>(geometry.origin_x);
  const double local_y = global_y -
      static_cast<double>(geometry.origin_y);
  if (local_x >= 3.0 && local_y >= 3.0 &&
      local_x < static_cast<double>(geometry.n_x - 4U) &&
      local_y < static_cast<double>(geometry.n_y - 4U)) {
    const double base_x_floor = std::floor(local_x);
    const double base_y_floor = std::floor(local_y);
    const std::size_t base_x = static_cast<std::size_t>(base_x_floor);
    const std::size_t base_y = static_cast<std::size_t>(base_y_floor);
    if (base_x >= 3U && base_y >= 3U &&
        base_x + 4U < geometry.n_x && base_y + 4U < geometry.n_y) {
      result.centred_base_x = static_cast<int>(base_x);
      result.centred_base_y = static_cast<int>(base_y);
      result.lagrange_weights_x = cubic_lagrange_weights(
          local_x - base_x_floor);
      result.lagrange_weights_y = cubic_lagrange_weights(
          local_y - base_y_floor);
      result.accepted = true;
      return result;
    }
  }

  const auto shifted_lagrange_weights = [](double coordinate,
                                            long interpolation_start) {
    const double t = coordinate -
        static_cast<double>(interpolation_start);
    if (!std::isfinite(t) || t < -0.5 || t > 3.5)
      throw std::invalid_argument(
          "invalid shifted phase in fourth-order adaptive gather mapping");
    return std::array<double, 4>{{
        -(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0,
        t * (t - 2.0) * (t - 3.0) / 2.0,
        -t * (t - 1.0) * (t - 3.0) / 2.0,
        t * (t - 1.0) * (t - 2.0) / 6.0}};
  };

  // Return the five global potential nodes used by the fourth-order
  // derivative at one cell-centre node.  Only the first two and final two
  // nodes of the *global* lattice receive one-sided closures.  A cropped
  // local face therefore cannot masquerade as a physical boundary.
  const auto derivative_support = [](std::size_t node,
                                     std::size_t global_cells) {
    if (node <= 1U)
      return std::array<std::size_t, 2>{{0U, 4U}};
    if (node >= global_cells - 2U)
      return std::array<std::size_t, 2>{{global_cells - 5U,
                                         global_cells - 1U}};
    return std::array<std::size_t, 2>{{node - 2U, node + 2U}};
  };

  const auto map_axis = [&cubic_lagrange_weights,
                         &shifted_lagrange_weights,
                         &derivative_support](
      double coordinate, CanonicalIndex signed_origin,
      std::size_t local_cells, std::size_t global_cells,
      int& centred_base, std::array<double, 4>& weights) {
    const long base = static_cast<long>(std::floor(coordinate));
    const long natural_start = base - 1L;
    const long maximum_start = static_cast<long>(global_cells - 4U);
    const long interpolation_start = std::max(
        0L, std::min(natural_start, maximum_start));
    const std::size_t origin = static_cast<std::size_t>(signed_origin);
    const std::size_t start = static_cast<std::size_t>(interpolation_start);
    if (start < origin || start - origin > local_cells - 4U)
      return false;
    const std::size_t local_end = origin + local_cells - 1U;
    for (std::size_t offset = 0U; offset < 4U; ++offset) {
      const std::array<std::size_t, 2> support = derivative_support(
          start + offset, global_cells);
      if (support[0] < origin || support[1] > local_end)
        return false;
    }
    const std::size_t local_start = start - origin;
    centred_base = static_cast<int>(local_start + 1U);
    // Retain the established interior expression and operation order exactly.
    // Only an actually shifted face stencil needs the general-node formula.
    if (interpolation_start == natural_start)
      weights = cubic_lagrange_weights(
          coordinate - static_cast<double>(base));
    else
      weights = shifted_lagrange_weights(coordinate,
                                          interpolation_start);
    return true;
  };

  if (!map_axis(global_x, geometry.origin_x, geometry.n_x,
                geometry.global_n_x, result.centred_base_x,
                result.lagrange_weights_x) ||
      !map_axis(global_y, geometry.origin_y, geometry.n_y,
                geometry.global_n_y, result.centred_base_y,
                result.lagrange_weights_y))
    return result;
  result.accepted = true;
  return result;
}

namespace {

struct DensityMoments {
  long double charge = 0.0L;
  long double x = 0.0L;
  long double y = 0.0L;
};

DensityMoments density_moments(const std::vector<double>& density,
                               const FieldLevelGeometry& geometry) {
  DensityMoments result;
  for (std::size_t ix = 0U; ix < geometry.n_x; ++ix) {
    const long double x =
        (static_cast<long double>(geometry.origin_x) + ix + 0.5L -
         geometry.offset_x) * geometry.delta_x;
    for (std::size_t iy = 0U; iy < geometry.n_y; ++iy) {
      const double value = density[ix * geometry.n_y + iy];
      if (!std::isfinite(value))
        throw std::runtime_error(
            "non-finite charge in coherent CIC density hierarchy");
      const long double y =
          (static_cast<long double>(geometry.origin_y) + iy + 0.5L -
           geometry.offset_y) * geometry.delta_y;
      const long double charge = value;
      result.charge += charge;
      result.x += charge * x;
      result.y += charge * y;
    }
  }
  return result;
}

}  // namespace

DensityTransferReport prolong_cic_charge_density(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination) {
  if (!source_geometry.valid() || !destination_geometry.valid() ||
      source.size() != source_geometry.cells() ||
      destination_geometry.refinement_factor <
          source_geometry.refinement_factor ||
      destination_geometry.refinement_factor %
              source_geometry.refinement_factor != 0U)
    throw std::invalid_argument(
        "invalid coherent CIC density hierarchy request");
  const std::size_t ratio = destination_geometry.refinement_factor /
                            source_geometry.refinement_factor;
  const double expected_delta_x = source_geometry.delta_x /
                                  static_cast<double>(ratio);
  const double expected_delta_y = source_geometry.delta_y /
                                  static_cast<double>(ratio);
  const double spacing_tolerance = 32.0 *
      std::numeric_limits<float>::epsilon();
  const double expected_offset_x =
      source_geometry.offset_x * static_cast<double>(ratio);
  const double expected_offset_y =
      source_geometry.offset_y * static_cast<double>(ratio);
  if (std::fabs(destination_geometry.delta_x / expected_delta_x - 1.0) >
          spacing_tolerance ||
      std::fabs(destination_geometry.delta_y / expected_delta_y - 1.0) >
          spacing_tolerance ||
      std::fabs(destination_geometry.offset_x - expected_offset_x) >
          spacing_tolerance *
              std::max(1.0, std::fabs(expected_offset_x)) ||
      std::fabs(destination_geometry.offset_y - expected_offset_y) >
          spacing_tolerance *
              std::max(1.0, std::fabs(expected_offset_y)) ||
      destination_geometry.global_n_x !=
          checked_product(source_geometry.global_n_x, ratio,
                          "density hierarchy global x") ||
      destination_geometry.global_n_y !=
          checked_product(source_geometry.global_n_y, ratio,
                          "density hierarchy global y"))
    throw std::invalid_argument(
        "noncommensurate coherent CIC density hierarchy lattices");

  // Preserve an identical hierarchy member bit-for-bit.
  if (source_geometry.refinement_factor ==
          destination_geometry.refinement_factor &&
      source_geometry.global_n_x == destination_geometry.global_n_x &&
      source_geometry.global_n_y == destination_geometry.global_n_y &&
      source_geometry.origin_x == destination_geometry.origin_x &&
      source_geometry.origin_y == destination_geometry.origin_y &&
      source_geometry.n_x == destination_geometry.n_x &&
      source_geometry.n_y == destination_geometry.n_y &&
      source_geometry.delta_x == destination_geometry.delta_x &&
      source_geometry.delta_y == destination_geometry.delta_y &&
      source_geometry.offset_x == destination_geometry.offset_x &&
      source_geometry.offset_y == destination_geometry.offset_y) {
    for (double value : source)
      if (!std::isfinite(value))
        throw std::runtime_error(
            "non-finite canonical charge in coherent CIC hierarchy");
    destination = source;
    DensityTransferReport identity;
    return identity;
  }

  destination.assign(destination_geometry.cells(), 0.0);
  bool nonnegative_source = true;
  bool nonpositive_source = true;
  const std::size_t destination_origin_x =
      static_cast<std::size_t>(destination_geometry.origin_x);
  const std::size_t destination_origin_y =
      static_cast<std::size_t>(destination_geometry.origin_y);
  const std::size_t destination_end_x = checked_add(
      destination_origin_x, destination_geometry.n_x,
      "density hierarchy destination end x");
  const std::size_t destination_end_y = checked_add(
      destination_origin_y, destination_geometry.n_y,
      "density hierarchy destination end y");
  const std::size_t children_per_cell = checked_product(
      ratio, ratio, "density hierarchy children per cell");
  const double inverse_children = 1.0 /
      static_cast<double>(children_per_cell);
  for (std::size_t ix = 0U; ix < source_geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < source_geometry.n_y; ++iy) {
      const double charge = source[ix * source_geometry.n_y + iy];
      if (!std::isfinite(charge))
        throw std::runtime_error(
            "non-finite canonical charge in coherent CIC hierarchy");
      nonnegative_source = nonnegative_source && charge >= 0.0;
      nonpositive_source = nonpositive_source && charge <= 0.0;
      if (charge == 0.0) continue;
      const std::size_t source_global_x = checked_add(
          static_cast<std::size_t>(source_geometry.origin_x), ix,
          "density hierarchy source global x");
      const std::size_t source_global_y = checked_add(
          static_cast<std::size_t>(source_geometry.origin_y), iy,
          "density hierarchy source global y");
      const std::size_t first_child_x = checked_product(
          source_global_x, ratio, "density hierarchy first child x");
      const std::size_t first_child_y = checked_product(
          source_global_y, ratio, "density hierarchy first child y");
      const std::size_t child_end_x = checked_add(
          first_child_x, ratio, "density hierarchy child end x");
      const std::size_t child_end_y = checked_add(
          first_child_y, ratio, "density hierarchy child end y");
      if (first_child_x < destination_origin_x ||
          child_end_x > destination_end_x ||
          first_child_y < destination_origin_y ||
          child_end_y > destination_end_y)
        throw std::runtime_error(
            "coherent cell source escaped its planned destination window");
      const double child_charge = charge * inverse_children;
      for (std::size_t child_x = first_child_x;
           child_x < child_end_x; ++child_x)
        for (std::size_t child_y = first_child_y;
             child_y < child_end_y; ++child_y) {
          const std::size_t local_x = child_x - destination_origin_x;
          const std::size_t local_y = child_y - destination_origin_y;
          destination[local_x * destination_geometry.n_y + local_y] =
              child_charge;
        }
    }

  DensityTransferReport report;
  const DensityMoments before = density_moments(source, source_geometry);
  const DensityMoments after = density_moments(
      destination, destination_geometry);
  const long double charge_scale = std::max(
      std::fabs(before.charge), static_cast<long double>(1.0));
  const long double x_scale = std::max(
      std::fabs(before.x), charge_scale * source_geometry.delta_x);
  const long double y_scale = std::max(
      std::fabs(before.y), charge_scale * source_geometry.delta_y);
  report.relative_charge_error = static_cast<double>(
      std::fabs(after.charge - before.charge) / charge_scale);
  report.normalized_first_moment_error = static_cast<double>(std::max(
      std::fabs(after.x - before.x) / x_scale,
      std::fabs(after.y - before.y) / y_scale));
  for (double value : destination) {
    if ((nonnegative_source && value < 0.0) ||
        (nonpositive_source && value > 0.0)) {
      report.sign_preserving = false;
      break;
    }
  }
  const double closure_tolerance = 2.0e-12;
  if (!report.sign_preserving ||
      report.relative_charge_error > closure_tolerance ||
      report.normalized_first_moment_error > closure_tolerance)
    throw std::runtime_error(
        "coherent CIC density hierarchy failed charge/centroid closure");
  return report;
}

void extract_registered_density_window(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination) {
  if (!source_geometry.valid() || !destination_geometry.valid() ||
      source.size() != source_geometry.cells() ||
      source_geometry.refinement_factor !=
          destination_geometry.refinement_factor ||
      source_geometry.global_n_x != destination_geometry.global_n_x ||
      source_geometry.global_n_y != destination_geometry.global_n_y ||
      source_geometry.delta_x != destination_geometry.delta_x ||
      source_geometry.delta_y != destination_geometry.delta_y ||
      source_geometry.offset_x != destination_geometry.offset_x ||
      source_geometry.offset_y != destination_geometry.offset_y ||
      destination_geometry.origin_x < source_geometry.origin_x ||
      destination_geometry.origin_y < source_geometry.origin_y)
    throw std::invalid_argument(
        "density subwindow is not registered with its hierarchy level");
  const std::size_t offset_x = static_cast<std::size_t>(
      destination_geometry.origin_x - source_geometry.origin_x);
  const std::size_t offset_y = static_cast<std::size_t>(
      destination_geometry.origin_y - source_geometry.origin_y);
  if (offset_x > source_geometry.n_x ||
      destination_geometry.n_x > source_geometry.n_x - offset_x ||
      offset_y > source_geometry.n_y ||
      destination_geometry.n_y > source_geometry.n_y - offset_y)
    throw std::invalid_argument(
        "density subwindow escaped its registered hierarchy level");
  destination.assign(destination_geometry.cells(), 0.0);
  for (std::size_t ix = 0U; ix < destination_geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < destination_geometry.n_y; ++iy)
      destination[ix * destination_geometry.n_y + iy] =
          source[(offset_x + ix) * source_geometry.n_y + offset_y + iy];
}

DensityTransferReport restrict_cic_charge_density(
    const std::vector<double>& source,
    const FieldLevelGeometry& source_geometry,
    const FieldLevelGeometry& destination_geometry,
    std::vector<double>& destination) {
  if (!source_geometry.valid() || !destination_geometry.valid() ||
      source.size() != source_geometry.cells() ||
      source_geometry.refinement_factor <
          destination_geometry.refinement_factor ||
      source_geometry.refinement_factor %
              destination_geometry.refinement_factor != 0U)
    throw std::invalid_argument(
        "invalid coherent CIC density restriction request");
  const std::size_t ratio = source_geometry.refinement_factor /
                            destination_geometry.refinement_factor;
  if (ratio == 1U) {
    extract_registered_density_window(
        source, source_geometry, destination_geometry, destination);
    return DensityTransferReport();
  }
  const double tolerance = 32.0 * std::numeric_limits<float>::epsilon();
  const double expected_delta_x = source_geometry.delta_x * ratio;
  const double expected_delta_y = source_geometry.delta_y * ratio;
  const double expected_offset_x = source_geometry.offset_x /
                                   static_cast<double>(ratio);
  const double expected_offset_y = source_geometry.offset_y /
                                   static_cast<double>(ratio);
  if (std::fabs(destination_geometry.delta_x / expected_delta_x - 1.0) >
          tolerance ||
      std::fabs(destination_geometry.delta_y / expected_delta_y - 1.0) >
          tolerance ||
      std::fabs(destination_geometry.offset_x - expected_offset_x) >
          tolerance * std::max(1.0, std::fabs(expected_offset_x)) ||
      std::fabs(destination_geometry.offset_y - expected_offset_y) >
          tolerance * std::max(1.0, std::fabs(expected_offset_y)) ||
      checked_product(destination_geometry.global_n_x, ratio,
                      "density restriction global x") !=
          source_geometry.global_n_x ||
      checked_product(destination_geometry.global_n_y, ratio,
                      "density restriction global y") !=
          source_geometry.global_n_y)
    throw std::invalid_argument(
        "noncommensurate coherent CIC density restriction lattices");

  destination.assign(destination_geometry.cells(), 0.0);
  bool nonnegative_source = true;
  bool nonpositive_source = true;
  const std::size_t source_origin_x =
      static_cast<std::size_t>(source_geometry.origin_x);
  const std::size_t source_origin_y =
      static_cast<std::size_t>(source_geometry.origin_y);
  const std::size_t source_end_x = checked_add(
      source_origin_x, source_geometry.n_x,
      "density restriction source end x");
  const std::size_t source_end_y = checked_add(
      source_origin_y, source_geometry.n_y,
      "density restriction source end y");
  for (double value : source) {
    if (!std::isfinite(value))
      throw std::runtime_error(
          "non-finite reference charge in coherent CIC restriction");
    nonnegative_source = nonnegative_source && value >= 0.0;
    nonpositive_source = nonpositive_source && value <= 0.0;
  }
  for (std::size_t ix = 0U; ix < destination_geometry.n_x; ++ix)
    for (std::size_t iy = 0U; iy < destination_geometry.n_y; ++iy) {
      const std::size_t destination_global_x = checked_add(
          static_cast<std::size_t>(destination_geometry.origin_x), ix,
          "density restriction destination global x");
      const std::size_t destination_global_y = checked_add(
          static_cast<std::size_t>(destination_geometry.origin_y), iy,
          "density restriction destination global y");
      const std::size_t first_child_x = checked_product(
          destination_global_x, ratio,
          "density restriction first child x");
      const std::size_t first_child_y = checked_product(
          destination_global_y, ratio,
          "density restriction first child y");
      const std::size_t child_end_x = checked_add(
          first_child_x, ratio, "density restriction child end x");
      const std::size_t child_end_y = checked_add(
          first_child_y, ratio, "density restriction child end y");
      if (first_child_x < source_origin_x || child_end_x > source_end_x ||
          first_child_y < source_origin_y || child_end_y > source_end_y)
        throw std::runtime_error(
            "registered density restriction escaped its source window");
      long double sum = 0.0L;
      for (std::size_t child_x = first_child_x;
           child_x < child_end_x; ++child_x)
        for (std::size_t child_y = first_child_y;
             child_y < child_end_y; ++child_y)
          sum += source[(child_x - source_origin_x) * source_geometry.n_y +
                        child_y - source_origin_y];
      destination[ix * destination_geometry.n_y + iy] =
          static_cast<double>(sum);
    }

  DensityTransferReport report;
  const DensityMoments before = density_moments(source, source_geometry);
  const DensityMoments after = density_moments(
      destination, destination_geometry);
  const long double charge_scale = std::max(
      std::fabs(before.charge), static_cast<long double>(1.0));
  const long double x_scale = std::max(
      std::fabs(before.x), charge_scale * destination_geometry.delta_x);
  const long double y_scale = std::max(
      std::fabs(before.y), charge_scale * destination_geometry.delta_y);
  report.relative_charge_error = static_cast<double>(
      std::fabs(after.charge - before.charge) / charge_scale);
  report.normalized_first_moment_error = static_cast<double>(std::max(
      std::fabs(after.x - before.x) / x_scale,
      std::fabs(after.y - before.y) / y_scale));
  for (double value : destination)
    if ((nonnegative_source && value < 0.0) ||
        (nonpositive_source && value > 0.0)) {
      report.sign_preserving = false;
      break;
    }
  if (!report.sign_preserving || report.relative_charge_error > 2.0e-12)
    throw std::runtime_error(
        "coherent CIC density restriction failed charge/sign closure");
  return report;
}

void accumulate_grouped_cic_sampling(
    const LevelDepositDescriptor& reference_descriptor,
    double absolute_charge,
    const FieldLevelGeometry& reference_geometry,
    const FieldLevelGeometry& sampling_geometry,
    std::vector<double>& sum_weight,
    std::vector<double>& sum_weight_squared) {
  if (!reference_descriptor.accepted || !reference_geometry.valid() ||
      !sampling_geometry.valid() || !std::isfinite(absolute_charge) ||
      absolute_charge < 0.0 ||
      reference_geometry.refinement_factor <
          sampling_geometry.refinement_factor ||
      reference_geometry.refinement_factor %
              sampling_geometry.refinement_factor != 0U ||
      sum_weight.size() != sampling_geometry.cells() ||
      sum_weight_squared.size() != sampling_geometry.cells())
    throw std::invalid_argument(
        "invalid grouped reference-CIC sampling request");
  const std::size_t ratio = reference_geometry.refinement_factor /
                            sampling_geometry.refinement_factor;
  const double tolerance = 32.0 * std::numeric_limits<float>::epsilon();
  const double expected_delta_x = reference_geometry.delta_x * ratio;
  const double expected_delta_y = reference_geometry.delta_y * ratio;
  const double expected_offset_x = reference_geometry.offset_x /
                                   static_cast<double>(ratio);
  const double expected_offset_y = reference_geometry.offset_y /
                                   static_cast<double>(ratio);
  if (checked_product(sampling_geometry.global_n_x, ratio,
                      "grouped sampling global x") !=
          reference_geometry.global_n_x ||
      checked_product(sampling_geometry.global_n_y, ratio,
                      "grouped sampling global y") !=
          reference_geometry.global_n_y ||
      std::fabs(sampling_geometry.delta_x / expected_delta_x - 1.0) >
          tolerance ||
      std::fabs(sampling_geometry.delta_y / expected_delta_y - 1.0) >
          tolerance ||
      std::fabs(sampling_geometry.offset_x - expected_offset_x) >
          tolerance * std::max(1.0, std::fabs(expected_offset_x)) ||
      std::fabs(sampling_geometry.offset_y - expected_offset_y) >
          tolerance * std::max(1.0, std::fabs(expected_offset_y)))
    throw std::invalid_argument(
        "grouped reference-CIC sampling lattices are not registered");
  if (reference_descriptor.base_x < 0 || reference_descriptor.base_y < 0 ||
      static_cast<std::size_t>(reference_descriptor.base_x) + 1U >=
          reference_geometry.n_x ||
      static_cast<std::size_t>(reference_descriptor.base_y) + 1U >=
          reference_geometry.n_y)
    throw std::invalid_argument(
        "grouped reference-CIC descriptor has no complete stencil");
  const std::size_t reference_base_x = checked_add(
      static_cast<std::size_t>(reference_geometry.origin_x),
      static_cast<std::size_t>(reference_descriptor.base_x),
      "grouped sampling reference base x");
  const std::size_t reference_base_y = checked_add(
      static_cast<std::size_t>(reference_geometry.origin_y),
      static_cast<std::size_t>(reference_descriptor.base_y),
      "grouped sampling reference base y");
  const double wx = reference_descriptor.weight_x;
  const double wy = reference_descriptor.weight_y;
  if (!std::isfinite(wx) || !std::isfinite(wy) || wx < 0.0 || wx > 1.0 ||
      wy < 0.0 || wy > 1.0)
    throw std::invalid_argument(
        "grouped reference-CIC descriptor has invalid weights");
  const std::size_t global_x[4] = {
      reference_base_x, reference_base_x + 1U,
      reference_base_x, reference_base_x + 1U};
  const std::size_t global_y[4] = {
      reference_base_y, reference_base_y,
      reference_base_y + 1U, reference_base_y + 1U};
  const double contribution[4] = {
      absolute_charge * (1.0 - wx) * (1.0 - wy),
      absolute_charge * wx * (1.0 - wy),
      absolute_charge * (1.0 - wx) * wy,
      absolute_charge * wx * wy};
  std::size_t grouped_cells[4] = {0U, 0U, 0U, 0U};
  double grouped_weights[4] = {0.0, 0.0, 0.0, 0.0};
  std::size_t groups = 0U;
  const std::size_t sampling_origin_x =
      static_cast<std::size_t>(sampling_geometry.origin_x);
  const std::size_t sampling_origin_y =
      static_cast<std::size_t>(sampling_geometry.origin_y);
  for (std::size_t corner = 0U; corner < 4U; ++corner) {
    const std::size_t parent_x = global_x[corner] / ratio;
    const std::size_t parent_y = global_y[corner] / ratio;
    if (parent_x < sampling_origin_x || parent_y < sampling_origin_y ||
        parent_x >= sampling_origin_x + sampling_geometry.n_x ||
        parent_y >= sampling_origin_y + sampling_geometry.n_y)
      throw std::runtime_error(
          "grouped reference-CIC sampling parent escaped its window");
    const std::size_t parent =
        (parent_x - sampling_origin_x) * sampling_geometry.n_y +
        (parent_y - sampling_origin_y);
    std::size_t group = 0U;
    while (group < groups && grouped_cells[group] != parent) ++group;
    if (group == groups) {
      grouped_cells[groups] = parent;
      ++groups;
    }
    grouped_weights[group] += contribution[corner];
  }
  for (std::size_t group = 0U; group < groups; ++group) {
    const double weight = grouped_weights[group];
    sum_weight[grouped_cells[group]] += weight;
    sum_weight_squared[grouped_cells[group]] += weight * weight;
  }
}

SourcePairRegime source_pair_regime(
    std::size_t coarse_factor, std::size_t fine_factor,
    std::size_t source_factor) noexcept {
  if (coarse_factor == 0U || fine_factor == 0U || source_factor == 0U ||
      coarse_factor > std::numeric_limits<std::size_t>::max() / 2U ||
      fine_factor != 2U * coarse_factor)
    return SourcePairRegime::Invalid;
  if (fine_factor <= source_factor && source_factor % fine_factor == 0U)
    return SourcePairRegime::SourceProjection;
  if (coarse_factor >= source_factor && coarse_factor % source_factor == 0U)
    return SourcePairRegime::FieldOnly;
  return SourcePairRegime::Invalid;
}

SameRegimeTriplet select_same_regime_triplet(
    const std::vector<std::size_t>& factors,
    std::size_t candidate_pair,
    std::size_t source_factor) {
  SameRegimeTriplet result;
  if (candidate_pair + 1U >= factors.size()) return result;
  result.regime = source_pair_regime(
      factors[candidate_pair], factors[candidate_pair + 1U], source_factor);
  if (result.regime == SourcePairRegime::Invalid) return result;
  if (candidate_pair + 2U < factors.size() &&
      source_pair_regime(factors[candidate_pair + 1U],
                         factors[candidate_pair + 2U], source_factor) ==
          result.regime) {
    result.valid = true;
    result.first_pair = candidate_pair;
    result.second_pair = candidate_pair + 1U;
    return result;
  }
  if (candidate_pair > 0U &&
      source_pair_regime(factors[candidate_pair - 1U],
                         factors[candidate_pair], source_factor) ==
          result.regime) {
    result.valid = true;
    result.first_pair = candidate_pair - 1U;
    result.second_pair = candidate_pair;
  }
  return result;
}

std::array<float, 2> gather_level_field(
    const double* phi, const FieldLevelGeometry& geometry,
    const LevelGatherDescriptor& d) {
  if (!phi || !geometry.valid() || !d.accepted)
    throw std::invalid_argument("invalid adaptive field gather");
  const auto at = [phi, &geometry](int x, int y) -> double {
    if (x < 0 || y < 0 || x >= static_cast<int>(geometry.n_x) ||
        y >= static_cast<int>(geometry.n_y))
      throw std::out_of_range("adaptive gather stencil escaped its level");
    return phi[static_cast<std::size_t>(x) * geometry.n_y +
               static_cast<std::size_t>(y)];
  };
  const float y_complement = 1.0f - d.y_weight;
  const double phi1_y = d.y_weight * at(d.y_base_x + 1, d.y_base_y + 1) +
      y_complement * at(d.y_base_x + 1, d.y_base_y);
  const double phi2_y = d.y_weight * at(d.y_base_x, d.y_base_y + 1) +
      y_complement * at(d.y_base_x, d.y_base_y);
  const double phi3_y = d.y_weight * at(d.y_base_x - 1, d.y_base_y + 1) +
      y_complement * at(d.y_base_x - 1, d.y_base_y);
  const float x_complement = 1.0f - d.x_weight;
  const double phi1_x = d.x_weight * at(d.x_base_x + 1, d.x_base_y + 1) +
      x_complement * at(d.x_base_x, d.x_base_y + 1);
  const double phi2_x = d.x_weight * at(d.x_base_x + 1, d.x_base_y) +
      x_complement * at(d.x_base_x, d.x_base_y);
  const double phi3_x = d.x_weight * at(d.x_base_x + 1, d.x_base_y - 1) +
      x_complement * at(d.x_base_x, d.x_base_y - 1);
  std::array<float, 2> field;
  // Preserve the historical narrowing point: GENERAL_GRID stores the raw
  // derivative in a float TRIVECTOR and only then applies the 0.5 factor.
  field[0] = static_cast<float>(
      (d.fractional_x * (phi1_y - phi2_y) +
       (1.0 - d.fractional_x) * (phi2_y - phi3_y)) *
      geometry.inverse_delta_x);
  field[1] = static_cast<float>(
      (d.fractional_y * (phi1_x - phi2_x) +
       (1.0 - d.fractional_y) * (phi2_x - phi3_x)) *
      geometry.inverse_delta_y);
  field[0] *= 0.5f;
  field[1] *= 0.5f;
  return field;
}

std::array<float, 2> gather_level_field(
    const double* phi, const FieldLevelGeometry& geometry,
    const LevelGatherDescriptor& d, FieldOperator field_operator) {
  if (field_operator == FieldOperator::LegacyCic12V1) {
    if (d.field_operator != FieldOperator::LegacyCic12V1)
      throw std::invalid_argument(
          "adaptive field gather descriptor/operator mismatch");
    // This call is intentionally the entire legacy branch: its established
    // operation order and narrowing points remain untouched above.
    return gather_level_field(phi, geometry, d);
  }
  if (field_operator != FieldOperator::Fd4Lagrange4V1)
    throw std::invalid_argument("unknown adaptive field operator");
  if (!phi || !geometry.valid() || !d.accepted ||
      d.field_operator != FieldOperator::Fd4Lagrange4V1)
    throw std::invalid_argument("invalid fourth-order adaptive field gather");

  const int base_x = d.centred_base_x;
  const int base_y = d.centred_base_y;
  if (geometry.global_n_x < 5U || geometry.global_n_y < 5U ||
      geometry.n_x < 5U || geometry.n_y < 5U ||
      base_x < 1 || base_y < 1 ||
      base_x > static_cast<int>(geometry.n_x) - 3 ||
      base_y > static_cast<int>(geometry.n_y) - 3)
    throw std::out_of_range(
        "fourth-order adaptive gather stencil escaped its level");

  const auto derivative_support = [](std::size_t node,
                                     std::size_t global_cells) {
    if (node <= 1U)
      return std::array<std::size_t, 2>{{0U, 4U}};
    if (node >= global_cells - 2U)
      return std::array<std::size_t, 2>{{global_cells - 5U,
                                         global_cells - 1U}};
    return std::array<std::size_t, 2>{{node - 2U, node + 2U}};
  };
  const auto interpolation_supported = [&derivative_support](
      int base, CanonicalIndex signed_origin, std::size_t local_cells,
      std::size_t global_cells) {
    const std::size_t origin = static_cast<std::size_t>(signed_origin);
    const std::size_t local_end = origin + local_cells - 1U;
    const std::size_t local_start = static_cast<std::size_t>(base - 1);
    for (std::size_t offset = 0U; offset < 4U; ++offset) {
      const std::size_t global_node = origin + local_start + offset;
      const std::array<std::size_t, 2> support = derivative_support(
          global_node, global_cells);
      if (support[0] < origin || support[1] > local_end)
        return false;
    }
    return true;
  };
  if (!interpolation_supported(base_x, geometry.origin_x, geometry.n_x,
                               geometry.global_n_x) ||
      !interpolation_supported(base_y, geometry.origin_y, geometry.n_y,
                               geometry.global_n_y))
    throw std::out_of_range(
        "fourth-order adaptive gather used a cropped-face closure");

  double sum_weights_x = 0.0;
  double sum_weights_y = 0.0;
  for (std::size_t i = 0U; i < 4U; ++i) {
    if (!std::isfinite(d.lagrange_weights_x[i]) ||
        !std::isfinite(d.lagrange_weights_y[i]))
      throw std::invalid_argument(
          "non-finite fourth-order adaptive gather weight");
    sum_weights_x += d.lagrange_weights_x[i];
    sum_weights_y += d.lagrange_weights_y[i];
  }
  const double weight_tolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  if (std::fabs(sum_weights_x - 1.0) > weight_tolerance ||
      std::fabs(sum_weights_y - 1.0) > weight_tolerance)
    throw std::invalid_argument(
        "fourth-order adaptive gather weights do not form a partition");

  const auto at = [phi, &geometry](int x_index, int y_index) {
    if (x_index < 0 || y_index < 0 ||
        x_index >= static_cast<int>(geometry.n_x) ||
        y_index >= static_cast<int>(geometry.n_y))
      throw std::out_of_range(
          "fourth-order adaptive gather potential read escaped its level");
    const double value = phi[static_cast<std::size_t>(x_index) *
                                 geometry.n_y +
                             static_cast<std::size_t>(y_index)];
    if (!std::isfinite(value))
      throw std::runtime_error(
          "non-finite potential in fourth-order adaptive gather");
    return value;
  };

  double field_x = 0.0;
  double field_y = 0.0;
  const double inverse_12_dx =
      static_cast<double>(geometry.inverse_delta_x) / 12.0;
  const double inverse_12_dy =
      static_cast<double>(geometry.inverse_delta_y) / 12.0;
  const auto derivative_x_at = [&at, &geometry, inverse_12_dx](
      int x_index, int y_index) {
    const std::size_t global_x =
        static_cast<std::size_t>(geometry.origin_x) +
        static_cast<std::size_t>(x_index);
    if (global_x == 0U)
      return (-25.0 * at(x_index, y_index) +
              48.0 * at(x_index + 1, y_index) -
              36.0 * at(x_index + 2, y_index) +
              16.0 * at(x_index + 3, y_index) -
               3.0 * at(x_index + 4, y_index)) * inverse_12_dx;
    if (global_x == 1U)
      return (-3.0 * at(x_index - 1, y_index) -
              10.0 * at(x_index, y_index) +
              18.0 * at(x_index + 1, y_index) -
               6.0 * at(x_index + 2, y_index) +
                     at(x_index + 3, y_index)) * inverse_12_dx;
    if (global_x == geometry.global_n_x - 2U)
      return (-at(x_index - 3, y_index) +
               6.0 * at(x_index - 2, y_index) -
              18.0 * at(x_index - 1, y_index) +
              10.0 * at(x_index, y_index) +
               3.0 * at(x_index + 1, y_index)) * inverse_12_dx;
    if (global_x == geometry.global_n_x - 1U)
      return ( 3.0 * at(x_index - 4, y_index) -
              16.0 * at(x_index - 3, y_index) +
              36.0 * at(x_index - 2, y_index) -
              48.0 * at(x_index - 1, y_index) +
              25.0 * at(x_index, y_index)) * inverse_12_dx;
    return (at(x_index - 2, y_index) -
            8.0 * at(x_index - 1, y_index) +
            8.0 * at(x_index + 1, y_index) -
            at(x_index + 2, y_index)) * inverse_12_dx;
  };
  const auto derivative_y_at = [&at, &geometry, inverse_12_dy](
      int x_index, int y_index) {
    const std::size_t global_y =
        static_cast<std::size_t>(geometry.origin_y) +
        static_cast<std::size_t>(y_index);
    if (global_y == 0U)
      return (-25.0 * at(x_index, y_index) +
              48.0 * at(x_index, y_index + 1) -
              36.0 * at(x_index, y_index + 2) +
              16.0 * at(x_index, y_index + 3) -
               3.0 * at(x_index, y_index + 4)) * inverse_12_dy;
    if (global_y == 1U)
      return (-3.0 * at(x_index, y_index - 1) -
              10.0 * at(x_index, y_index) +
              18.0 * at(x_index, y_index + 1) -
               6.0 * at(x_index, y_index + 2) +
                     at(x_index, y_index + 3)) * inverse_12_dy;
    if (global_y == geometry.global_n_y - 2U)
      return (-at(x_index, y_index - 3) +
               6.0 * at(x_index, y_index - 2) -
              18.0 * at(x_index, y_index - 1) +
              10.0 * at(x_index, y_index) +
               3.0 * at(x_index, y_index + 1)) * inverse_12_dy;
    if (global_y == geometry.global_n_y - 1U)
      return ( 3.0 * at(x_index, y_index - 4) -
              16.0 * at(x_index, y_index - 3) +
              36.0 * at(x_index, y_index - 2) -
              48.0 * at(x_index, y_index - 1) +
              25.0 * at(x_index, y_index)) * inverse_12_dy;
    return (at(x_index, y_index - 2) -
            8.0 * at(x_index, y_index - 1) +
            8.0 * at(x_index, y_index + 1) -
            at(x_index, y_index + 2)) * inverse_12_dy;
  };
  for (int a = 0; a < 4; ++a) {
    const int x_index = base_x + a - 1;
    const double weight_x = d.lagrange_weights_x[
        static_cast<std::size_t>(a)];
    for (int b = 0; b < 4; ++b) {
      const int y_index = base_y + b - 1;
      const double weight = weight_x * d.lagrange_weights_y[
          static_cast<std::size_t>(b)];
      const double derivative_x = derivative_x_at(x_index, y_index);
      const double derivative_y = derivative_y_at(x_index, y_index);
      if (!std::isfinite(derivative_x) || !std::isfinite(derivative_y))
        throw std::runtime_error(
            "non-finite derivative in fourth-order adaptive gather");
      field_x += weight * derivative_x;
      field_y += weight * derivative_y;
    }
  }
  field_x *= 0.5;
  field_y *= 0.5;
  const double float_limit = std::numeric_limits<float>::max();
  if (!std::isfinite(field_x) || !std::isfinite(field_y) ||
      std::fabs(field_x) > float_limit || std::fabs(field_y) > float_limit)
    throw std::overflow_error(
        "fourth-order adaptive gathered field is not representable");
  return std::array<float, 2>{{static_cast<float>(field_x),
                               static_cast<float>(field_y)}};
}

RichardsonEstimate estimate_same_particle_field_error(
    const std::vector<float>& coarse1,
    const std::vector<float>& coarse2,
    const std::vector<float>& fine1,
    const std::vector<float>& fine2,
    std::size_t ratio,
    double order,
    double field_floor,
    const std::vector<unsigned char>* accepted1,
    const std::vector<unsigned char>* accepted2) {
  if (coarse1.size() != fine1.size() || coarse2.size() != fine2.size() ||
      (coarse1.size() & 1U) != 0U || (coarse2.size() & 1U) != 0U ||
      (accepted1 && accepted1->size() * 2U != coarse1.size()) ||
      (accepted2 && accepted2->size() * 2U != coarse2.size()) ||
      ratio < 2U || !power_of_two(ratio) || !std::isfinite(order) ||
      !(order > 0.0) || !std::isfinite(field_floor) || field_floor < 0.0)
    throw std::invalid_argument("invalid same-particle Richardson inputs");
  RichardsonEstimate result;
  bool compared_any_target = false;
  const auto accumulate = [&](const std::vector<float>& coarse,
                              const std::vector<float>& fine,
                              const std::vector<unsigned char>* accepted,
                              std::size_t component_offset) {
    for (std::size_t component = 0U; component < 2U; ++component) {
      long double difference2 = 0.0L;
      long double fine2_norm = 0.0L;
      double max_difference = 0.0;
      double max_fine = 0.0;
      std::size_t count = 0U;
      for (std::size_t particle = 0U; particle < coarse.size() / 2U;
           ++particle) {
        if (accepted && !(*accepted)[particle]) continue;
        const std::size_t i = 2U * particle + component;
        if (!std::isfinite(coarse[i]) || !std::isfinite(fine[i]))
        throw std::runtime_error(
            "non-finite gathered field in Richardson comparison");
        const double delta = static_cast<double>(fine[i]) - coarse[i];
        difference2 += static_cast<long double>(delta) * delta;
        fine2_norm += static_cast<long double>(fine[i]) * fine[i];
        max_difference = std::max(max_difference, std::fabs(delta));
        max_fine = std::max(max_fine,
                            std::fabs(static_cast<double>(fine[i])));
        ++count;
      }
      const std::size_t slot = component_offset + component;
      if (count == 0U) continue;
      compared_any_target = true;
      const double absolute_rms = std::sqrt(static_cast<double>(
          difference2 / static_cast<long double>(count)));
      const double fine_rms = std::sqrt(static_cast<double>(
          fine2_norm / static_cast<long double>(count)));
      result.absolute_difference_rms_components[slot] = absolute_rms;
      result.absolute_difference_linf_components[slot] = max_difference;
      result.observed_difference_rms_components[slot] =
          absolute_rms / std::max(fine_rms, field_floor > 0.0
                                                ? field_floor
                                                : std::numeric_limits<double>::min());
      result.observed_difference_linf_components[slot] =
          max_difference / std::max(max_fine, field_floor > 0.0
                                                  ? field_floor
                                                  : std::numeric_limits<double>::min());
    }
  };
  accumulate(coarse1, fine1, accepted1, 0U);
  accumulate(coarse2, fine2, accepted2, 2U);
  result.absolute_difference_rms = *std::max_element(
      result.absolute_difference_rms_components.begin(),
      result.absolute_difference_rms_components.end());
  result.observed_difference_rms = *std::max_element(
      result.observed_difference_rms_components.begin(),
      result.observed_difference_rms_components.end());
  result.observed_difference_linf = *std::max_element(
      result.observed_difference_linf_components.begin(),
      result.observed_difference_linf_components.end());
  const double ratio_to_order = std::pow(static_cast<double>(ratio), order);
  const double divisor = ratio_to_order - 1.0;
  if (!std::isfinite(ratio_to_order) || !(divisor > 0.0)) return result;
  result.coarse_error_rms = ratio_to_order *
      result.observed_difference_rms / divisor;
  result.coarse_error_linf = ratio_to_order *
      result.observed_difference_linf / divisor;
  result.fine_error_rms = result.observed_difference_rms / divisor;
  result.fine_error_linf = result.observed_difference_linf / divisor;
  result.coarse_error = std::max(result.coarse_error_rms,
                                 result.coarse_error_linf);
  result.fine_error = std::max(result.fine_error_rms,
                               result.fine_error_linf);
  result.order = order;
  result.ratio = static_cast<double>(ratio);
  result.reliable = std::isfinite(result.absolute_difference_rms) &&
                    compared_any_target &&
                    std::isfinite(result.coarse_error_rms) &&
                    std::isfinite(result.coarse_error_linf) &&
                    std::isfinite(result.fine_error_rms) &&
                    std::isfinite(result.fine_error_linf) &&
                    std::isfinite(result.coarse_error) &&
                    std::isfinite(result.fine_error);
  return result;
}

SymmetricVectorFieldSensitivity estimate_symmetric_vector_field_sensitivity(
    const std::vector<float>& coarse1,
    const std::vector<float>& coarse2,
    const std::vector<float>& fine1,
    const std::vector<float>& fine2,
    double field_floor,
    const std::vector<unsigned char>* accepted1,
    const std::vector<unsigned char>* accepted2) {
  if (coarse1.size() != fine1.size() || coarse2.size() != fine2.size() ||
      (coarse1.size() & 1U) != 0U || (coarse2.size() & 1U) != 0U ||
      (accepted1 && accepted1->size() != coarse1.size() / 2U) ||
      (accepted2 && accepted2->size() != coarse2.size() / 2U) ||
      !std::isfinite(field_floor) || !(field_floor > 0.0))
    throw std::invalid_argument(
        "invalid symmetric vector field-sensitivity inputs");

  SymmetricVectorFieldSensitivity result;
  const auto bounded_ratio = [](long double numerator,
                                long double denominator) {
    if (!std::isfinite(numerator) || !std::isfinite(denominator) ||
        numerator < 0.0L || !(denominator > 0.0L))
      throw std::runtime_error(
          "non-finite symmetric vector field-sensitivity norm");
    const long double ratio = numerator / denominator;
    if (!std::isfinite(ratio) || ratio < 0.0L)
      throw std::runtime_error(
          "non-finite symmetric vector field-sensitivity ratio");
    // The mathematical ratio is at most two by the triangle inequality.
    // Clamp only the final roundoff ulp so the public contract remains exact.
    const long double tolerance = 64.0L *
        std::numeric_limits<long double>::epsilon();
    if (ratio > 2.0L + tolerance)
      throw std::runtime_error(
          "symmetric vector field sensitivity violated its norm bound");
    return static_cast<double>(std::min(2.0L, ratio));
  };
  const auto accumulate = [&result, field_floor, &bounded_ratio](
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>* accepted,
      std::size_t beam) {
    long double coarse_squared_sum = 0.0L;
    long double fine_squared_sum = 0.0L;
    long double difference_squared_sum = 0.0L;
    long double coarse_linf = 0.0L;
    long double fine_linf = 0.0L;
    long double difference_linf = 0.0L;
    std::size_t count = 0U;
    for (std::size_t particle = 0U; particle < coarse.size() / 2U;
         ++particle) {
      if (accepted && !(*accepted)[particle]) continue;
      const std::size_t index = 2U * particle;
      const float coarse_x_value = coarse[index];
      const float coarse_y_value = coarse[index + 1U];
      const float fine_x_value = fine[index];
      const float fine_y_value = fine[index + 1U];
      if (!std::isfinite(coarse_x_value) ||
          !std::isfinite(coarse_y_value) ||
          !std::isfinite(fine_x_value) ||
          !std::isfinite(fine_y_value))
        throw std::runtime_error(
            "non-finite gathered field in symmetric vector comparison");
      const long double coarse_x = coarse_x_value;
      const long double coarse_y = coarse_y_value;
      const long double fine_x = fine_x_value;
      const long double fine_y = fine_y_value;
      const long double difference_x = fine_x - coarse_x;
      const long double difference_y = fine_y - coarse_y;
      const long double coarse_squared =
          coarse_x * coarse_x + coarse_y * coarse_y;
      const long double fine_squared =
          fine_x * fine_x + fine_y * fine_y;
      const long double difference_squared =
          difference_x * difference_x + difference_y * difference_y;
      if (!std::isfinite(coarse_squared) ||
          !std::isfinite(fine_squared) ||
          !std::isfinite(difference_squared))
        throw std::runtime_error(
            "non-finite squared norm in symmetric vector comparison");
      coarse_squared_sum += coarse_squared;
      fine_squared_sum += fine_squared;
      difference_squared_sum += difference_squared;
      coarse_linf = std::max(coarse_linf, std::sqrt(coarse_squared));
      fine_linf = std::max(fine_linf, std::sqrt(fine_squared));
      difference_linf = std::max(
          difference_linf, std::sqrt(difference_squared));
      ++count;
    }
    result.accepted_targets[beam] = count;
    if (count == 0U) return;
    const long double inverse_count =
        1.0L / static_cast<long double>(count);
    const long double coarse_rms = std::sqrt(
        coarse_squared_sum * inverse_count);
    const long double fine_rms = std::sqrt(
        fine_squared_sum * inverse_count);
    const long double difference_rms = std::sqrt(
        difference_squared_sum * inverse_count);
    const long double floor = static_cast<long double>(field_floor);
    result.rms_ratios[beam] = bounded_ratio(
        difference_rms, std::max(floor, std::max(coarse_rms, fine_rms)));
    result.linf_ratios[beam] = bounded_ratio(
        difference_linf, std::max(floor, std::max(coarse_linf, fine_linf)));
  };

  accumulate(coarse1, fine1, accepted1, 0U);
  accumulate(coarse2, fine2, accepted2, 1U);
  result.aggregate = std::max(
      *std::max_element(result.rms_ratios.begin(), result.rms_ratios.end()),
      *std::max_element(result.linf_ratios.begin(), result.linf_ratios.end()));
  result.reliable = result.accepted_targets[0] != 0U ||
                    result.accepted_targets[1] != 0U;
  return result;
}

VectorFieldErrorStatistics estimate_vector_field_error_statistics(
    const std::vector<float>& coarse1,
    const std::vector<float>& coarse2,
    const std::vector<float>& fine1,
    const std::vector<float>& fine2,
    double quantile_probability,
    double error_target,
    const std::vector<unsigned char>* accepted1,
    const std::vector<unsigned char>* accepted2) {
  if (coarse1.size() != fine1.size() || coarse2.size() != fine2.size() ||
      (coarse1.size() & 1U) != 0U || (coarse2.size() & 1U) != 0U ||
      (accepted1 && accepted1->size() != coarse1.size() / 2U) ||
      (accepted2 && accepted2->size() != coarse2.size() / 2U) ||
      !std::isfinite(quantile_probability) ||
      quantile_probability < 0.95 || quantile_probability >= 1.0 ||
      !std::isfinite(error_target) || error_target < 0.0 ||
      error_target >= 1.0)
    throw std::invalid_argument(
        "invalid vector RMS/quantile field-error inputs");

  VectorFieldErrorStatistics result;
  result.quantile_probability = quantile_probability;
  const auto accumulate = [&result, quantile_probability, error_target](
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>* accepted,
      std::size_t beam) {
    result.accepted_target_hash[beam] = target_identity_hash(
        coarse.size() / 2U, accepted);
    std::vector<double> ratios;
    ratios.reserve(coarse.size() / 2U);
    long double ratio_squared_sum = 0.0L;
    double maximum = 0.0;
    std::size_t above = 0U;
    for (std::size_t particle = 0U; particle < coarse.size() / 2U;
         ++particle) {
      if (accepted && !(*accepted)[particle]) continue;
      const std::size_t index = 2U * particle;
      const long double coarse_x = coarse[index];
      const long double coarse_y = coarse[index + 1U];
      const long double fine_x = fine[index];
      const long double fine_y = fine[index + 1U];
      if (!std::isfinite(coarse_x) || !std::isfinite(coarse_y) ||
          !std::isfinite(fine_x) || !std::isfinite(fine_y))
        throw std::runtime_error(
            "non-finite gathered field in vector RMS/quantile comparison");
      const long double coarse_norm = std::hypotl(coarse_x, coarse_y);
      const long double fine_norm = std::hypotl(fine_x, fine_y);
      const long double difference_norm =
          std::hypotl(fine_x - coarse_x, fine_y - coarse_y);
      const long double scale = std::max(coarse_norm, fine_norm);
      const long double raw_ratio = scale == 0.0L
          ? 0.0L : difference_norm / scale;
      const long double tolerance = 64.0L *
          std::numeric_limits<long double>::epsilon();
      if (!std::isfinite(raw_ratio) || raw_ratio < 0.0L ||
          raw_ratio > 2.0L + tolerance)
        throw std::runtime_error(
            "vector field-error ratio violated its exact [0,2] bound");
      const double ratio = static_cast<double>(
          std::min(2.0L, raw_ratio));
      ratios.push_back(ratio);
      ratio_squared_sum += static_cast<long double>(ratio) * ratio;
      maximum = std::max(maximum, ratio);
      if (ratio > error_target) ++above;
    }
    const std::size_t count = ratios.size();
    result.accepted_targets[beam] = count;
    result.above_target[beam] = above;
    if (count == 0U) return;
    result.rms[beam] = std::sqrt(static_cast<double>(
        ratio_squared_sum / static_cast<long double>(count)));
    const long double rank_real = std::ceil(
        static_cast<long double>(quantile_probability) *
        static_cast<long double>(count));
    const std::size_t rank = std::max<std::size_t>(
        1U, static_cast<std::size_t>(rank_real));
    std::nth_element(ratios.begin(), ratios.begin() + (rank - 1U),
                     ratios.end());
    result.quantile[beam] = ratios[rank - 1U];
    result.maximum[beam] = maximum;
  };

  accumulate(coarse1, fine1, accepted1, 0U);
  accumulate(coarse2, fine2, accepted2, 1U);
  if (result.accepted_targets[0] >
      std::numeric_limits<std::size_t>::max() -
          result.accepted_targets[1])
    throw std::overflow_error("vector field-error target count overflowed");
  if (result.above_target[0] >
      std::numeric_limits<std::size_t>::max() - result.above_target[1])
    throw std::overflow_error(
        "vector field-error exceedance count overflowed");
  result.total_accepted_targets = result.accepted_targets[0] +
                                  result.accepted_targets[1];
  result.total_above_target = result.above_target[0] +
                              result.above_target[1];
  result.aggregate_rms = std::max(result.rms[0], result.rms[1]);
  result.aggregate_quantile = std::max(
      result.quantile[0], result.quantile[1]);
  result.aggregate_maximum = std::max(
      result.maximum[0], result.maximum[1]);
  result.above_target_exposure = result.total_accepted_targets == 0U
      ? 0.0
      : static_cast<double>(result.total_above_target) /
            static_cast<double>(result.total_accepted_targets);
  result.reliable = result.total_accepted_targets != 0U &&
                    std::isfinite(result.aggregate_rms) &&
                    std::isfinite(result.aggregate_quantile) &&
                    std::isfinite(result.aggregate_maximum) &&
                    std::isfinite(result.above_target_exposure);
  return result;
}

VectorFieldErrorStatistics estimate_mixed_vector_field_error_statistics_v2(
    const std::vector<float>& coarse1,
    const std::vector<float>& coarse2,
    const std::vector<float>& fine1,
    const std::vector<float>& fine2,
    double quantile_probability,
    double error_target,
    double reference_field,
    double absolute_floor_fraction,
    double catastrophe_relative_limit,
    double catastrophe_absolute_fraction,
    const std::vector<unsigned char>* accepted1,
    const std::vector<unsigned char>* accepted2,
    const std::vector<std::uint64_t>* target_ids1,
    const std::vector<std::uint64_t>* target_ids2) {
  const double absolute_floor = reference_field * absolute_floor_fraction;
  const double catastrophe_absolute_threshold =
      reference_field * catastrophe_absolute_fraction;
  if (coarse1.size() != fine1.size() || coarse2.size() != fine2.size() ||
      (coarse1.size() & 1U) != 0U || (coarse2.size() & 1U) != 0U ||
      (accepted1 && accepted1->size() != coarse1.size() / 2U) ||
      (accepted2 && accepted2->size() != coarse2.size() / 2U) ||
      (target_ids1 && target_ids1->size() != coarse1.size() / 2U) ||
      (target_ids2 && target_ids2->size() != coarse2.size() / 2U) ||
      !std::isfinite(quantile_probability) ||
      quantile_probability < 0.95 || quantile_probability >= 1.0 ||
      !std::isfinite(error_target) || error_target < 0.0 ||
      error_target >= 1.0 || !std::isfinite(reference_field) ||
      !(reference_field > 0.0) ||
      !std::isfinite(absolute_floor_fraction) ||
      !(absolute_floor_fraction > 0.0) ||
      absolute_floor_fraction > 1.0 ||
      !std::isfinite(catastrophe_relative_limit) ||
      !(catastrophe_relative_limit > 0.0) ||
      catastrophe_relative_limit > 2.0 ||
      !std::isfinite(catastrophe_absolute_fraction) ||
      !(catastrophe_absolute_fraction > 0.0) ||
      catastrophe_absolute_fraction > 2.0 ||
      !std::isfinite(absolute_floor) || !(absolute_floor > 0.0) ||
      !std::isfinite(catastrophe_absolute_threshold) ||
      !(catastrophe_absolute_threshold > 0.0))
    throw std::invalid_argument(
        "invalid mixed absolute/relative vector field-error inputs");

  VectorFieldErrorStatistics result;
  result.policy_version = 2U;
  result.quantile_probability = quantile_probability;
  result.reference_field = reference_field;
  result.absolute_floor_fraction = absolute_floor_fraction;
  result.absolute_floor = absolute_floor;
  result.catastrophe_relative_limit = catastrophe_relative_limit;
  result.catastrophe_absolute_fraction =
      catastrophe_absolute_fraction;
  result.catastrophe_absolute_threshold =
      catastrophe_absolute_threshold;

  const auto accumulate = [
      &result, quantile_probability, error_target,
      absolute_floor, catastrophe_relative_limit,
      catastrophe_absolute_threshold](
      const std::vector<float>& coarse,
      const std::vector<float>& fine,
      const std::vector<unsigned char>* accepted,
      const std::vector<std::uint64_t>* target_ids,
      std::size_t beam) {
    result.accepted_target_hash[beam] = target_identity_hash(
        coarse.size() / 2U, accepted, target_ids);
    result.accepted_target_ids_bound[beam] = target_ids != NULL;
    std::vector<double> ratios;
    std::vector<double> absolute_differences;
    ratios.reserve(coarse.size() / 2U);
    absolute_differences.reserve(coarse.size() / 2U);
    long double ratio_squared_sum = 0.0L;
    long double absolute_squared_sum = 0.0L;
    double maximum = 0.0;
    double absolute_maximum = 0.0;
    double significant_unregularized_maximum = 0.0;
    std::size_t above = 0U;
    std::size_t catastrophes = 0U;
    for (std::size_t particle = 0U; particle < coarse.size() / 2U;
         ++particle) {
      if (accepted && !(*accepted)[particle]) continue;
      const std::size_t index = 2U * particle;
      const long double coarse_x = coarse[index];
      const long double coarse_y = coarse[index + 1U];
      const long double fine_x = fine[index];
      const long double fine_y = fine[index + 1U];
      if (!std::isfinite(coarse_x) || !std::isfinite(coarse_y) ||
          !std::isfinite(fine_x) || !std::isfinite(fine_y))
        throw std::runtime_error(
            "non-finite gathered field in mixed vector comparison");
      const long double coarse_norm = std::hypotl(coarse_x, coarse_y);
      const long double fine_norm = std::hypotl(fine_x, fine_y);
      const long double difference_norm =
          std::hypotl(fine_x - coarse_x, fine_y - coarse_y);
      const long double unregularized_scale =
          std::max(coarse_norm, fine_norm);
      const long double mixed_scale = std::max(
          unregularized_scale,
          static_cast<long double>(absolute_floor));
      const long double raw_ratio = difference_norm / mixed_scale;
      const long double unregularized_ratio =
          unregularized_scale == 0.0L
              ? 0.0L : difference_norm / unregularized_scale;
      const long double tolerance = 64.0L *
          std::numeric_limits<long double>::epsilon();
      if (!std::isfinite(raw_ratio) || raw_ratio < 0.0L ||
          raw_ratio > 2.0L + tolerance ||
          !std::isfinite(unregularized_ratio) ||
          unregularized_ratio < 0.0L ||
          unregularized_ratio > 2.0L + tolerance ||
          !std::isfinite(difference_norm) || difference_norm < 0.0L)
        throw std::runtime_error(
            "mixed vector field-error metric violated its exact bounds");
      const double ratio = static_cast<double>(
          std::min(2.0L, raw_ratio));
      const double absolute = static_cast<double>(difference_norm);
      const double unregularized = static_cast<double>(
          std::min(2.0L, unregularized_ratio));
      ratios.push_back(ratio);
      absolute_differences.push_back(absolute);
      ratio_squared_sum += static_cast<long double>(ratio) * ratio;
      absolute_squared_sum += difference_norm * difference_norm;
      maximum = std::max(maximum, ratio);
      absolute_maximum = std::max(absolute_maximum, absolute);
      if (absolute > catastrophe_absolute_threshold) {
        significant_unregularized_maximum = std::max(
            significant_unregularized_maximum, unregularized);
        if (unregularized > catastrophe_relative_limit) ++catastrophes;
      }
      if (ratio > error_target) ++above;
    }
    const std::size_t count = ratios.size();
    result.accepted_targets[beam] = count;
    result.above_target[beam] = above;
    result.catastrophe_targets[beam] = catastrophes;
    if (count == 0U) return;
    const long double inverse_count =
        1.0L / static_cast<long double>(count);
    result.rms[beam] = std::sqrt(static_cast<double>(
        ratio_squared_sum * inverse_count));
    result.absolute_rms[beam] = std::sqrt(static_cast<double>(
        absolute_squared_sum * inverse_count));
    const long double rank_real = std::ceil(
        static_cast<long double>(quantile_probability) *
        static_cast<long double>(count));
    const std::size_t rank = std::max<std::size_t>(
        1U, static_cast<std::size_t>(rank_real));
    std::nth_element(ratios.begin(), ratios.begin() + (rank - 1U),
                     ratios.end());
    std::nth_element(absolute_differences.begin(),
                     absolute_differences.begin() + (rank - 1U),
                     absolute_differences.end());
    result.quantile[beam] = ratios[rank - 1U];
    result.absolute_quantile[beam] = absolute_differences[rank - 1U];
    result.maximum[beam] = maximum;
    result.absolute_maximum[beam] = absolute_maximum;
    result.significant_unregularized_maximum[beam] =
        significant_unregularized_maximum;
  };

  accumulate(coarse1, fine1, accepted1, target_ids1, 0U);
  accumulate(coarse2, fine2, accepted2, target_ids2, 1U);
  if (result.accepted_targets[0] >
      std::numeric_limits<std::size_t>::max() -
          result.accepted_targets[1] ||
      result.above_target[0] >
      std::numeric_limits<std::size_t>::max() - result.above_target[1] ||
      result.catastrophe_targets[0] >
      std::numeric_limits<std::size_t>::max() -
          result.catastrophe_targets[1])
    throw std::overflow_error(
        "mixed vector field-error target count overflowed");
  result.total_accepted_targets = result.accepted_targets[0] +
                                  result.accepted_targets[1];
  result.total_above_target = result.above_target[0] +
                              result.above_target[1];
  result.total_catastrophe_targets = result.catastrophe_targets[0] +
                                     result.catastrophe_targets[1];
  result.aggregate_rms = std::max(result.rms[0], result.rms[1]);
  result.aggregate_quantile = std::max(
      result.quantile[0], result.quantile[1]);
  result.aggregate_maximum = std::max(
      result.maximum[0], result.maximum[1]);
  result.aggregate_absolute_rms = std::max(
      result.absolute_rms[0], result.absolute_rms[1]);
  result.aggregate_absolute_quantile = std::max(
      result.absolute_quantile[0], result.absolute_quantile[1]);
  result.aggregate_absolute_maximum = std::max(
      result.absolute_maximum[0], result.absolute_maximum[1]);
  result.aggregate_significant_unregularized_maximum = std::max(
      result.significant_unregularized_maximum[0],
      result.significant_unregularized_maximum[1]);
  result.above_target_exposure = result.total_accepted_targets == 0U
      ? 0.0
      : static_cast<double>(result.total_above_target) /
            static_cast<double>(result.total_accepted_targets);
  for (std::size_t beam = 0U; beam < 2U; ++beam)
    result.above_target_exposure_by_beam[beam] =
        result.accepted_targets[beam] == 0U
            ? 0.0
            : static_cast<double>(result.above_target[beam]) /
                  static_cast<double>(result.accepted_targets[beam]);
  result.maximum_beam_above_target_exposure = std::max(
      result.above_target_exposure_by_beam[0],
      result.above_target_exposure_by_beam[1]);
  result.catastrophe_exposure = result.total_accepted_targets == 0U
      ? 0.0
      : static_cast<double>(result.total_catastrophe_targets) /
            static_cast<double>(result.total_accepted_targets);
  result.reliable = result.total_accepted_targets != 0U &&
                    std::isfinite(result.aggregate_rms) &&
                    std::isfinite(result.aggregate_quantile) &&
                    std::isfinite(result.aggregate_maximum) &&
                    std::isfinite(result.aggregate_absolute_rms) &&
                    std::isfinite(result.aggregate_absolute_quantile) &&
                    std::isfinite(result.aggregate_absolute_maximum) &&
                    std::isfinite(
                        result.aggregate_significant_unregularized_maximum) &&
                    std::isfinite(result.above_target_exposure) &&
                    std::isfinite(
                        result.maximum_beam_above_target_exposure) &&
                    std::isfinite(result.catastrophe_exposure);
  return result;
}

std::uint64_t accepted_target_identity_hash(
    const std::vector<unsigned char>& accepted_targets) {
  return target_identity_hash(accepted_targets.size(), &accepted_targets);
}

VectorFieldRichardsonEstimate estimate_vector_field_richardson_error(
    const VectorFieldErrorStatistics& earlier,
    const VectorFieldErrorStatistics& later,
    const VectorFieldErrorStatistics& current,
    std::size_t refinement_ratio,
    double minimum_observed_order) {
  if (!earlier.reliable || !later.reliable || !current.reliable ||
      (earlier.policy_version != 1U && earlier.policy_version != 2U) ||
      later.policy_version != earlier.policy_version ||
      current.policy_version != earlier.policy_version ||
      refinement_ratio < 2U || !power_of_two(refinement_ratio) ||
      !std::isfinite(minimum_observed_order) ||
      !(minimum_observed_order > 0.0) || minimum_observed_order > 16.0 ||
      !std::isfinite(earlier.quantile_probability) ||
      earlier.quantile_probability < 0.95 ||
      earlier.quantile_probability >= 1.0 ||
      later.quantile_probability != earlier.quantile_probability ||
      current.quantile_probability != earlier.quantile_probability ||
      earlier.accepted_targets != later.accepted_targets ||
      earlier.accepted_targets != current.accepted_targets ||
      earlier.accepted_target_hash != later.accepted_target_hash ||
      earlier.accepted_target_hash != current.accepted_target_hash ||
      earlier.accepted_target_hash[0] == 0U ||
      earlier.accepted_target_hash[1] == 0U ||
      (earlier.policy_version == 2U &&
       (!earlier.accepted_target_ids_bound[0] ||
        !earlier.accepted_target_ids_bound[1] ||
        !later.accepted_target_ids_bound[0] ||
        !later.accepted_target_ids_bound[1] ||
        !current.accepted_target_ids_bound[0] ||
        !current.accepted_target_ids_bound[1] ||
        !(earlier.reference_field > 0.0) ||
        !std::isfinite(earlier.reference_field) ||
        later.reference_field != earlier.reference_field ||
        current.reference_field != earlier.reference_field ||
        later.absolute_floor != earlier.absolute_floor ||
        current.absolute_floor != earlier.absolute_floor ||
        later.catastrophe_absolute_threshold !=
            earlier.catastrophe_absolute_threshold ||
        current.catastrophe_absolute_threshold !=
            earlier.catastrophe_absolute_threshold)))
    throw std::invalid_argument(
        "invalid vector three-level Richardson inputs");

  VectorFieldRichardsonEstimate result;
  const long double tail_probability =
      1.0L - static_cast<long double>(earlier.quantile_probability);
  const long double minimum_targets_real = std::ceil(1.0L /
                                                      tail_probability);
  if (!std::isfinite(minimum_targets_real) || minimum_targets_real < 1.0L ||
      minimum_targets_real > static_cast<long double>(
          std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error(
        "vector empirical-quantile support requirement overflowed");
  result.minimum_quantile_targets =
      static_cast<std::size_t>(minimum_targets_real);
  result.minimum_accepted_targets =
      std::numeric_limits<std::size_t>::max();
  result.quantile_support_met = true;
  result.asymptotic = true;
  result.order_from_absolute_differences = earlier.policy_version == 2U;
  result.roundoff_limited = false;
  result.plausible_order = true;
  result.observed_order = result.order_from_absolute_differences
      ? 8.0 : 16.0;

  const double ratio = static_cast<double>(refinement_ratio);
  const double absolute_roundoff_floor =
      result.order_from_absolute_differences
          ? 64.0 * static_cast<double>(
                std::numeric_limits<float>::epsilon()) *
                earlier.reference_field
          : 0.0;
  const auto component_order = [
      ratio, absolute_roundoff_floor, &result](
      double first, double second) {
    if (!std::isfinite(first) || !std::isfinite(second) || first < 0.0 ||
        second < 0.0 ||
        (!result.order_from_absolute_differences &&
         (first > 2.0 || second > 2.0)))
      return -1.0;
    if (result.order_from_absolute_differences &&
        (first <= absolute_roundoff_floor ||
         second <= absolute_roundoff_floor)) {
      // A zero/roundoff plateau is useful diagnostic information but is not
      // evidence of arbitrarily high convergence order.
      result.roundoff_limited = true;
      result.plausible_order = false;
      return 0.0;
    }
    if (first == 0.0 && second == 0.0) return 16.0;
    if (!(first > 0.0) || !(second < first)) {
      result.plausible_order = false;
      return 0.0;
    }
    if (second == 0.0) return 16.0;
    const double order = std::log(first / second) / std::log(ratio);
    if (!std::isfinite(order) || !(order > 0.0)) {
      result.plausible_order = false;
      return 0.0;
    }
    if (result.order_from_absolute_differences && order > 8.0) {
      result.plausible_order = false;
      return 0.0;
    }
    return std::min(16.0, order);
  };
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    const std::size_t count = current.accepted_targets[beam];
    if (count == 0U)
      throw std::invalid_argument(
          "vector Richardson estimate is missing one colliding beam");
    result.minimum_accepted_targets = std::min(
        result.minimum_accepted_targets, count);
    if (count < result.minimum_quantile_targets)
      result.quantile_support_met = false;
    const double orders[2] = {
        component_order(
            result.order_from_absolute_differences
                ? earlier.absolute_rms[beam] : earlier.rms[beam],
            result.order_from_absolute_differences
                ? later.absolute_rms[beam] : later.rms[beam]),
        component_order(
            result.order_from_absolute_differences
                ? earlier.absolute_quantile[beam]
                : earlier.quantile[beam],
            result.order_from_absolute_differences
                ? later.absolute_quantile[beam]
                : later.quantile[beam])};
    for (double order : orders) {
      if (order < 0.0)
        throw std::invalid_argument(
            "non-finite or out-of-range vector Richardson component");
      result.observed_order = std::min(result.observed_order, order);
      if (!(order >= minimum_observed_order)) result.asymptotic = false;
    }
    const double current_metrics[2] = {
        current.rms[beam], current.quantile[beam]};
    for (double value : current_metrics)
      if (!std::isfinite(value) || value < 0.0 || value > 2.0)
        throw std::invalid_argument(
            "non-finite or out-of-range current vector metric");
  }
  // A noncontracting sequence is never certifiable.  Retain a finite,
  // conservative diagnostic by scaling the current difference with the
  // preregistered minimum order; acceptance still requires asymptotic=true.
  const double scaling_order = result.asymptotic
      ? result.observed_order : minimum_observed_order;
  const double ratio_to_order = std::pow(ratio, scaling_order);
  const double divisor = ratio_to_order - 1.0;
  if (!std::isfinite(divisor) || !(divisor > 0.0))
    throw std::runtime_error(
        "vector Richardson scaling divisor is invalid");
  for (std::size_t beam = 0U; beam < 2U; ++beam) {
    result.fine_rms_error[beam] = current.rms[beam] / divisor;
    result.fine_quantile_error[beam] =
        current.quantile[beam] / divisor;
    result.coarse_rms_error[beam] =
        ratio_to_order * result.fine_rms_error[beam];
    result.coarse_quantile_error[beam] =
        ratio_to_order * result.fine_quantile_error[beam];
    result.fine_error = std::max(
        result.fine_error,
        std::max(result.fine_rms_error[beam],
                 result.fine_quantile_error[beam]));
    result.coarse_error = std::max(
        result.coarse_error,
        std::max(result.coarse_rms_error[beam],
                 result.coarse_quantile_error[beam]));
  }
  result.reliable = std::isfinite(result.coarse_error) &&
                    std::isfinite(result.fine_error);
  return result;
}

double kish_effective_sample_size(long double sum_weight,
                                  long double sum_weight_squared) {
  if (!std::isfinite(sum_weight) || !std::isfinite(sum_weight_squared) ||
      sum_weight < 0.0L || sum_weight_squared < 0.0L)
    throw std::invalid_argument("invalid Kish effective-sample inputs");
  if (!(sum_weight > 0.0L) || !(sum_weight_squared > 0.0L)) return 0.0;
  const long double value = sum_weight * sum_weight / sum_weight_squared;
  if (!std::isfinite(value) || value < 0.0L)
    throw std::overflow_error("Kish effective-sample result is invalid");
  return static_cast<double>(value);
}

NestedPatchBox select_nested_patch(
    const std::vector<double>& rho1,
    const std::vector<double>& rho2,
    std::size_t nx,
    std::size_t ny,
    double overlap_fraction,
    double curvature_threshold,
    double rho_floor,
    unsigned int smoothing_passes,
    std::size_t buffer_cells,
    std::size_t transition_cells,
    std::size_t blocking_factor) {
  if (nx < 4U || ny < 4U || rho1.size() != checked_product(nx, ny, "rho") ||
      rho2.size() != rho1.size() || !std::isfinite(overlap_fraction) ||
      overlap_fraction < 0.0 || overlap_fraction > 1.0 ||
      !std::isfinite(curvature_threshold) || curvature_threshold < 0.0 ||
      !std::isfinite(rho_floor) || rho_floor < 0.0 ||
      smoothing_passes > 2U ||
      (blocking_factor != 0U && !power_of_two(blocking_factor)))
    throw std::invalid_argument("invalid nested-patch indicator inputs");
  std::vector<double> charge(rho1.size(), 0.0);
  std::vector<double> overlap(rho1.size(), 0.0);
  for (std::size_t i = 0; i < rho1.size(); ++i) {
    if (!std::isfinite(rho1[i]) || !std::isfinite(rho2[i]))
      throw std::runtime_error("non-finite charge in nested-patch indicator");
    const double a = std::fabs(rho1[i]);
    const double b = std::fabs(rho2[i]);
    charge[i] = a + b;
    overlap[i] = std::sqrt(a * b);
  }
  charge = smooth_binomial(charge, nx, ny, smoothing_passes);
  overlap = smooth_binomial(overlap, nx, ny, smoothing_passes);
  const auto maximum = std::max_element(overlap.begin(), overlap.end());
  const double max_overlap = maximum == overlap.end() ? 0.0 : *maximum;
  std::size_t seed = maximum == overlap.end()
      ? 0U : static_cast<std::size_t>(maximum - overlap.begin());
  const double max_charge = *std::max_element(charge.begin(), charge.end());
  if (!(max_overlap > rho_floor)) {
    const auto charge_maximum = std::max_element(charge.begin(), charge.end());
    if (charge_maximum == charge.end() || !(*charge_maximum > rho_floor))
      return NestedPatchBox();
    seed = static_cast<std::size_t>(charge_maximum - charge.begin());
  }
  std::vector<unsigned char> flagged(rho1.size(), 0U);
  const double overlap_cut = overlap_fraction * max_overlap;
  const double charge_cut = overlap_fraction * max_charge;
  for (std::size_t x = 1U; x + 1U < nx; ++x) {
    for (std::size_t y = 1U; y + 1U < ny; ++y) {
      const std::size_t i = x * ny + y;
      const double center = charge[i];
      const double curvature =
          std::fabs(charge[(x + 1U) * ny + y] - 2.0 * center +
                    charge[(x - 1U) * ny + y]) +
          std::fabs(charge[x * ny + y + 1U] - 2.0 * center +
                    charge[x * ny + y - 1U]);
      const double eta = curvature / (center + rho_floor +
          std::numeric_limits<double>::min());
      const bool overlap_selected = max_overlap > rho_floor
          ? overlap[i] >= overlap_cut : charge[i] >= charge_cut;
      const bool curvature_selected = curvature_threshold > 0.0 &&
          eta >= curvature_threshold && charge[i] >= charge_cut;
      flagged[i] = overlap_selected || curvature_selected ? 1U : 0U;
    }
  }
  flagged[seed] = 1U;
  std::deque<std::size_t> queue;
  std::vector<unsigned char> visited(rho1.size(), 0U);
  queue.push_back(seed);
  visited[seed] = 1U;
  std::size_t min_x = seed / ny;
  std::size_t max_x = min_x;
  std::size_t min_y = seed % ny;
  std::size_t max_y = min_y;
  while (!queue.empty()) {
    const std::size_t i = queue.front();
    queue.pop_front();
    const std::size_t x = i / ny;
    const std::size_t y = i % ny;
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
    const long neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto& delta : neighbors) {
      const long xx = static_cast<long>(x) + delta[0];
      const long yy = static_cast<long>(y) + delta[1];
      if (xx < 0 || yy < 0 || xx >= static_cast<long>(nx) ||
          yy >= static_cast<long>(ny)) continue;
      const std::size_t next = static_cast<std::size_t>(xx) * ny +
                               static_cast<std::size_t>(yy);
      if (flagged[next] && !visited[next]) {
        visited[next] = 1U;
        queue.push_back(next);
      }
    }
  }
  const std::size_t total_buffer = checked_add(
      checked_add(buffer_cells, transition_cells, "patch buffer"), 2U,
      "patch stencil buffer");
  std::size_t begin_x = min_x > total_buffer ? min_x - total_buffer : 0U;
  std::size_t begin_y = min_y > total_buffer ? min_y - total_buffer : 0U;
  std::size_t end_x = std::min(nx, checked_add(max_x + 1U, total_buffer,
                                               "patch end x"));
  std::size_t end_y = std::min(ny, checked_add(max_y + 1U, total_buffer,
                                               "patch end y"));
  const std::size_t block = blocking_factor == 0U ? 2U : blocking_factor;
  begin_x -= begin_x % block;
  begin_y -= begin_y % block;
  end_x = std::min(nx, checked_add(end_x, block - 1U,
                                   "blocked patch end x") / block * block);
  end_y = std::min(ny, checked_add(end_y, block - 1U,
                                   "blocked patch end y") / block * block);
  if ((end_x - begin_x) < 4U || (end_y - begin_y) < 4U)
    return NestedPatchBox();
  NestedPatchBox result;
  result.valid = true;
  result.begin_x = begin_x;
  result.end_x = end_x;
  result.begin_y = begin_y;
  result.end_y = end_y;
  return result;
}

FieldLevelGeometry make_patch_geometry(
    const FieldLevelGeometry& parent, const NestedPatchBox& patch) {
  if (!parent.valid() || !patch.valid || patch.end_x > parent.n_x ||
      patch.end_y > parent.n_y || patch.n_x() < 4U || patch.n_y() < 4U)
    throw std::invalid_argument("invalid parent/nested patch geometry");
  FieldLevelGeometry result(parent);
  result.origin_x += static_cast<CanonicalIndex>(patch.begin_x);
  result.origin_y += static_cast<CanonicalIndex>(patch.begin_y);
  result.n_x = patch.n_x();
  result.n_y = patch.n_y();
  if (!result.valid())
    throw std::invalid_argument("nested parent-level patch is invalid");
  return result;
}

FieldLevelGeometry refine_patch_geometry(
    const FieldLevelGeometry& patch, std::size_t ratio) {
  if (!patch.valid() || ratio < 2U || !power_of_two(ratio))
    throw std::invalid_argument("invalid nested refinement request");
  FieldLevelGeometry result;
  result.refinement_factor = checked_product(
      patch.refinement_factor, ratio, "nested refinement factor");
  result.global_n_x = checked_product(patch.global_n_x, ratio,
                                      "nested global n_x");
  result.global_n_y = checked_product(patch.global_n_y, ratio,
                                      "nested global n_y");
  result.origin_x = static_cast<CanonicalIndex>(checked_product(
      static_cast<std::size_t>(patch.origin_x), ratio,
      "nested origin x"));
  result.origin_y = static_cast<CanonicalIndex>(checked_product(
      static_cast<std::size_t>(patch.origin_y), ratio,
      "nested origin y"));
  result.n_x = checked_product(patch.n_x, ratio, "nested n_x");
  result.n_y = checked_product(patch.n_y, ratio, "nested n_y");
  result.delta_x = patch.delta_x / static_cast<float>(ratio);
  result.delta_y = patch.delta_y / static_cast<float>(ratio);
  result.inverse_delta_x = patch.inverse_delta_x * static_cast<float>(ratio);
  result.inverse_delta_y = patch.inverse_delta_y * static_cast<float>(ratio);
  result.offset_x = patch.offset_x * static_cast<float>(ratio);
  result.offset_y = patch.offset_y * static_cast<float>(ratio);
  if (!result.valid())
    throw std::invalid_argument("constructed nested fine geometry is invalid");
  return result;
}

double patch_transition_weight(float x, float y,
                               const FieldLevelGeometry& patch,
                               std::size_t transition_cells,
                               std::size_t stencil_guard) {
  if (!patch.valid() || !std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("invalid patch-transition request");
  const double gx = x * patch.inverse_delta_x + patch.offset_x -
                    static_cast<double>(patch.origin_x);
  const double gy = y * patch.inverse_delta_y + patch.offset_y -
                    static_cast<double>(patch.origin_y);
  if (!std::isfinite(gx) || !std::isfinite(gy))
    throw std::invalid_argument(
        "non-finite coordinate in patch-transition mapping");
  if (transition_cells == 0U)
    throw std::invalid_argument(
        "patch transition requires a positive C1 transition width");
  // gx/gy are measured from the physical lower patch face.  The opposite
  // face is at n, not n-1: actual field samples lie at 0.5 and n-0.5.
  // Multiplying independent lower/upper face ramps avoids the gradient kink
  // produced by smoothstep(min(distance-to-face)) on equal-distance
  // diagonals and at the centre of a narrow patch.  Each clamped smoothstep
  // has zero derivative at both endpoints, so their product is globally C1,
  // mirror symmetric, zero in the stencil guard, and one in a sufficiently
  // wide fine interior.
  const auto face_ramp = [transition_cells, stencil_guard](double distance) {
    const double t = std::max(0.0, std::min(
        1.0, (distance - static_cast<double>(stencil_guard)) /
                 static_cast<double>(transition_cells)));
    return t * t * (3.0 - 2.0 * t);
  };
  return face_ramp(gx) *
      face_ramp(static_cast<double>(patch.n_x) - gx) *
      face_ramp(gy) *
      face_ramp(static_cast<double>(patch.n_y) - gy);
}

double interpolate_level_potential(
    const double* phi, const FieldLevelGeometry& geometry,
    float x, float y) {
  if (!phi || !geometry.valid() || !std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("invalid potential-prolongation request");
  const double global_x = x * geometry.inverse_delta_x + geometry.offset_x -
                          0.5;
  const double global_y = y * geometry.inverse_delta_y + geometry.offset_y -
                          0.5;
  if (!std::isfinite(global_x) || !std::isfinite(global_y))
    throw std::invalid_argument(
        "non-finite coordinate in potential prolongation");
  const double local_x_coordinate =
      global_x - static_cast<double>(geometry.origin_x);
  const double local_y_coordinate =
      global_y - static_cast<double>(geometry.origin_y);
  if (local_x_coordinate < 0.0 || local_y_coordinate < 0.0 ||
      local_x_coordinate >= static_cast<double>(geometry.n_x - 1U) ||
      local_y_coordinate >= static_cast<double>(geometry.n_y - 1U))
    throw std::out_of_range(
        "coarse potential interpolation escaped its field window");
  // Derive the interpolation index and fraction from the already-bounded
  // local coordinates.  Casting a rounded global coordinate near LONG_MAX is
  // otherwise undefined even though its local index is small.
  const double base_x_floor = std::floor(local_x_coordinate);
  const double base_y_floor = std::floor(local_y_coordinate);
  const std::size_t ix = static_cast<std::size_t>(base_x_floor);
  const std::size_t iy = static_cast<std::size_t>(base_y_floor);
  const double wx = local_x_coordinate - base_x_floor;
  const double wy = local_y_coordinate - base_y_floor;
  const auto at = [phi, &geometry](std::size_t sample_x,
                                   std::size_t sample_y) {
    return phi[sample_x * geometry.n_y + sample_y];
  };
  return (1.0 - wx) * (1.0 - wy) * at(ix, iy) +
         wx * (1.0 - wy) * at(ix + 1U, iy) +
         (1.0 - wx) * wy * at(ix, iy + 1U) +
         wx * wy * at(ix + 1U, iy + 1U);
}

HarmonicSolveReport solve_harmonic_dirichlet(
    std::vector<double>& correction,
    std::size_t nx,
    std::size_t ny,
    double dx,
    double dy,
    double tolerance,
    std::size_t maximum_iterations) {
  if (nx < 4U || ny < 4U ||
      correction.size() != checked_product(nx, ny, "harmonic patch") ||
      !std::isfinite(dx) || !std::isfinite(dy) || dx <= 0.0 || dy <= 0.0 ||
      !std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0 ||
      maximum_iterations == 0U)
    throw std::invalid_argument("invalid harmonic Dirichlet solve contract");
  for (double value : correction)
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "non-finite value in harmonic Dirichlet boundary");
  const double ax = 1.0 / (dx * dx);
  const double ay = 1.0 / (dy * dy);

  // Remove the arbitrary additive potential gauge before solving.  Restoring
  // the mean afterward produces exactly the same field while preventing a
  // large coarse/fine gauge offset from weakening the residual test.
  long double boundary_sum = 0.0L;
  std::size_t boundary_count = 0U;
  for (std::size_t x = 0U; x < nx; ++x) {
    boundary_sum += correction[x * ny];
    boundary_sum += correction[x * ny + ny - 1U];
    boundary_count += 2U;
  }
  for (std::size_t y = 1U; y + 1U < ny; ++y) {
    boundary_sum += correction[y];
    boundary_sum += correction[(nx - 1U) * ny + y];
    boundary_count += 2U;
  }
  const double boundary_mean = static_cast<double>(
      boundary_sum / static_cast<long double>(boundary_count));
  double boundary_variation = 0.0;
  for (std::size_t x = 0U; x < nx; ++x) {
    correction[x * ny] -= boundary_mean;
    correction[x * ny + ny - 1U] -= boundary_mean;
    boundary_variation = std::max(
        boundary_variation, std::fabs(correction[x * ny]));
    boundary_variation = std::max(
        boundary_variation, std::fabs(correction[x * ny + ny - 1U]));
  }
  for (std::size_t y = 1U; y + 1U < ny; ++y) {
    correction[y] -= boundary_mean;
    correction[(nx - 1U) * ny + y] -= boundary_mean;
    boundary_variation = std::max(boundary_variation,
                                  std::fabs(correction[y]));
    boundary_variation = std::max(
        boundary_variation,
        std::fabs(correction[(nx - 1U) * ny + y]));
  }
  for (std::size_t x = 1U; x + 1U < nx; ++x)
    for (std::size_t y = 1U; y + 1U < ny; ++y)
      correction[x * ny + y] = 0.0;

  HarmonicSolveReport report;

  const auto measure_residual = [&]() {
    long double residual2 = 0.0L;
    long double gradient2 = 0.0L;
    std::size_t residual_count = 0U;
    std::size_t gradient_count = 0U;
    double maximum_residual = 0.0;
    double maximum_gradient = 0.0;
    for (std::size_t x = 1U; x + 1U < nx; ++x)
      for (std::size_t y = 1U; y + 1U < ny; ++y) {
        const double residual =
            ax * (correction[(x - 1U) * ny + y] -
                  2.0 * correction[x * ny + y] +
                  correction[(x + 1U) * ny + y]) +
            ay * (correction[x * ny + y - 1U] -
                  2.0 * correction[x * ny + y] +
                  correction[x * ny + y + 1U]);
        residual2 += static_cast<long double>(residual) * residual;
        maximum_residual = std::max(maximum_residual, std::fabs(residual));
        ++residual_count;
      }
    for (std::size_t x = 0U; x + 1U < nx; ++x)
      for (std::size_t y = 0U; y < ny; ++y) {
        const double gradient =
            (correction[(x + 1U) * ny + y] - correction[x * ny + y]) / dx;
        gradient2 += static_cast<long double>(gradient) * gradient;
        maximum_gradient = std::max(maximum_gradient, std::fabs(gradient));
        ++gradient_count;
      }
    for (std::size_t x = 0U; x < nx; ++x)
      for (std::size_t y = 0U; y + 1U < ny; ++y) {
        const double gradient =
            (correction[x * ny + y + 1U] - correction[x * ny + y]) / dy;
        gradient2 += static_cast<long double>(gradient) * gradient;
        maximum_gradient = std::max(maximum_gradient, std::fabs(gradient));
        ++gradient_count;
      }
    const double residual_rms = residual_count == 0U ? 0.0 :
        std::sqrt(static_cast<double>(
            residual2 / static_cast<long double>(residual_count)));
    const double gradient_rms = gradient_count == 0U ? 0.0 :
        std::sqrt(static_cast<double>(
            gradient2 / static_cast<long double>(gradient_count)));
    const double h_effective = 1.0 / std::sqrt(ax + ay);
    const double gradient_floor = std::numeric_limits<double>::epsilon() *
        std::max(boundary_variation / h_effective,
                 std::numeric_limits<double>::min());
    report.normalized_residual_l2 = h_effective * residual_rms /
        std::max(gradient_rms, gradient_floor);
    report.normalized_residual_linf = h_effective * maximum_residual /
        std::max(maximum_gradient, gradient_floor);
    if (residual_rms == 0.0 && maximum_residual == 0.0) {
      report.normalized_residual_l2 = 0.0;
      report.normalized_residual_linf = 0.0;
    }
    report.normalized_residual = std::max(
        report.normalized_residual_l2, report.normalized_residual_linf);
  };

#if defined(USE_FFTW3)
  const std::size_t interior_nx = nx - 2U;
  const std::size_t interior_ny = ny - 2U;
  if (interior_nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      interior_ny > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("harmonic DST dimensions exceed FFTW int range");
  std::vector<double> transformed(
      checked_product(interior_nx, interior_ny, "harmonic DST"), 0.0);
  fftw_plan plan = NULL;
  {
    std::lock_guard<std::mutex> lock(
        feebl::fftw_planning::process_mutex());
    plan = fftw_plan_r2r_2d(
        static_cast<int>(interior_nx), static_cast<int>(interior_ny),
        transformed.data(), transformed.data(), FFTW_RODFT00, FFTW_RODFT00,
        FFTW_ESTIMATE);
  }
  if (!plan) throw std::runtime_error("FFTW DST-I plan creation failed");
  for (std::size_t x = 0U; x < interior_nx; ++x)
    for (std::size_t y = 0U; y < interior_ny; ++y) {
      double rhs = 0.0;
      if (x == 0U) rhs += ax * correction[y + 1U];
      if (x + 1U == interior_nx)
        rhs += ax * correction[(nx - 1U) * ny + y + 1U];
      if (y == 0U) rhs += ay * correction[(x + 1U) * ny];
      if (y + 1U == interior_ny)
        rhs += ay * correction[(x + 1U) * ny + ny - 1U];
      transformed[x * interior_ny + y] = rhs;
    }
  fftw_execute(plan);
  const double pi = std::acos(-1.0);
  for (std::size_t x = 0U; x < interior_nx; ++x)
    for (std::size_t y = 0U; y < interior_ny; ++y) {
      const double eigenvalue =
          2.0 * ax * (1.0 - std::cos(
              pi * static_cast<double>(x + 1U) /
              static_cast<double>(interior_nx + 1U))) +
          2.0 * ay * (1.0 - std::cos(
              pi * static_cast<double>(y + 1U) /
              static_cast<double>(interior_ny + 1U)));
      transformed[x * interior_ny + y] /= eigenvalue;
    }
  fftw_execute(plan);
  const double normalization = 4.0 *
      static_cast<double>(interior_nx + 1U) *
      static_cast<double>(interior_ny + 1U);
  for (std::size_t x = 0U; x < interior_nx; ++x)
    for (std::size_t y = 0U; y < interior_ny; ++y)
      correction[(x + 1U) * ny + y + 1U] =
          transformed[x * interior_ny + y] / normalization;
  {
    std::lock_guard<std::mutex> lock(
        feebl::fftw_planning::process_mutex());
    fftw_destroy_plan(plan);
  }
  report.iterations = 1U;
  measure_residual();
  report.converged = report.normalized_residual <= tolerance;
#else
  // Compatibility path for legacy FFTW2 builds.  Red-black SOR is retained
  // only here; supported FFTW3 builds use the O(N log N) direct DST solve.
  const double diagonal = 2.0 * (ax + ay);
  const double omega = 2.0 / (1.0 + std::sin(
      std::acos(-1.0) / static_cast<double>(std::max(nx, ny))));
  for (std::size_t iteration = 1U; iteration <= maximum_iterations;
       ++iteration) {
    for (unsigned int color = 0U; color < 2U; ++color)
      for (std::size_t x = 1U; x + 1U < nx; ++x)
        for (std::size_t y = 1U; y + 1U < ny; ++y)
          if (((x + y) & 1U) == color) {
            const double candidate =
                (ax * (correction[(x - 1U) * ny + y] +
                       correction[(x + 1U) * ny + y]) +
                 ay * (correction[x * ny + y - 1U] +
                       correction[x * ny + y + 1U])) /
                diagonal;
            correction[x * ny + y] +=
                omega * (candidate - correction[x * ny + y]);
          }
    report.iterations = iteration;
    if (iteration == 1U || (iteration & 7U) == 0U ||
        iteration == maximum_iterations) {
      measure_residual();
      if (report.normalized_residual <= tolerance) {
        report.converged = true;
        break;
      }
    }
  }
#endif
  for (double& value : correction) value += boundary_mean;
  return report;
}

}  // namespace adaptive_grid
}  // namespace feebl
