#include "uniformWindowPlanCPP.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using feebl::adaptive_grid::CanonicalDepositDescriptor;
using feebl::adaptive_grid::CanonicalGatherDescriptor;
using feebl::adaptive_grid::CanonicalLatticeInput;
using feebl::adaptive_grid::CanonicalLatticeSnapshot;
using feebl::adaptive_grid::CanonicalNode;
using feebl::adaptive_grid::CanonicalWindowInputs;
using feebl::adaptive_grid::UniformWindowPlan;
using feebl::adaptive_grid::UniformWindowPolicy;
using feebl::adaptive_grid::WindowPlanError;
using feebl::adaptive_grid::WindowPlanErrorCode;
using feebl::adaptive_grid::WindowPlanStatus;
using feebl::adaptive_grid::parse_size_classes;

int failures = 0;

void check(bool condition, const std::string& description) {
  std::cout << "  " << (condition ? "PASS" : "FAIL") << "  "
            << description << '\n';
  if (!condition) ++failures;
}

template <class Function>
void check_error(WindowPlanErrorCode expected, Function function,
                 const std::string& description) {
  bool matched = false;
  try {
    function();
  } catch (const WindowPlanError& error) {
    matched = error.code() == expected;
    if (!matched)
      std::cerr << "    wrong error code for " << description << ": "
                << static_cast<int>(error.code()) << " message="
                << error.what() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "    wrong exception for " << description << ": "
              << error.what() << '\n';
  }
  check(matched, description);
}

CanonicalLatticeInput lattice_input(std::size_t n_x = 16U,
                                    std::size_t n_y = 16U) {
  CanonicalLatticeInput input;
  input.n_x = n_x;
  input.n_y = n_y;
  input.delta_x = 0.25f;
  input.delta_y = 0.5f;
  input.delta_z = 0.75f;
  input.inverse_delta_x = 4.0f;
  input.inverse_delta_y = 2.0f;
  input.offset_x = static_cast<float>(n_x / 2U);
  input.offset_y = static_cast<float>(n_y / 2U);
  input.intended_min_x = -1.75f;
  input.intended_max_x = 1.75f;
  input.intended_min_y = -3.5f;
  input.intended_max_y = 3.5f;
  input.charge_sign = -1.0f;
  input.field_kernel_factor = -0.03125;
  input.geometry_revision = 19U;
  return input;
}

UniformWindowPolicy policy(const std::string& x = "4,5,8,12,16",
                           const std::string& y = "4,5,8,12,16") {
  UniformWindowPolicy result;
  result.size_classes_x = parse_size_classes(x);
  result.size_classes_y = parse_size_classes(y);
  return result;
}

template <class Collection>
bool contains(const Collection& values, const CanonicalNode& node) {
  return std::find(values.begin(), values.end(), node) != values.end();
}

void test_lattice_and_class_parser() {
  std::cout << "\nCanonical snapshot and deterministic class parsing\n";
  const std::vector<std::size_t> parsed = parse_size_classes(" 16,4, 8,4 ");
  check(parsed == std::vector<std::size_t>({4U, 8U, 16U}),
        "class parser trims, sorts, and deduplicates deterministically");

  check_error(WindowPlanErrorCode::InvalidClassBank,
              [] { (void)parse_size_classes(""); },
              "empty class table fails closed");
  check_error(WindowPlanErrorCode::InvalidClassBank,
              [] { (void)parse_size_classes("4,,8"); },
              "empty class-table entry fails closed");
  check_error(WindowPlanErrorCode::InvalidClassBank,
              [] { (void)parse_size_classes("-4,8"); },
              "signed class-table entry is rejected");
  check_error(WindowPlanErrorCode::InvalidClassBank,
              [] { (void)parse_size_classes("3,8"); },
              "class smaller than the supported field grid is rejected");
  check_error(WindowPlanErrorCode::ArithmeticOverflow,
              [] { (void)parse_size_classes(
                       "999999999999999999999999999999999999999"); },
              "decimal class overflow is rejected before wraparound");

  const CanonicalLatticeInput input = lattice_input();
  const CanonicalLatticeSnapshot first(input);
  const CanonicalLatticeSnapshot again(input);
  check(first.hash() == again.hash() && first.geometry_revision() == 19U,
        "identical exact canonical inputs have a reproducible snapshot hash");
  check(first.hash() == 360135084128434996ULL,
        "canonical snapshot hash has a stable version-1 serialization fixture");
  CanonicalLatticeInput changed = input;
  changed.delta_x = std::nextafter(changed.delta_x, 1.0f);
  check(CanonicalLatticeSnapshot(changed).hash() != first.hash(),
        "a one-ULP lattice-spacing change changes the exact hash");
  changed = input;
  changed.charge_sign = std::nextafter(changed.charge_sign, 0.0f);
  check(CanonicalLatticeSnapshot(changed).hash() != first.hash(),
        "the exact solver charge sign participates in the lattice hash");
  changed = input;
  changed.geometry_revision++;
  check(CanonicalLatticeSnapshot(changed).hash() != first.hash(),
        "geometry revision participates in the lattice hash");

  changed = input;
  changed.n_x = 3U;
  check_error(WindowPlanErrorCode::InvalidLattice,
              [&] { (void)CanonicalLatticeSnapshot(changed); },
              "undersized canonical lattice is rejected");
  changed = input;
  changed.inverse_delta_x = std::numeric_limits<float>::infinity();
  check_error(WindowPlanErrorCode::InvalidLattice,
              [&] { (void)CanonicalLatticeSnapshot(changed); },
              "non-finite canonical affine input is rejected");
}

void test_exact_descriptor_nodes_and_nextafter_faces() {
  std::cout << "\nExact canonical descriptor inputs and one-ULP faces\n";
  const float almost_one = std::nextafter(1.0f, 0.0f);
  const CanonicalDepositDescriptor deposit =
      CanonicalDepositDescriptor::from_canonical(3, 4, almost_one, 0.0f);
  const std::array<CanonicalNode, 4> expected_deposit =
      {{{3, 4}, {4, 4}, {3, 5}, {4, 5}}};
  check(deposit.accepted() && deposit.nodes() == expected_deposit &&
            deposit.weight_x() == almost_one,
        "four-node descriptor preserves canonical CIC order and exact weight");
  check_error(WindowPlanErrorCode::InvalidDescriptor,
              [] { (void)CanonicalDepositDescriptor::from_canonical(
                       3, 4, 1.0f, 0.0f); },
              "a non-canonical unit CIC weight is rejected");

  // These are the precomputed float-narrowed decisions made by
  // interpolePotential for absolute coordinates immediately below integer
  // faces.  The planner receives them; it never repeats the affine mapping.
  const float absolute_x = std::nextafter(9.0f, 0.0f);
  const float absolute_y = std::nextafter(11.0f, 0.0f);
  const std::int64_t y_base_x =
      static_cast<std::int64_t>(std::floor(absolute_x));
  const std::int64_t y_base_y =
      static_cast<std::int64_t>(std::floor(absolute_y - 0.5f));
  const float y_weight = absolute_y - 0.5f -
                         static_cast<float>(y_base_y);
  const std::int64_t x_base_x =
      static_cast<std::int64_t>(std::floor(absolute_x - 0.5f));
  const std::int64_t x_base_y =
      static_cast<std::int64_t>(std::floor(absolute_y));
  const float x_weight = absolute_x - 0.5f -
                         static_cast<float>(x_base_x);
  const double fractional_x = static_cast<double>(absolute_x) -
                              std::floor(static_cast<double>(absolute_x));
  const double fractional_y = static_cast<double>(absolute_y) -
                              std::floor(static_cast<double>(absolute_y));
  const CanonicalGatherDescriptor gather =
      CanonicalGatherDescriptor::from_canonical(
          y_base_x, y_base_y, y_weight,
          x_base_x, x_base_y, x_weight,
          fractional_x, fractional_y);
  const CanonicalGatherDescriptor gathered_by_helper =
      CanonicalGatherDescriptor::from_legacy_absolute_coordinates(
          static_cast<double>(absolute_x), static_cast<double>(absolute_y));
  const std::array<CanonicalNode, 12> expected_gather =
      {{{9, 11}, {9, 10}, {8, 11}, {8, 10}, {7, 11}, {7, 10},
        {9, 11}, {8, 11}, {9, 10}, {8, 10}, {9, 9}, {8, 9}}};
  check(y_base_x == 8 && x_base_y == 10 &&
            gather.nodes() == expected_gather &&
            gathered_by_helper.nodes() == gather.nodes() &&
            gathered_by_helper.y_weight() == gather.y_weight() &&
            gathered_by_helper.x_weight() == gather.x_weight() &&
            gathered_by_helper.fractional_x() == gather.fractional_x() &&
            gathered_by_helper.fractional_y() == gather.fractional_y(),
        "nextafter-below indices retain the exact twelve Y-then-X reads");

  const CanonicalGatherDescriptor on_face =
      CanonicalGatherDescriptor::from_canonical(
          9, 10, 0.5f, 8, 11, 0.5f, 0.0, 0.0);
  check(on_face.y_base_x() == gather.y_base_x() + 1 &&
            on_face.x_base_y() == gather.x_base_y() + 1,
        "the exact-face and nextafter-below mapping decisions stay distinct");
  check_error(WindowPlanErrorCode::InvalidDescriptor,
              [] { (void)CanonicalGatherDescriptor::from_canonical(
                       4, 4, 0.5f, 4, 4, 0.5f,
                       std::numeric_limits<double>::quiet_NaN(), 0.5); },
              "non-finite canonical gather fractions are rejected");
}

void test_exact_union_and_odd_slack() {
  std::cout << "\nExact node union, class selection, and odd slack\n";
  const CanonicalLatticeSnapshot lattice(lattice_input());
  CanonicalWindowInputs inputs;
  const CanonicalDepositDescriptor source =
      CanonicalDepositDescriptor::from_canonical(5, 6, 0.25f, 0.75f);
  inputs.beam1_sources.push_back(source);
  const UniformWindowPlan plan = UniformWindowPlan::build(
      lattice, inputs, policy("5,8,16", "5,8,16"));
  std::vector<CanonicalNode> expected_single(source.nodes().begin(),
                                              source.nodes().end());
  std::sort(expected_single.begin(), expected_single.end());
  check(plan.status() == WindowPlanStatus::Ready &&
            plan.local_n_x() == 5U && plan.local_n_y() == 5U,
        "smallest independently fitting x/y classes are selected");
  const UniformWindowPlan anisotropic = UniformWindowPlan::build(
      lattice, inputs, policy("5,8,16", "8,16"));
  check(anisotropic.local_n_x() == 5U && anisotropic.local_n_y() == 8U,
        "x and y size-class banks are selected independently");
  check(plan.required_nodes() == expected_single,
        "a single CIC input exposes its exact four-node union");
  check(plan.origin_x() == 3 && plan.origin_y() == 4,
        "odd three-cell slack assigns two cells to the low side");
  check(plan.padded_n_x() == 10U && plan.padded_n_y() == 10U &&
            plan.estimated_bytes() == 5600U,
        "doubled-domain geometry and conservative byte estimate are exact");
  check(plan.to_local({5, 6}) == CanonicalNode{2, 2} &&
            plan.local_linear_index({5, 6}) == 12U,
        "canonical nodes translate only by the chosen integer origin");
  check(plan.beam1_sources().size() == 1U &&
            plan.beam1_sources()[0].nodes() == source.nodes(),
        "the immutable plan retains exact per-particle mapping descriptors");
  check_error(WindowPlanErrorCode::NodeNotInWindow,
              [&] { (void)plan.to_local({15, 15}); },
              "translation of a node outside the selected window fails closed");

  CanonicalWindowInputs overlap;
  overlap.beam1_sources.push_back(source);
  overlap.beam2_sources.push_back(source);
  overlap.beam1_sources.push_back(CanonicalDepositDescriptor::rejected());
  const CanonicalGatherDescriptor target =
      CanonicalGatherDescriptor::from_canonical(
          6, 6, 0.2f, 5, 7, 0.8f, 0.3, 0.7);
  overlap.beam2_targets.push_back(target);
  std::vector<CanonicalNode> expected_source(source.nodes().begin(),
                                              source.nodes().end());
  std::vector<CanonicalNode> expected_gather(target.nodes().begin(),
                                              target.nodes().end());
  std::vector<CanonicalNode> expected_union = expected_source;
  expected_union.insert(expected_union.end(), expected_gather.begin(),
                        expected_gather.end());
  const auto unique = [](std::vector<CanonicalNode>& nodes) {
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  };
  unique(expected_source);
  unique(expected_gather);
  unique(expected_union);
  const UniformWindowPlan union_plan = UniformWindowPlan::build(
      lattice, overlap, policy("8,16", "8,16"));
  check(union_plan.source_nodes() == expected_source &&
            union_plan.gather_nodes() == expected_gather &&
            union_plan.required_nodes() == expected_union,
        "source, gather, and combined unions are independently exact/deduplicated");
  check(union_plan.accepted_source_count() == 2U &&
            union_plan.accepted_target_count() == 1U,
        "accepted counts retain duplicates while rejected descriptors need no nodes");
  CanonicalWindowInputs moved_between_beams;
  moved_between_beams.beam2_sources.push_back(source);
  const UniformWindowPlan moved_plan = UniformWindowPlan::build(
      lattice, moved_between_beams, policy("5,8,16", "5,8,16"));
  check(moved_plan.required_nodes() == plan.required_nodes() &&
            moved_plan.plan_hash() != plan.plan_hash(),
        "beam identity binds the plan even when the exact union is unchanged");
}

void test_guard_and_boundary_shift() {
  std::cout << "\nGuard saturation and deterministic boundary shift\n";
  const CanonicalLatticeSnapshot lattice(lattice_input());
  UniformWindowPolicy guarded_policy = policy("8,16", "8,16");
  guarded_policy.extra_guard_cells = 2;

  CanonicalWindowInputs low;
  low.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(0, 5, 0.1f, 0.2f));
  const UniformWindowPlan low_plan =
      UniformWindowPlan::build(lattice, low, guarded_policy);
  check(low_plan.guard_clipped_low_x() &&
            !low_plan.guard_clipped_high_x() &&
            low_plan.guarded_box().min_x == 0 &&
            low_plan.guarded_box().max_x == 3 && low_plan.origin_x() == 0,
        "low-boundary guard saturates explicitly and shifts the window to zero");
  for (const CanonicalNode& node : low_plan.required_nodes())
    check(low_plan.contains_canonical_node(node),
          "low-boundary placement retains every required node");

  CanonicalWindowInputs high;
  high.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(14, 5, 0.1f, 0.2f));
  const UniformWindowPlan high_plan =
      UniformWindowPlan::build(lattice, high, guarded_policy);
  check(high_plan.guard_clipped_high_x() &&
            !high_plan.guard_clipped_low_x() &&
            high_plan.guarded_box().min_x == 12 &&
            high_plan.guarded_box().max_x == 15 && high_plan.origin_x() == 8,
        "high-boundary guard saturates and shifts against canonical storage");
  for (const CanonicalNode& node : high_plan.required_nodes())
    check(high_plan.contains_canonical_node(node),
          "high-boundary placement retains every required node");
}

void test_fail_closed_selection_and_bank_bounds() {
  std::cout << "\nFail-closed descriptor, class-bank, and full-window policy\n";
  const CanonicalLatticeSnapshot lattice(lattice_input());

  CanonicalWindowInputs outside;
  outside.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(15, 4, 0.5f, 0.5f));
  check_error(WindowPlanErrorCode::OutOfCanonicalBank,
              [&] { (void)UniformWindowPlan::build(
                       lattice, outside, policy("8,16", "8,16")); },
              "accepted source whose +1 node leaves canonical storage is rejected");

  CanonicalWindowInputs negative_gather;
  negative_gather.beam1_targets.push_back(
      CanonicalGatherDescriptor::from_canonical(
          0, 4, 0.5f, 4, 4, 0.5f, 0.5, 0.5));
  check_error(WindowPlanErrorCode::OutOfCanonicalBank,
              [&] { (void)UniformWindowPlan::build(
                       lattice, negative_gather, policy("8,16", "8,16")); },
              "accepted twelve-node stencil crossing the low bank is rejected");

  CanonicalWindowInputs too_wide;
  too_wide.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(0, 5, 0.5f, 0.5f));
  too_wide.beam2_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(10, 5, 0.5f, 0.5f));
  check_error(WindowPlanErrorCode::NoFittingClass,
              [&] { (void)UniformWindowPlan::build(
                       lattice, too_wide, policy("4,8", "4,8")); },
              "required union wider than every admitted class fails closed");

  UniformWindowPolicy invalid_bank = policy("32", "8");
  check_error(WindowPlanErrorCode::InvalidClassBank,
              [&] { (void)UniformWindowPlan::build(
                       lattice, CanonicalWindowInputs{}, invalid_bank); },
              "class bank wholly outside canonical max is invalid even when empty");
  UniformWindowPolicy invalid_guard = policy("8", "8");
  invalid_guard.extra_guard_cells = -1;
  check_error(WindowPlanErrorCode::InvalidPolicy,
              [&] { (void)UniformWindowPlan::build(
                       lattice, CanonicalWindowInputs{}, invalid_guard); },
              "negative optional guard is rejected");
  UniformWindowPolicy invalid_limits = policy("8,16", "8,16");
  invalid_limits.max_n_x = 17U;
  check_error(WindowPlanErrorCode::InvalidPolicy,
              [&] { (void)UniformWindowPlan::build(
                       lattice, CanonicalWindowInputs{}, invalid_limits); },
              "configured maximum beyond canonical storage is rejected");

  const CanonicalLatticeSnapshot small(lattice_input(8U, 8U));
  CanonicalWindowInputs full_inputs;
  full_inputs.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(0, 0, 0.5f, 0.5f));
  full_inputs.beam2_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(6, 6, 0.5f, 0.5f));
  UniformWindowPolicy forbid = policy("8", "8");
  forbid.allow_full_window = false;
  check_error(WindowPlanErrorCode::FullWindowForbidden,
              [&] { (void)UniformWindowPlan::build(small, full_inputs, forbid); },
              "canonical-sized selection requires explicit full-window permission");
  forbid.allow_full_window = true;
  const UniformWindowPlan full =
      UniformWindowPlan::build(small, full_inputs, forbid);
  check(full.full_window() && full.origin_x() == 0 && full.origin_y() == 0,
        "permitted full window remains an explicit counted topology class");
  const UniformWindowPlan compact = full.compact_full_window_identity();
  check(compact.full_window() && compact.plan_hash() == full.plan_hash() &&
            compact.lattice_hash() == full.lattice_hash() &&
            compact.accepted_source_count() ==
                full.accepted_source_count() &&
            compact.accepted_target_count() ==
                full.accepted_target_count() &&
            compact.required_node_count() == full.required_nodes().size() &&
            !compact.descriptor_payload_retained() &&
            compact.beam1_sources().empty() &&
            compact.beam2_sources().empty() &&
            compact.beam1_targets().empty() &&
            compact.beam2_targets().empty() &&
            compact.required_nodes().empty(),
        "compact full-window identity preserves binding/count metadata "
        "without retaining execution descriptors");
}

void test_overflow_memory_empty_and_reproducibility() {
  std::cout << "\nOverflow, memory budget, empty plan, and reproducibility\n";
  const CanonicalLatticeSnapshot lattice(lattice_input());
  CanonicalWindowInputs inputs;
  inputs.beam1_sources.push_back(
      CanonicalDepositDescriptor::from_canonical(5, 5, 0.25f, 0.75f));

  UniformWindowPolicy overflow = policy("4", "4");
  overflow.bytes_per_local_cell =
      std::numeric_limits<std::size_t>::max();
  overflow.bytes_per_padded_cell = 0U;
  check_error(WindowPlanErrorCode::ArithmeticOverflow,
              [&] { (void)UniformWindowPlan::build(lattice, inputs, overflow); },
              "per-cell byte multiplication is checked before allocation");

  UniformWindowPolicy addition_overflow = policy("4", "4");
  addition_overflow.bytes_per_local_cell = 1U;
  addition_overflow.bytes_per_padded_cell = 0U;
  addition_overflow.fixed_bytes = std::numeric_limits<std::size_t>::max();
  check_error(WindowPlanErrorCode::ArithmeticOverflow,
              [&] { (void)UniformWindowPlan::build(
                       lattice, inputs, addition_overflow); },
              "fixed-byte addition is checked before wraparound");

  UniformWindowPolicy constrained = policy("4", "4");
  constrained.max_bytes = 100U;
  check_error(WindowPlanErrorCode::MemoryBudgetExceeded,
              [&] { (void)UniformWindowPlan::build(
                       lattice, inputs, constrained); },
              "a representable plan over the declared memory budget is rejected");

  CanonicalWindowInputs empty_inputs;
  empty_inputs.beam1_sources.push_back(
      CanonicalDepositDescriptor::rejected());
  empty_inputs.beam2_targets.push_back(
      CanonicalGatherDescriptor::rejected());
  UniformWindowPolicy unsorted = policy();
  unsorted.size_classes_x = {16U, 4U, 8U, 4U, 5U, 12U};
  unsorted.size_classes_y = {12U, 8U, 5U, 4U, 16U, 8U};
  const UniformWindowPlan empty_first =
      UniformWindowPlan::build(lattice, empty_inputs, unsorted);
  const UniformWindowPlan empty_again =
      UniformWindowPlan::build(lattice, empty_inputs, policy());
  check(empty_first.empty() && empty_first.local_n_x() == 0U &&
            empty_first.local_n_y() == 0U &&
            empty_first.estimated_bytes() == 0U &&
            empty_first.required_nodes().empty(),
        "all-rejected encounter has a deterministic allocation-free empty status");
  check(empty_first.plan_hash() == empty_again.plan_hash() &&
            empty_first.lattice_hash() == lattice.hash(),
        "semantically identical normalized class banks reproduce the same plan hash");
  check(empty_first.plan_hash() == 10985285129460039600ULL,
        "empty plan hash has a stable version-1 serialization fixture");

  const UniformWindowPlan ready_first =
      UniformWindowPlan::build(lattice, inputs, policy("4,8", "4,8"));
  const UniformWindowPlan ready_again =
      UniformWindowPlan::build(lattice, inputs, policy("8,4,4", "8,4"));
  check(ready_first.plan_hash() == ready_again.plan_hash(),
        "ready-plan hash is reproducible under class-table reordering/duplicates");
  check(ready_first.plan_hash() == 16742494756907126528ULL,
        "ready plan hash has a stable version-1 serialization fixture");
  CanonicalWindowInputs changed_weight = inputs;
  changed_weight.beam1_sources[0] =
      CanonicalDepositDescriptor::from_canonical(
          5, 5, std::nextafter(0.25f, 1.0f), 0.75f);
  const UniformWindowPlan rebound = UniformWindowPlan::build(
      lattice, changed_weight, policy("4,8", "4,8"));
  check(rebound.required_nodes() == ready_first.required_nodes() &&
            rebound.plan_hash() != ready_first.plan_hash(),
        "exact descriptor bits bind the plan even when the node union is unchanged");
}

}  // namespace

int main() {
  test_lattice_and_class_parser();
  test_exact_descriptor_nodes_and_nextafter_faces();
  test_exact_union_and_odd_slack();
  test_guard_and_boundary_shift();
  test_fail_closed_selection_and_bank_bounds();
  test_overflow_memory_empty_and_reproducibility();

  if (failures != 0) {
    std::cerr << "\nuniform-window plan tests failed: " << failures << '\n';
    return EXIT_FAILURE;
  }
  std::cout << "\nall uniform-window plan tests passed\n";
  return EXIT_SUCCESS;
}
