#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "gridCPP.h"

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    {
      std::cerr << "sparse-cell test failure: " << message << std::endl;
      std::exit(1);
    }
}

void require_cells(const std::vector<int>& actual,
                   const std::vector<int>& expected,
                   const char* message)
{
  require(actual == expected, message);
}

void test_default_dense_clear()
{
  feebl::sparse_cells::ActiveCellSet active(12U);
  std::vector<std::vector<int> > bins(12U);
  bins[2].push_back(10);
  bins[9].push_back(20);
  active.clear_touched(bins);
  require(bins[2].empty() && bins[9].empty(),
          "default mode must retain dense clearing");
  require(active.dense_clear_visits() == 12U,
          "default dense-equivalent counter");
  require(active.actual_clear_visits() == 12U,
          "default actual-clear counter");
  require(active.avoided_clear_visits() == 0U,
          "default mode must claim no avoided work");
}

void test_order_dedup_and_epoch_reuse()
{
  feebl::sparse_cells::ActiveCellSet active(12U);
  active.set_enabled(true);
  std::vector<std::vector<int> > bins(12U);

  const int cells[] = {7, 2, 7, 9, 2};
  const int values[] = {70, 20, 71, 90, 21};
  for (std::size_t index = 0; index < 5U; ++index)
    {
      active.touch(cells[index]);
      bins[static_cast<std::size_t>(cells[index])].push_back(values[index]);
    }
  active.sort_cells();
  require_cells(active.cells(), std::vector<int>({2, 7, 9}),
                "active cells must be unique and ascending");
  require(bins[2] == std::vector<int>({20, 21}),
          "within-cell insertion order for cell 2");
  require(bins[7] == std::vector<int>({70, 71}),
          "within-cell insertion order for cell 7");

  // An untracked sentinel proves the enabled clear does not scan every bin.
  bins[11].push_back(111);
  active.clear_touched(bins);
  require(bins[2].empty() && bins[7].empty() && bins[9].empty(),
          "all touched cells must clear");
  require(bins[11] == std::vector<int>({111}),
          "untouched cell must not be visited by sparse clear");
  require(active.dense_clear_visits() == 12U,
          "sparse dense-equivalent counter");
  require(active.actual_clear_visits() == 3U,
          "sparse actual-clear counter");
  require(active.avoided_clear_visits() == 9U,
          "sparse avoided-clear counter");

  // Empty -> nonempty -> empty -> nonempty must work across generations.
  active.touch(7);
  bins[7].push_back(72);
  active.touch(7);
  active.sort_cells();
  require_cells(active.cells(), std::vector<int>({7}),
                "generation must deduplicate a reused cell");
  active.clear_touched(bins);
  require(bins[7].empty(), "reused cell must clear in the next epoch");
}

void test_sorted_set_operations()
{
  const std::vector<int> a({1, 4, 8, 12});
  const std::vector<int> b({0, 4, 7, 8, 15});
  const std::vector<int> c({3, 7, 12});
  std::vector<int> result;
  feebl::sparse_cells::intersect_sorted(a, b, &result);
  require_cells(result, std::vector<int>({4, 8}),
                "intersection must preserve ascending cell order");
  const std::vector<int>* sets[] = {&a, &b, &c};
  feebl::sparse_cells::sorted_union(sets, 3U, &result);
  require_cells(result, std::vector<int>({0, 1, 3, 4, 7, 8, 12, 15}),
                "union must be unique and ascending");
}

void test_charge_tracking_and_symmetry()
{
  SLICE_ON_GRID slice(4, 4);
  slice.set_macroparticles(1);
  slice.update_nb_part_per_macro(1.0F);
  slice.enableSparseCells(true);

  slice.assignChargeToNGP(0, 0, 1.0F);
  slice.assignChargeToCIC(1, 1, 0.25F, 0.75F, 1.0F);
  slice.finalizeActiveRhoCells();
  require_cells(slice.activeRhoCells(), std::vector<int>({0, 5, 6, 9, 10}),
                "NGP and all four CIC nodes must be tracked");
  slice.razRho();
  const PHI_FLOAT* rho = slice.get_rho();
  for (int cell = 0; cell < 16; ++cell)
    require(rho[cell] == static_cast<PHI_FLOAT>(0.0),
            "sparse rho clear must remove every deposited value");
  require(slice.denseRhoClearVisits() == 16U,
          "rho dense-equivalent counter");
  require(slice.actualRhoClearVisits() == 5U,
          "rho actual-clear counter");

  slice.assignChargeToNGP(1, 1, 1.0F);
  slice.symmetrizeCharges(4, 4);
  slice.finalizeActiveRhoCells();
  require(slice.activeRhoCells().size() == 16U,
          "four-way symmetrization must track every written cell");
  slice.razRho();
  for (int cell = 0; cell < 16; ++cell)
    require(rho[cell] == static_cast<PHI_FLOAT>(0.0),
            "symmetrized rho must clear without stale cells");
}

}  // namespace

int main()
{
  test_default_dense_clear();
  test_order_dedup_and_epoch_reuse();
  test_sorted_set_operations();
  test_charge_tracking_and_symmetry();
  std::cout << "sparse-cell tests passed" << std::endl;
  return 0;
}
