#include "particleBeamCPP.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t mix64(std::uint64_t value)
{
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  return value;
}

void randomized_delay(std::uint64_t seed, int slice)
{
  const std::uint64_t mixed =
      mix64(seed ^ (static_cast<std::uint64_t>(slice) << 32));
  const unsigned int iterations = static_cast<unsigned int>(mixed & 0x3fffu);
  volatile std::uint64_t sink = mixed;
  for (unsigned int i = 0; i < iterations; ++i)
    sink = mix64(sink + i);
  (void)sink;
}

struct Snapshot
{
  std::uint64_t count;
  std::uint64_t energy_bits;
  std::vector<std::vector<PHOTON::id_type> > ids;
  std::vector<std::vector<float> > energies;
  std::vector<std::vector<PHOTON::id_type> > dump_samples;

  bool operator==(const Snapshot& other) const
  {
    return count == other.count && energy_bits == other.energy_bits &&
           ids == other.ids && energies == other.energies &&
           dump_samples == other.dump_samples;
  }
};

float photon_energy(int slice, int ordinal)
{
  // Exactly representable binary fractions make any difference in reduction
  // order obvious while retaining non-uniform per-slice totals.
  return static_cast<float>(32 + 3 * slice + ordinal) / 64.0f;
}

int photons_in_slice(int slice)
{
  return 3 + ((slice * 7) % 17);
}

Snapshot run_case(int requested_threads, std::uint64_t delay_seed,
                  int* observed_threads)
{
  const int n_slices = 37;
  PHOTON_BEAM beam(n_slices);
  std::vector<PARTICLE> parents(static_cast<std::size_t>(n_slices));
  for (int slice = 0; slice < n_slices; ++slice)
    parents[static_cast<std::size_t>(slice)].setParticle(
        static_cast<float>(slice), -static_cast<float>(slice),
        0.25f * static_cast<float>(slice), 0.0f, 0.0f, 45.6f);

  int team_size = 1;
#ifdef _OPENMP
  omp_set_dynamic(0);
#pragma omp parallel num_threads(requested_threads)
  {
#pragma omp single
    team_size = omp_get_num_threads();

#pragma omp for schedule(dynamic, 1)
    for (int slice = 0; slice < n_slices; ++slice)
      {
        randomized_delay(delay_seed, slice);
        const int count = photons_in_slice(slice);
        for (int ordinal = 0; ordinal < count; ++ordinal)
          {
            randomized_delay(delay_seed + static_cast<std::uint64_t>(ordinal),
                             n_slices - slice);
            beam.new_photon(photon_energy(slice, ordinal),
                            parents[static_cast<std::size_t>(slice)], slice);
          }
      }
  }
#else
  (void)requested_threads;
  for (int slice = 0; slice < n_slices; ++slice)
    {
      randomized_delay(delay_seed, slice);
      const int count = photons_in_slice(slice);
      for (int ordinal = 0; ordinal < count; ++ordinal)
        beam.new_photon(photon_energy(slice, ordinal),
                        parents[static_cast<std::size_t>(slice)], slice);
    }
#endif
  if (observed_threads != NULL) *observed_threads = team_size;

  const PHOTON_COUNTER counter = beam.get_photon_counter();
  Snapshot snapshot;
  snapshot.count = counter.getNumber();
  static_assert(sizeof(snapshot.energy_bits) == sizeof(double),
                "unexpected double width");
  const double energy_sum = counter.getSum();
  std::memcpy(&snapshot.energy_bits, &energy_sum, sizeof(energy_sum));
  snapshot.ids.resize(static_cast<std::size_t>(n_slices));
  snapshot.energies.resize(static_cast<std::size_t>(n_slices));
  snapshot.dump_samples.resize(5u);
  const int dump_periods[] = {1, 2, 7, 16, 31};

  std::set<PHOTON::id_type> unique_ids;
  std::uint64_t expected_count = 0;
  for (int slice = 0; slice < n_slices; ++slice)
    {
      const std::vector<PHOTON>& photons = beam.getPhotonVector(slice);
      require(static_cast<int>(photons.size()) == photons_in_slice(slice),
              "slice photon count mismatch");
      for (std::size_t ordinal = 0; ordinal < photons.size(); ++ordinal)
        {
          const PHOTON::id_type expected_id =
              static_cast<PHOTON::id_type>(ordinal) *
                  static_cast<PHOTON::id_type>(n_slices) +
              static_cast<PHOTON::id_type>(slice) + 1u;
          require(photons[ordinal].no() == expected_id,
                  "stable photon ID mismatch");
          require(photons[ordinal].energy() ==
                      photon_energy(slice, static_cast<int>(ordinal)),
                  "photon energy mismatch");
          require(unique_ids.insert(expected_id).second,
                  "duplicate stable photon ID");
          snapshot.ids[static_cast<std::size_t>(slice)].push_back(expected_id);
          snapshot.energies[static_cast<std::size_t>(slice)].push_back(
              photons[ordinal].energy());
          for (std::size_t period = 0; period < 5u; ++period)
            if ((expected_id %
                 static_cast<PHOTON::id_type>(dump_periods[period])) == 0u)
              snapshot.dump_samples[period].push_back(expected_id);
          ++expected_count;
        }
    }
  require(snapshot.count == expected_count, "aggregate photon count mismatch");

  double surviving_energy = 0.0;
  int surviving_count = 0;
  beam.photon_info(surviving_energy, surviving_count);
  require(static_cast<std::uint64_t>(surviving_count) == snapshot.count,
          "surviving photon count mismatch");
  require(surviving_energy == energy_sum,
          "canonical emitted/surviving energy sums differ");

  // A rejected append cannot perturb any per-slice or aggregate accounting.
  bool rejected = false;
  try
    {
      beam.new_photon(1.0f, parents.front(), n_slices);
    }
  catch (const std::out_of_range&)
    {
      rejected = true;
    }
  require(rejected, "invalid slice append did not fail");
  const PHOTON_COUNTER after_rejection = beam.get_photon_counter();
  require(after_rejection.getNumber() == counter.getNumber(),
          "rejected append changed aggregate count");
  require(after_rejection.getSum() == counter.getSum(),
          "rejected append changed aggregate energy");

  return snapshot;
}

} // namespace

int main()
{
  int observed = 1;
  const Snapshot reference = run_case(1, UINT64_C(0x1020304050607080),
                                      &observed);
  require(observed == 1, "one-thread reference used a larger OpenMP team");

#ifdef _OPENMP
  const int thread_counts[] = {1, 2, 4, 8};
#else
  const int thread_counts[] = {1};
#endif
  const std::uint64_t seeds[] = {
      UINT64_C(0x1020304050607080),
      UINT64_C(0x8877665544332211),
      UINT64_C(0xfedcba9876543210)};

  for (std::size_t t = 0;
       t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
    for (std::size_t seed = 0; seed < sizeof(seeds) / sizeof(seeds[0]); ++seed)
      {
        const Snapshot candidate =
            run_case(thread_counts[t], seeds[seed], &observed);
        require(candidate == reference,
                "photon signature changed with worker schedule");
        std::cout << "PASS photon accounting requested_threads="
                  << thread_counts[t] << " observed_threads=" << observed
                  << " delay_seed=" << seeds[seed] << '\n';
      }

  return 0;
}
