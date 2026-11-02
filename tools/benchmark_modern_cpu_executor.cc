#include "encounterDAGCPP.h"
#include "modernCpuExecutorCPP.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#error "The modern CPU executor benchmark requires an OpenMP compiler build"
#endif

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

using feebl::modern::ConstByteLane;
using feebl::modern::CpuContextResetPolicy;
using feebl::modern::CpuEncounterExecutor;
using feebl::modern::CpuExecutionKind;
using feebl::modern::CpuExecutionReport;
using feebl::modern::CpuExecutorConfig;
using feebl::modern::CpuScheduleStrategy;
using feebl::modern::EncounterDAG;
using feebl::modern::EncounterId;
using feebl::modern::EncounterLogicalTime;
using feebl::modern::EncounterNode;
using feebl::modern::MutableByteLane;
using feebl::modern::StepContext;

namespace {

constexpr const char* kSchema =
    "feebl.modern_cpu_executor.microbenchmark.v2";

struct Options {
  std::string preset;
  std::filesystem::path json_path;
  std::filesystem::path csv_path;
  std::filesystem::path metadata_path;
  std::uint32_t warmups = 0;
  std::uint32_t repetitions = 0;
  std::uint32_t worker_capacity = 0;
  std::uint64_t seed = 0;
  std::string git_revision;
  std::string compiler_command;
  std::string compile_flags;
};

struct Shape {
  std::string name;
  std::uint64_t beam1 = 0;
  std::uint64_t beam2 = 0;
  std::uint64_t substeps = 0;
};

struct Intensity {
  std::string name;
  std::uint32_t iterations = 0;
};

struct Variant {
  std::string name;
  CpuExecutionKind execution = CpuExecutionKind::SerialReference;
  std::uint32_t workers = 1;
};

struct alignas(64) SliceState {
  std::uint64_t value = 0;
};

struct ResultRecord {
  std::uint64_t id = 0;
  std::uint64_t wave = 0;
  std::uint64_t beam1 = 0;
  std::uint64_t beam2 = 0;
};

struct EmissionRecord {
  std::uint64_t id = 0;
  std::uint64_t value = 0;
};

struct KernelState {
  std::vector<SliceState> beam1;
  std::vector<SliceState> beam2;
  std::uint32_t iterations = 0;

  KernelState(std::uint64_t n1, std::uint64_t n2, std::uint32_t work)
      : beam1(static_cast<std::size_t>(n1)),
        beam2(static_cast<std::size_t>(n2)),
        iterations(work) {
    reset();
  }

  void reset();
};

struct Sample {
  std::uint32_t repetition = 0;
  std::uint32_t order_index = 0;
  double seconds = 0.0;
  std::uint64_t checksum = 0;
  CpuExecutionReport report;
};

struct Aggregate {
  Variant variant;
  double construction_seconds = 0.0;
  std::uint64_t preflight_bytes = 0;
  std::uint64_t checksum = 0;
  bool deterministic_checksum_match = false;
  std::vector<Sample> samples;
  double median = 0.0;
  double p95 = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double speedup = 0.0;
};

struct CaseResult {
  Shape shape;
  Intensity intensity;
  std::uint64_t encounters = 0;
  std::uint64_t waves = 0;
  std::uint64_t oracle_checksum = 0;
  std::vector<Aggregate> aggregates;
};

struct ExecutorHolder {
  Aggregate aggregate;
  std::unique_ptr<CpuEncounterExecutor> executor;
  KernelState state;

  ExecutorHolder(Aggregate aggregate_in,
                 std::unique_ptr<CpuEncounterExecutor> executor_in,
                 KernelState state_in)
      : aggregate(std::move(aggregate_in)),
        executor(std::move(executor_in)),
        state(std::move(state_in)) {}
};

struct BindingMetadata {
  std::vector<int> process_cpus;
  std::vector<int> worker_cpus;
  std::vector<int> worker_places;
  std::string proc_bind;
  int omp_num_places = 0;
};

std::uint64_t mix(std::uint64_t state, std::uint64_t value) noexcept {
  state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
  state ^= state >> 30U;
  state *= 0xbf58476d1ce4e5b9ULL;
  state ^= state >> 27U;
  state *= 0x94d049bb133111ebULL;
  return state ^ (state >> 31U);
}

std::uint64_t rotate_left(std::uint64_t value, unsigned shift) noexcept {
  shift &= 63U;
  return shift == 0 ? value : (value << shift) | (value >> (64U - shift));
}

void KernelState::reset() {
  for (std::size_t index = 0; index < beam1.size(); ++index)
    beam1[index].value = mix(0x123456789abcdef0ULL, index);
  for (std::size_t index = 0; index < beam2.size(); ++index)
    beam2[index].value = mix(0xfedcba9876543210ULL, index);
}

void kernel_callback(const EncounterNode& node,
                     const EncounterLogicalTime& time, StepContext& context,
                     MutableByteLane result_lane,
                     MutableByteLane emission_lane, void* opaque) {
  auto& state = *static_cast<KernelState*>(opaque);
  SliceState& left = state.beam1.at(
      static_cast<std::size_t>(node.beam1_slice));
  SliceState& right = state.beam2.at(
      static_cast<std::size_t>(node.beam2_slice));
  std::uint64_t a = mix(left.value, node.id ^ (time.wave << 17U));
  std::uint64_t b = mix(right.value, node.id ^ (time.substep << 41U));
  for (std::uint32_t iteration = 0; iteration < state.iterations;
       ++iteration) {
    a = mix(a, rotate_left(b + iteration + node.id, iteration % 61U + 1U));
    b = mix(b, rotate_left(a ^ time.diagonal, iteration % 59U + 1U));
  }
  left.value = mix(left.value, a ^ b);
  right.value = mix(right.value, b + rotate_left(a, 23U));
  const ResultRecord result{node.id, time.wave, left.value, right.value};
  const EmissionRecord emission{node.id, mix(a, b)};
  result_lane.store(result);
  emission_lane.store(emission);
  if (context.scratch().size() >= sizeof(std::uint64_t))
    context.scratch().store(a ^ b);
}

std::uint64_t checksum(const CpuEncounterExecutor& executor,
                       const KernelState& state) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const EncounterNode& node : executor.dag().nodes()) {
    const ResultRecord result =
        executor.result_lane(node.id).load<ResultRecord>();
    const EmissionRecord emission =
        executor.emission_lane(node.id).load<EmissionRecord>();
    hash = mix(hash, result.id);
    hash = mix(hash, result.wave);
    hash = mix(hash, result.beam1);
    hash = mix(hash, result.beam2);
    hash = mix(hash, emission.id);
    hash = mix(hash, emission.value);
  }
  for (const SliceState& value : state.beam1) hash = mix(hash, value.value);
  for (const SliceState& value : state.beam2) hash = mix(hash, value.value);
  return hash;
}

std::uint64_t parse_u64(const std::string& text, const char* description) {
  if (text.empty() || text[0] == '-')
    throw std::invalid_argument(std::string(description) + " must be unsigned");
  std::size_t consumed = 0;
  const unsigned long long value = std::stoull(text, &consumed, 0);
  if (consumed != text.size())
    throw std::invalid_argument(std::string(description) + " is not an integer");
  return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const std::string& text, const char* description) {
  const std::uint64_t value = parse_u64(text, description);
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::out_of_range(std::string(description) + " exceeds uint32_t");
  return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc)
      throw std::invalid_argument("every benchmark option requires a value");
    const std::string value = argv[++index];
    if (argument == "--preset") options.preset = value;
    else if (argument == "--output-json") options.json_path = value;
    else if (argument == "--output-csv") options.csv_path = value;
    else if (argument == "--metadata-json") options.metadata_path = value;
    else if (argument == "--warmups")
      options.warmups = parse_u32(value, "warmups");
    else if (argument == "--repetitions")
      options.repetitions = parse_u32(value, "repetitions");
    else if (argument == "--worker-capacity")
      options.worker_capacity = parse_u32(value, "worker capacity");
    else if (argument == "--seed") options.seed = parse_u64(value, "seed");
    else if (argument == "--git-revision") options.git_revision = value;
    else if (argument == "--compiler-command") options.compiler_command = value;
    else if (argument == "--compile-flags") options.compile_flags = value;
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.preset != "smoke" && options.preset != "full")
    throw std::invalid_argument("--preset must be smoke or full");
  if (options.json_path.empty() || options.csv_path.empty() ||
      options.metadata_path.empty())
    throw std::invalid_argument("JSON, CSV, and metadata paths are required");
  if (options.warmups == 0 || options.repetitions < 3)
    throw std::invalid_argument("at least one warmup and three repetitions are required");
  if (options.worker_capacity == 0 || options.worker_capacity > 10)
    throw std::invalid_argument("--worker-capacity must be in [1,10]");
  if (options.json_path == options.csv_path)
    throw std::invalid_argument("JSON and CSV output paths must differ");
  return options;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open metadata file: " + path.string());
  std::ostringstream content;
  content << stream.rdbuf();
  if (!stream.good() && !stream.eof())
    throw std::runtime_error("cannot read metadata file: " + path.string());
  std::string result = content.str();
  const auto first = result.find_first_not_of(" \t\r\n");
  const auto last = result.find_last_not_of(" \t\r\n");
  if (first == std::string::npos || result[first] != '{' || result[last] != '}')
    throw std::runtime_error("metadata file is not a JSON object");
  return result.substr(first, last - first + 1);
}

std::string json_escape(const std::string& input) {
  std::ostringstream stream;
  for (const unsigned char value : input) {
    switch (value) {
      case '"': stream << "\\\""; break;
      case '\\': stream << "\\\\"; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (value < 0x20U) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(value) << std::dec;
        } else {
          stream << static_cast<char>(value);
        }
    }
  }
  return stream.str();
}

std::string hex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

std::vector<Shape> shapes(const std::string& preset) {
  if (preset == "smoke") return {{"smoke_24x24_t3", 24, 24, 3}};
  return {{"legacy_64x64_t5", 64, 64, 5},
          {"unequal_96x48_t5", 96, 48, 5},
          {"wide_128x128_t5", 128, 128, 5}};
}

std::vector<Intensity> intensities(const std::string& preset) {
  if (preset == "smoke") return {{"light", 8}, {"medium", 64}};
  return {{"light", 8}, {"medium", 64}, {"heavy", 512}};
}

std::vector<Variant> variants(std::uint32_t worker_capacity) {
  std::vector<Variant> result = {
      {"serial_ref", CpuExecutionKind::SerialReference, 1}};
  for (const std::uint32_t workers : {1U, 2U, 4U, 8U, 10U}) {
    if (workers <= worker_capacity)
      result.push_back({"openmp_" + std::to_string(workers),
                        CpuExecutionKind::OpenMP, workers});
  }
  return result;
}

std::vector<std::size_t> interleaved_order(std::size_t count,
                                           std::uint64_t seed,
                                           std::uint64_t case_index,
                                           std::uint64_t phase,
                                           std::uint64_t repetition) {
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::seed_seq sequence{
      static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32U),
      static_cast<std::uint32_t>(case_index), static_cast<std::uint32_t>(phase),
      static_cast<std::uint32_t>(repetition)};
  std::mt19937 generator(sequence);
  std::shuffle(order.begin(), order.end(), generator);
  return order;
}

void validate_report(const CpuExecutionReport& report, const Variant& variant,
                     std::uint64_t encounters) {
  if (!report.success || report.used_serial_fallback ||
      report.completed_encounters != encounters ||
      report.attempted_encounters != encounters ||
      report.requested_workers != variant.workers ||
      report.actual_workers != variant.workers)
    throw std::runtime_error("executor report failed completeness/team checks for " +
                             variant.name);
  const std::uint32_t expected_regions =
      variant.execution == CpuExecutionKind::OpenMP ? 1U : 0U;
  if (report.parallel_regions != expected_regions)
    throw std::runtime_error("executor report has an unexpected parallel-region count for " +
                             variant.name);
}

Sample execute_once(ExecutorHolder& holder, std::uint32_t repetition,
                    std::uint32_t order_index) {
  holder.state.reset();
  const auto begin = std::chrono::steady_clock::now();
  const CpuExecutionReport report =
      holder.executor->execute(kernel_callback, &holder.state);
  const auto end = std::chrono::steady_clock::now();
  validate_report(report, holder.aggregate.variant,
                  holder.executor->dag().encounter_count());
  const std::uint64_t digest = checksum(*holder.executor, holder.state);
  const double seconds = std::chrono::duration<double>(end - begin).count();
  if (!std::isfinite(seconds) || seconds <= 0.0)
    throw std::runtime_error("non-positive or non-finite timing sample");
  return Sample{repetition, order_index, seconds, digest, report};
}

double percentile(std::vector<double> values, double probability) {
  if (values.empty()) throw std::runtime_error("cannot summarize empty samples");
  std::sort(values.begin(), values.end());
  const std::size_t index = std::min<std::size_t>(
      values.size() - 1,
      static_cast<std::size_t>(std::ceil(probability * values.size())) - 1U);
  return values[index];
}

double median(std::vector<double> values) {
  if (values.empty()) throw std::runtime_error("cannot summarize empty samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if (values.size() % 2U != 0U) return values[middle];
  return (values[middle - 1U] + values[middle]) / 2.0;
}

BindingMetadata verify_binding(std::uint32_t worker_capacity) {
#if !defined(_OPENMP) || !defined(__linux__)
  throw std::runtime_error("benchmark requires Linux and an OpenMP build");
#else
  omp_set_dynamic(0);
  omp_set_max_active_levels(1);
  BindingMetadata metadata;
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    throw std::runtime_error("sched_getaffinity failed: " +
                             std::string(std::strerror(errno)));
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) metadata.process_cpus.push_back(cpu);
  metadata.worker_cpus.assign(worker_capacity, -1);
  metadata.worker_places.assign(worker_capacity, -1);
  int actual = 0;
#pragma omp parallel num_threads(worker_capacity) shared(actual, metadata)
  {
#pragma omp single
    actual = omp_get_num_threads();
    const int thread = omp_get_thread_num();
    metadata.worker_cpus[static_cast<std::size_t>(thread)] = sched_getcpu();
    metadata.worker_places[static_cast<std::size_t>(thread)] = omp_get_place_num();
  }
  if (actual != static_cast<int>(worker_capacity))
    throw std::runtime_error("OpenMP binding probe did not create the requested workers");
  std::vector<int> unique = metadata.worker_cpus;
  std::sort(unique.begin(), unique.end());
  if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
    throw std::runtime_error("OpenMP workers do not occupy distinct physical CPUs");
  metadata.omp_num_places = omp_get_num_places();
  if (metadata.omp_num_places != static_cast<int>(worker_capacity))
    throw std::runtime_error("OpenMP place count differs from worker capacity");
  std::vector<int> place_cpus;
  place_cpus.reserve(worker_capacity);
  for (int place = 0; place < metadata.omp_num_places; ++place) {
    const int count = omp_get_place_num_procs(place);
    if (count != 1)
      throw std::runtime_error("each OpenMP place must contain exactly one CPU");
    int cpu = -1;
    omp_get_place_proc_ids(place, &cpu);
    place_cpus.push_back(cpu);
  }
  std::vector<int> sorted_place_cpus = place_cpus;
  std::sort(sorted_place_cpus.begin(), sorted_place_cpus.end());
  if (sorted_place_cpus != unique)
    throw std::runtime_error("OpenMP worker CPUs do not match configured places");
  for (std::size_t worker = 0; worker < metadata.worker_places.size(); ++worker) {
    const int place = metadata.worker_places[worker];
    if (place < 0 || place >= metadata.omp_num_places ||
        metadata.worker_cpus[worker] !=
            place_cpus[static_cast<std::size_t>(place)])
      throw std::runtime_error(
          "OpenMP worker is not resident in its reported place");
  }
  switch (omp_get_proc_bind()) {
    case omp_proc_bind_false: metadata.proc_bind = "false"; break;
    case omp_proc_bind_true: metadata.proc_bind = "true"; break;
    case omp_proc_bind_master: metadata.proc_bind = "master"; break;
    case omp_proc_bind_close: metadata.proc_bind = "close"; break;
    case omp_proc_bind_spread: metadata.proc_bind = "spread"; break;
    default: metadata.proc_bind = "unknown"; break;
  }
  if (metadata.proc_bind == "false" || metadata.proc_bind == "unknown")
    throw std::runtime_error("OpenMP processor binding is disabled or unknown");
  return metadata;
#endif
}

CaseResult run_case(const Options& options, const Shape& shape,
                    const Intensity& intensity, std::uint64_t case_index) {
  EncounterDAG source = EncounterDAG::build_legacy(
      shape.beam1, shape.beam2, shape.substeps);
  CaseResult result;
  result.shape = shape;
  result.intensity = intensity;
  result.encounters = source.encounter_count();
  result.waves = source.wave_count();

  std::vector<ExecutorHolder> holders;
  const std::vector<Variant> selected_variants = variants(options.worker_capacity);
  holders.reserve(selected_variants.size());
  for (const Variant& variant : selected_variants) {
    CpuExecutorConfig config;
    config.execution = variant.execution;
    config.strategy = CpuScheduleStrategy::Static;
    config.context_reset = CpuContextResetPolicy::Preserve;
    config.worker_count = variant.workers;
    config.context_count = variant.workers;
    config.context_bytes = 64;
    config.result_lane_bytes = sizeof(ResultRecord);
    config.emission_lane_bytes = sizeof(EmissionRecord);
    const auto begin = std::chrono::steady_clock::now();
    auto executor = std::make_unique<CpuEncounterExecutor>(source, config);
    const auto end = std::chrono::steady_clock::now();
    Aggregate aggregate;
    aggregate.variant = variant;
    aggregate.construction_seconds =
        std::chrono::duration<double>(end - begin).count();
    aggregate.preflight_bytes = executor->preflight_bytes();
    holders.emplace_back(std::move(aggregate), std::move(executor),
                         KernelState(shape.beam1, shape.beam2,
                                     intensity.iterations));
  }

  bool oracle_set = false;
  for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
    const auto order = interleaved_order(holders.size(), options.seed,
                                         case_index, 0, warmup);
    for (std::size_t position = 0; position < order.size(); ++position) {
      ExecutorHolder& holder = holders[order[position]];
      const Sample sample = execute_once(holder, warmup,
                                         static_cast<std::uint32_t>(position));
      if (holder.aggregate.variant.name == "serial_ref") {
        if (!oracle_set) {
          result.oracle_checksum = sample.checksum;
          oracle_set = true;
        } else if (sample.checksum != result.oracle_checksum) {
          throw std::runtime_error("serial warmup checksum is not stable");
        }
      }
    }
  }
  if (!oracle_set) throw std::runtime_error("serial oracle warmup was not executed");

  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    const auto order = interleaved_order(holders.size(), options.seed,
                                         case_index, 1, repetition);
    for (std::size_t position = 0; position < order.size(); ++position) {
      ExecutorHolder& holder = holders[order[position]];
      Sample sample = execute_once(holder, repetition,
                                   static_cast<std::uint32_t>(position));
      if (sample.checksum != result.oracle_checksum)
        throw std::runtime_error("exact checksum mismatch for " +
                                 holder.aggregate.variant.name);
      holder.aggregate.samples.push_back(sample);
    }
  }

  for (ExecutorHolder& holder : holders) {
    Aggregate& aggregate = holder.aggregate;
    if (aggregate.samples.size() != options.repetitions)
      throw std::runtime_error("incomplete timing sample set");
    aggregate.checksum = aggregate.samples.front().checksum;
    aggregate.deterministic_checksum_match =
        aggregate.checksum == result.oracle_checksum;
    std::vector<double> times;
    times.reserve(aggregate.samples.size());
    for (const Sample& sample : aggregate.samples) {
      if (sample.checksum != aggregate.checksum)
        throw std::runtime_error("variant checksum changed between repetitions");
      times.push_back(sample.seconds);
    }
    aggregate.median = median(times);
    aggregate.p95 = percentile(times, 0.95);
    aggregate.minimum = *std::min_element(times.begin(), times.end());
    aggregate.maximum = *std::max_element(times.begin(), times.end());
  }
  const double serial_median = holders.front().aggregate.median;
  for (ExecutorHolder& holder : holders) {
    holder.aggregate.speedup = serial_median / holder.aggregate.median;
    result.aggregates.push_back(std::move(holder.aggregate));
  }
  return result;
}

void write_int_array(std::ostream& stream, const std::vector<int>& values) {
  stream << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ',';
    stream << values[index];
  }
  stream << ']';
}

void write_json(std::ostream& stream, const Options& options,
                const std::string& host_metadata,
                const BindingMetadata& binding,
                const std::vector<CaseResult>& cases) {
  stream << std::setprecision(17);
  stream << "{\n"
         << "  \"schema\": \"" << kSchema << "\",\n"
         << "  \"valid\": false,\n"
         << "  \"validation_status\": \"unvalidated_staging\",\n"
         << "  \"classification\": \"exploratory_microkernel\",\n"
         << "  \"eligible_for_project_level_2x_claim\": false,\n"
         << "  \"disclaimer\": \"Executor-only microkernel timings exclude FeeBL physics, field solves, data conversion, I/O, and end-to-end initialization; they cannot support the project-level 2x claim.\",\n"
         << "  \"oracle_scope\": \"SerialReference is a same-implementation scheduling oracle; checksum agreement demonstrates deterministic self-consistency, not independent physics correctness.\",\n"
         << "  \"timing_scope\": \"CpuEncounterExecutor::execute only; state reset, checksum, construction, and serialization excluded\",\n"
         << "  \"timing_scope_id\": \"executor_execute_only\",\n"
         << "  \"preset\": \"" << json_escape(options.preset) << "\",\n"
         << "  \"warmups_per_variant_case\": " << options.warmups << ",\n"
         << "  \"repetitions_per_variant_case\": " << options.repetitions << ",\n"
         << "  \"interleaving\": \"deterministic Fisher-Yates order per phase/case/repetition\",\n"
         << "  \"interleaving_seed\": " << options.seed << ",\n"
         << "  \"git_revision\": \"" << json_escape(options.git_revision) << "\",\n"
         << "  \"compiler_command\": \"" << json_escape(options.compiler_command) << "\",\n"
         << "  \"compile_flags\": \"" << json_escape(options.compile_flags) << "\",\n"
         << "  \"openmp_macro\": " << _OPENMP << ",\n"
         << "  \"host_metadata\": " << host_metadata << ",\n"
         << "  \"runtime_binding\": {\n"
         << "    \"initial_thread_affinity_at_probe\": ";
  write_int_array(stream, binding.process_cpus);
  stream << ",\n    \"worker_cpus_at_probe\": ";
  write_int_array(stream, binding.worker_cpus);
  stream << ",\n    \"worker_places_at_probe\": ";
  write_int_array(stream, binding.worker_places);
  stream << ",\n    \"omp_proc_bind\": \"" << json_escape(binding.proc_bind) << "\",\n"
         << "    \"omp_num_places\": " << binding.omp_num_places << "\n"
         << "  },\n"
         << "  \"statistics\": {\"median\": \"standard median (mean of middle pair for even N)\", \"p95\": \"nearest-rank p95\"},\n"
         << "  \"cases\": [\n";
  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    const CaseResult& result = cases[case_index];
    if (case_index != 0) stream << ",\n";
    stream << "    {\n"
           << "      \"shape\": \"" << json_escape(result.shape.name) << "\",\n"
           << "      \"beam1_slices\": " << result.shape.beam1 << ",\n"
           << "      \"beam2_slices\": " << result.shape.beam2 << ",\n"
           << "      \"substeps\": " << result.shape.substeps << ",\n"
           << "      \"encounters\": " << result.encounters << ",\n"
           << "      \"waves\": " << result.waves << ",\n"
           << "      \"intensity\": \"" << json_escape(result.intensity.name) << "\",\n"
           << "      \"integer_mix_iterations\": " << result.intensity.iterations << ",\n"
           << "      \"oracle_checksum\": \"" << hex64(result.oracle_checksum) << "\",\n"
           << "      \"aggregates\": [\n";
    for (std::size_t aggregate_index = 0;
         aggregate_index < result.aggregates.size(); ++aggregate_index) {
      const Aggregate& aggregate = result.aggregates[aggregate_index];
      if (aggregate_index != 0) stream << ",\n";
      stream << "        {\n"
             << "          \"variant\": \"" << json_escape(aggregate.variant.name) << "\",\n"
             << "          \"execution\": \""
             << (aggregate.variant.execution == CpuExecutionKind::OpenMP ? "OpenMP" : "SerialReference") << "\",\n"
             << "          \"workers\": " << aggregate.variant.workers << ",\n"
             << "          \"schedule\": \"Static\",\n"
             << "          \"construction_seconds\": " << aggregate.construction_seconds << ",\n"
             << "          \"preflight_bytes\": " << aggregate.preflight_bytes << ",\n"
             << "          \"checksum\": \"" << hex64(aggregate.checksum) << "\",\n"
             << "          \"deterministic_checksum_match_same_implementation_oracle\": "
             << (aggregate.deterministic_checksum_match ? "true" : "false") << ",\n"
             << "          \"median_seconds\": " << aggregate.median << ",\n"
             << "          \"p95_seconds\": " << aggregate.p95 << ",\n"
             << "          \"min_seconds\": " << aggregate.minimum << ",\n"
             << "          \"max_seconds\": " << aggregate.maximum << ",\n"
             << "          \"exploratory_speedup_vs_serial_median\": " << aggregate.speedup << ",\n"
             << "          \"samples\": [";
      for (std::size_t sample_index = 0; sample_index < aggregate.samples.size();
           ++sample_index) {
        const Sample& sample = aggregate.samples[sample_index];
        if (sample_index != 0) stream << ',';
        stream << "\n            {\"repetition\": " << sample.repetition
               << ", \"order_index\": " << sample.order_index
               << ", \"seconds\": " << sample.seconds
               << ", \"checksum\": \"" << hex64(sample.checksum)
               << "\", \"report_success\": "
               << (sample.report.success ? "true" : "false")
               << ", \"used_serial_fallback\": "
               << (sample.report.used_serial_fallback ? "true" : "false")
               << ", \"attempted_encounters\": "
               << sample.report.attempted_encounters
               << ", \"completed_encounters\": "
               << sample.report.completed_encounters
               << ", \"published_encounters\": "
               << sample.report.published_encounters
               << ", \"requested_workers\": "
               << sample.report.requested_workers
               << ", \"actual_workers\": " << sample.report.actual_workers
               << ", \"parallel_regions\": " << sample.report.parallel_regions
               << '}';
      }
      stream << "\n          ]\n        }";
    }
    stream << "\n      ]\n    }";
  }
  stream << "\n  ]\n}\n";
}

std::string csv_quote(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
  std::string result = "\"";
  for (char character : value) {
    if (character == '"') result += '"';
    result += character;
  }
  return result + '"';
}

void write_csv(std::ostream& stream, const Options& options,
               const std::vector<CaseResult>& cases) {
  stream << std::setprecision(17);
  stream << "schema,valid,classification,claim_eligible,preset,shape,beam1_slices,beam2_slices,substeps,encounters,waves,intensity,iterations,variant,execution,workers,schedule,timing_scope,warmups,repetition,order_index,seconds,checksum,oracle_checksum,deterministic_checksum_match,report_success,used_serial_fallback,attempted_encounters,completed_encounters,published_encounters,requested_workers,actual_workers,parallel_regions,preflight_bytes,construction_seconds,median_seconds,p95_seconds,min_seconds,max_seconds,exploratory_speedup_vs_serial_median\n";
  for (const CaseResult& result : cases) {
    for (const Aggregate& aggregate : result.aggregates) {
      for (const Sample& sample : aggregate.samples) {
        stream << kSchema << ",false,exploratory_microkernel,false,"
               << csv_quote(options.preset) << ',' << csv_quote(result.shape.name)
               << ',' << result.shape.beam1 << ',' << result.shape.beam2 << ','
               << result.shape.substeps << ',' << result.encounters << ','
               << result.waves << ',' << csv_quote(result.intensity.name) << ','
               << result.intensity.iterations << ','
               << csv_quote(aggregate.variant.name) << ','
               << (aggregate.variant.execution == CpuExecutionKind::OpenMP
                       ? "OpenMP" : "SerialReference")
               << ',' << aggregate.variant.workers
               << ",Static,executor_execute_only," << options.warmups << ','
               << sample.repetition << ',' << sample.order_index << ','
               << sample.seconds << ',' << hex64(sample.checksum) << ','
               << hex64(result.oracle_checksum) << ','
               << (aggregate.deterministic_checksum_match ? "true" : "false")
               << ',' << (sample.report.success ? "true" : "false")
               << ',' << (sample.report.used_serial_fallback ? "true" : "false")
               << ',' << sample.report.attempted_encounters
               << ',' << sample.report.completed_encounters
               << ',' << sample.report.published_encounters
               << ',' << sample.report.requested_workers
               << ','
               << sample.report.actual_workers << ','
               << sample.report.parallel_regions << ','
               << aggregate.preflight_bytes << ','
               << aggregate.construction_seconds << ',' << aggregate.median << ','
               << aggregate.p95 << ',' << aggregate.minimum << ','
               << aggregate.maximum << ',' << aggregate.speedup << '\n';
      }
    }
  }
}

template <class Writer>
void write_checked(const std::filesystem::path& path, Writer writer) {
  std::ofstream stream(path, std::ios::out | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create output: " + path.string());
  writer(stream);
  stream.flush();
  if (!stream) throw std::runtime_error("failed writing output: " + path.string());
  stream.close();
  if (!stream) throw std::runtime_error("failed closing output: " + path.string());
}

void publish_outputs(const Options& options, const std::string& host_metadata,
                     const BindingMetadata& binding,
                     const std::vector<CaseResult>& cases) {
  if (std::filesystem::exists(options.json_path) ||
      std::filesystem::exists(options.csv_path))
    throw std::runtime_error("refusing to overwrite an existing result file");
  if (options.json_path.parent_path() != options.csv_path.parent_path())
    throw std::runtime_error("JSON and CSV outputs must share a directory");
  if (!std::filesystem::is_directory(options.json_path.parent_path()))
    throw std::runtime_error("output directory does not exist");
  const auto json_temp = options.json_path.string() + ".tmp";
  const auto csv_temp = options.csv_path.string() + ".tmp";
  if (std::filesystem::exists(json_temp) || std::filesystem::exists(csv_temp))
    throw std::runtime_error("refusing to overwrite an existing temporary output");
  try {
    write_checked(json_temp, [&](std::ostream& stream) {
      write_json(stream, options, host_metadata, binding, cases);
    });
    write_checked(csv_temp, [&](std::ostream& stream) {
      write_csv(stream, options, cases);
    });
    std::filesystem::rename(csv_temp, options.csv_path);
    try {
      std::filesystem::rename(json_temp, options.json_path);
    } catch (...) {
      std::filesystem::remove(options.csv_path);
      throw;
    }
  } catch (...) {
    std::filesystem::remove(json_temp);
    std::filesystem::remove(csv_temp);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::string host_metadata = read_file(options.metadata_path);
    const BindingMetadata binding = verify_binding(options.worker_capacity);
    std::vector<CaseResult> cases;
    std::uint64_t case_index = 0;
    for (const Shape& shape : shapes(options.preset)) {
      for (const Intensity& intensity : intensities(options.preset)) {
        std::cerr << "microkernel case " << shape.name << '/' << intensity.name
                  << " (not project-claim eligible)\n";
        cases.push_back(run_case(options, shape, intensity, case_index++));
      }
    }
    publish_outputs(options, host_metadata, binding, cases);
    std::cout << "wrote unvalidated exploratory staging JSON "
              << options.json_path << '\n'
              << "wrote unvalidated exploratory staging CSV "
              << options.csv_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "modern CPU executor benchmark failed closed: "
              << error.what() << '\n';
    return 1;
  }
}
