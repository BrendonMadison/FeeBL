#!/usr/bin/env bash
set -euo pipefail

# Exploratory executor-only microbenchmark. These results are deliberately
# ineligible for FeeBL's project-level 2x claim: no production physics, field
# solve, data conversion, initialization, or I/O path is measured.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
preset=full
output_dir=
warmups=
repetitions=
seed=1442695040888963407
requested_workers=10

usage() {
  echo "usage: $0 [--smoke|--full] [--output-dir DIR] [--warmups N] [--repetitions N] [--seed N]" >&2
}

while (($#)); do
  case "$1" in
    --smoke) preset=smoke; shift ;;
    --full) preset=full; shift ;;
    --output-dir)
      (($# >= 2)) || { usage; exit 2; }
      output_dir=$2; shift 2 ;;
    --warmups)
      (($# >= 2)) || { usage; exit 2; }
      warmups=$2; shift 2 ;;
    --repetitions)
      (($# >= 2)) || { usage; exit 2; }
      repetitions=$2; shift 2 ;;
    --seed)
      (($# >= 2)) || { usage; exit 2; }
      seed=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ${preset} == smoke ]]; then
  warmups=${warmups:-1}
  repetitions=${repetitions:-3}
else
  warmups=${warmups:-2}
  repetitions=${repetitions:-11}
fi
[[ ${warmups} =~ ^[1-9][0-9]*$ ]] || { echo "warmups must be positive" >&2; exit 2; }
[[ ${repetitions} =~ ^[0-9]+$ ]] && ((repetitions >= 3)) || {
  echo "repetitions must be at least 3" >&2
  exit 2
}
[[ ${seed} =~ ^[0-9]+$ ]] || { echo "seed must be unsigned" >&2; exit 2; }

for command in git ldd lscpu python3 taskset; do
  command -v "${command}" >/dev/null || {
    echo "required command is unavailable: ${command}" >&2
    exit 1
  }
done
read -r -a cxx_command <<<"${CXX:-g++}"
((${#cxx_command[@]} > 0)) && command -v "${cxx_command[0]}" >/dev/null || {
  echo "CXX compiler command is unavailable: ${CXX:-g++}" >&2
  exit 1
}

# The mutation suite runs before compilation or result-directory creation.
python3 "${script_dir}/test_modern_cpu_executor_benchmark_validator.py"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -z ${output_dir} ]]; then
  output_dir="${repo_root}/benchmark_results/modern_cpu_executor_${preset}_${timestamp}"
fi
if [[ -e ${output_dir} ]]; then
  echo "refusing to reuse existing output directory: ${output_dir}" >&2
  exit 1
fi
mkdir -p -- "${output_dir}"

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/feebl-modern-cpu-benchmark.XXXXXX")
cleanup() {
  rm -rf -- "${build_dir}"
}
trap cleanup EXIT

compile_flags=(
  -std=c++17 -O3 -DNDEBUG -fopenmp -pthread
  -Wall -Wextra -Wpedantic -Werror
  -I"${repo_root}/src"
)
benchmark_binary="${build_dir}/modern-cpu-executor-benchmark"
"${cxx_command[@]}" "${compile_flags[@]}" \
  "${repo_root}/src/encounterDAGCPP.cc" \
  "${repo_root}/src/modernCpuExecutorCPP.cc" \
  "${script_dir}/benchmark_modern_cpu_executor.cc" \
  -o "${benchmark_binary}"

compiler_version_output=$("${cxx_command[@]}" --version)
compiler_version=${compiler_version_output%%$'\n'*}
compiler_invocation="${cxx_command[*]}"
compile_description="${compile_flags[*]}; ${compiler_version}"
git_revision=$(git -C "${repo_root}" rev-parse HEAD)

omp_dynamic=FALSE
omp_thread_limit=${requested_workers}
omp_max_active_levels=1
omp_proc_bind=close
omp_wait_policy=ACTIVE
omp_schedule=static

metadata_json="${build_dir}/host-metadata.json"
mapfile -t affinity_lines < <(
  python3 "${script_dir}/modern_cpu_executor_benchmark_support.py" affinity \
    --requested-workers "${requested_workers}" \
    --metadata "${metadata_json}" \
    --binary "${benchmark_binary}" \
    --git-root "${repo_root}" \
    --compiler "${compiler_invocation}" \
    --hash-file "source_encounter_dag_cc=${repo_root}/src/encounterDAGCPP.cc" \
    --hash-file "source_encounter_dag_h=${repo_root}/src/encounterDAGCPP.h" \
    --hash-file "source_executor_cc=${repo_root}/src/modernCpuExecutorCPP.cc" \
    --hash-file "source_executor_h=${repo_root}/src/modernCpuExecutorCPP.h" \
    --hash-file "source_modern_types_h=${repo_root}/src/modernTypesCPP.h" \
    --hash-file "harness_cpp=${script_dir}/benchmark_modern_cpu_executor.cc" \
    --hash-file "harness_runner=${script_dir}/run_modern_cpu_executor_benchmark.sh" \
    --hash-file "harness_validator=${script_dir}/modern_cpu_executor_benchmark_support.py" \
    --hash-file "harness_validator_tests=${script_dir}/test_modern_cpu_executor_benchmark_validator.py" \
    --openmp-env "OMP_DYNAMIC=${omp_dynamic}" \
    --openmp-env "OMP_THREAD_LIMIT=${omp_thread_limit}" \
    --openmp-env "OMP_MAX_ACTIVE_LEVELS=${omp_max_active_levels}" \
    --openmp-env "OMP_PROC_BIND=${omp_proc_bind}" \
    --openmp-env "OMP_WAIT_POLICY=${omp_wait_policy}" \
    --openmp-env "OMP_SCHEDULE=${omp_schedule}" \
    --openmp-env "GOMP_CPU_AFFINITY=<unset>" \
    --openmp-env "KMP_AFFINITY=<unset>"
)
if ((${#affinity_lines[@]} != 3)); then
  echo "physical-core affinity helper returned malformed output" >&2
  exit 1
fi
physical_cpus=${affinity_lines[0]}
omp_places=${affinity_lines[1]}
worker_capacity=${affinity_lines[2]}
[[ ${worker_capacity} =~ ^[1-9][0-9]*$ ]] && ((worker_capacity <= requested_workers)) || {
  echo "invalid worker capacity returned by affinity helper" >&2
  exit 1
}

staging_json="${build_dir}/modern_cpu_executor_microbenchmark.staging.json"
staging_csv="${build_dir}/modern_cpu_executor_microbenchmark.staging.csv"
json_path="${output_dir}/modern_cpu_executor_microbenchmark.json"
csv_path="${output_dir}/modern_cpu_executor_microbenchmark.csv"

echo "Exploratory microkernel only; NOT eligible for FeeBL's project-level 2x claim."
echo "Same-implementation checksum oracle only; not independent physics validation."
echo "Pinned physical CPUs: ${physical_cpus}"
echo "Worker capacity=${worker_capacity}; retaining supported counts from 1/2/4/8/10."
echo "Preset=${preset}, warmups=${warmups}, repetitions=${repetitions}, seed=${seed}"

taskset --cpu-list "${physical_cpus}" env \
  -u GOMP_CPU_AFFINITY -u KMP_AFFINITY \
  OMP_DYNAMIC="${omp_dynamic}" \
  OMP_THREAD_LIMIT="${omp_thread_limit}" \
  OMP_NUM_THREADS="${worker_capacity}" \
  OMP_MAX_ACTIVE_LEVELS="${omp_max_active_levels}" \
  OMP_PROC_BIND="${omp_proc_bind}" \
  OMP_PLACES="${omp_places}" \
  OMP_WAIT_POLICY="${omp_wait_policy}" \
  OMP_SCHEDULE="${omp_schedule}" \
  "${benchmark_binary}" \
    --preset "${preset}" \
    --output-json "${staging_json}" \
    --output-csv "${staging_csv}" \
    --metadata-json "${metadata_json}" \
    --warmups "${warmups}" \
    --repetitions "${repetitions}" \
    --worker-capacity "${worker_capacity}" \
    --seed "${seed}" \
    --git-revision "${git_revision}" \
    --compiler-command "${compiler_invocation}" \
    --compile-flags "${compile_description}"

# The C++ program can emit only valid=false staging files. The independent
# validator recomputes all statistics and invariants before it creates the
# valid=true final pair. JSON is renamed last and is the publication marker.
python3 "${script_dir}/modern_cpu_executor_benchmark_support.py" validate \
  --json "${staging_json}" --csv "${staging_csv}" \
  --repetitions "${repetitions}" --expected-valid false \
  --promote-json "${json_path}" --promote-csv "${csv_path}"
python3 "${script_dir}/modern_cpu_executor_benchmark_support.py" validate \
  --json "${json_path}" --csv "${csv_path}" \
  --repetitions "${repetitions}" --expected-valid true

printf 'JSON: %s\nCSV:  %s\n' "${json_path}" "${csv_path}"
