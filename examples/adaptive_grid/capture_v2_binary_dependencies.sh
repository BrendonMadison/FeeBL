#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <release-build-root> <evidence-root>\n' "$0" >&2
  exit 64
fi

readonly release_build_root="$(realpath "$1")"
readonly evidence_root="$(realpath "$2")"
readonly output_root="${evidence_root}/binary_dependency_evidence"

if [[ -e "${output_root}" ]]; then
  printf 'refusing to overwrite binary-dependency evidence: %s\n' \
    "${output_root}" >&2
  exit 73
fi
mkdir -p "${output_root}"

readonly -a binary_names=(
  feebl
  stage4_validator
  poisson_unit
  coupling_unit
  collision_unit
  stage4_asan
)
readonly -a binary_paths=(
  "${release_build_root}/src/feebl"
  "${release_build_root}/Testing/feebl_adaptive_composite_physics_validation"
  "${release_build_root}/Testing/feebl_adaptive_composite_poisson_unit"
  "${release_build_root}/Testing/feebl_adaptive_composite_coupling_unit"
  "${release_build_root}/Testing/feebl_electron_lumi_spatial_unit"
  "${evidence_root}/adaptive_composite_physics_validation_asan"
)

: > "${output_root}/binary_sha256.txt"
: > "${output_root}/resolved_dependency_sha256.txt"

for index in "${!binary_names[@]}"; do
  binary_name="${binary_names[${index}]}"
  binary_path="${binary_paths[${index}]}"
  if [[ ! -x "${binary_path}" ]]; then
    printf 'missing dependency-audit binary: %s\n' "${binary_path}" >&2
    exit 66
  fi

  printf '%s  %s\n' "$(sha256sum "${binary_path}" | awk '{print $1}')" \
    "${binary_name}" >> "${output_root}/binary_sha256.txt"
  ldd "${binary_path}" > "${output_root}/${binary_name}.ldd.txt"

  while IFS= read -r dependency; do
    printf '%s  %s  %s\n' \
      "$(sha256sum "${dependency}" | awk '{print $1}')" \
      "${binary_name}" "${dependency}"
  done < <(
    ldd "${binary_path}" | awk '
      /=> \/[^ ]+/ {print $3}
      /^\// {print $1}
    ' | LC_ALL=C sort -u
  ) >> "${output_root}/resolved_dependency_sha256.txt"
done

{
  printf 'python=%s\n' "$(python3 --version 2>&1)"
  printf 'python_dependency_note=V2 campaign and spectrum tools use the Python standard library only\n'
  printf 'pip_environment_begin\n'
  python3 -m pip freeze --all 2>&1 || true
  printf 'pip_environment_end\n'
  printf 'debian_packages_begin\n'
  dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' \
    cmake g++ gcc libfftw3-dev libfftw3-double3 libgomp1 libc6 libstdc++6 \
    2>&1 || true
  printf 'debian_packages_end\n'
} > "${output_root}/toolchain_packages.txt"

printf 'binary dependency capture complete for %d executables\n' \
  "${#binary_names[@]}"
