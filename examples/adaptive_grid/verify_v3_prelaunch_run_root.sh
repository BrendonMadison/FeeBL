#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s RUN_ROOT\n' "$0" >&2
  exit 64
fi

run_root="$1"
if [[ ! -d "$run_root" ]]; then
  printf '{"schema":"feebl.adaptive-convergence-v3.prelaunch-run-root.v1","status":"fail","reason":"run_root_missing"}\n' >&2
  exit 2
fi

canonical_root="$(realpath -e -- "$run_root")"
entry_count="$(find "$canonical_root" -mindepth 1 -print | wc -l)"
if [[ "$entry_count" -ne 0 ]]; then
  printf '{"schema":"feebl.adaptive-convergence-v3.prelaunch-run-root.v1","status":"fail","reason":"run_root_not_empty","entry_count":%s,"run_root":"%s"}\n' \
    "$entry_count" "$canonical_root" >&2
  exit 3
fi

printf '{"schema":"feebl.adaptive-convergence-v3.prelaunch-run-root.v1","status":"pass","entry_count":0,"persistent_campaign_seeds":0,"persistent_campaign_runs":0,"run_root":"%s"}\n' \
  "$canonical_root"
