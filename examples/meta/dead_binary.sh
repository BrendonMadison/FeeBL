#!/bin/bash
# Verify that simulation test wrappers propagate executable failures.
# Replace FeeBL with a recording stub in a temporary source copy and discover
# the shell suites. Scripts that call the stub must return nonzero. Standalone
# unit tests that never invoke FeeBL are reported separately.
#
# --quick skips the long refinement scripts.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"   # examples/meta/ -> repo root

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# Scripts excluded from the sweep, with the reason. Anything excluded here is
# NOT covered, so keep the list short and justified.
#   bitcheck      -- needs a v0.5.0 reference binary, absent in a bare checkout
#   reprocheck    -- copies the tree and swaps binaries; would fight this harness
#   dead_binary   -- this file
#
# Include suite aggregators to check propagation of member failures. Exclude
# run_all.sh because it invokes this meta-test recursively and writes results.
SKIP_ALWAYS='bitcheck\.sh$|reprocheck\.sh$|dead_binary\.sh$|run_all\.sh$'
# Long ones: full FCC-ee refinements, hours each.
SKIP_QUICK='s3_convergence\.sh$|s3_headon_control\.sh$|converge\.sh$'

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -a "$ROOT" "$W/tree"

export FEEBL_STUB_LOG="$W/invoked.log"
STUB="$W/tree/src/feebl"
cat > "$STUB" <<'STUBEOF'
#!/bin/sh
if [ -n "${FEEBL_STUB_LOG:-}" ]; then echo "called" >> "$FEEBL_STUB_LOG"; fi
echo "DEAD-BINARY STUB: deliberate failure" >&2
exit 7
STUBEOF
chmod +x "$STUB"
# feeblroot links from the same objects; stub it too so nothing falls back to it.
if [ -e "$W/tree/src/feeblroot" ]; then cp "$STUB" "$W/tree/src/feeblroot"; fi

# Reserve the token FAIL for verdicts consumed by aggregate reporters.
echo "=== every suite that runs FEEBL must report an error when FEEBL does ==="
if [ "$QUICK" = 1 ]; then echo "    (--quick: long refinement scripts skipped)"; fi
echo

false_pass=0
ran=0
skipped=0
not_applicable=0

while IFS= read -r rel; do
  case "$rel" in
    *lib/*) continue ;;                       # helpers, not suites
  esac
  if printf '%s' "$rel" | grep -Eq "$SKIP_ALWAYS"; then
    skipped=$((skipped+1)); continue
  fi
  if [ "$QUICK" = 1 ] && printf '%s' "$rel" | grep -Eq "$SKIP_QUICK"; then
    printf "  %-46s skipped (--quick)\n" "$rel"
    skipped=$((skipped+1)); continue
  fi

  rc=0
  : > "$FEEBL_STUB_LOG"
  # Isolate suite stdin from the discovery stream so a child cannot consume
  # filenames and silently truncate coverage.
  ( cd "$W/tree" && timeout 600 bash "$rel" >/dev/null 2>&1 </dev/null ) || rc=$?

  if [ ! -s "$FEEBL_STUB_LOG" ]; then
    printf "  %-46s n/a (never invokes FEEBL)\n" "$rel"
    not_applicable=$((not_applicable+1))
    continue
  fi

  ran=$((ran+1))
  if [ "$rc" -eq 0 ]; then
    printf "  %-46s *** FALSE PASS (exit 0) ***\n" "$rel"
    false_pass=$((false_pass+1))
  else
    printf "  %-46s ok (exit %s)\n" "$rel" "$rc"
  fi
done < <(cd "$W/tree" && find tests -name '*.sh' -type f | sort)

echo
echo "  ran FEEBL: $ran   never did: $not_applicable   skipped: $skipped   false passes: $false_pass"

if [ "$false_pass" -ne 0 ]; then
  echo
  echo "  A suite that passes without a working binary is not a gate. Fix it by"
  echo "  sourcing examples/lib/harness.sh and using feebl_run + need_num, which"
  echo "  abort rather than returning an empty string nobody checks."
  exit 1
fi

# Non-vacuity: if almost nothing actually exercised the binary, this file proved
# nothing and must not report success.
# Discovery must have seen every .sh under examples/ except the lib helpers.
expected=$(cd "$W/tree" && find tests -name '*.sh' -type f -not -path '*lib/*' | wc -l)
accounted=$((ran + not_applicable + skipped))
if [ "$accounted" -ne "$expected" ]; then
  echo "  ERROR: $accounted scripts accounted for, but $expected exist."
  echo "  The sweep truncated. See the </dev/null note above."
  exit 3
fi

if [ "$ran" -lt 5 ]; then
  echo "  ERROR: only $ran scripts actually ran FEEBL -- discovery is broken, and"
  echo "  this test would report success while checking almost nothing."
  exit 2
fi
echo "  every suite that runs FEEBL fails when FEEBL does."
