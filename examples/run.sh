#!/bin/bash
# FEEBL regression runner.
#
#   ./examples/run.sh quick                     fast suite, every commit (~30 s)
#   ./examples/run.sh benchmark16 -j 16         full suite in parallel   (~1 h)
#   ./examples/run.sh quick -b /path/to/feebl   run the suite with another binary
#   ./examples/run.sh quick --exact /path/to/x  BIT-IDENTICAL differential test:
#                                            run both binaries and require their
#                                            output to match byte for byte
#
# --exact is the real differential check. The tolerance modes compare ONE
# scalar against a golden value; --exact compares the entire output of two
# binaries, excluding only the banner (which carries the program name and
# version) and the timing line.
#
# Golden values are what the code actually produced. Tolerance is the measured
# seed-to-seed scatter of the code itself (1.6%), not a guess.
set -u
SUITE="quick"; JOBS=1; BIN=""; EXACT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS="$2"; shift 2;;
    -b) BIN="$2";  shift 2;;
    --exact) EXACT="$2"; shift 2;;
    -*) echo "unknown option: $1"; exit 2;;
    *)  SUITE="$1"; shift;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -n "$BIN" ] || BIN="$HERE/../src/feebl"
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
GOLD="$HERE/golden/$SUITE.tsv"
case "$SUITE" in quick) DECK="$HERE/decks/quick.dat";; *) DECK="$HERE/decks/benchmark16.dat";; esac
[ -f "$GOLD" ] || { echo "no such suite: $SUITE"; exit 2; }
[ -x "$BIN"  ] || { echo "no binary: $BIN"; exit 2; }

if [ -n "$EXACT" ]; then
  EXACT="$(cd "$(dirname "$EXACT")" && pwd)/$(basename "$EXACT")"
  [ -x "$EXACT" ] || { echo "no binary: $EXACT"; exit 2; }
  echo "differential (bit-identical) test"
  echo "  A: $BIN"
  echo "  B: $EXACT"
  echo
  W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
  cp "$DECK" "$W/acc.dat"; cd "$W"
  # Strip the banner: it carries the program name and version, which differ by
  # design. Everything below it must match byte for byte.
  strip () { sed -n '/^ \*\{10,\}/,$p' "$1" | grep -v "^ *\*" ; }

  pair () {                       # tag name
    local tag="$1" name="$2" st en
    st=$(date +%s)
    # Both statuses matter. A binary that dies at the dynamic loader writes no
    # output at all, and comparing two absent files must not read as agreement.
    "$BIN"   "$tag" "p_$tag" "a_$tag.dat" >/dev/null 2>&1; local ra=$?
    "$EXACT" "$tag" "p_$tag" "b_$tag.dat" >/dev/null 2>&1; local rb=$?
    en=$(date +%s)
    if [ "$ra" -ne 0 ] || [ "$rb" -ne 0 ]; then
      printf "  %-26s RUN FAILED     (exit %s / %s)\n" "$name" "$ra" "$rb" > "x_$tag"
      return
    fi
    if strip "a_$tag.dat" > "A_$tag" 2>/dev/null \
       && strip "b_$tag.dat" > "B_$tag" 2>/dev/null \
       && [ -s "A_$tag" ] && cmp -s "A_$tag" "B_$tag"; then
      printf "  %-26s BIT-IDENTICAL  %5s lines  %5ss\n" \
             "$name" "$(wc -l < "A_$tag")" "$((en-st))" > "x_$tag"
    else
      { printf "  %-26s *** DIFFERS ***  %5ss\n" "$name" "$((en-st))"
        diff "A_$tag" "B_$tag" 2>/dev/null | head -6 | sed 's/^/      /'
      } > "x_$tag"
    fi
  }

  running=0; tags=""
  while IFS=$'\t' read -r tag name want tol; do
    case "$tag" in \#*|"") continue;; esac
    tags="$tags $tag"
    pair "$tag" "$name" &
    running=$((running+1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done < "$GOLD"
  wait

  # A worker that died wrote no result file. Skipping it silently means that
  # when EVERY worker dies the loop counts nothing, prints "0/0" and exits 0 --
  # a total failure reported as success. Count missing results as failures,
  # including dynamic-loader errors before the simulation starts.
  ntot=0; nfail=0
  for tag in $tags; do
    if [ -f "x_$tag" ]; then
      cat "x_$tag"; ntot=$((ntot+1))
      grep -q "BIT-IDENTICAL" "x_$tag" || nfail=$((nfail+1))
    else
      printf "  %-26s NO RESULT -- the scenario produced nothing\n" "$tag"
      ntot=$((ntot+1)); nfail=$((nfail+1))
    fi
  done
  echo
  echo "$((ntot-nfail))/$ntot bit-identical"
  [ "$ntot" -gt 0 ] || { echo "  no scenario ran at all"; exit 1; }
  exit $([ "$nfail" -eq 0 ] && echo 0 || echo 1)
fi

echo "suite  : $SUITE"
echo "binary : $BIN"
echo "jobs   : $JOBS"
echo

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cp "$DECK" "$WORK/acc.dat"; cd "$WORK"

one () {   # tag name want tol
  local tag="$1" name="$2" want="$3" tol="$4" st en
  st=$(date +%s)
  "$BIN" "$tag" "p_$tag" "o_$tag.dat" >/dev/null 2>"e_$tag"
  rc=$?
  en=$(date +%s)
  if [ "$rc" -ne 0 ]; then
    printf "%-24s %-13s %-13s %8s %6ss  %s\n" "$name" "$want" "--" "--" \
      "$((en-st))" "EXIT $rc: $(head -1 "e_$tag" 2>/dev/null)" > "r_$tag"
    return
  fi
  local got; got=$(grep -m1 "lumi\[0\]\[1\]" "o_$tag.dat" 2>/dev/null | sed 's/.*= *//')
  if [ -z "$got" ]; then
    printf "%-24s %-13s %-13s %8s %6ss  %s\n" "$name" "$want" "--" "--" "$((en-st))" "NO OUTPUT" > "r_$tag"
    return
  fi
  local dev ok
  dev=$(awk -v a="$got" -v b="$want" 'BEGIN{printf "%+.4f",(a-b)/b}')
  ok=$(awk -v d="$dev" -v t="$tol" 'BEGIN{d=(d<0?-d:d); print (d<=t)?"PASS":"FAIL"}')
  printf "%-24s %-13s %-13s %8s %6ss  %s\n" "$name" "$want" "$got" "$dev" "$((en-st))" "$ok" > "r_$tag"
}

running=0; tags=""
while IFS=$'\t' read -r tag name want tol; do
  case "$tag" in \#*|"") continue;; esac
  tags="$tags $tag"
  one "$tag" "$name" "$want" "$tol" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done < "$GOLD"
wait

printf "%-24s %-13s %-13s %8s %7s  %s\n" scenario expected got dev time result
echo "----------------------------------------------------------------------------------------"
# Same hazard as the differential path above: a scenario whose worker died
# leaves no r_<tag>, and skipping it silently turns a total failure into "0/0
# passed" with a zero exit status.
n=0; fail=0
for tag in $tags; do
  if [ -f "r_$tag" ]; then
    cat "r_$tag"; n=$((n+1))
    grep -q "PASS" "r_$tag" || fail=$((fail+1))
  else
    printf "%-24s %-13s %-13s %8s %7s  %s\n" "$tag" "-" "-" "-" "-" "NO RESULT"
    n=$((n+1)); fail=$((fail+1))
  fi
done
echo "----------------------------------------------------------------------------------------"
echo "$((n-fail))/$n passed"
[ "$n" -gt 0 ] || { echo "no scenario ran at all"; exit 1; }
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
