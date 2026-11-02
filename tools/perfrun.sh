#!/bin/bash
# Now that perf is available, three things worth doing that hwcount.c could not:
#
#  1. CROSS-VALIDATE hwcount.c against real perf on the same workload. The
#     work-bound diagnosis rests on my custom harness; if it was wrong, so is
#     the conclusion. This is the check that matters most.
#  2. Detailed cache breakdown (-d): L1 vs LLC, plus stall attribution. My
#     harness only had the generic cache-references/cache-misses pair, and I
#     ASSUMED cache-misses == DRAM traffic. Verify that against LLC-load-misses.
#  3. perf record sampling instead of gprof. gprof instruments every call and
#     distorts fully-inlined code -- and assignBeamSliceCIC is fully inlined
#     (129 instructions, zero calls), exactly the case gprof handles worst.
set -e
cd $HOME/flag/run
B=$HOME/feebl/src/feebl

echo "############ 1+2. perf stat -d, ILC 250 ############"
perf stat -d -e cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,\
stalled-cycles-frontend,stalled-cycles-backend,branch-misses \
  $B s9 p_s9 perf_s9.dat 2>&1 >/dev/null | tail -22

echo
echo "############ 3. perf record, ILC 250 ############"
perf record -q --call-graph=fp -F 299 -o perf_s9.data $B s9 p_s9 rec_s9.dat >/dev/null 2>&1 || true
echo "--- flat profile, top 16 by self time ---"
perf report -i perf_s9.data --stdio --no-children --percent-limit 0.5 2>/dev/null \
  | grep -E "^ +[0-9]" | head -16

echo
echo "############ perf stat -d, FCC-ee Z ############"
perf stat -d -e cycles,instructions,cache-misses,LLC-load-misses,\
stalled-cycles-backend $B s3 p_s3 perf_s3.dat 2>&1 >/dev/null | tail -14
