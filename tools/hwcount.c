// Hardware-counter harness using perf_event_open(2) directly.
//
// `perf` is not installed and sudo needs a password, but neither is required:
// this WSL2 kernel exposes a real Skylake PMU (/sys/bus/event_source/devices/
// cpu/caps/pmu_name = skylake) and perf_event_paranoid=2 already permits
// user-space counting on one's own descendants. Only kernel profiling is
// blocked, which we do not need.
//
// Usage:  hwcount <program> [args...]
//
// The decisive question this answers: is FEEBL limited by DRAM *bandwidth* or
// by memory *latency*? Those have different fixes. Achieved bandwidth is
// estimated as (cache-misses x 64 B)/elapsed and compared against the
// platform's practical peak; a low IPC combined with bandwidth far below peak
// indicates latency-bound pointer chasing, not saturation.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <time.h>

static int perf_open(unsigned type, unsigned long long config, pid_t pid, int group)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.type = type;
    a.size = sizeof(a);
    a.config = config;
    a.disabled = (group == -1);
    a.exclude_kernel = 1;      // required at perf_event_paranoid = 2
    a.exclude_hv = 1;
    a.inherit = 1;             // follow the child
    a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(__NR_perf_event_open, &a, pid, -1, group, 0);
}

struct ctr { const char *name; unsigned type; unsigned long long config; int fd; double val; };

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: hwcount <program> [args...]\n"); return 2; }

    struct ctr c[] = {
        {"cycles",           PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES,        -1, 0},
        {"instructions",     PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS,      -1, 0},
        {"cache-references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES,  -1, 0},
        {"cache-misses",     PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES,      -1, 0},
        {"branch-insns",     PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS,-1, 0},
        {"branch-misses",    PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES,     -1, 0},
    };
    const int N = sizeof(c) / sizeof(c[0]);

    pid_t pid = fork();
    if (pid == 0) {
        // child: pause until the parent has the counters armed
        raise(SIGSTOP);
        execvp(argv[1], &argv[1]);
        perror("execvp");
        _exit(127);
    }
    // wait for the child to stop itself
    int st; waitpid(pid, &st, WUNTRACED);

    int nopen = 0;
    for (int i = 0; i < N; i++) {
        c[i].fd = perf_open(c[i].type, c[i].config, pid, -1);
        if (c[i].fd < 0)
            fprintf(stderr, "  (could not open %s: %s)\n", c[i].name, strerror(errno));
        else { ioctl(c[i].fd, PERF_EVENT_IOC_RESET, 0);
               ioctl(c[i].fd, PERF_EVENT_IOC_ENABLE, 0); nopen++; }
    }
    if (!nopen) { fprintf(stderr, "no counters available\n"); kill(pid, SIGKILL); return 1; }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    kill(pid, SIGCONT);
    waitpid(pid, &st, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    for (int i = 0; i < N; i++) {
        if (c[i].fd < 0) continue;
        unsigned long long v[3] = {0, 0, 0};
        if (read(c[i].fd, v, sizeof(v)) > 0) {
            // scale if the counter was multiplexed
            double scale = (v[2] > 0 && v[1] > 0) ? (double)v[1] / (double)v[2] : 1.0;
            c[i].val = (double)v[0] * scale;
        }
        close(c[i].fd);
    }

    printf("\n  elapsed              %12.3f s\n", el);
    for (int i = 0; i < N; i++)
        if (c[i].fd >= 0 || c[i].val > 0)
            printf("  %-20s %12.0f\n", c[i].name, c[i].val);

    double cyc = c[0].val, ins = c[1].val, cref = c[2].val, cmis = c[3].val;
    double bri = c[4].val, brm = c[5].val;
    printf("\n  derived:\n");
    if (cyc > 0) printf("    IPC                %12.3f\n", ins / cyc);
    if (cref > 0) printf("    cache miss rate    %11.2f%%   (misses/references)\n", 100.0 * cmis / cref);
    if (bri > 0) printf("    branch miss rate   %11.2f%%\n", 100.0 * brm / bri);
    if (el > 0 && cmis > 0) {
        double gbs = cmis * 64.0 / el / 1e9;
        printf("    est. DRAM traffic  %12.2f GB/s   (misses x 64 B / elapsed)\n", gbs);
        printf("    ... vs ~41 GB/s practical peak for dual-channel DDR4-2666: %.1f%% of peak\n",
               100.0 * gbs / 41.0);
    }
    if (cyc > 0) printf("    effective GHz      %12.2f\n", cyc / el / 1e9);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
