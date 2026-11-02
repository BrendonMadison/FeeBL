"use strict";
const assert = require("node:assert/strict"),
    stats = require("./collider_statistics.js");
let checks = 0;

function equal(a, b) {
    assert.deepEqual(a, b);
    checks++
}

function near(a, b, tolerance = 2e-13) {
    assert(Number.isFinite(a) && Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(
        b)), `${a} != ${b}`);
    checks++
}
const hist = bins => ({
    bins: bins.map(([low, high, weight]) => ({
        low,
        high,
        weight
    })),
    underflow: 0,
    overflow: 0
});
// Unequal widths: highest weight is not necessarily highest density.
const asymmetric = hist([
        [0, 1, 2],
        [1, 3, 3],
        [3, 7, 1]
    ]),
    a = stats.histogram(asymmetric);
equal(a.available, true);
near(a.mean, 2);
near(a.variance, 2.25);
equal(a.mode_intervals, [{
    low: 0,
    high: 1
}]);
equal(a.minimum, 0);
equal(a.maximum, 7);
equal(a.selected_weight, 6);
equal(a.provenance.moments, "bin-centres");
// Explicit density ties, including disjoint modes; never invent one midpoint.
const ties = stats.histogram(hist([
    [0, 1, 2],
    [1, 3, 4],
    [3, 4, 0],
    [4, 8, 8]
]));
equal(ties.mode_intervals, [{
    low: 0,
    high: 1
}, {
    low: 1,
    high: 3
}, {
    low: 4,
    high: 8
}]);
// Flow weights do not have invented positions and cannot shift binned moments.
const flowing = {
        ...asymmetric,
        underflow: 1e8,
        overflow: 2e8
    },
    f = stats.histogram(flowing);
near(f.mean, a.mean);
near(f.variance, a.variance);
equal(f.excluded.underflow, 1e8);
equal(f.excluded.overflow, 2e8);
// Whole-contained crop with both cut-straddling and wholly outside bins.
const crop = stats.histogram(hist([
    [-2, 0, 7],
    [0, 2, 11],
    [2, 3, 13],
    [3, 4, 17],
    [4, 6, 19],
    [6, 8, 23]
]), {
    window: [1, 5]
});
equal(crop.bin_count, 2);
equal(crop.occupied_bins, 2);
equal(crop.selected_weight, 30);
equal(crop.excluded.cut_straddling_weight, 30);
equal(crop.excluded.outside_window_weight, 30);
near(crop.mean, (13 * 2.5 + 17 * 3.5) / 30);
equal(crop.minimum, 2);
equal(crop.maximum, 4);
equal(crop.mode_intervals, [{
    low: 3,
    high: 4
}]);
equal(crop.provenance.scope, "whole-contained-window-bins");
const exact = stats.histogram(hist([
    [0, 1, 3],
    [1, 2, 7],
    [2, 3, 11]
]), {
    window: [1, 2]
});
equal(exact.selected_weight, 7);
equal(exact.excluded.cut_straddling_weight, 0);
equal(exact.excluded.outside_window_weight, 14);
// Scaling physical rate or normalization cannot alter any shape statistic.
for (const scale of [1e-200, 1e200, 6560]) {
    const h = structuredClone(asymmetric);
    h.bins.forEach(b => b.weight *= scale);
    const s = stats.histogram(h);
    near(s.mean, a.mean);
    near(s.variance, a.variance);
    equal(s.mode_intervals, a.mode_intervals);
}
// Narrow spectra at a large absolute offset need a shifted central variance.
const offset = 1e15,
    stable = stats.histogram(hist([
        [offset, offset + 1, 1],
        [offset + 2, offset + 3, 3]
    ]));
equal(stable.mean, offset + 2);
near(stable.variance, .75);
const hugeWeights = stats.histogram(hist([
    [0, 1, 1e300],
    [1, 2, 1e300]
]));
near(hugeWeights.mean, 1);
near(hugeWeights.variance, .25);
const tinyWidths = stats.histogram(hist([
    [0, 1e-300, 1],
    [1e-300, 3e-300, 1]
]));
equal(tinyWidths.available, true);
equal(tinyWidths.mode_intervals, [{
    low: 0,
    high: 1e-300
}]);
// An empty computed histogram has unavailable statistics, not a zero mean.
for (const h of [hist([]), hist([
        [0, 1, 0],
        [1, 2, 0]
    ]), {
        ...hist([
            [0, 1, 0]
        ]),
        underflow: 5
    }]) {
    const s = stats.histogram(h);
    equal(s.available, false);
    equal(s.mean, null);
    equal(s.variance, null);
    equal(s.mode_intervals, []);
}
const cutEmpty = stats.histogram(hist([
    [0, 2, 1]
]), {
    window: [.5, 1.5]
});
equal(cutEmpty.available, false);
equal(cutEmpty.excluded.cut_straddling_weight, 1);
// Validated native all-pair moments override only mean/variance, never mode/bounds.
const native = stats.histogram(flowing, {
    nativeMoments: {
        mean: 4,
        rms: 3,
        validated: true
    }
});
equal(native.mean, 4);
equal(native.variance, 9);
equal(native.native_moments_status, "accepted");
equal(native.provenance.moments, "native-all-pairs");
equal(native.mode_intervals, a.mode_intervals);
equal(native.minimum, a.minimum);
equal(native.excluded.overflow, 2e8);
for (const value of [null, {}, {
        mean: 4,
        rms: 3
    }, {
        mean: 4,
        rms: 3,
        validated: false
    }, {
        mean: NaN,
        rms: 3,
        validated: true
    }, {
        mean: 4,
        rms: -1,
        validated: true
    }, {
        mean: 4,
        rms: Infinity,
        validated: true
    }, {
        mean: 4,
        rms: 1e200,
        validated: true
    }]) {
    const s = stats.histogram(asymmetric, {
        nativeMoments: value
    });
    equal(s.available, true);
    equal(s.native_moments_status, "rejected");
    near(s.mean, a.mean);
    near(s.variance, a.variance);
}
const croppedNative = stats.histogram(asymmetric, {
    window: [0, 7],
    nativeMoments: {
        mean: 4,
        rms: 3,
        validated: true
    }
});
equal(croppedNative.native_moments_status, "rejected");
near(croppedNative.mean, a.mean);
const nativeZero = stats.histogram(asymmetric, {
    nativeMoments: {
        mean: 0,
        rms: 0,
        validated: true
    }
});
equal(nativeZero.native_moments_status, "accepted");
equal(nativeZero.mean, 0);
equal(nativeZero.variance, 0);
// Full native moments can survive an all-flow histogram, but cannot invent an
// occupied modal bin, sample extrema, or observations in an entirely empty run.
for (const bins of [
        [],
        [
            [0, 1, 0]
        ]
    ]) {
    const flowOnly = {
        ...hist(bins),
        overflow: 2
    };
    const s = stats.histogram(flowOnly, {
        nativeMoments: {
            mean: 4,
            rms: 3,
            validated: true
        }
    });
    equal(s.available, true);
    equal(s.mean, 4);
    equal(s.variance, 9);
    equal(s.mode_intervals, []);
    equal(s.minimum, null);
    equal(s.maximum, null);
    equal(s.native_moments_status, "accepted");
    equal(s.occupied_bins, 0);
}
const zeroNative = stats.histogram(hist([
    [0, 1, 0]
]), {
    nativeMoments: {
        mean: 0,
        rms: 0,
        validated: true
    }
});
equal(zeroNative.available, false);
equal(zeroNative.mean, null);
const noNativeFlow = stats.histogram({
    ...hist([]),
    overflow: 2
}, {
    nativeMoments: {
        mean: 4,
        rms: 3
    }
});
equal(noNativeFlow.available, false);
equal(noNativeFlow.mean, null);
const cropFlow = stats.histogram({
    ...hist([]),
    overflow: 2
}, {
    window: [0, 1],
    nativeMoments: {
        mean: 4,
        rms: 3,
        validated: true
    }
});
equal(cropFlow.available, false);
equal(cropFlow.mean, null);
// Valid native moments are useful even if coarse extreme bin-centre variance
// cannot be represented. Modal bins and bounds retain their binned provenance.
const extremeNative = stats.histogram(hist([
    [-1e200, -.5e200, 1],
    [.5e200, 1e200, 1]
]), {
    nativeMoments: {
        mean: 0,
        rms: 1,
        validated: true
    }
});
equal(extremeNative.available, true);
equal(extremeNative.variance, 1);
// Invalid inputs fail safely; no NaN/Infinity reaches a legend as a statistic.
for (const value of [null, {}, hist([
            [0, 0, 1]
        ]), hist([
            [0, 1, -1]
        ]), hist([
            [0, 2, 1],
            [1, 3, 2]
        ]),
        hist([
            [0, 1, NaN]
        ]), hist([
            [-Infinity, 1, 2]
        ]), hist([
            [-1e308, 1e308, 1]
        ]), {
            ...asymmetric,
            underflow: -1
        }, {
            ...asymmetric,
            overflow: Infinity
        },
        hist([
            [0, 1, 1e308],
            [1, 2, 1e308]
        ])
    ]) {
    const s = stats.histogram(value);
    equal(s.available, false);
    equal(s.mean, null);
    equal(Number.isFinite(s.selected_weight), true);
    equal(Object.values(s.excluded).every(Number.isFinite), true);
}
for (const window of [null, [0, 0],
        [2, 1],
        [0, Infinity],
        [0],
        [false, 1]
    ]) equal(stats.histogram(asymmetric, {
    window
}).available, false);
equal(stats.histogram(asymmetric, null).available, false);
// No mutations of inputs or options while switching views.
const untouched = structuredClone(asymmetric),
    options = {
        window: [0, 7],
        nativeMoments: {
            mean: 4,
            rms: 3,
            validated: true
        }
    },
    saved = structuredClone(options);
stats.histogram(asymmetric, options);
equal(asymmetric, untouched);
equal(options, saved);
console.log(JSON.stringify({
    status: "PASS_FOR_STATED_TARGET",
    checks,
    scope: "Binned and native histogram statistics, unequal-width modes/ties, flow/window exclusions, stable moments, invalid/empty handling, no physics rerun"
}));
