/* Histogram statistics only: integrated weights, no event reconstruction.
 * histogram({bins:[{low,high,weight}],underflow,overflow}, options={})
 * options.window=[low,high] selects whole-contained bins (never fractions).
 * options.nativeMoments={mean,rms,validated:true} is reserved for authenticated,
 * schema-validated native all-pair full-W moments. The caller establishes that
 * provenance; this module also checks finite mean/RMS, RMS >= 0 and finite RMS^2.
 * Native moments cannot describe a cropped window. They override only mean and
 * population variance, including native flow; modes and bounds remain binned.
 * An all-flow spectrum can expose valid native moments with no modal intervals
 * or occupied-bin bounds. An entirely zero-weight spectrum stays unavailable.
 * All numeric values are in the histogram coordinate unit (variance squared).
 * Invalid/unrepresentable input returns unavailable; inputs are never mutated.
 */
(function(root) {
    "use strict";
    const LIMITS = {
        bins: 20000
    };
    const finite = Number.isFinite;

    function sum(values) {
        // Neumaier compensation protects smaller contributions from summation order.
        let s = 0,
            c = 0;
        for (const x of values) {
            const t = s + x;
            c += Math.abs(s) >= Math.abs(x) ? (s - t) + x : (x - t) + s;
            s = t
        }
        return s + c;
    }

    function unavailable(reason, result = {}) {
        return Object.assign({
            available: false,
            reason,
            mean: null,
            variance: null,
            mode_intervals: [],
            minimum: null,
            maximum: null,
            selected_weight: 0,
            excluded: {
                underflow: 0,
                overflow: 0,
                outside_window_weight: 0,
                cut_straddling_weight: 0
            },
            bin_count: 0,
            occupied_bins: 0,
            provenance: {
                moments: "bin-centres",
                mode: "maximum-weight-density-bin-intervals",
                bounds: "occupied-bin-edges",
                scope: "full-bins"
            },
            native_moments_status: "not-requested",
            notes: []
        }, result, {
            available: false,
            reason
        });
    }

    function histogram(h, options = {}) {
        if (!options || typeof options !== "object" || Array.isArray(options))
        return unavailable("Invalid statistics options.");
        const hasWindow = Object.prototype.hasOwnProperty.call(options, "window");
        const nativeRequested = Object.prototype.hasOwnProperty.call(options, "nativeMoments");
        const base = unavailable("Statistics unavailable.");
        base.native_moments_status = nativeRequested ? "rejected" : "not-requested";
        base.provenance.scope = hasWindow ? "whole-contained-window-bins" : "full-bins";
        if (!h || typeof h !== "object" || !Array.isArray(h.bins) || h.bins.length > LIMITS
            .bins)
            return unavailable("Invalid or oversized histogram.", base);
        if (![h.underflow, h.overflow].every(v => finite(v) && v >= 0))
            return unavailable("Invalid underflow or overflow weight.", base);
        base.excluded.underflow = h.underflow;
        base.excluded.overflow = h.overflow;
        let windowLow = -Infinity,
            windowHigh = Infinity;
        if (hasWindow) {
            const window = options.window;
            if (!Array.isArray(window) || window.length !== 2 || !window.every(finite) || !(
                    window[1] > window[0]))
                return unavailable("Statistics window needs two finite increasing bounds.",
                    base);
            [windowLow, windowHigh] = window;
        }
        const selected = [],
            outside = [],
            straddling = [];
        let previous = -Infinity;
        for (const b of h.bins) {
            if (!b || ![b.low, b.high, b.weight].every(finite) || b.low < previous ||
                !(b.high > b.low) || !finite(b.high - b.low) || b.weight < 0)
                return unavailable("Invalid, overlapping, or unordered histogram bins.", base);
            previous = b.high;
            if (b.low >= windowLow && b.high <= windowHigh) selected.push(b);
            else if (b.high <= windowLow || b.low >= windowHigh) outside.push(b.weight);
            else straddling.push(b.weight);
        }
        base.bin_count = selected.length;
        const occupied = selected.filter(b => b.weight > 0);
        base.occupied_bins = occupied.length;
        const selectedWeight = sum(selected.map(b => b.weight)),
            outsideWeight = sum(outside),
            straddlingWeight = sum(straddling);
        if (![selectedWeight, outsideWeight, straddlingWeight].every(finite))
            return unavailable("Histogram weight sums exceed the finite numeric range.", base);
        base.selected_weight = selectedWeight;
        base.excluded.outside_window_weight = outsideWeight;
        base.excluded.cut_straddling_weight = straddlingWeight;
        base.notes = [
            "Mean and population variance use bin centres and integrated bin weights; no within-bin variance is inferred.",
            "Mode identifies maximum weight per unit x; minimum and maximum are occupied bin edges, not exact sample extrema.",
            "Underflow and overflow have no coordinates and are excluded from all binned statistics."
        ];
        if (hasWindow) base.notes.push(
            "Window statistics include only whole-contained bins; bins crossing either cut and bins outside the window are excluded."
            );
        if (nativeRequested) base.notes.push(
            "Native moments were not applied unless explicitly validated for a full histogram with finite mean and nonnegative finite RMS."
            );
        const native = options.nativeMoments;
        const nativeValid = nativeRequested && !hasWindow && native && native.validated ===
            true &&
            finite(native.mean) && finite(native.rms) && native.rms >= 0 && finite(native.rms *
                native.rms);
        const anyWeight = occupied.length > 0 || h.underflow > 0 || h.overflow > 0;

        function applyNative() {
            base.mean = native.mean;
            base.variance = native.rms * native.rms;
            base.provenance.moments = "native-all-pairs";
            base.native_moments_status = "accepted";
            base.notes[0] =
                "Mean and population variance use validated native moments over all pairs, including flow; mode and bounds remain binned and exclude flow.";
            base.notes.pop();
        }
        if (!occupied.length) {
            if (nativeValid && anyWeight) {
                applyNative();
                base.notes.push(
                    "No occupied bins are available: modal intervals and occupied-bin bounds are unavailable; native moments remain available."
                    );
                base.available = true;
                base.reason = null;
                return base;
            }
            return unavailable(hasWindow ?
                "No positive-weight bins are wholly contained in this window." :
                "No positive-weight bins are available.", base);
        }

        // Scale weights before products. Shift coordinates before squaring to avoid
        // catastrophic E[x^2]-E[x]^2 cancellation for narrow high-offset spectra.
        const maximumWeight = Math.max(...occupied.map(b => b.weight));
        const weights = occupied.map(b => b.weight / maximumWeight);
        const centres = occupied.map(b => b.low + (b.high - b.low) / 2);
        const weightSum = sum(weights);
        let anchor = centres[0];
        let offsets = centres.map(x => x - anchor);
        if (!offsets.every(finite)) {
            anchor = 0;
            offsets = centres.slice()
        }
        const coordinateScale = Math.max(...offsets.map(Math.abs));
        const scaled = coordinateScale > 0 ? offsets.map(x => x / coordinateScale) : offsets;
        const scaledMean = sum(scaled.map((x, i) => x * weights[i])) / weightSum;
        const mean = anchor + coordinateScale * scaledMean;
        const scaledVariance = sum(scaled.map((x, i) => weights[i] * (x - scaledMean) ** 2)) /
            weightSum;
        const rms = coordinateScale * Math.sqrt(Math.max(0, scaledVariance));
        const variance = rms * rms;
        if ((!finite(mean) || !finite(variance)) && !nativeValid)
            return unavailable("Histogram moments exceed the finite numeric range.", base);
        if (finite(mean) && finite(variance)) {
            base.mean = mean;
            base.variance = variance
        }
        base.minimum = occupied[0].low;
        base.maximum = occupied[occupied.length - 1].high;
        // Log density avoids overflow for narrow bins and underflow for tiny
        // positive weights. Ties use only a small floating-point rounding tolerance.
        const logDensity = occupied.map(b => Math.log(b.weight) - Math.log(b.high - b.low));
        const maximumDensity = Math.max(...logDensity);
        base.mode_intervals = occupied.filter((b, i) => Math.abs(logDensity[i] -
                maximumDensity) <=
                16 * Number.EPSILON * Math.max(1, Math.abs(logDensity[i]), Math.abs(
                    maximumDensity)))
            .map(b => ({
                low: b.low,
                high: b.high
            }));
        if (nativeValid) applyNative();
        base.available = true;
        base.reason = null;
        return base;
    }
    const api = {
        LIMITS,
        histogram
    };
    root.ColliderStatistics = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);
