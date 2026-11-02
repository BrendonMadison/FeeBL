/* Browser validation of authenticated Eq(A.1) curves and native-weight residuals.
   Python checks the beta quadrature/optimizer. This module independently computes
   Gaussian bin integrals and nonnegative component amplitudes; it does not rerun
   the beta quadrature or claim independent proof of an optimizer minimum. */
(function(root) {
    "use strict";
    const SCHEMA = "feebl.collider.beta-gaussian-peak-fit/1";
    const ESTIMATOR =
        "bin-integrated beta-delta Gaussian convolution density least-squares shape fit; nonnegative component amplitudes profiled";
    const MODEL =
        "Wilson Appendix A.1 beta distribution plus delta at x=1, convolved with a Gaussian response";
    const RULES = {
        minimum_occupied_bins: 12,
        minimum_sigma_over_bin_width: 1.5,
        maximum_relative_l2_residual: .25,
        minimum_positive_bins_each_side: 2,
        maximum_sigma_fraction_of_nominal: .01,
        minimum_parameter_boundary_fraction: 1e-4,
        alpha_bounds: [.05, 500],
        beta_bounds: [.02, 20],
        endpoint_fraction_bounds: [.98, 1.02],
        minimum_component_fraction: 1e-6,
        maximum_near_optimal_sigma_spread: .02,
        near_optimal_objective_absolute: 1e-6,
        maximum_quadrature_relative_l2_error: 2e-6,
        maximum_quadrature_sigma_relative_change: .002,
        maximum_jacobian_condition_number: 1e10,
        sigma_profile_probe_fraction: .10,
        minimum_sigma_profile_objective_increase: 1e-4,
        quadrature_fit_order: 48,
        quadrature_refit_order: 96,
        quadrature_check_order: 192,
        gaussian_truncation_sigma: 8,
        maximum_function_evaluations_per_start: 240
    };
    const TOLERANCE = {
        endpoint_absolute_fraction_of_nominal: 2e-8,
        sigma_relative: 2e-5,
        shape_parameter_relative: 2e-4,
        area_relative: 2e-5,
        residual_absolute: 2e-6,
        curve_probability_absolute: 2e-7,
        curve_probability_relative: 2e-4,
        optimizer_counts: "recorded provenance; not a cross-platform equality condition"
    };
    const check = (ok, message) => {
            if (!ok) throw new Error(message)
        },
        finite = Number.isFinite;

    function same(a, b) {
        if (a === b) return true;
        if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(a) !==
            Array.isArray(b)) return false;
        const x = Object.keys(a).sort(),
            y = Object.keys(b).sort();
        return x.length === y.length && x.every((k, i) => k === y[i] && same(a[k], b[k]))
    }

    function numeric(a, b, relative = 1e-9, absolute = 1e-12) {
        return finite(a) && finite(b) && Math.abs(a - b) <= absolute + relative * Math.max(Math
            .abs(a), Math.abs(b))
    }

    function text(v, name) {
        check(typeof v === "string" && v.length > 0 && v.length <= 4096, "Invalid " + name)
    }

    function gaussianBins(bins, mean, sigma) {
        const nodes = [.1834346424956498, .525532409916329, .7966664774136267,
                .9602898564975363],
            weights = [.362683783378362, .3137066458778873, .2223810344533745,
                .1012285362903763],
            normal = Math.sqrt(2 * Math.PI);
        return bins.map(b => {
            const lo = Math.max(-12, (b.low - mean) / sigma),
                hi = Math.min(12, (b.high - mean) / sigma);
            if (hi <= lo) return 0;
            const pieces = Math.ceil(hi - lo),
                width = (hi - lo) / pieces;
            let sum = 0;
            for (let i = 0; i < pieces; i++) {
                const mid = lo + (i + .5) * width,
                    half = width / 2;
                for (let j = 0; j < 4; j++) sum += half * weights[j] * (Math.exp(-.5 * (
                    mid - half * nodes[j]) ** 2) + Math.exp(-.5 * (mid + half *
                    nodes[j]) ** 2)) / normal
            }
            return sum
        });
    }

    function profile(bins, g, t = null) {
        const sum = bins.reduce((s, b) => s + b.weight, 0),
            y = bins.map(b => b.weight / sum);
        let gg = 0,
            tt = 0,
            gt = 0,
            gy = 0,
            ty = 0,
            yy = 0;
        bins.forEach((b, i) => {
            const w = b.high - b.low;
            gg += g[i] * g[i] / w;
            gy += g[i] * y[i] / w;
            yy += y[i] * y[i] / w;
            if (t) {
                tt += t[i] * t[i] / w;
                gt += g[i] * t[i] / w;
                ty += t[i] * y[i] / w
            }
        });
        const score = (a, b) => bins.reduce((s, r, i) => s + (y[i] - a * g[i] - b * (t?.[i] ||
            0)) ** 2 / (r.high - r.low), 0) / yy;
        const candidates = [
            [Math.max(0, gy / gg), 0]
        ];
        if (t) {
            candidates.push([0, Math.max(0, ty / tt)]);
            const det = gg * tt - gt * gt;
            if (det > 0) {
                const a = (gy * tt - ty * gt) / det,
                    b = (ty * gg - gy * gt) / det;
                if (a >= 0 && b >= 0) candidates.push([a, b])
            }
        }
        candidates.sort((a, b) => score(...a) - score(...b));
        const c = candidates[0],
            area = c[0] + c[1];
        return {
            area,
            fraction: c[0] / area,
            residual: Math.sqrt(score(...c)),
            coefficients: c
        };
    }

    function curveBins(candidate, histogram, peak = false) {
        const curve = candidate[peak ? "peak_curve" : "curve"],
            bins = histogram.bins,
            g = gaussianBins(bins, candidate.endpoint_W_GeV, candidate.sigma_W_GeV),
            t = curve.tail_bin_probabilities,
            f = candidate.peak_fraction;
        return bins.map((b, i) => ({
            low: b.low,
            high: b.high,
            probability: f * g[i] + (1 - f) * (t?.[i] || 0),
            density: candidate.model_area_native * (f * g[i] + (1 - f) * (t?.[i] ||
                0)) / (b.high - b.low)
        }));
    }

    /** Check serialized beta/delta-convolved Gaussian fit curves against native bin weights and admission criteria. Energies are GeV; return the same fit object or throw. */
    function validate(f, reference, diagnostics, source) {
        check(f?.schema_version === SCHEMA && f.estimator === ESTIMATOR && f.model === MODEL &&
            same(f.rules, RULES) && same(f.verification_tolerance, TOLERANCE),
            "Unknown beta-Gaussian model/rules");
        check(diagnostics?.total?.peak && same(f.source_reference, source) && same(f.selection,
                diagnostics.total.peak.selection) && (f.reference_sigma_W_GeV ===
                reference || numeric(f.reference_sigma_W_GeV, reference)),
            "Beta-Gaussian source/reference mismatch");
        check(same(f.fit_source, {
            histogram: "luminosity_spectrum",
            selection: "all exported native bins; no rebinning; flow must be zero"
        }), "Unknown beta-Gaussian source scope");
        for (const k of ["formula", "objective", "normalization", "scope",
                "statistical_interpretation"
            ]) text(f[k], "beta fit " + k);
        check(/^[a-f0-9]{64}$/.test(f.adapter_sha256) && Array.isArray(f.frames) && f.frames
            .length === diagnostics.frames.length && f.frames.length <= 64,
            "Missing/bounded beta fit provenance");
        const quad = f.quadrature;
        check(quad && quad.fit_order === 48 && quad.refit_order === 96 && quad.check_order ===
            192 && quad.gaussian_window_sigma === 8 && numeric(quad
                .omitted_gaussian_probability_bound, 1.2441921148543639e-15, 1e-8, 1e-25),
            "Unknown beta quadrature bound");
        const fields = ["endpoint_W_GeV", "sigma_W_GeV", "sigma_fractional", "alpha", "beta",
            "peak_fraction", "model_area_native", "model_area_over_selected_luminosity",
            "input_reference_ratio"
        ];

        function curve(c, key, h) {
            const v = c[key];
            check(v && same(Object.keys(v).sort(), ["source_bin_start", "source_bin_stop",
                    "tail_bin_probabilities"
                ]) && v.source_bin_start === 0 && v.source_bin_stop === h.bins.length,
                "Beta curve geometry mismatch");
            if (c.model_regime === "gaussian_delta_limit") check(v.tail_bin_probabilities ===
                null, "Pure delta carries tail curve");
            else check(Array.isArray(v.tail_bin_probabilities) && v.tail_bin_probabilities
                .length === h.bins.length && v.tail_bin_probabilities.every(p => finite(
                    p) && p >= 0 && p <= 1) && v.tail_bin_probabilities.reduce((a, b) => a +
                    b, 0) <= 1 + 2e-6, "Invalid beta convolution bin probabilities");
        }

        function record(r, n, total) {
            check(r && typeof r.available === "boolean" && r.estimator === ESTIMATOR && r
                .source_histogram === "luminosity_spectrum" && r.diagonal_start === n
                .diagonal_start && r.diagonal_end === n.diagonal_end && (total || r.id === n
                    .id), "Beta fit interval/source mismatch");
            const peak = n.peak,
                h = peak.available ? n.luminosity_spectrum : null,
                nominal = peak.selection?.nominal_W_GeV ?? null;
            check(r.nominal_W_GeV === nominal && r.requested_lower_W_GeV === null && r
                .selected_bin_start === 0, "Beta nominal/full-spectrum window mismatch");
            const allow = r.candidate_display_allowed,
                c = r.candidate;
            check(typeof allow === "boolean" && ["unavailable", "resolved_shape_mismatch",
                    "admitted"
                ].includes(r.candidate_status) && allow === (r.candidate_status !==
                    "unavailable") && r.available === (r.candidate_status === "admitted") &&
                (c !== null) === allow, "Inconsistent beta candidate state");
            if (!r.available) {
                text(r.reason, "beta unavailability");
                for (const key of fields) check(r[key] === null,
                    "Unavailable beta fit carries admitted value")
            } else check(r.reason === null, "Admitted beta has failure reason");
            if (h) {
                check(r.selected_bin_stop === h.bins.length && r.selected_bins === h.bins
                    .length && r.occupied_bins === h.bins.filter(b => b.weight > 0)
                    .length && numeric(r.selected_luminosity_weight, h.bins.reduce((s, b) =>
                        s + b.weight, 0)) && r.fit_lower_W_GeV === h.bins[0].low && r
                    .fit_upper_W_GeV === h.bins.at(-1).high,
                    "Beta fit native-bin binding mismatch")
            } else check(r.selected_bin_stop === null && r.selected_bins === 0 && r
                .occupied_bins === 0 && r.selected_luminosity_weight === 0 && r
                .fit_lower_W_GeV === null && r.fit_upper_W_GeV === null && !allow,
                "Missing full spectrum carries beta fit");
            if (!allow) return;
            check(h && h.underflow === 0 && h.overflow === 0 && r.occupied_bins >= 12 && r
                .selected_luminosity_weight > 0 && c.model_regime === r.model_regime && [
                    "beta_delta_gaussian", "gaussian_delta_limit"
                ].includes(c.model_regime), "Unresolved beta spectrum/regime");
            for (const k of ["endpoint_W_GeV", "sigma_W_GeV", "sigma_fractional",
                    "model_area_native", "model_area_over_selected_luminosity"
                ]) check(finite(c[k]) && c[k] > 0, "Invalid beta parameter " + k);
            const e = c.endpoint_W_GeV,
                s = c.sigma_W_GeV,
                delta = c.model_regime === "gaussian_delta_limit",
                bins = h.bins;
            check(numeric(c.sigma_fractional, s / e) && numeric(c.model_area_native, c
                    .model_area_over_selected_luminosity * r.selected_luminosity_weight) &&
                (reference > 0 ? numeric(c.input_reference_ratio, reference / s) : c
                    .input_reference_ratio === null),
                "Beta core units/area/input ratio mismatch");
            if (delta) check(c.alpha === null && c.beta === null && c.peak_fraction === 1,
                "Invalid pure-delta Gaussian limit");
            else check(finite(c.alpha) && c.alpha > .05 && c.alpha < 500 && finite(c.beta) && c
                .beta > .02 && c.beta < 20 && finite(c.peak_fraction) && c.peak_fraction >
                1e-6 && c.peak_fraction < 1 - 1e-6, "Unidentified beta mixture parameters");
            curve(c, "curve", h);
            curve(c, "peak_curve", peak.luminosity_spectrum);
            const q = r.quality,
                solver = r.solver,
                size = delta ? 2 : 4;
            check(solver?.converged === true && Number.isSafeInteger(solver.evaluations) &&
                solver.evaluations > 0 && solver.evaluations <= 100000 && Number
                .isSafeInteger(solver.starts) && solver.starts > 0 && solver.starts <= 32 &&
                Array.isArray(solver.parameter_position) && solver.parameter_position
                .length === size && solver.parameter_position.every(v => finite(v) && v >
                    1e-4 && v < 1 - 1e-4), "Unresolved beta optimizer");
            const sigmaLo = Math.min(...bins.map(b => b.high - b.low)) / 4,
                sigmaHi = .01 * nominal,
                bounds = [
                    [.98 * nominal, Math.log(sigmaLo), Math.log(.05), Math.log(.02)],
                    [1.02 * nominal, Math.log(sigmaHi), Math.log(500), Math.log(20)]
                ].map(v => v.slice(0, size));
            check(Array.isArray(solver.parameter_bounds) && solver.parameter_bounds.length ===
                2 && solver.parameter_bounds.every((a, j) => Array.isArray(a) && a
                    .length === size && a.every((v, i) => numeric(v, bounds[j][i]))) &&
                same(solver.sigma_bounds.length, 2) && numeric(solver.sigma_bounds[0],
                    sigmaLo) && numeric(solver.sigma_bounds[1], sigmaHi),
                "Beta optimizer bounds mismatch");
            const params = [e, Math.log(s), Math.log(c.alpha), Math.log(c.beta)].slice(0, size);
            check(params.every((v, i) => numeric(v, bounds[0][i] + solver.parameter_position[
                    i] * (bounds[1][i] - bounds[0][i]))),
                "Beta optimizer coordinate mismatch");
            const width = Math.max(...bins.filter(b => b.high >= e - 2 * s && b.low <= e + 2 *
                s).map(b => b.high - b.low));
            const left = bins.filter(b => b.weight > 0 && (b.low + b.high) / 2 < e && (b.low + b
                    .high) / 2 >= e - 2 * s).length,
                right = bins.filter(b => b.weight > 0 && (b.low + b.high) / 2 >= e && (b.low + b
                    .high) / 2 <= e + 2 * s).length;
            check(q?.interior_parameters === true && q.bracketed_one_sigma === true && q
                .component_identifiable === true && e - s > bins[0].low && e + s < bins.at(-
                    1).high && numeric(q.sigma_over_bin_width, s / width) && q
                .sigma_over_bin_width >= 1.5 && numeric(q.sigma_fraction_of_nominal, s /
                    nominal) && q.sigma_fraction_of_nominal <= .01 && q
                .positive_bins_left === left && q.positive_bins_right === right && Math.min(
                    left, right) >= 2, "Unresolved Gaussian response core");
            for (const [k, limit] of [
                    ["near_optimal_sigma_spread", .02],
                    ["quadrature_relative_l2_error", 2e-6],
                    ["quadrature_sigma_relative_change", .002],
                    ["jacobian_condition_number", 1e10]
                ]) check(finite(q[k]) && q[k] >= 0 && q[k] <= limit,
                "Failed beta quality guard " + k);
            check(Array.isArray(q.sigma_profile_probes) && q.sigma_profile_probes.length ===
                2 && q.sigma_profile_probes.every((p, i) => p.converged === true && finite(p
                    .objective_increase) && p.objective_increase >= 1e-4 && numeric(p
                    .sigma_W_GeV, s * (i === 0 ? .9 : 1.1))),
                "Weakly identified Gaussian response width");
            const fit = profile(bins, gaussianBins(bins, e, s), c.curve.tail_bin_probabilities);
            check(numeric(c.model_area_over_selected_luminosity, fit.area, 2e-5, 2e-8) &&
                numeric(c.peak_fraction, fit.fraction, 2e-5, 2e-8) && numeric(r
                    .relative_l2_residual, fit.residual, 2e-5, 2e-7),
                "Beta component amplitudes/residual differ from native weights");
            check(r.available === (r.relative_l2_residual <= .25),
                "Beta residual admission mismatch");
            if (r.available)
                for (const key of fields) check(r[key] === c[key],
                    "Admitted beta/candidate values disagree");
        }
        record(f.total, diagnostics.total, true);
        f.frames.forEach((r, i) => record(r, diagnostics.frames[i], false));
        return f;
    }
    const api = {
        SCHEMA,
        ESTIMATOR,
        MODEL,
        RULES,
        TOLERANCE,
        gaussianBins,
        profile,
        curveBins,
        validate
    };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    root.ColliderBetaGaussian = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
