/* Authenticated input-reference and collision-width ratios; no control gain inference. */
(function(root) {
    "use strict";
    const LIMITS = {
        fileBytes: 4 * 1024 * 1024
    };
    const TOTAL = "native unbinned centered full-spectrum RMS; printed native precision",
        FRAME = "luminosity-weighted bin-center population RMS; not a core fit";

    function check(ok, message) {
        if (!ok) throw new Error(message)
    }

    function text(v, name) {
        check(typeof v === "string" && v.length > 0 && v.length <= 2048, "Invalid " + name)
    }

    function close(a, b) {
        return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-10 * Math.max(
            1e-100, Math.abs(a), Math.abs(b))
    }

    function idealValue(i) {
        for (const key of ["energy_GeV", "sigma_beta_x_m", "sigma_delta", "requested_espread",
                "ip_dispersion_m", "which_espread", "dist_x", "native_beam_angles_x_rad",
                "native_beam_angles_y_rad", "crab_mode", "waist_mode"
            ])
            check(Array.isArray(i[key]) && i[key].length === 2 && i[key].every(Number.isFinite),
                "Invalid ideal input " + key);
        check(i.energy_GeV.every(v => v > 0) && i.sigma_beta_x_m.every(v => v > 0) && i
            .sigma_delta.every(v => v >= 0) && [0, 1].includes(i.do_espread),
            "Invalid input width/size");
        for (let k = 0; k < 2; k++) {
            check(Number.isInteger(i.which_espread[k]) && i.which_espread[k] >= 0 && i
                .which_espread[k] <= 3 && i.requested_espread[k] >= 0,
                "Unknown energy-spread model");
            const expected = !i.do_espread || i.which_espread[k] === 0 ? 0 : i
                .requested_espread[k] * (i.which_espread[k] === 1 ? 1.7 / Math.sqrt(3) : 1);
            check(close(i.sigma_delta[k], expected) || i.sigma_delta[k] === expected,
                "Effective spread mismatch");
        }
        const [s1, s2] = i.sigma_delta, [d1, d2] = i.ip_dispersion_m, [b1, b2] = i
            .sigma_beta_x_m;
        if (d1 === 0 && d2 === 0 || s1 === 0 && s2 === 0) return 1;
        if (i.sigma_delta.some((s, k) => s > 0 && i.which_espread[k] !== 3) || i.dist_x.some(
                v => v !== 0)) return null;
        if (["native_beam_angles_x_rad", "native_beam_angles_y_rad", "crab_mode", "waist_mode"]
            .some(k => i[k].some(v => v !== 0))) return null;
        const v = s1 * s1 + s2 * s2,
            b = b1 * b1 + b2 * b2,
            variance = (v * b + (s1 * s2 * (d1 + d2)) ** 2) / (b + (d1 * s1) ** 2 + (d2 * s2) **
                2),
            value = Math.sqrt(v / variance);
        return Number.isFinite(value) ? value : null;
    }

    function frameWidth(h) {
        if (!h || h.underflow || h.overflow) return {
            available: false,
            sigma: null
        };
        const bins = h.bins.filter(b => b.weight > 0),
            sum = bins.reduce((a, b) => a + b.weight, 0);
        if (sum <= 0 || bins.length < 2) return {
            available: false,
            sigma: null
        };
        const origin = (bins[0].low + bins[0].high) / 2,
            mean = bins.reduce((a, b) => a + ((b.low + b.high) / 2 - origin) * b.weight / sum,
                0);
        const sigma = Math.sqrt(bins.reduce((a, b) => a + b.weight / sum * ((b.low + b.high) /
            2 - origin - mean) ** 2, 0));
        return {
            available: sigma > 0 && sigma >= Math.max(...bins.map(b => b.high - b.low)),
            sigma
        };
    }
    const PEAK_ESTIMATOR =
        "bin-integrated Gaussian density least-squares shape fit; profiled amplitude";
    const PEAK_RULES = {
        minimum_occupied_bins: 6,
        minimum_sigma_over_bin_width: 1.5,
        maximum_relative_l2_residual: .25,
        minimum_positive_bins_each_side: 2,
        maximum_sigma_over_fit_span: .25,
        minimum_parameter_boundary_fraction: 1e-4
    };
    const PEAK_TOLERANCE = {
        mean_absolute_fraction_of_nominal: 1e-9,
        sigma_relative: 2e-6,
        area_relative: 2e-6,
        residual_absolute: 2e-7,
        dimensionless_quality_relative: 3e-6,
        dimensionless_quality_absolute: 2e-7,
        optimizer_counts: "recorded provenance; not a cross-platform equality condition"
    };

    function sameObject(a, b) {
        if (a === b) return true;
        if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(a) !==
            Array.isArray(b)) return false;
        const x = Object.keys(a).sort(),
            y = Object.keys(b).sort();
        return x.length === y.length && x.every((k, i) => k === y[i] && sameObject(a[k], b[k]))
    }

    function gaussianDensity(fit, x) {
        return fit.gaussian_area_native * Math.exp(-.5 * ((x - fit.mean_W_GeV) / fit
            .sigma_W_GeV) ** 2) / (fit.sigma_W_GeV * Math.sqrt(2 * Math.PI))
    }
    // Eight-point Gauss-Legendre bin integrals are accurate for the admitted resolved
    // uniform bins (sigma >= 1.5 bin widths). This checks the supplied shape, without
    // rerunning or claiming to independently prove the Python optimizer's minimum.
    function gaussianProfile(bins, mean, sigma) {
        const nodes = [.1834346424956498, .525532409916329, .7966664774136267,
                .9602898564975363],
            weights = [.362683783378362, .3137066458778873, .2223810344533745,
                .1012285362903763];
        const sum = bins.reduce((s, b) => s + b.weight, 0),
            normal = Math.sqrt(2 * Math.PI),
            p = [],
            q = [];
        let yy = 0,
            qq = 0,
            pq = 0;
        for (const b of bins) {
            const half = (b.high - b.low) / (2 * sigma),
                mid = ((b.high + b.low) / 2 - mean) / sigma;
            let prob = 0;
            for (let j = 0; j < 4; j++) prob += weights[j] * (Math.exp(-.5 * (mid - half *
                nodes[j]) ** 2) + Math.exp(-.5 * (mid + half * nodes[j]) ** 2));
            prob *= half / normal;
            const y = b.weight / sum,
                width = b.high - b.low;
            p.push(y);
            q.push(prob);
            yy += y * y / width;
            qq += prob * prob / width;
            pq += y * prob / width;
        }
        const area = pq / qq,
            residual = Math.sqrt(bins.reduce((s, b, i) => s + (p[i] - area * q[i]) ** 2 / (b
                .high - b.low), 0) / yy);
        return {
            area,
            residual
        };
    }

    function validatePeakFit(f, reference, diagnostics, source) {
        if (f?.schema_version === "feebl.collider.beta-gaussian-peak-fit/1") {
            const adapter = typeof module !== "undefined" && module.exports ? require(
                "./collider_beta_gaussian.js") : root.ColliderBetaGaussian;
            return adapter.validate(f, reference, diagnostics, source)
        }
        check(f?.schema_version === "feebl.collider.gaussian-peak-fit/1" && f.estimator ===
            PEAK_ESTIMATOR && sameObject(f.rules, PEAK_RULES) && sameObject(f
                .verification_tolerance, PEAK_TOLERANCE),
            "Unknown Gaussian peak-fit contract/rules");
        check(diagnostics?.total?.peak && sameObject(f.source_reference, source) && sameObject(f
            .selection, diagnostics.total.peak.selection) && close(f
            .reference_sigma_W_GeV, reference), "Peak fit source/reference mismatch");
        for (const k of ["objective", "normalization", "scope", "statistical_interpretation"])
            text(f[k], "peak " + k);
        check(/^[a-f0-9]{64}$/.test(f.adapter_sha256), "Missing peak estimator provenance");
        check(Array.isArray(f.frames) && f.frames.length === diagnostics.frames.length,
            "Peak fit frame cardinality mismatch");
        const numeric = (a, b, relative = 1e-9, absolute = 1e-12) => Number.isFinite(a) &&
            Number.isFinite(b) && Math.abs(a - b) <= absolute + relative * Math.max(Math.abs(a),
                Math.abs(b));

        function record(r, n, isTotal) {
            check(r && typeof r.available === "boolean" && r.estimator === PEAK_ESTIMATOR && r
                .diagonal_start === n.diagonal_start && r.diagonal_end === n.diagonal_end &&
                (isTotal || r.id === n.id), "Peak fit interval mismatch");
            const p = n.peak,
                h = p.available ? p.luminosity_spectrum : null,
                s = p.selection,
                nominal = s?.nominal_W_GeV ?? null;
            check(r.nominal_W_GeV === nominal && r.requested_lower_W_GeV === (s
                ?.minimum_W_GeV ?? null), "Peak fit nominal/cut mismatch");
            const allow = r.candidate_display_allowed,
                c = r.candidate;
            check(typeof allow === "boolean" && ["unavailable", "resolved_shape_mismatch",
                    "admitted"
                ].includes(r.candidate_status) && allow === (r.candidate_status !==
                    "unavailable") && r.available === (r.candidate_status === "admitted") &&
                (c !== null) === allow, "Inconsistent Gaussian candidate state");
            if (!r.available) {
                text(r.reason, "peak unavailability");
                for (const k of ["mean_W_GeV", "sigma_W_GeV", "gaussian_area_native",
                        "gaussian_area_over_selected_luminosity", "input_reference_ratio"
                    ]) check(r[k] === null, "Unavailable peak fit carries admitted value")
            } else check(r.reason === null, "Admitted peak has failure reason");
            let bins = [];
            if (h) {
                bins = h.bins.filter(b => b.low >= s.minimum_W_GeV || numeric(b.low, s
                    .minimum_W_GeV, 1e-12));
                const start = h.bins.length - bins.length;
                check(r.selected_bin_start === start && r.selected_bin_stop === h.bins.length &&
                    r.selected_bins === bins.length && r.occupied_bins === bins.filter(b =>
                        b.weight > 0).length && numeric(r.selected_luminosity_weight, bins
                        .reduce((v, b) => v + b.weight, 0)) && r.fit_lower_W_GeV === bins[0]
                    ?.low && r.fit_upper_W_GeV === bins.at(-1)?.high,
                    "Peak fit native-bin binding mismatch");
            } else check(r.selected_bin_start === null && r.selected_bin_stop === null && r
                .selected_bins === 0 && r.occupied_bins === 0 && r
                .selected_luminosity_weight === 0 && r.fit_lower_W_GeV === null && r
                .fit_upper_W_GeV === null && !allow, "Missing native peak carries fit bins");
            if (!allow) return;
            check(h && h.overflow === 0 && h.underflow === 0 && bins.length >= 8 && r
                .occupied_bins >= 6 && r.selected_luminosity_weight > 0,
                "Unresolved native peak candidate");
            for (const k of ["mean_W_GeV", "sigma_W_GeV", "gaussian_area_native",
                    "gaussian_area_over_selected_luminosity"
                ]) check(Number.isFinite(c[k]) && c[k] > 0, "Invalid Gaussian candidate " + k);
            check(numeric(c.gaussian_area_native, c.gaussian_area_over_selected_luminosity * r
                .selected_luminosity_weight), "Gaussian area normalization mismatch");
            check(reference > 0 ? numeric(c.input_reference_ratio, reference / c.sigma_W_GeV) :
                c.input_reference_ratio === null, "Gaussian input-reference ratio mismatch");
            const q = r.quality,
                solver = r.solver,
                lo = bins[0].low,
                hi = bins.at(-1).high,
                mu = c.mean_W_GeV,
                sigma = c.sigma_W_GeV,
                span = hi - lo;
            check(solver?.converged === true && Number.isSafeInteger(solver.iterations) &&
                solver.iterations > 0 && solver.iterations <= 240 && Number.isSafeInteger(
                    solver.evaluations) && solver.evaluations > 0 && solver.evaluations <=
                10000 && Array.isArray(solver.parameter_position) && solver
                .parameter_position.length === 2 && solver.parameter_position.every(v =>
                    Number.isFinite(v) && v > 1e-4 && v < 1 - 1e-4),
                "Unresolved optimizer candidate");
            const widths = bins.filter(b => b.high >= mu - 2 * sigma && b.low <= mu + 2 * sigma)
                .map(b => b.high - b.low),
                width = Math.max(...widths);
            const sigmaLo = Math.min(...bins.map(b => b.high - b.low)) / nominal / 4,
                sigmaHi = span / nominal / 2;
            check(Array.isArray(solver.sigma_bounds) && solver.sigma_bounds.length === 2 &&
                numeric(solver.sigma_bounds[0], sigmaLo) && numeric(solver.sigma_bounds[1],
                    sigmaHi) && numeric(mu, lo + solver.parameter_position[0] * span) &&
                numeric(sigma, nominal * sigmaLo * Math.exp(solver.parameter_position[1] *
                    Math.log(sigmaHi / sigmaLo))), "Gaussian optimizer coordinate mismatch");
            const left = bins.filter(b => b.weight > 0 && (b.low + b.high) / 2 < mu && (b.low +
                    b.high) / 2 >= mu - 2 * sigma).length,
                right = bins.filter(b => b.weight > 0 && (b.low + b.high) / 2 >= mu && (b.low +
                    b.high) / 2 <= mu + 2 * sigma).length;
            check(q?.interior_parameters === true && q.bracketed_one_sigma === true && mu -
                sigma > lo && mu + sigma < hi && numeric(q.sigma_over_bin_width, sigma /
                    width) && numeric(q.sigma_over_fit_span, sigma / span) && q
                .positive_bins_left === left && q.positive_bins_right === right && sigma /
                width >= 1.5 && sigma / span <= .25 && Math.min(left, right) >= 2,
                "Unresolved Gaussian shape candidate");
            const profile = gaussianProfile(bins, mu, sigma);
            check(numeric(r.relative_l2_residual, profile.residual, 2e-6, 2e-8) && numeric(c
                    .gaussian_area_over_selected_luminosity, profile.area, 2e-6, 2e-8),
                "Gaussian amplitude/residual differs from authenticated native bins");
            check(r.available === (r.relative_l2_residual <= .25),
                "Gaussian residual admission mismatch");
            if (r.available)
                for (const k of Object.keys(c)) check(r[k] === c[k],
                    "Admitted Gaussian/candidate mismatch");
        }
        record(f.total, diagnostics.total, true);
        f.frames.forEach((r, i) => record(r, diagnostics.frames[i], false));
        return f;
    }

    /** Verify Gaussian input-reference factors and observed width ratios against authenticated optics and diagnostics. Ratios are dimensionless; return the same object or throw. */
    function validate(a, refs = [], context = {}) {
        check(a?.schema_version === "feebl.collider.monochromatization/1",
            "Unknown monochromatization contract");
        const input = a.inputs,
            value = idealValue(input),
            ideal = a.ideal;
        check(ideal && ideal.available === (value !== null) && ideal.unit === "1" && ideal
            .scope ===
            "static evaluated-optics reference; not a frame or luminosity-weighted measurement",
            "Invalid ideal factor scope");
        text(ideal.label, "ideal label");
        if (value === null) {
            check(ideal.value === null, "Unavailable ideal carries value");
            text(ideal.reason, "ideal reason")
        } else check(ideal.reason === null && close(ideal.value, value),
            "Ideal factor formula mismatch");
        for (const k of ["ip_dispersion_m", "crab_mode", "waist_mode"]) {
            check(JSON.stringify(input[k]) === JSON.stringify(a.native_manifest_parameters?.[
                k]), "Native map receipt mismatch");
            if (context.manifest) check(JSON.stringify(input[k]) === JSON.stringify(context
                .manifest[k]), "Dataset map receipt mismatch")
        }
        for (const key of ["energy_GeV", "espread", "ip_dispersion_m",
                "native_beam_angles_x_rad", "native_beam_angles_y_rad", "crab_mode",
                "waist_mode"
            ]) {
            const native = a.native_manifest_parameters?.[key],
                evaluated = input[key === "espread" ? "requested_espread" : key];
            check(Array.isArray(native) && native.length === 2 && native.every(Number.isFinite),
                "Missing native input receipt " + key);
            check(evaluated.every((v, i) => Math.abs(v - native[i]) <= 1e-5 * Math.max(1e-12,
                    Math.abs(v), Math.abs(native[i]))), "Evaluated/native input mismatch " +
                key);
            if (context.manifest) check(JSON.stringify(native) === JSON.stringify(context
                .manifest[key]), "Dataset native input receipt mismatch " + key);
        }
        const measured = a.measured,
            reference = Math.sqrt(input.energy_GeV[0] * input.energy_GeV[1]) * Math.hypot(...
                input.sigma_delta);
        check(measured && close(measured.reference_sigma_W_GeV, reference) || measured
            ?.reference_sigma_W_GeV === reference, "Input reference width mismatch");
        text(measured.label, "width ratio label");
        text(measured.definition, "width ratio definition");
        check(measured.reference_scope ===
            "independent head-on input energies; first-order approximation; not a matched collision control",
            "Unknown input width reference scope");

        function record(r, estimator) {
            check(r && typeof r.available === "boolean" && r.estimator === estimator,
                "Invalid width estimator");
            check(r.sigma_W_GeV === null || Number.isFinite(r.sigma_W_GeV) && r.sigma_W_GeV >=
                0, "Invalid observed width");
            if (r.available) check(reference > 0 && r.sigma_W_GeV > 0 && r.reason === null &&
                close(r.value, reference / r.sigma_W_GeV), "Width ratio formula mismatch");
            else {
                check(r.value === null, "Unavailable width ratio carries value");
                text(r.reason, "width unavailability")
            }
        }
        record(measured.total, TOTAL);
        if (measured.total.sigma_W_GeV !== null) check(context.analysis && a.source_artifacts
            ?.analysis, "Observed total width requires validated spectrum source");
        if (context.analysis) {
            const sigma = context.analysis.native_statistics?.full_RMS_W_GeV ?? null;
            check(sigma === null ? measured.total.sigma_W_GeV === null : close(sigma, measured
                    .total.sigma_W_GeV) || sigma === measured.total.sigma_W_GeV,
                "Total RMS source mismatch");
            check(measured.total.available === (reference > 0 && sigma > 0),
                "Total width availability mismatch")
        }
        check(Array.isArray(measured.frames) && measured.frames.length <= 64,
            "Width frame bound");
        let previous = -1;
        if (measured.frames.length) check(context.diagnostics && a.source_artifacts
            ?.diagnostics, "Frame widths require validated native diagnostics");
        if (context.diagnostics) check(measured.frames.length === context.diagnostics.frames
            .length, "Width frame cardinality mismatch");
        measured.frames.forEach((r, k) => {
            record(r, FRAME);
            check(r.id === k && Number.isInteger(r.diagonal_start) && Number.isInteger(r
                    .diagonal_end) && r.diagonal_start === previous + 1 && r
                .diagonal_end >= r.diagonal_start, "Width frame alignment mismatch");
            previous = r.diagonal_end;
            if (context.manifest) check(r.diagonal_end === context.manifest.frames[k]
                ?.diagonal, "Width/display frame alignment mismatch");
            if (context.diagnostics) {
                const expected = frameWidth(context.diagnostics.frames[k]
                    .luminosity_spectrum);
                check(r.available === (reference > 0 && expected.available),
                    "Frame width admission mismatch");
                check(expected.sigma === null ? r.sigma_W_GeV === null : close(expected
                        .sigma, r.sigma_W_GeV) || expected.sigma === r.sigma_W_GeV,
                    "Frame RMS estimator mismatch")
            }
        });
        check(a.matched_control?.available === false && a.matched_control.value === null,
            "Unadmitted matched-control claim");
        text(a.matched_control.reason, "matched-control reason");
        check(a.definition && Array.isArray(a.definition.limitations) && a.definition
            .limitations.length <= 16, "Missing ideal definition");
        for (const k of ["model", "formula", "meaning"]) text(a.definition[k], k);
        for (const v of a.definition.limitations) text(v, "model limitation");
        check(a.source_artifacts?.native_summary, "Missing evaluated input provenance");
        for (const ref of Object.values(a.source_artifacts)) {
            check(ref && /^[A-Za-z0-9_.-]+$/.test(ref.path) && ![".", ".."].includes(ref
                .path) && Number.isSafeInteger(ref.bytes) && ref.bytes > 0 && ref.bytes <=
                32 * 1024 * 1024 && /^[a-f0-9]{64}$/.test(ref.sha256),
                "Invalid factor source receipt");
            check(refs.some(r => r.path === ref.path && r.bytes === ref.bytes && r.sha256 ===
                ref.sha256), "Factor source is not authenticated")
        }
        for (const key of ["analysis", "diagnostics"])
            if (a.source_artifacts[key]) {
                const actual = context.manifest?.[key],
                    ref = a.source_artifacts[key];
                check(actual && ["path", "bytes", "sha256"].every(k => actual[k] === ref[k]),
                    "Factor source does not match dataset " + key)
            }
        if (context.diagnostics?.total?.peak) check(a.peak_fit !== undefined,
            "Native peak diagnostics require the bound peak-fit analysis");
        if (a.peak_fit !== undefined) validatePeakFit(a.peak_fit, reference, context
            .diagnostics, a.source_artifacts.diagnostics);
        check(/^[a-f0-9]{64}$/.test(a.adapter_sha256), "Missing factor adapter provenance");
        return a;
    }
    const api = {
        LIMITS,
        TOTAL,
        FRAME,
        idealValue,
        frameWidth,
        PEAK_ESTIMATOR,
        PEAK_RULES,
        PEAK_TOLERANCE,
        gaussianDensity,
        gaussianProfile,
        validatePeakFit,
        validate
    };
    root.ColliderMonochromatization = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);
