/* Bounded collider data contract. Node exports support independent parser tests. */
(function(root) {
    "use strict";
    const LIMITS = {
        frames: 64,
        rows: 4096,
        fileBytes: 4 * 1024 * 1024,
        totalBytes: 128 * 1024 * 1024
    };

    function check(ok, msg) {
        if (!ok) throw new Error(msg)
    }

    function safeName(s) {
        check(typeof s === "string" && /^[A-Za-z0-9_.-]+$/.test(s) && s !== "." && s !== "..",
            "Unsafe artifact path");
        return s
    }

    /** Validate a bounded manifest and frame references; return the same object or throw. Display units are GeV, um, urad and seconds. */
    function manifest(m) {
        check(m.schema_version === "feebl.collider/2",
            "Unsupported collider schema; finalize raw output first");
        check(JSON.stringify(m.units) === JSON.stringify({
            energy: "GeV",
            position: "um",
            angle: "urad",
            time: "s"
        }), "Incompatible units");
        check(m.coordinates?.frame === "legacy_collider_dump", "Unsupported coordinates");
        check(Array.isArray(m.frames) && m.frames.length > 0 && m.frames.length <= LIMITS
            .frames, "Invalid frame count");
        check(Array.isArray(m.energy_GeV) && m.energy_GeV.length === 2 && m.energy_GeV.every(
            v => Number.isFinite(v) && v > 0), "Invalid nominal energies");
        check(m.validation?.status && m.backend_actual_receipt && m.source?.commit,
            "Missing provenance or validation status");
        let prev = -1,
            total = 0;
        const seen = new Set();
        m.frames.forEach((f, i) => {
            check(f.id === i && Number.isFinite(f.elapsed_s) && f.elapsed_s > prev,
                "Nonmonotone frame ID or time");
            prev = f.elapsed_s;
            ["beam1", "beam2"].forEach(k => {
                const a = f[k];
                safeName(a.path);
                check(!seen.has(a.path), "Duplicate artifact");
                seen.add(a.path);
                check(Number.isInteger(a.rows) && a.rows > 0 && a.rows <= LIMITS
                    .rows, "Invalid row count");
                check([6, 9].includes(a.columns), "Invalid columns");
                check(Number.isInteger(a.bytes) && a.bytes > 0 && a.bytes <=
                    LIMITS.fileBytes, "Artifact size bound");
                check(/^[a-f0-9]{64}$/.test(a.sha256), "Invalid digest");
                check(Array.isArray(a.bounds) && a.bounds.length === 6 && a
                    .bounds.every(x => Array.isArray(x) && x.length === 2 &&
                        x.every(Number.isFinite) && x[0] <= x[1]),
                    "Invalid bounds");
                total += a.bytes;
            });
        });
        check(total <= LIMITS.totalBytes, "Dataset size bound");
        return m;
    }

    /** Parse finite native display rows against authenticated count/bounds metadata; return numeric arrays in the declared native display units. */
    function rows(text, a) {
        const result = [];
        for (const line of text.split(/\r?\n/)) {
            if (!line.trim()) continue;
            const row = line.trim().split(/\s+/).map(Number);
            check(row.length === a.columns && row.every(Number.isFinite),
                "Truncated/nonfinite particle row");
            result.push(row);
            check(result.length <= LIMITS.rows, "Row bound exceeded");
        }
        check(result.length === a.rows, "Particle row count mismatch");
        const bounds = Array.from({
            length: 6
        }, (_, i) => [Math.min(...result.map(r => r[i])), Math.max(...result.map(r => r[
            i]))]);
        check(JSON.stringify(bounds) === JSON.stringify(a.bounds), "Particle bounds mismatch");
        return result;
    }
    /** Authenticate bytes and parse one bounded particle frame; reject mismatched hash, count or bounds. */
    async function artifact(bytes, a) {
        check(bytes.byteLength === a.bytes, "Artifact byte count mismatch");
        check(root.crypto?.subtle, "SHA256 unavailable: serve on loopback");
        const hash = Array.from(new Uint8Array(await root.crypto.subtle.digest("SHA-256",
            bytes)), v => v.toString(16).padStart(2, "0")).join("");
        check(hash === a.sha256, "Artifact hash mismatch: " + a.path);
        return rows(new TextDecoder("utf-8", {
            fatal: true
        }).decode(bytes), a);
    }

    function fileRef(a, maxBytes = 32 * 1024 * 1024, allowEmpty = false) {
        safeName(a.path);
        check(Number.isInteger(a.bytes) && a.bytes >= (allowEmpty ? 0 : 1) && a.bytes <=
            maxBytes, "Analysis file size bound");
        check(/^[a-f0-9]{64}$/.test(a.sha256), "Analysis digest missing");
        return a;
    }
    async function bytesHash(bytes, a, maxBytes = 32 * 1024 * 1024, allowEmpty = false) {
        fileRef(a, maxBytes, allowEmpty);
        check(bytes.byteLength === a.bytes, "Analysis byte count mismatch");
        check(root.crypto?.subtle, "SHA256 unavailable: serve on loopback");
        const hash = Array.from(new Uint8Array(await root.crypto.subtle.digest("SHA-256",
            bytes)), v => v.toString(16).padStart(2, "0")).join("");
        check(hash === a.sha256, "Analysis hash mismatch: " + a.path);
        return bytes;
    }

    /** Validate luminosity bin integrals, densities and normalization metadata; return the same object or throw. */
    function analysis(a) {
        check(a.schema_version === "feebl.collider.analysis/1",
            "Unknown spectrum analysis schema");
        check(a.observable === "e+e- luminosity spectrum", "Unknown spectrum observable");
        check(Number.isFinite(a.definition?.nominal_W_GeV) && a.definition.nominal_W_GeV > 0,
            "Missing nominal W");
        check(a.plot?.crop_renormalization === false, "Unsupported crop renormalization");
        check(a.plot.peak.x_min_GeV === .98 * a.definition.nominal_W_GeV,
            "Incorrect peak lower edge");
        check(a.plot.full.y_scale === "log" && a.plot.peak.y_scale === "linear",
            "Incorrect spectrum axes");
        check(Array.isArray(a.bins) && a.bins.length <= 20000, "Spectrum bin limit");
        let prev = -Infinity;
        for (const b of a.bins) {
            check([b.W_low_GeV, b.W_high_GeV, b.dL_dW_m2_per_crossing_GeV, b
                .weight_m2_per_crossing
            ].every(Number.isFinite), "Nonfinite spectrum");
            check(b.W_low_GeV >= prev && b.W_high_GeV > b.W_low_GeV && b
                .dL_dW_m2_per_crossing_GeV > 0 && b.weight_m2_per_crossing > 0,
                "Invalid spectrum bin");
            check(b.uncertainty === null, "Unadmitted spectrum error model");
            for (const key of ["dL_dW_cm2_s_GeV", "shape_density_per_GeV"])
                check(b[key] === null || (Number.isFinite(b[key]) && b[key] > 0),
                    "Invalid spectrum normalization");
            prev = b.W_high_GeV;
        }
        check(a.normalization && ["underflow_m2_per_crossing", "overflow_m2_per_crossing",
                "total_spectrum_m2_per_crossing"
            ].every(k => Number.isFinite(a.normalization[k]) && a.normalization[k] >= 0),
            "Invalid flow normalization");
        // Optional older datasets may omit native moments. If supplied, validate
        // every value now consumed by the statistics legend before rendering it.
        if (a.native_statistics !== undefined && a.native_statistics !== null) {
            const s = a.native_statistics;
            check(typeof s === "object" && !Array.isArray(s) && Number.isFinite(s.mean_W_GeV) &&
                Number.isFinite(s.full_RMS_W_GeV) && s.full_RMS_W_GeV >= 0 &&
                Number.isFinite(s.full_RMS_W_GeV * s.full_RMS_W_GeV),
                "Invalid native spectrum moments");
        }
        return a;
    }

    function close(a, b) {
        return Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b))
    }

    function finiteText(s, name) {
        check(typeof s === "string" && s.length > 0 && s.length <= 2048, "Invalid " + name)
    }

    /** Validate bin-integrated weights, finite ordered edges and flow closure in the stated units; throw on inconsistency. */
    function histogram(h, name) {
        if (h === null) return null;
        check(h && typeof h === "object", "Missing " + name + " histogram");
        for (const key of ["quantity", "unit", "weight_unit", "weighting"]) finiteText(h[key],
            name + " " + key);
        check(h.normalization === "bin-integrated", "Unsupported histogram normalization");
        check(Array.isArray(h.bins) && h.bins.length > 0 && h.bins.length <= 2048,
            "Histogram bin bound");
        let prev = -Infinity,
            sum = 0;
        for (const b of h.bins) {
            check([b.low, b.high, b.weight].every(Number.isFinite) && b.low >= prev && b.high >
                b.low && b.weight >= 0, "Invalid histogram bin");
            prev = b.high;
            sum += b.weight;
        }
        for (const key of ["underflow", "overflow", "total_weight"])
            check(Number.isFinite(h[key]) && h[key] >= 0, "Invalid histogram flow or total");
        check(close(sum + h.underflow + h.overflow, h.total_weight),
        "Histogram weight closure");
        return h;
    }

    function peakDiagnostics(d, m) {
        const records = [...d.frames, d.total],
            has = records.some(r => Object.prototype.hasOwnProperty.call(r, "peak"));
        if (!has) return;
        check(records.every(r => Object.prototype.hasOwnProperty.call(r, "peak")),
            "Partial peak diagnostic extension");
        const equal = (a, b) => {
            if (a === b) return true;
            if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(
                    a) !== Array.isArray(b)) return false;
            const x = Object.keys(a).sort(),
                y = Object.keys(b).sort();
            return x.length === y.length && x.every((k, i) => k === y[i] && equal(a[k], b[
                k]))
        };
        const finite = (x) => Number.isFinite(x),
            positive = (x) => finite(x) && x > 0,
            count = (x) => Number.isSafeInteger(x) && x >= 0;
        const near = (a, b) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-10 * Math.max(1,
            Math.abs(a), Math.abs(b));
        const total = d.total.peak,
            scale = d.total.luminosity_spectrum?.total_weight || 0;
        const weightClose = (a, b) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-6 * Math
            .max(1, Math.abs(a), Math.abs(b)) + 1e-12 * scale;

        function selection(s) {
            if (s === null) return;
            check(s && s.threshold_fraction === .98 && positive(s.nominal_W_GeV) && near(s
                    .minimum_W_GeV, .98 * s.nominal_W_GeV) && s.upper_cut_GeV === null && s
                .comparison === "W >= minimum_W_GeV", "Invalid native peak selection");
            const r = s.nominal_reference;
            check(r && Array.isArray(r.energy_GeV) && r.energy_GeV.length === 2 && r.energy_GeV
                .every(positive) && r.electron_mass_GeV === .00051099895 && finite(r
                    .full_crossing_angle_rad) && r.full_crossing_angle_rad >= 0 && r
                .full_crossing_angle_rad < Math.PI, "Invalid peak nominal reference");
            check(r.definition ===
                "sqrt(2*m_e^2 + 2*(E1*E2 + p1*p2*cos(full_crossing_angle)))" && r
                .native_W ===
                "Existing massive invariant mass of native simulated incoming directions; no design crossing rotation; beam energies before optional ISR sampling",
                "Unknown peak invariant-mass convention");
            check(["evaluated head-on beam angles",
                "evaluated crab_full_crossing_angle matched to beam-angle receipt"
            ].includes(r.angle_source), "Unadmitted peak nominal angle source");
            check((r.full_crossing_angle_rad === 0) === (r.angle_source ===
                "evaluated head-on beam angles"), "Peak nominal angle/source mismatch");
            check(m.energy_GeV.every((e, i) => near(e, r.energy_GeV[i])),
                "Peak nominal beam-energy source mismatch");
            const mass = r.electron_mass_GeV,
                [e1, e2] = r.energy_GeV;
            check(e1 > mass && e2 > mass, "Invalid peak nominal particle energies");
            const p1 = Math.sqrt((e1 - mass) * (e1 + mass)),
                p2 = Math.sqrt((e2 - mass) * (e2 + mass));
            check(near(s.nominal_W_GeV, Math.sqrt(2 * mass * mass + 2 * (e1 * e2 + p1 * p2 *
                    Math.cos(r.full_crossing_angle_rad)))),
                "Peak nominal mass formula mismatch");
        }
        selection(total?.selection);

        function moment(v, name) {
            check(v && positive(v.mean) && finite(v.rms) && v.rms >= 0, "Invalid peak " + name +
                " moment")
        }
        for (const r of records) {
            const p = r.peak;
            check(p && typeof p.available === "boolean" && p.available === total.available &&
                equal(p.selection, total.selection),
                "Inconsistent native peak availability/selection");
            check([p.weight_sum, p.spin_weight_sum].every(x => finite(x) && x >= 0) && [p
                    .contributions, p.spin_contributions
                ].every(count) && p.spin_contributions <= p.contributions,
                "Invalid peak weights/contributions");
            check((p.spin_contributions === 0) === (p.spin_weight_sum === 0),
                "Peak spin count/weight presence mismatch");
            check(p.spin_weight_sum <= p.weight_sum || weightClose(p.spin_weight_sum, p
                .weight_sum), "Peak spin weight exceeds selected weight");
            if (!p.available) {
                finiteText(p.reason, "native peak unavailability");
                check(p.weight_sum === 0 && p.contributions === 0 && p.spin_weight_sum === 0 &&
                    p.spin_contributions === 0 && p.beam_energy_GeV === null && p.W_GeV ===
                    null && p.polarization === null && p.luminosity_spectrum === null,
                    "Unavailable peak carries data");
                finiteText(p.polarization_unavailable_reason,
                    "peak polarization unavailability");
                continue
            }
            check(p.reason === null && p.selection !== null,
            "Computed peak requires selection");
            const h = histogram(p.luminosity_spectrum, "peak luminosity");
            check(h && h.quantity === "W" && h.unit === "GeV" && h.weight_unit ===
                "m^-2 per BX" && h.weighting === "native collision pair luminosity" && h
                .bins.length === 512 && h.samples === p.contributions,
                "Invalid dedicated peak spectrum");
            check(near(h.bins[0].low, p.selection.minimum_W_GeV) && h.underflow === 0 && h.bins
                .every((b, i) => !i || near(b.low, h.bins[i - 1].high)) && weightClose(h
                    .total_weight, p.weight_sum), "Peak histogram gate/weight mismatch");
            const high = h.bins.at(-1).high,
                low = p.selection.minimum_W_GeV;
            check(h.bins.every((b, i) => near(b.low, low + (high - low) * i / 512) && near(b
                    .high, low + (high - low) * (i + 1) / 512)),
                "Peak histogram must retain native uniform bins");
            check(h.bins.at(-1).high >= 1.02 * p.selection.nominal_W_GeV || near(h.bins.at(-1)
                    .high, 1.02 * p.selection.nominal_W_GeV),
                "Peak spectrum upper domain too small");
            if (r === d.total && r.luminosity_spectrum) check(p.weight_sum <= r
                .luminosity_spectrum.total_weight || weightClose(p.weight_sum, r
                    .luminosity_spectrum.total_weight),
                "Peak luminosity exceeds full spectrum");
            if (p.weight_sum === 0) {
                check(p.contributions === 0 && p.spin_weight_sum === 0 && p
                    .spin_contributions === 0 && p.beam_energy_GeV === null && p.W_GeV ===
                    null, "Empty peak carries moments")
            } else {
                check(p.contributions > 0 && p.beam_energy_GeV,
                    "Selected peak requires moments");
                moment(p.beam_energy_GeV.beam1, "beam1 energy");
                moment(p.beam_energy_GeV.beam2, "beam2 energy");
                moment(p.W_GeV, "W");
                check(p.W_GeV.mean >= p.selection.minimum_W_GeV || near(p.W_GeV.mean, p
                    .selection.minimum_W_GeV), "Peak W mean falls below gate")
            }
            const spinComplete = p.weight_sum > 0 && p.spin_contributions === p.contributions;
            check((p.polarization !== null) === spinComplete,
                "Peak polarization must use every selected contribution");
            if (spinComplete) {
                const q = p.polarization;
                check(q.weighting === "native collision pair luminosity" && weightClose(q
                        .weight_sum, p.weight_sum) && weightClose(p.spin_weight_sum, p
                        .weight_sum) && [q.beam1, q.beam2].every(v => finite(v) && Math.abs(
                        v) <= 1 + 1e-9) && p.polarization_unavailable_reason === null,
                    "Invalid peak polarization/weight coverage")
            } else finiteText(p.polarization_unavailable_reason,
                "peak polarization unavailability");
        }
        if (!total.available) return;
        const parts = d.frames.map(r => r.peak);
        for (const key of ["contributions", "spin_contributions"]) check(parts.reduce((s, p) =>
            s + p[key], 0) === total[key], "Peak frame/total count closure");
        for (const key of ["weight_sum", "spin_weight_sum"]) check(weightClose(parts.reduce((s,
            p) => s + p[key], 0), total[key]), "Peak frame/total weight closure");
        for (const key of ["underflow", "overflow", "total_weight"]) check(weightClose(parts
            .reduce((s, p) => s + p.luminosity_spectrum[key], 0), total
            .luminosity_spectrum[key]), "Peak spectrum flow closure");
        total.luminosity_spectrum.bins.forEach((b, i) => {
            check(parts.every(p => near(p.luminosity_spectrum.bins[i].low, b.low) &&
                    near(p.luminosity_spectrum.bins[i].high, b.high)) &&
                weightClose(parts.reduce((s, p) => s + p.luminosity_spectrum.bins[i]
                    .weight, 0), b.weight), "Peak spectrum interval bin closure")
        });
        if (total.weight_sum > 0) {
            const nonempty = parts.filter(p => p.weight_sum > 0);
            for (const get of[p => p.beam_energy_GeV.beam1, p => p.beam_energy_GeV.beam2, p => p
                    .W_GeV]) {
                const target = get(total),
                    mean = nonempty.reduce((s, p) => s + p.weight_sum / total.weight_sum * get(
                        p).mean, 0),
                    variance = nonempty.reduce((s, p) => s + p.weight_sum / total.weight_sum * (
                        get(p).rms ** 2 + (get(p).mean - target.mean) ** 2), 0);
                check(Math.abs(mean - target.mean) <= 1e-12 + 2e-9 * Math.max(Math.abs(mean),
                        Math.abs(target.mean)) && Math.abs(variance - target.rms ** 2) <=
                    2e-8 * Math.max(variance, target.rms ** 2) + 2e-10 * Math.max(1,
                        variance), "Peak weighted mean/RMS closure");
            }
            if (total.polarization)
                for (const key of ["beam1", "beam2"]) check(near(nonempty.reduce((s, p) => s + p
                        .weight_sum / total.weight_sum * p.polarization[key], 0), total
                    .polarization[key]), "Peak weighted polarization closure");
        }
    }

    function bindPeakReference(d, m, a) {
        const s = d?.total?.peak?.selection;
        if (!s) return;
        const r = s.nominal_reference,
            v = a?.definition,
            near = (x, y) => Number.isFinite(x) && Number.isFinite(y) && Math.abs(x - y) <=
            1e-12 * Math.max(1, Math.abs(x), Math.abs(y));
        check(v && v.angle_receipt_matches_design === true && v.design_W === r.definition &&
            Array.isArray(v.energy_GeV) && v.energy_GeV.length === 2 && v.energy_GeV.every((
                x, i) => near(x, r.energy_GeV[i])) && near(v.full_crossing_angle_rad, r
                .full_crossing_angle_rad) && near(v.nominal_W_GeV, s.nominal_W_GeV),
            "Native peak/design analysis binding mismatch");
        const x = m.native_beam_angles_x_rad,
            y = m.native_beam_angles_y_rad;
        check([x, y].every(z => Array.isArray(z) && z.length === 2 && z.every(Number.isFinite)),
            "Missing peak beam-angle receipt");
        const sum = [...x, ...y].reduce((a, b) => a + Math.abs(b), 0),
            angle = r.full_crossing_angle_rad;
        if (angle === 0) check(sum === 0, "Head-on peak receipt has crossing angles");
        else check((x.every(v => v === 0) || y.every(v => v === 0)) && x[0] * x[1] >= 0 && y[
            0] * y[1] >= 0 && Math.abs(angle - sum) <= 1e-6 * (1 + sum),
            "Peak crossing reference/native-angle mismatch");
    }

    /** Validate frame/total native diagnostics, weighting and centered-moment closure; return the same object or throw. */
    function diagnostics(d, m) {
        check(d?.schema_version === "feebl.collider.diagnostics/1",
            "Unknown native diagnostics schema");
        check(Array.isArray(d.frames) && d.frames.length === m.frames.length && d.frames
            .length <= LIMITS.frames, "Diagnostics frame count mismatch");
        if (d.nominal_polarization !== null) {
            const p = d.nominal_polarization;
            check(p && typeof p === "object", "Missing nominal polarization");
            finiteText(p.convention, "polarization convention");
            check(typeof p.enabled === "boolean", "Missing spin transport/storage flag");
            for (const k of ["beam1", "beam2"]) check(Array.isArray(p[k]) && p[k].length ===
                3 && p[k].every(v => Number.isFinite(v) && Math.abs(v) <= 1 + 1e-9),
                "Invalid nominal spin vector");
            check(Array.isArray(p.reference_helicity) && p.reference_helicity.length === 2 && p
                .reference_helicity.every(v => Number.isFinite(v) && Math.abs(v) <= 1 +
                    1e-9), "Invalid reference helicity");
            check(close(p.reference_helicity[0], p.beam1[2]) && close(p.reference_helicity[1], -
                p.beam2[2]), "Reference helicity convention mismatch");
        }
        check(d.unavailable_reasons && typeof d.unavailable_reasons === "object" && !Array
            .isArray(d.unavailable_reasons), "Missing unavailable reasons");
        check(Object.keys(d.unavailable_reasons).length <= 32, "Unavailable-reason bound");
        for (const s of Object.values(d.unavailable_reasons)) finiteText(s,
            "unavailable reason");
        if (d.nominal_polarization === null) finiteText(d.unavailable_reasons
            .nominal_polarization, "nominal polarization unavailable reason");

        function moments(v, weighting) {
            if (v === null) return;
            check(v && v.weighting === weighting, "Unexpected beam-moment weighting");
            for (const beam of ["beam1", "beam2"]) {
                if (v[beam] === null && weighting !== "native collision pair luminosity")
                    continue;
                check(v[beam] && Number.isFinite(v[beam].weight_sum) && v[beam].weight_sum > 0,
                    "Invalid beam-moment weight sum");
                for (const key of ["energy_GeV", "x_nm", "y_nm", "vx", "vy"]) {
                    const pair = v[beam][key];
                    check(pair && Number.isFinite(pair.mean) && Number.isFinite(pair.rms) &&
                        pair.rms >= 0, "Invalid native beam moment");
                }
            }
        }
        const speciesKeys = ["chi_e_electron", "chi_e_positron"],
            species = d.primary_species;
        const hasSpecies = Object.prototype.hasOwnProperty.call(d, "primary_species");
        if (hasSpecies) {
            check(species && typeof species === "object" && !Array.isArray(species) &&
                typeof species.supported === "boolean" && Number.isFinite(species
                    .charge_sign) && species.basis ===
                "canonical e+e- primary beam assignment; not per-particle species measurement",
                "Invalid primary species mapping");
            check(species.supported === (species.charge_sign === -1),
                "Unsupported primary charge/species mapping");
            check(species.supported ? species.beam1_pdg === 11 && species.beam2_pdg === -11 :
                species.beam1_pdg === null && species.beam2_pdg === null,
                "Invalid primary species PDG assignment");
        } else check([...d.frames, d.total].every(r => speciesKeys.every(key => !Object
                .prototype.hasOwnProperty.call(r?.distributions || {}, key))),
            "Species histograms require mapping metadata");
        const distKeys = ["chi_e", "chi_gamma", "photon_energy", "pair_energy", ...(hasSpecies ?
            speciesKeys : [])];

        function speciesRecord(r) {
            if (!hasSpecies) return;
            const combined = r.distributions.chi_e,
                electron = r.distributions.chi_e_electron,
                positron = r.distributions.chi_e_positron;
            const available = species.supported && combined !== null;
            check((electron !== null) === available && (positron !== null) === available,
                "Inconsistent primary species histogram availability");
            if (!available) return;
            for (const [h, quantity] of [
                    [combined, "native primary kick chi_e"],
                    [electron, "native primary electron kick chi_e"],
                    [positron, "native primary positron kick chi_e"]
                ]) {
                check(h.quantity === quantity && h.unit === "1" && h.weight_unit ===
                    "macroparticle-kick evaluations" && h.weighting ===
                    "one per actual primary field kick; repeated across substeps; before radiation",
                    "Invalid primary chi definition");
                check([h.samples, h.underflow, h.overflow, h.total_weight, ...h.bins.map(b => b
                        .weight)].every(v => Number.isSafeInteger(v) && v >= 0) && h
                    .samples === h.total_weight, "Primary chi requires integer kick counts");
                check(h.bins.reduce((sum, b) => sum + b.weight, h.underflow + h.overflow) === h
                    .total_weight, "Primary chi exact histogram closure failed");
                check(h.bins.length === combined.bins.length && h.bins.every((b, i) => b.low ===
                        combined.bins[i].low && b.high === combined.bins[i].high),
                    "Species chi histogram edges mismatch");
            }
            check(combined.bins.every((b, i) => b.weight === electron.bins[i].weight + positron
                .bins[i].weight), "Species/combined chi bin closure failed");
            check(["underflow", "overflow", "total_weight", "samples"].every(k => combined[
                    k] === electron[k] + positron[k]),
                "Species/combined chi count closure failed");
        }
        const lumiScale = d.total?.luminosity_spectrum?.total_weight || 0;
        const weightClose = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math
            .abs(b)) + 1e-12 * lumiScale;

        function record(r, total = false) {
            check(r && Number.isInteger(r.diagonal_start) && Number.isInteger(r.diagonal_end) &&
                r.diagonal_start >= 0 && r.diagonal_end >= r.diagonal_start,
                "Invalid diagnostic interval");
            const h = histogram(r.luminosity_spectrum, "luminosity");
            if (h) check(h.quantity === "W" && h.unit === "GeV" && h.weight_unit ===
                "m^-2 per BX" && h.weighting === "native collision pair luminosity",
                "Unsupported luminosity histogram units or weighting");
            const p = r.polarization;
            if (p !== null) {
                check(p && p.weighting === "native collision pair luminosity" && Number
                    .isFinite(p.weight_sum) && p.weight_sum > 0,
                    "Invalid polarization weighting");
                check([p.beam1, p.beam2].every(v => Number.isFinite(v) && Math.abs(v) <= 1 +
                    1e-9), "Invalid luminosity helicity");
                if (h) check(weightClose(p.weight_sum, h.total_weight),
                    "Polarization luminosity closure");
            }
            moments(r.beam_moments, "native collision pair luminosity");
            if (r.beam_moments && h)
                for (const b of ["beam1", "beam2"])
                    check(weightClose(r.beam_moments[b].weight_sum, h.total_weight),
                        "Beam-moment luminosity weight closure");
            if (!total || r.bunch_moments !== undefined) moments(r.bunch_moments,
                "all stored native primary macroparticles at diagonal barrier; no inactive-slice drift synchronization"
                );
            if (r.bunch_polarization !== undefined && r.bunch_polarization !== null)
                check(Array.isArray(r.bunch_polarization) && r.bunch_polarization.length ===
                    2 && r.bunch_polarization.every(v => v === null || (Number.isFinite(v) &&
                        Math.abs(v) <= 1 + 1e-9)), "Invalid bunch polarization");
            check(r.distributions && typeof r.distributions === "object",
                "Missing distributions");
            for (const key of distKeys) {
                histogram(r.distributions[key], key);
                if (r.distributions[key] === null) finiteText(d.unavailable_reasons[key], key +
                    " unavailable reason");
            }
            speciesRecord(r);
        }
        let prev = -1;
        d.frames.forEach((r, i) => {
            record(r);
            check(r.id === i && r.diagonal_start === prev + 1 && r.diagonal_end === m
                .frames[i].diagonal, "Diagnostics frame alignment mismatch");
            prev = r.diagonal_end;
        });
        record(d.total, true);
        check(d.total.diagonal_start === 0 && d.total.diagonal_end === prev,
            "Diagnostics total interval mismatch");

        function closure(total, frames, name) {
            if (total === null) {
                check(frames.every(x => x === null), "Partial " + name + " availability");
                return
            }
            check(frames.every(x => x !== null), "Partial " + name + " availability");
            const sum = new Array(total.bins.length).fill(0);
            let under = 0,
                over = 0,
                weight = 0;
            for (const f of frames) {
                check(f.quantity === total.quantity && f.unit === total.unit && f
                    .weight_unit === total.weight_unit && f.weighting === total.weighting,
                    "Histogram metadata differs across intervals");
                check(f.bins.length === total.bins.length,
                    "Histogram edges differ across intervals");
                f.bins.forEach((b, i) => {
                    check(b.low === total.bins[i].low && b.high === total.bins[i].high,
                        "Histogram edges differ across intervals");
                    sum[i] += b.weight
                });
                under += f.underflow;
                over += f.overflow;
                weight += f.total_weight;
            }
            check(sum.every((v, i) => close(v, total.bins[i].weight)) && close(under, total
                .underflow) && close(over, total.overflow) && close(weight, total
                .total_weight), "Frame-to-total " + name + " weight closure");
        }
        closure(d.total.luminosity_spectrum, d.frames.map(f => f.luminosity_spectrum),
            "luminosity");
        for (const key of distKeys) closure(d.total.distributions[key], d.frames.map(f => f
            .distributions[key]), key);
        if (hasSpecies)
            for (const key of ["chi_e", ...speciesKeys]) {
                const h = d.total.distributions[key];
                if (h && species.supported) {
                    const parts = d.frames.map(f => f.distributions[key]);
                    check(h.bins.every((b, i) => parts.reduce((sum, f) => sum + f.bins[i]
                            .weight, 0) === b.weight) && ["underflow", "overflow",
                            "total_weight", "samples"
                        ].every(k => parts.reduce((sum, f) => sum + f[k], 0) === h[k]),
                        "Species chi exact frame/total count closure failed");
                }
            }
        const totalMoments = d.total.beam_moments,
            frameMoments = d.frames.map(f => f.beam_moments).filter(Boolean);
        if (totalMoments) {
            for (const b of ["beam1", "beam2"]) {
                const sum = frameMoments.reduce((a, v) => a + v[b].weight_sum, 0);
                check(close(sum, totalMoments[b].weight_sum),
                    "Beam-moment interval weight closure");
                for (const key of ["energy_GeV", "x_nm", "y_nm", "vx", "vy"]) {
                    const target = totalMoments[b][key],
                        mean = frameMoments.reduce((a, v) => a + v[b].weight_sum * v[b][key]
                            .mean, 0) / sum;
                    const variance = frameMoments.reduce((a, v) => a + v[b].weight_sum * (v[b][
                        key
                    ].rms ** 2 + (v[b][key].mean - target.mean) ** 2), 0) / sum;
                    const scale = Math.max(Math.abs(mean), Math.abs(target.mean),
                        Math.sqrt(variance), target.rms, 1e-140);
                    const meanRoundoff = 128 * Number.EPSILON * scale;
                    const varianceTolerance = 2e-8 * Math.max(variance, target.rms ** 2) +
                        4 * meanRoundoff * Math.max(Math.sqrt(variance), target.rms) +
                        meanRoundoff ** 2;
                    check(Math.abs(mean - target.mean) <= 1e-12 + 2e-8 * Math.abs(target.mean) &&
                        Math.abs(variance - target.rms ** 2) <= varianceTolerance,
                        "Beam-moment weighted mean/RMS closure");
                }
            }
        } else check(frameMoments.length === 0, "Partial beam-moment availability");
        if (d.total.polarization) {
            const ps = d.frames.map(f => f.polarization).filter(Boolean),
                sum = ps.reduce((a, p) => a + p.weight_sum, 0);
            check(close(sum, d.total.polarization.weight_sum),
                "Polarization interval weight closure");
            for (const b of ["beam1", "beam2"]) check(close(ps.reduce((a, p) => a + p
                    .weight_sum * p[b], 0) / sum, d.total.polarization[b]),
                "Polarization mean closure");
        }
        peakDiagnostics(d, m);
        return d;
    }

    const api = {
        LIMITS,
        safeName,
        manifest,
        rows,
        artifact,
        fileRef,
        bytesHash,
        analysis,
        histogram,
        diagnostics,
        peakDiagnostics,
        bindPeakReference
    };
    root.ColliderContract = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);
