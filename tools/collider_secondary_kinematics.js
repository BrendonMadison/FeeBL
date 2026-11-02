/* Bounded final/snapshot native-record kinematics. Frames are not additive. */
(function(root) {
    "use strict";
    const LIMITS = {
        fileBytes: 32 * 1024 * 1024,
        records: 1000000,
        side: 128,
        frames: 64
    };
    const WEIGHTING =
        "one per final stored tracked secondary record; no luminosity or pair-ratio reweighting";
    const FRAME_WEIGHTING =
        "one per current stored tracked secondary record; no luminosity or pair-ratio reweighting";
    const FRAME_PHASE = "post-diagonal-tracker-state";

    function check(ok, message) {
        if (!ok) throw new Error(message)
    }

    function text(v, name) {
        check(typeof v === "string" && v.length > 0 && v.length <= 2048, "Invalid " + name)
    }

    function count(v) {
        return Number.isSafeInteger(v) && v >= 0 && v <= LIMITS.records
    }

    function classificationLabel(f) {
        if (f.do_bhabhas) return "Bhabha/imported particles after beam-field tracking";
        if (f.do_pairs) return f.beam_pair || f.load_events || f.do_comp || f.do_prod ?
            "Pairs + deflected primary/secondary particles (inseparable native output)" :
            "Produced pairs after beam-field tracking";
        return f.load_events || f.do_comp || f.do_prod ?
            "Imported/scattered particles after beam-field tracking" : null;
    }

    function same(a, b) {
        if (a === b) return true;
        if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(a) !==
            Array.isArray(b)) return false;
        const ak = Object.keys(a).sort(),
            bk = Object.keys(b).sort();
        return ak.length === bk.length && ak.every((k, i) => k === bk[i] && same(a[k], b[k]));
    }

    function snapshotAxes(axes) {
        const keys = ["theta_edges_rad", "theta_folded_edges_rad", "pT_edges_GeV",
            "marginal_theta_edges_rad", "marginal_theta_folded_edges_rad",
            "marginal_pT_edges_GeV"
        ];
        check(axes && Object.keys(axes).sort().join(",") === keys.sort().join(","),
            "Unknown native snapshot axes");
        const near = (x, y) => Number.isFinite(x) && Math.abs(x - y) <= Math.max(1e-12 * Math
            .max(Math.abs(x), Math.abs(y)), 1e-30);
        for (const [key, end] of [
                ["theta", Math.PI],
                ["theta_folded", Math.PI / 2],
                ["pT", 1e6]
            ]) {
            const suffix = key === "pT" ? "pT_edges_GeV" : key + "_edges_rad",
                edges = axes[suffix],
                marginal = axes["marginal_" + suffix];
            check(Array.isArray(edges) && edges.length === 129 && edges.every((v, i) => near(v,
                10 ** (-12 + (Math.log10(end) + 12) * i / 128)) && v > 0 && (!i ||
                v > edges[i - 1])), "Native snapshot joint geometry mismatch");
            check(Array.isArray(marginal) && marginal.length === 161 && marginal.every((v, i) =>
                    near(v, key === "pT" ? (i ? 10 ** (-12 + 18 * (i - 1) / 159) : 0) :
                        end * i / 160) && v >= 0 && (!i || v > marginal[i - 1])),
                "Native snapshot marginal geometry mismatch");
        }
    }

    function source(ref, refs) {
        check(ref && /^[A-Za-z0-9_.-]+$/.test(ref.path) && ![".", ".."].includes(ref.path) &&
            Number.isSafeInteger(ref.bytes) && ref.bytes >= 0 && ref.bytes <= 32 * 1024 *
            1024 && /^[a-f0-9]{64}$/.test(ref.sha256), "Invalid kinematics source receipt");
        check(refs.some(r => r.path === ref.path && r.bytes === ref.bytes && r.sha256 === ref
            .sha256), "Kinematics source is not authenticated");
    }

    function joint(h, n) {
        check(h && h.storage_order === "row-major-pT-theta", "Unknown joint storage order");
        for (const edges of [h.theta_edges_rad, h.pT_edges_GeV]) check(Array.isArray(edges) &&
            edges.length >= 2 && edges.length <= LIMITS.side + 1 && edges.every((v, i) =>
                Number.isFinite(v) && v > 0 && (!i || v > edges[i - 1])),
            "Invalid positive logarithmic axis");
        const cells = (h.theta_edges_rad.length - 1) * (h.pT_edges_GeV.length - 1);
        check(Array.isArray(h.counts) && h.counts.length === cells && h.counts.every(count),
            "Invalid joint counts");
        const e = h.excluded,
            keys = ["theta_nonpositive", "pT_nonpositive", "both_nonpositive",
                "nonpositive_records", "underflow_records", "overflow_records"
            ];
        check(e && Object.keys(e).length === keys.length && keys.every(k => count(e[k])),
            "Invalid joint exclusion counts");
        check(e.both_nonpositive <= Math.min(e.theta_nonpositive, e.pT_nonpositive) && e
            .nonpositive_records === e.theta_nonpositive + e.pT_nonpositive - e
            .both_nonpositive, "Nonpositive count union mismatch");
        const sum = h.counts.reduce((a, b) => a + b, 0);
        check(h.samples === n && h.in_range_records === sum && sum + e.nonpositive_records + e
            .underflow_records + e.overflow_records === n, "Joint count closure failed");
    }

    function histogram(h, n, key, weighting = WEIGHTING) {
        const quantities = {
            theta: "native polar angle from +z",
            theta_folded: "native polar angle from nearer +z or -z axis",
            pT: "transverse momentum"
        };
        check(h && h.quantity === quantities[key] && h.unit === (key === "pT" ? "GeV/c" :
            "rad") && h.weight_unit === "stored particle records" && h.weighting ===
            weighting && h.normalization === "bin-integrated",
            "Invalid marginal units or weighting");
        check(h.samples === n && h.total_weight === n && count(h.underflow) && count(h
            .overflow), "Invalid marginal counts");
        check(Array.isArray(h.bins) && h.bins.length > 0 && h.bins.length <= 2048,
            "Marginal bin bound");
        let previous = -Infinity,
            sum = h.underflow + h.overflow;
        for (const b of h.bins) {
            check(Number.isFinite(b.low) && Number.isFinite(b.high) && b.low >= 0 && b.low >=
                previous && b.high > b.low && count(b.weight), "Invalid marginal bin");
            previous = b.high;
            sum += b.weight
        }
        check(sum === n, "Marginal count closure failed");
        if (n === 0) {
            check(h.statistics === null, "Empty frame has invented record statistics");
            return
        }
        const s = h.statistics;
        if (weighting === FRAME_WEIGHTING && key !== "pT") check(s && s.max <= (key ===
                "theta_folded" ? Math.PI / 2 : Math.PI) + 1e-12,
            "Snapshot angle exceeds physical range");
        if (weighting === FRAME_WEIGHTING && h.bins.every(b => b.weight === 0)) {
            check(s && [s.mean, s.variance, s.min, s.max].every(Number.isFinite) && s
                .variance >= 0 && s.min >= 0 && s.min <= s.mean && s.mean <= s.max && s
                .mode === null && s.mode_method ===
                "unavailable: no in-range marginal records",
                "Invalid all-flow snapshot statistics");
            return;
        }
        check(s && [s.mean, s.variance, s.min, s.max, s.mode].every(Number.isFinite) && s
            .variance >= 0 && s.min >= 0 && s.min <= s.mean && s.mean <= s.max && s.mode >=
            h.bins[0].low && s.mode <= h.bins.at(-1).high,
            "Invalid stored-record statistics");
        check(s.mode_method === "centre of highest count-density histogram bin; approximate",
            "Unknown mode estimator");
        const maximum = Math.max(...h.bins.map(b => b.weight / (b.high - b.low)));
        check(h.bins.some(b => b.weight / (b.high - b.low) === maximum && Math.abs(s.mode - (b
                .low + b.high) / 2) <= 1e-12 * Math.max(1, s.mode)),
            "Marginal density mode mismatch");
    }

    function validateFrames(a, refs, context, expected) {
        const frames = a.frames ?? [],
            raw = a.source_artifacts.frame_snapshots;
        check(Array.isArray(frames) && frames.length <= LIMITS.frames,
            "Particle snapshot count bound");
        if (raw != null) {
            source(raw, refs);
            check(raw.path === "gui_secondary_frames.json" && raw.bytes > 0 && raw.bytes <= 16 *
                1024 * 1024, "Native particle snapshot role or size bound")
        }
        if (!frames.length) {
            check(a.frame_availability === false && raw == null,
                "Missing advertised particle snapshots");
            return
        }
        const manifest = context.manifest;
        check(raw && manifest && Array.isArray(manifest.frames) && frames.length === manifest
            .frames.length, "Particle snapshots require matching authenticated manifest");
        check(manifest.secondary_frames_file === raw.path,
            "Particle snapshot source role differs from manifest");
        const native = context.frameSnapshots;
        check(native?.schema_version === "feebl.collider.secondary-frames/1" && native.phase ===
            FRAME_PHASE && native.coordinate_frame === "native_pair_tracking" && native
            .weighting === FRAME_WEIGHTING, "Missing authenticated native snapshot payload");
        check(native.definitions && Object.keys(native.definitions).sort().join(",") ===
            "coordinates,normalization,precision,selection,timing",
            "Invalid native snapshot definitions");
        for (const [k, v] of Object.entries(native.definitions)) text(v, k);
        const definition = {
            schema_version: native.schema_version,
            phase: native.phase,
            coordinate_frame: native.coordinate_frame,
            weighting: native.weighting,
            definitions: native.definitions,
            axes: native.axes
        };
        if (native.position_z_definition !== undefined) definition.position_z_definition =
            native.position_z_definition;
        check(same(a.frame_definition, definition),
        "Snapshot model differs from native source");
        check(native.evaluated_flags && Object.keys(native.evaluated_flags).sort().join(",") ===
            "beam_pair,do_bhabhas,do_comp,do_pairs,do_prod,load_events,pair_ecut,pair_ratio,store_pairs,track_pairs",
            "Missing native snapshot flag contract");
        for (const [key, v] of Object.entries(native.evaluated_flags)) {
            const expectedFlag = a.evaluated_flags[key];
            if (key === "pair_ratio" || key === "pair_ecut") check(Number.isFinite(v) && Math
                .abs(v - expectedFlag) <= Math.max(5e-6 * Math.max(Math.abs(v), Math.abs(
                    expectedFlag)), 5e-12), "Native snapshot flag mismatch");
            else check(Number.isSafeInteger(v) && v >= 0 && (v === expectedFlag || key ===
                "do_bhabhas" && expectedFlag === null && v === 0 && a.evaluated_flags
                .do_pairs > 0), "Native snapshot flag mismatch");
        }
        snapshotAxes(native.axes);
        check(Array.isArray(native.frames) && native.frames.length === frames.length,
            "Native snapshot frames missing");
        let computed = 0;
        const geometry = new Map();
        for (let i = 0; i < frames.length; i++) {
            const f = frames[i],
                m = manifest.frames[i],
                rawFrame = native.frames[i];
            check(f && f.id === i && f.id === m.id && Number.isSafeInteger(f.diagonal) && f
                .diagonal >= 0 && (!i || f.diagonal > frames[i - 1].diagonal) && f
                .diagonal === m.diagonal && Number.isFinite(f.elapsed_s) && f.elapsed_s ===
                m.elapsed_s && f.phase === FRAME_PHASE,
                "Particle snapshot timing does not match displayed frame");
            check(rawFrame && rawFrame.id === f.id && rawFrame.diagonal === f.diagonal &&
                rawFrame.available === f.available && rawFrame.reason === f.reason,
                "Snapshot availability differs from native source");
            check(typeof f.available === "boolean" && f.populations && Object.keys(f
                    .populations).sort().join(",") === "combined,deflected,pairs",
                "Invalid particle snapshot availability");
            if (f.available) {
                computed++;
                check(f.reason === null && a.evaluated_flags.track_pairs > 0,
                    "Computed snapshot requires active native tracking")
            } else text(f.reason, "particle snapshot unavailability");
            const verification = f.tracker_count_verification;
            check(verification && verification.available === f.available && count(verification
                    .actual_records) && verification.actual_records === rawFrame.nrecords,
                "Invalid tracker count verification");
            if (f.available) check(verification.reason === null && verification
                .admitted_tracker_count === verification.actual_records && verification
                .authoritative_reserve_count === verification.actual_records,
                "Authoritative tracker count closure failed");
            else {
                text(verification.reason, "tracker count unavailability");
                check(verification.admitted_tracker_count === null && verification
                    .authoritative_reserve_count === null && verification.actual_records ===
                    0 && ["joint", "joint_folded", "marginals", "selection"].every(k =>
                        rawFrame[k] === null), "Unavailable tracker carries records")
            }
            let observed = 0,
                available = 0;
            for (const [key, p] of Object.entries(f.populations)) {
                text(p.label, "snapshot population label");
                check(typeof p.available === "boolean" && count(p.nrecords),
                    "Invalid snapshot population");
                if (!p.available) {
                    text(p.reason, "snapshot population unavailability");
                    check(p.nrecords === 0 && p.joint === null && p.joint_folded === null && p
                        .selection === null && p.normalization === null && p
                        .distributions && ["theta", "theta_folded", "pT"].every(k => p
                            .distributions[k] === null),
                        "Unavailable snapshot population carries data");
                    continue
                }
                available++;
                observed += p.nrecords;
                check(f.available && key === expected && p.label === classificationLabel(a
                        .evaluated_flags) && p.reason === null,
                    "Unadmitted snapshot population or label");
                joint(p.joint, p.nrecords);
                joint(p.joint_folded, p.nrecords);
                for (const name of ["joint", "joint_folded"]) {
                    const h = p[name],
                        edges = JSON.stringify([h.theta_edges_rad, h.pT_edges_GeV]),
                        old = geometry.get(name);
                    check(!old || old === edges,
                        "Native snapshot histogram axes changed within run");
                    geometry.set(name, edges);
                    check(same(h.theta_edges_rad, native.axes[name === "joint" ?
                            "theta_edges_rad" : "theta_folded_edges_rad"]) && same(h
                            .pT_edges_GeV, native.axes.pT_edges_GeV),
                        "Joint axes differ from native source");
                    for (const field of ["counts", "samples", "in_range_records", "excluded"])
                        check(same(h[field], rawFrame[name]?.[field]),
                            "Joint data differ from native source");
                }
                for (const name of ["theta", "theta_folded", "pT"]) {
                    const h = p.distributions[name],
                        r = rawFrame.marginals?.[name],
                        edges = native.axes["marginal_" + (name === "pT" ? "pT_edges_GeV" :
                            name + "_edges_rad")];
                    histogram(h, p.nrecords, name, FRAME_WEIGHTING);
                    check(r && Array.isArray(edges) && edges.length === h.bins.length + 1 && h
                        .bins.every((b, j) => b.low === edges[j] && b.high === edges[j +
                            1] && b.weight === r.counts?.[j]),
                        "Marginal bins differ from native source");
                    for (const field of ["samples", "underflow", "overflow", "total_weight",
                            "statistics"
                        ]) check(same(h[field], r[field]),
                        "Marginal data differ from native source");
                }
                const sel = p.selection,
                    flags = a.evaluated_flags;
                check(sel && sel.origin_separation === false && sel.pair_ratio === flags
                    .pair_ratio && sel.pair_ecut_GeV === flags.pair_ecut && sel
                    .admitted_tracker_count === p.nrecords && sel
                    .authoritative_reserve_count === p.nrecords,
                    "Snapshot selection or tracker count mismatch");
                check(sel.process_counts && Object.keys(sel.process_counts).length <= 128 &&
                    Object.entries(sel.process_counts).every(([k, v]) => Number.isInteger(
                            Number(k)) && String(Number(k)) === k && Number(k) >= -999 &&
                        Number(k) <= 999 && count(v)) && Object.values(sel.process_counts)
                    .reduce((x, y) => x + y, 0) === p.nrecords,
                    "Snapshot process count closure failed");
                check(sel.signed_energy_counts && Object.keys(sel.signed_energy_counts).sort()
                    .join(",") === "negative_energy,positive_energy" && Object.values(sel
                        .signed_energy_counts).every(count) && Object.values(sel
                        .signed_energy_counts).reduce((x, y) => x + y, 0) === p.nrecords,
                    "Snapshot signed-energy count closure failed");
                for (const name of ["process_counts", "signed_energy_counts",
                        "admitted_tracker_count", "authoritative_reserve_count"
                    ]) check(same(sel[name], rawFrame.selection?.[name]),
                    "Tracker selection differs from native source");
                const norm = p.normalization;
                check(norm?.kind === "stored-record counts" && norm.weight_unit ===
                    "stored particle records" && norm.weighting === FRAME_WEIGHTING && norm
                    .physical_weight_available === false && norm.completeness_verified ===
                    false, "Snapshot normalization cannot claim final-record completeness");
            }
            check(available === (f.available ? 1 : 0) && observed === verification
                .actual_records, "Snapshot record population closure failed");
        }
        check(a.frame_availability === (computed > 0),
            "Particle snapshot availability mismatch");
    }
    const POSITION_Z_DEFINITION = {
        coordinate: "stored native z position",
        unit: "nm",
        positive: "z > 0",
        negative: "z < 0",
        zero: "z == 0; included only in All",
        direction_not_used: true
    };

    function sparseCounts(h, cells) {
        check(h && Object.keys(h).sort().join(",") ===
            "counts_sparse,excluded,in_range_records,samples" && Array.isArray(h
                .counts_sparse) && h.counts_sparse.length <= cells,
            "Invalid sparse z-selected joint map");
        const result = new Array(cells).fill(0);
        let previous = -1;
        for (const pair of h.counts_sparse) {
            check(Array.isArray(pair) && pair.length === 2 && Number.isSafeInteger(pair[0]) &&
                pair[0] > previous && pair[0] < cells && count(pair[1]) && pair[1] > 0,
                "Sparse z joint indices/counts must be sorted unique and positive");
            previous = pair[0];
            result[pair[0]] = pair[1]
        }
        return result;
    }

    function subsetPopulation(pop, key, weighting) {
        const s = pop.position_z[key];
        const buildJoint = (h, all) => ({
            ...all,
            ...h,
            counts: sparseCounts(h, all.counts.length)
        });
        const distributions = {};
        for (const name of ["theta", "theta_folded", "pT"]) {
            const h = s.marginals[name],
                all = pop.distributions[name];
            distributions[name] = {
                ...all,
                ...h,
                weighting,
                bins: all.bins.map((b, i) => ({
                    ...b,
                    weight: h.counts[i]
                }))
            }
        }
        return {
            ...pop,
            nrecords: s.nrecords,
            joint: buildJoint(s.joint, pop.joint),
            joint_folded: buildJoint(s.joint_folded, pop.joint_folded),
            distributions,
            selection: {
                ...pop.selection,
                process_counts: s.selection.process_counts,
                signed_energy_counts: s.selection.signed_energy_counts
            },
            normalization: {
                ...pop.normalization,
                completeness_verified: false
            },
            position_z_selection: key
        };
    }

    /** Return the inclusive, positive-z or negative-z stored-position population without changing the authenticated input. The selection is not a momentum-hemisphere cut. */
    function selectZ(pop, key = "all", weighting = WEIGHTING) {
        check(["all", "positive", "negative"].includes(key), "Unknown stored-z selection");
        if (!pop || key === "all" || !pop.available) return pop;
        if (!pop.position_z) return {
            ...pop,
            available: false,
            reason: "Stored z-position subsets were not exported for this dataset; All remains available.",
            nrecords: 0,
            joint: null,
            joint_folded: null,
            distributions: {
                theta: null,
                theta_folded: null,
                pT: null
            },
            selection: null,
            normalization: null
        };
        const selected = subsetPopulation(pop, key, weighting);
        // Full tracker ownership/completeness receipts do not apply to a z subset.
        delete selected.selection.admitted_tracker_count;
        delete selected.selection.authoritative_reserve_count;
        return selected;
    }

    function validateZPopulation(pop, weighting) {
        const z = pop.position_z;
        check(z && Object.keys(z).sort().join(",") === "negative,positive,zero_records" &&
            count(z.zero_records), "Invalid stored-z partition");
        const selected = [];
        for (const key of ["positive", "negative"]) {
            const s = z[key];
            check(s && Object.keys(s).sort().join(",") ===
                "joint,joint_folded,marginals,nrecords,selection" && count(s.nrecords),
                "Invalid z subset population");
            check(s.marginals && Object.keys(s.marginals).sort().join(",") ===
                "pT,theta,theta_folded", "Missing z subset marginals");
            for (const name of ["theta", "theta_folded", "pT"]) {
                const h = s.marginals[name];
                check(h && Object.keys(h).sort().join(",") ===
                    "counts,overflow,samples,statistics,total_weight,underflow" && Array
                    .isArray(h.counts) && h.counts.length === pop.distributions[name].bins
                    .length && h.counts.every(count),
                    "Z marginal bin geometry/count mismatch")
            }
            const p = subsetPopulation(pop, key, weighting);
            selected.push(p);
            joint(p.joint, p.nrecords);
            joint(p.joint_folded, p.nrecords);
            for (const name of ["theta", "theta_folded", "pT"]) histogram(p.distributions[name],
                p.nrecords, name, weighting);
            const sel = s.selection;
            check(sel && Object.keys(sel).sort().join(",") ===
                "process_counts,signed_energy_counts",
                "Z subset cannot claim full tracker ownership");
            check(sel.process_counts && Object.keys(sel.process_counts).length <= 128 && Object
                .entries(sel.process_counts).every(([k, v]) => Number.isInteger(Number(
                    k)) && String(Number(k)) === k && Number(k) >= -999 && Number(k) <=
                    999 && count(v)) && Object.values(sel.process_counts).reduce((x, y) =>
                    x + y, 0) === p.nrecords, "Z subset process count closure");
            check(sel.signed_energy_counts && Object.keys(sel.signed_energy_counts).sort().join(
                    ",") === "negative_energy,positive_energy" && Object.values(sel
                    .signed_energy_counts).every(count) && Object.values(sel
                    .signed_energy_counts).reduce((x, y) => x + y, 0) === p.nrecords,
                "Z subset signed-energy count closure");
        }
        const [positive, negative] = selected;
        check(positive.nrecords + negative.nrecords + z.zero_records === pop.nrecords,
            "Stored-z partition record closure");
        for (const name of ["joint", "joint_folded"]) {
            const all = pop[name],
                p = positive[name],
                n = negative[name];
            check(all.counts.every((v, i) => p.counts[i] + n.counts[i] <= v),
                "Z subsets exceed inclusive joint bin");
            for (const k of Object.keys(all.excluded)) check(p.excluded[k] + n.excluded[k] <=
                all.excluded[k], "Z subsets exceed inclusive joint exclusions");
            const counts = all.counts.map((v, i) => v - p.counts[i] - n.counts[i]),
                excluded = Object.fromEntries(Object.keys(all.excluded).map(k => [k, all
                    .excluded[k] - p.excluded[k] - n.excluded[k]
                ]));
            joint({
                ...all,
                counts,
                excluded,
                samples: z.zero_records,
                in_range_records: counts.reduce((a, b) => a + b, 0)
            }, z.zero_records);
        }
        const near = (a, b) => Math.abs(a - b) <= 2e-9 * Math.max(Math.abs(a), Math.abs(b)) +
            1e-12;
        for (const name of ["theta", "theta_folded", "pT"]) {
            const all = pop.distributions[name],
                p = positive.distributions[name],
                n = negative.distributions[name];
            check(all.bins.every((b, i) => p.bins[i].weight + n.bins[i].weight <= b.weight) && p
                .underflow + n.underflow <= all.underflow && p.overflow + n.overflow <= all
                .overflow, "Z subsets exceed inclusive marginal counts");
            if (pop.nrecords) {
                const occupied = [positive, negative].filter(p => p.nrecords > 0),
                    s = all.statistics;
                check(occupied.every(p => p.distributions[name].statistics.min >= s.min -
                        1e-12 * Math.max(1, Math.abs(s.min)) && p.distributions[name]
                        .statistics.max <= s.max + 1e-12 * Math.max(1, Math.abs(s.max))),
                    "Z subset extrema exceed inclusive record bounds");
                if (!z.zero_records) {
                    const mean = occupied.reduce((v, p) => v + p.nrecords / pop.nrecords * p
                            .distributions[name].statistics.mean, 0),
                        variance = occupied.reduce((v, p) => {
                            const t = p.distributions[name].statistics;
                            return v + p.nrecords / pop.nrecords * (t.variance + (t.mean - s
                                .mean) ** 2)
                        }, 0);
                    check(near(mean, s.mean) && near(variance, s.variance) && near(Math.min(...
                                occupied.map(p => p.distributions[name].statistics.min)), s
                            .min) && near(Math.max(...occupied.map(p => p.distributions[
                            name].statistics.max)), s.max),
                        "Z subset marginal moment closure");
                }
            }
        }
        for (const group of ["process_counts", "signed_energy_counts"]) {
            const all = pop.selection[group],
                p = positive.selection[group],
                n = negative.selection[group];
            for (const k of new Set([...Object.keys(p), ...Object.keys(n)])) check((p[k] || 0) +
                (n[k] || 0) <= (all[k] || 0), "Z subsets exceed inclusive record selection")
        }
    }

    function validateZSelections(a, context) {
        const native = context.frameSnapshots,
            advertised = Object.prototype.hasOwnProperty.call(a, "position_z_definition"),
            rawAdvertised = !!native && Object.prototype.hasOwnProperty.call(native,
                "position_z_definition");
        check(rawAdvertised || !native?.frames?.some(f => f.position_z != null),
            "Native z partitions omitted their definition");
        const records = [a, ...(a.frames || [])],
            any = records.some(r => Object.values(r.populations).some(p => p.position_z !=
                null));
        if (!advertised) {
            check(!any && !rawAdvertised, "Z partition definition omitted");
            return
        }
        check(same(a.position_z_definition, POSITION_Z_DEFINITION),
            "Unknown stored-z selection definition");
        if (rawAdvertised) check(same(native.position_z_definition, a.position_z_definition),
            "Z selection differs from authenticated native definition");
        records.forEach((r, i) => {
            const raw = i ? native?.frames[i - 1] : null;
            if (rawAdvertised && raw && !raw.available) check(raw.position_z === null,
                "Unavailable native frame carries z partitions");
            for (const pop of Object.values(r.populations)) {
                if (!pop.available) {
                    check(pop.position_z == null,
                        "Unavailable population carries z partitions");
                    continue
                }
                if (i && !rawAdvertised) {
                    check(pop.position_z == null,
                        "Old native frames cannot supply z partitions");
                    continue
                }
                validateZPopulation(pop, i ? FRAME_WEIGHTING : WEIGHTING);
                if (i) check(raw && same(pop.position_z, raw.position_z),
                    "Z subsets differ from authenticated native frame");
            }
        });
    }

    /** Validate bounded native secondary populations and optional frame/z partitions against authenticated inputs. Angles are radians and pT is GeV/c; throw on mismatch. */
    function validate(a, refs = [], context = {}) {
        check(a?.schema_version === "feebl.collider.secondary-kinematics/1" && a.phase ===
            "final" && typeof a.frame_availability === "boolean" && a.coordinate_frame ===
            "native_pair_tracking", "Unknown final kinematics contract");
        check(a.source_artifacts && a.source_artifacts.summary, "Missing summary receipt");
        source(a.source_artifacts.summary, refs);
        if (a.source_artifacts.pairs !== null) source(a.source_artifacts.pairs, refs);
        if (a.source_artifacts.imported_input != null) source(a.source_artifacts.imported_input,
            refs);
        check(a.definitions && Object.keys(a.definitions).length <= 16,
            "Missing kinematics definitions");
        for (const [k, v] of Object.entries(a.definitions)) text(v, k);
        const f = a.evaluated_flags;
        check(f && typeof f === "object", "Missing evaluated flags");
        for (const k of ["do_pairs", "load_events", "track_pairs", "store_pairs", "beam_pair",
                "do_comp", "do_prod"
            ]) check(Number.isSafeInteger(f[k]) && f[k] >= 0, "Invalid evaluated flag " + k);
        check(f.do_bhabhas === null || Number.isSafeInteger(f.do_bhabhas) && f.do_bhabhas >= 0,
            "Invalid Bhabha flag");
        check(Number.isFinite(f.pair_ratio) && f.pair_ratio >= 0 && f.pair_ratio <= 1 && Number
            .isFinite(f.pair_ecut) && f.pair_ecut >= 0, "Invalid pair storage selection");
        check(!(f.do_bhabhas > 0 && (f.do_pairs > 0 || f.do_comp > 0)),
            "Bhabha output cannot be mixed with native pair/Compton columns");
        if (a.source_artifacts.imported_input != null) check(f.load_events > 0 && a
            .source_artifacts.imported_input.path === "event.ini",
            "Unadmitted imported-input provenance");
        const pops = a.populations;
        check(pops && Object.keys(pops).sort().join(",") === "combined,deflected,pairs",
            "Unknown populations");
        let available = 0;
        const extra = Boolean(f.load_events || f.do_comp || f.do_prod || f.do_bhabhas),
            expected = f.do_pairs ? (f.beam_pair || extra ? "combined" : "pairs") : (extra ?
                "deflected" : null);
        for (const [key, p] of Object.entries(pops)) {
            text(p.label, "population label");
            check(typeof p.available === "boolean" && count(p.nrecords),
                "Invalid population availability/count");
            if (!p.available) {
                text(p.reason, "population unavailability");
                check(p.nrecords === 0 && p.joint === null && p.joint_folded === null && p
                    .selection === null && p.normalization === null && p.distributions && [
                        "theta", "theta_folded", "pT"
                    ].every(k => p.distributions[k] === null),
                    "Unavailable population carries data");
                continue
            }
            available++;
            check(key === expected && f.track_pairs > 0 && p.nrecords > 0 && p.reason ===
                null && a.source_artifacts.pairs !== null,
                "Unadmitted population classification");
            check(f.do_pairs > 0 || f.do_bhabhas !== null, "Unknown nonpair column contract");
            joint(p.joint, p.nrecords);
            joint(p.joint_folded, p.nrecords);
            for (const k of ["theta", "theta_folded", "pT"]) histogram(p.distributions[k], p
                .nrecords, k);
            const sel = p.selection;
            check(sel && sel.origin_separation === false && sel.pair_ratio === f.pair_ratio &&
                sel.pair_ecut_GeV === f.pair_ecut,
                "Unsupported source separation or selection");
            check(sel.process_counts && Object.keys(sel.process_counts).length <= 128 && Object
                .entries(sel.process_counts).every(([k, v]) => /^-?\d+$/.test(k) && count(
                    v)) && Object.values(sel.process_counts).reduce((a, b) => a + b, 0) ===
                p.nrecords, "Process count closure failed");
            check(sel.signed_energy_counts && Object.keys(sel.signed_energy_counts).sort().join(
                    ",") === "negative_energy,positive_energy" && Object.values(sel
                    .signed_energy_counts).every(count) && Object.values(sel
                    .signed_energy_counts).reduce((a, b) => a + b, 0) === p.nrecords,
                "Signed-energy count closure failed");
            const norm = p.normalization;
            check(norm?.kind === "stored-record counts" && norm.weight_unit ===
                "stored particle records" && norm.weighting === WEIGHTING && norm
                .physical_weight_available === false && typeof norm
                .completeness_verified === "boolean", "Unadmitted secondary normalization");
            check(norm.completeness_verified === a.record_count_verification?.available,
                "Conflicting completeness receipts");
            if (norm.completeness_verified) check(a.record_count_verification?.available ===
                true && a.record_count_verification.expected_records === p.nrecords && a
                .record_count_verification.actual_records === p.nrecords,
                "Native record completeness mismatch");
        }
        check(available <= 1,
            "Legacy output cannot be split into separate simultaneous origins");
        const verification = a.record_count_verification,
            observed = Object.values(pops).reduce((sum, p) => sum + p.nrecords, 0);
        check(verification && typeof verification.available === "boolean" && count(verification
                .actual_records) && verification.actual_records === observed,
            "Stored-record verification mismatch");
        if (verification.available) {
            check(expected === "pairs" && f.pair_ratio === 1 && f.pair_ecut <= 0.00051099895,
                "Unadmitted native completeness inference");
            check(verification.expected_records === observed,
                "Expected stored-record count mismatch")
        } else check(verification.expected_records === null,
            "Unavailable record expectation carries count");
        text(verification.reason, "record count qualification");
        validateFrames(a, refs, context, expected);
        validateZSelections(a, context);
        return a;
    }
    const axisCache = new WeakMap();

    /** Return fixed log10(theta/rad), log10(pT/(GeV/c)) bounds from all available scopes with 0.25-decade padding; cache by dataset and projection. */
    function axisBounds(a, key, folded = false) {
        let cache = axisCache.get(a);
        if (!cache) {
            cache = new Map();
            axisCache.set(a, cache)
        }
        const cacheKey = key + ":" + folded;
        if (cache.has(cacheKey)) return cache.get(cacheKey);
        const joints = [a.populations?.[key], ...(a.frames || []).map(f => f.populations?.[
            key])].filter(p => p?.available).map(p => p[folded ? "joint_folded" : "joint"]);
        let xmin = Infinity,
            xmax = -Infinity,
            ymin = Infinity,
            ymax = -Infinity;
        const final = a.populations?.[key];
        if (final?.available) {
            const h = final[folded ? "joint_folded" : "joint"];
            xmin = h.theta_edges_rad[0];
            xmax = h.theta_edges_rad.at(-1);
            ymin = h.pT_edges_GeV[0];
            ymax = h.pT_edges_GeV.at(-1)
        }
        for (const h of joints) {
            const nx = h.theta_edges_rad.length - 1;
            for (let cell = 0; cell < h.counts.length; cell++)
                if (h.counts[cell] > 0) {
                    const i = cell % nx,
                        j = Math.floor(cell / nx);
                    xmin = Math.min(xmin, h.theta_edges_rad[i]);
                    xmax = Math.max(xmax, h.theta_edges_rad[i + 1]);
                    ymin = Math.min(ymin, h.pT_edges_GeV[j]);
                    ymax = Math.max(ymax, h.pT_edges_GeV[j + 1]);
                }
        }
        const occupied = Number.isFinite(xmin);
        if (!occupied)
            for (const h of joints) {
                xmin = Math.min(xmin, h.theta_edges_rad[0]);
                xmax = Math.max(xmax, h.theta_edges_rad.at(-1));
                ymin = Math.min(ymin, h.pT_edges_GeV[0]);
                ymax = Math.max(ymax, h.pT_edges_GeV.at(-1))
            }
        if (!Number.isFinite(xmin)) {
            xmin = 1e-12;
            xmax = Math.PI;
            ymin = 1e-12;
            ymax = 1e6
        }
        const result = {
            xmin: Math.log10(xmin) - .25,
            xmax: Math.log10(xmax) + .25,
            ymin: Math.log10(ymin) - .25,
            ymax: Math.log10(ymax) + .25,
            padding_decades: .25,
            scope: occupied ? "final-axes-and-frame-occupied-bin-union" :
                "empty-histogram-axis-fallback"
        };
        cache.set(cacheKey, result);
        return result;
    }
    const api = {
        LIMITS,
        WEIGHTING,
        FRAME_WEIGHTING,
        FRAME_PHASE,
        POSITION_Z_DEFINITION,
        sparseCounts,
        selectZ,
        validateZSelections,
        validate,
        axisBounds
    };
    root.ColliderSecondaryKinematics = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);
