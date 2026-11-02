/* Identified native trajectories, never inferred from final output order or charge. */
(function(root) {
    "use strict";
    const LIMITS = {
        fileBytes: 16 * 1024 * 1024,
        records: 8192,
        events: 8192,
        plots: 32,
        frames: 64,
        bins: 160
    };
    const WEIGHTING = "one count per admitted tracked record; no luminosity or event weighting";
    const DEFINITIONS = {
        theta: "atan2(hypot(vx,vy),vz)",
        phi: "atan2(vy,vx)",
        delta_theta: "theta_current - theta_initial",
        delta_phi: "wrapped current phi minus initial phi in [-180,180) degrees",
        initial_reference: "actual native float state at admission",
        weighting: WEIGHTING
    };
    const INITIAL = ["label", "source_record", "scheduled_time_counter",
            "admitted_time_counter", "signed_energy_GeV", "x_nm", "y_nm", "z_nm", "vx", "vy",
            "vz"
        ],
        CURRENT = ["label", "vx", "vy", "vz", "main_mesh_gathers", "outer_mesh_gathers",
            "monopole_gathers"
        ];
    const check = (ok, msg) => {
            if (!ok) throw new Error(msg)
        },
        integer = (v, limit = Number.MAX_SAFE_INTEGER) => Number.isSafeInteger(v) && v >= 0 &&
        v <= limit;
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b),
        finite = Number.isFinite;

    function objectSame(a, b) {
        if (a === b) return true;
        if (!a || !b || typeof a !== "object" || typeof b !== "object" || Array.isArray(a) !==
            Array.isArray(b)) return false;
        const ka = Object.keys(a).sort(),
            kb = Object.keys(b).sort();
        return same(ka, kb) && ka.every(k => objectSame(a[k], b[k]))
    }

    function text(v, name) {
        check(typeof v === "string" && v.length > 0 && v.length <= 2048, "Invalid " + name)
    }

    function near(a, b) {
        return finite(a) && finite(b) && Math.abs(a - b) <= 1e-12 + 2e-9 * Math.max(Math.abs(a),
            Math.abs(b))
    }

    function ref(r, refs, path) {
        check(r && r.path === path && integer(r.bytes, 32 * 1024 * 1024) && /^[a-f0-9]{64}$/
            .test(r.sha256) && refs.some(x => objectSame(x, r)),
            "Unauthenticated event source " + path)
    }

    /** Compute current-minus-initial polar and wrapped azimuth changes in degrees from native identity/trajectory rows. Undefined azimuth returns null. */
    function angles(initial, current) {
        const a = initial.slice(8, 11),
            b = current.slice(1, 4),
            theta = v => Math.atan2(Math.hypot(v[0], v[1]), v[2]);
        let phi = null;
        if (Math.hypot(a[0], a[1]) > 0 && Math.hypot(b[0], b[1]) > 0) {
            const delta = Math.atan2(b[1], b[0]) * 180 / Math.PI - Math.atan2(a[1], a[0]) *
                180 / Math.PI;
            phi = ((delta + 180) % 360 + 360) % 360 - 180
        }
        return {
            delta_theta: theta(b) * 180 / Math.PI - theta(a) * 180 / Math.PI,
            delta_phi: phi
        };
    }

    /** Bin signed degree values with explicit underflow/overflow and centered statistics; undefined angles are excluded. */
    function histogramValues(values, edges) {
        const counts = Array(edges.length - 1).fill(0);
        let underflow = 0,
            overflow = 0;
        for (const value of values) {
            if (value < edges[0]) {
                underflow++;
                continue
            }
            if (value > edges.at(-1)) {
                overflow++;
                continue
            }
            let low = 0,
                high = counts.length;
            while (high - low > 1) {
                const mid = (low + high) >> 1;
                if (value < edges[mid]) high = mid;
                else low = mid
            }
            counts[low]++
        }
        if (!values.length) return {
            counts,
            underflow,
            overflow,
            statistics: null
        };
        const origin = values[0],
            meanOffset = values.reduce((s, v) => s + (v - origin) / values.length, 0),
            mean = origin + meanOffset;
        const variance = values.reduce((s, v) => s + ((v - origin) - meanOffset) ** 2 / values
            .length, 0);
        let maximum = 0,
            index = -1;
        counts.forEach((count, i) => {
            const density = count / (edges[i + 1] - edges[i]);
            if (density > maximum) {
                maximum = density;
                index = i
            }
        });
        return {
            counts,
            underflow,
            overflow,
            statistics: {
                mean,
                variance,
                min: Math.min(...values),
                max: Math.max(...values),
                mode: index < 0 ? null : (edges[index] + edges[index + 1]) / 2
            }
        };
    }

    /** Bind event identity, native input and trajectory artifacts; verify angular histograms in degrees for each event slot. Return the same object or throw. */
    function validate(a, refs = [], context = {}) {
        check(a?.schema_version === "feebl.collider.event-deflections/1" && typeof a
            .available === "boolean", "Unknown event-deflection contract");
        check(a.coordinate_frame === "native_pair_tracking" && a.phase ===
            "post-diagonal-tracker-state" && objectSame(a.definitions, DEFINITIONS),
            "Unknown event angle/weight convention");
        if (!a.available) {
            text(a.reason, "event unavailability");
            check(Array.isArray(a.plots) && a.plots.length === 0 && Array.isArray(a.frames) && a
                .frames.length === 0 && a.total === null,
                "Unavailable events carry histograms");
            return a
        }
        check(a.reason === null, "Available event analysis carries failure reason");
        const sources = a.source_artifacts;
        check(sources && Object.keys(sources).length === 4, "Event source roles missing");
        for (const [key, path] of [
                ["trajectories", "gui_imported_trajectories.json"],
                ["identity", "event_identity.json"],
                ["input", "event.ini"],
                ["summary", "out.dat"]
            ]) ref(sources[key], refs, path);
        const {
            manifest,
            trajectories: raw,
            identity
        } = context;
        check(typeof context.inputText === "string" && context.inputText.length <= 32 * 1024 *
            1024, "Authenticated imported input bytes missing");
        const inputLines = context.inputText.replace(/\r/g, "").replace(/\n$/, "").split("\n"),
            input = [];
        for (const line of inputLines) {
            check(line.length > 0 && line.length <= 2048, "Invalid imported input line");
            const row = line.trim().split(/\s+/).map(Number);
            check(row.length === 8 && row.every(finite) && integer(row[0], 2147483647) && row[
                1] !== 0 && Math.hypot(...row.slice(5)) > 0 && row.slice(5).reduce((s, v) =>
                    s + v * v, 0) <= 1 + 4e-6 && (!input.length || row[0] >= input.at(-1)[
                    0]), "Invalid imported input row");
            input.push([row[0], ...row.slice(1).map(Math.fround)])
        }
        check(input.length > 0 && input.length <= LIMITS.records && input.flat().every(finite),
            "Imported input exceeds native bound");
        check(["native_beam_angles_x_rad", "native_beam_angles_y_rad"].every(k => Array.isArray(
            manifest?.[k]) && manifest[k].length === 2 && manifest[k].every(v =>
            v === 0)), "Event deflections require head-on native axes");
        check(manifest && manifest.imported_trajectories_file === sources.trajectories.path,
            "Native manifest trajectory role mismatch");
        check(raw?.schema_version === "feebl.collider.imported-trajectories/1" && raw
            .available === true && raw.reason === null && raw.coordinate_frame === a
            .coordinate_frame && raw.phase === a.phase && raw.source_file === "event.ini",
            "Missing admitted native trajectories");
        const flags = raw.evaluated_flags;
        check(flags && flags.load_events === 1 && flags.track_pairs === 1 && ["do_pairs",
                "beam_pair", "do_comp", "do_prod", "do_bhabhas"
            ].every(k => flags[k] === 0) && finite(flags.pair_ratio) && flags.pair_ratio >=
            0 && flags.pair_ratio <= 1 && finite(flags.pair_ecut) && flags.pair_ecut >= 0 &&
            [-1, 0].includes(flags.charge_sign), "Unsupported/mixed imported tracker flags");
        check(typeof context.summaryText === "string",
            "Authenticated evaluated summary missing");
        for (const [key, value] of Object.entries(flags)) {
            const pattern = new RegExp("(?<![\\w.])" + key +
                    "\\s*=\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)(?=\\s|;|$)",
                    "g"),
                matches = [...context.summaryText.matchAll(pattern)];
            check(matches.length === 1 && finite(value) && Math.abs(Number(matches[0][1]) -
                value) <= Math.max(5e-12, 5e-6 * Math.max(Math.abs(value), Math.abs(
                Number(matches[0][1])))), "Native/summary flag mismatch: " + key)
        }
        check(same(raw.initial_columns, INITIAL) && same(raw.current_columns, CURRENT),
            "Unknown native trajectory columns");
        check(Array.isArray(raw.initial_records) && raw.initial_records.length <= LIMITS
            .records && Array.isArray(raw.frames) && raw.frames.length === manifest.frames
            .length && raw.frames.length <= LIMITS.frames, "Native trajectory count bound");
        check(identity?.schema_version === "feebl.collider.manufactured-events/1" && identity
            .coordinate_frame === "native_pair_tracking" && objectSame(identity
                .source_artifact, sources.input) && Array.isArray(identity.events) &&
            identity.events.length > 0 && identity.events.length <= LIMITS.events,
            "Event identity/input source mismatch");
        const eventIDs = new Set(),
            ordinals = new Map(),
            slots = new Map();
        let template = null;
        for (const event of identity.events) {
            check(integer(event.event_id) && !eventIDs.has(event.event_id) && Array.isArray(
                    event.particles) && event.particles.length > 0 && event.particles
                .length <= 16, "Invalid/duplicate event identity");
            eventIDs.add(event.event_id);
            const used = new Set();
            for (const p of event.particles) {
                check(integer(p.source_record, input.length) && p.source_record > 0 && !ordinals
                    .has(p.source_record) && typeof p.event_slot === "string" &&
                    /^[a-z][a-z0-9_]{0,47}$/.test(p.event_slot) && !used.has(p
                    .event_slot) && [11, -11].includes(p.pdg_id),
                    "Invalid event slot/ordinal/PDG");
                check(typeof p.role === "string" && p.role.length > 0 && p.role.length <= 64,
                    "Missing explicit particle role");
                used.add(p.event_slot);
                if (slots.has(p.event_slot)) check(objectSame(slots.get(p.event_slot), {
                    pdg_id: p.pdg_id,
                    role: p.role
                }), "Event slot changes identity");
                slots.set(p.event_slot, {
                    pdg_id: p.pdg_id,
                    role: p.role
                });
                ordinals.set(p.source_record, {
                    ...p,
                    event: event.event_id
                })
            }
            const present = [...used].sort();
            if (template) check(same(template, present), "Event slots differ between events");
            else template = present;
        }
        check(ordinals.size === input.length && input.every((_, i) => ordinals.has(i + 1)),
            "Identity must bind every input row");
        check(slots.size * 2 <= LIMITS.plots && Array.isArray(a.plots) && a.plots.length ===
            slots.size * 2, "Expected two plots per explicit event slot");
        const plotIDs = new Set(),
            plotKeys = new Set();
        for (const p of a.plots) {
            check(/^[a-zA-Z][a-zA-Z0-9_]{0,95}$/.test(p.id) && !plotIDs.has(p.id) && slots.get(p
                    .event_slot)?.pdg_id === p.pdg_id && ["delta_theta", "delta_phi"]
                .includes(p.observable) && p.unit === "deg", "Invalid deflection descriptor"
                );
            plotIDs.add(p.id);
            text(p.label, "plot label");
            check(p.id === p.event_slot + "_" + p.observable && p.label === slots.get(p
                    .event_slot).role + (p.observable === "delta_theta" ? " Δθ" : " Δφ"),
                "Plot identity/role label mismatch");
            const key = p.event_slot + ":" + p.observable;
            check(!plotKeys.has(key), "Duplicate event observable");
            plotKeys.add(key);
            check(Array.isArray(p.edges) && p.edges.length === LIMITS.bins + 1 && p.edges.every(
                    (v, i) => finite(v) && (!i || v > p.edges[i - 1])) && p.edges[0] <= 0 &&
                p.edges.at(-1) > 0, "Invalid fixed deflection edges");
            const width = (p.edges.at(-1) - p.edges[0]) / LIMITS.bins;
            check(p.edges.every((v, i) => near(v, p.edges[0] + i * width)),
                "Deflection bins must be linear");
        }
        const initial = new Map(),
            seenSource = new Set();
        for (const row of raw.initial_records) {
            check(Array.isArray(row) && row.length === 11 && row.every(finite) && integer(row[
                    0], 2147483647) && row[0] > 0 && !initial.has(row[0]) && integer(row[1],
                    LIMITS.records) && row[1] > 0 && !seenSource.has(row[1]) && ordinals
                .has(row[1]) && integer(row[2]) && integer(row[3]) && row[3] >= row[2] &&
                Math.hypot(...row.slice(8)) > 0, "Invalid native initial identity/state");
            check(row[2] === input[row[1] - 1][0] && same(row.slice(4), input[row[1] - 1].slice(
                1)), "Native initial float state differs from bound input row");
            const ident = ordinals.get(row[1]);
            check(row[4] !== 0 && (ident.pdg_id === 11 ? row[4] > 0 : row[4] < 0),
                "Explicit PDG disagrees with audited native input charge convention");
            initial.set(row[0], {
                row,
                ident
            });
            seenSource.add(row[1]);
        }
        check(Array.isArray(a.frames) && a.frames.length === raw.frames.length,
            "Derived event frame count mismatch");
        const frameLabels = new Set(),
            previousRows = new Map(),
            extents = new Map(a.plots.map(p => [p.id, 0]));

        function scope(r, current, frameIndex) {
            check(r?.available === true && Array.isArray(current) && current.length <= LIMITS
                .records && r.nrecords === current.length,
                "Event scope record count mismatch");
            if (frameIndex !== null) {
                const f = manifest.frames[frameIndex],
                    nf = raw.frames[frameIndex];
                check(r.id === f.id && r.diagonal === f.diagonal && r.elapsed_s === f
                    .elapsed_s && nf.id === f.id && nf.diagonal === f.diagonal,
                    "Event frame identity mismatch")
            }
            const labels = new Set(),
                eventCounts = new Map(),
                values = new Map(a.plots.map(p => [p.id, []])),
                excluded = new Map(a.plots.map(p => [p.id, 0])),
                gathers = [0, 0, 0];
            for (const row of current) {
                check(Array.isArray(row) && row.length === 7 && row.every(finite) && integer(
                        row[0]) && !labels.has(row[0]) && initial.has(row[0]) && row.slice(
                        4).every(v => integer(v)) && Math.hypot(...row.slice(1, 4)) > 0 &&
                    row.slice(1, 4).reduce((sum, v) => sum + v * v, 0) <= 1 + 8e-6,
                    "Invalid current native trajectory");
                labels.add(row[0]);
                const state = initial.get(row[0]),
                    d = angles(state.row, row);
                eventCounts.set(state.ident.event, (eventCounts.get(state.ident.event) || 0) +
                    1);
                row.slice(4).forEach((v, i) => gathers[i] += v);
                for (const p of a.plots)
                    if (p.event_slot === state.ident.event_slot) {
                        if (d[p.observable] === null) excluded.set(p.id, excluded.get(p.id) +
                        1);
                        else {
                            values.get(p.id).push(d[p.observable]);
                            extents.set(p.id, Math.max(extents.get(p.id), Math.abs(d[p
                                .observable])))
                        }
                    }
            }
            if (frameIndex !== null) {
                for (const row of current) {
                    const old = previousRows.get(row[0]);
                    if (old) check(old.slice(4).every((v, i) => v <= row[i + 4]),
                        "Cumulative native field gathers regressed");
                    previousRows.set(row[0], row)
                }
                for (const label of frameLabels) check(labels.has(label),
                    "Native snapshot lost an admitted event identity");
                for (const label of labels) frameLabels.add(label)
            } else {
                check(labels.size === initial.size && [...initial.keys()].every(k => labels.has(
                    k)), "Final event identities do not close");
                check(current.every(row => same(row, previousRows.get(row[0]))),
                    "Final state differs from final saved diagonal")
            }
            let complete = 0,
                incomplete = 0,
                notYet = 0;
            for (const e of identity.events) {
                const n = eventCounts.get(e.event_id) || 0;
                if (n === 0) notYet++;
                else if (n === e.particles.length) complete++;
                else incomplete++
            }
            check(objectSame(r.event_counts, {
                expected: identity.events.length,
                complete,
                incomplete,
                not_yet_admitted: notYet
            }) && objectSame(r.field_gathers, {
                main_mesh: gathers[0],
                outer_mesh: gathers[1],
                monopole: gathers[2]
            }), "Event/gather receipts do not match native identities");
            check(r.histograms && Object.keys(r.histograms).length === a.plots.length,
                "Event histogram cardinality mismatch");
            for (const p of a.plots) {
                const h = r.histograms[p.id],
                    v = values.get(p.id),
                    expected = histogramValues(v, p.edges);
                check(h && h.quantity === p.id && h.unit === "deg" && h.weight_unit ===
                    "stored particle records" && h.weighting === WEIGHTING && h
                    .normalization === "bin-integrated" && h.samples === v.length && h
                    .total_weight === v.length && h.underflow === expected.underflow && h
                    .overflow === expected.overflow && h.excluded_undefined_azimuth ===
                    excluded.get(p.id), "Deflection count/weight/exclusion mismatch");
                check(Array.isArray(h.bins) && h.bins.length === LIMITS.bins && h.bins.every((b,
                            i) => b.low === p.edges[i] && b.high === p.edges[i + 1] && b
                        .weight === expected.counts[i]),
                    "Deflection histogram differs from native trajectories");
                if (!v.length) check(h.statistics === null,
                    "Empty event histogram carries statistics");
                else {
                    check(h.statistics && ["mean", "variance", "min", "max"].every(k => near(h
                            .statistics[k], expected.statistics[k])) && h.statistics
                        .variance >= 0, "Deflection moments differ from native trajectories"
                        );
                    check(h.statistics.mode_method ===
                        "centre of highest count-density histogram bin; approximate",
                        "Unknown deflection mode estimator");
                    check(expected.statistics.mode === null ? h.statistics.mode === null : near(
                            h.statistics.mode, expected.statistics.mode),
                        "Deflection mode differs from native density bins")
                }
            }
        }
        a.frames.forEach((r, i) => scope(r, raw.frames[i].records, i));
        scope(a.total, raw.final?.records, null);
        for (const p of a.plots) {
            const half = Math.max(1e-9, Math.min(180, extents.get(p.id) * 1.1));
            check(p.edges.every((v, i) => near(v, half * (2 * i / LIMITS.bins - 1))),
                "Fixed event axes differ from all-scope native extents")
        }
        return a;
    }
    const api = {
        LIMITS,
        WEIGHTING,
        DEFINITIONS,
        angles,
        histogramValues,
        validate
    };
    if (typeof module !== "undefined" && module.exports) module.exports = api;
    root.ColliderEventDeflections = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
