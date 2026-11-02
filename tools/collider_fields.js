/* Authenticated native field overlays; no physics reconstruction. */
(function(root) {
    "use strict";

    function check(ok, message) {
        if (!ok) throw new Error(message)
    }
    const LIMITS = {
        frames: 64,
        sources: 8,
        side: 64,
        scalars: 2097152,
        longitudinalScalars: 524288,
        snapshotScalars: 1048576,
        fileBytes: 64 * 1024 * 1024
    };

    function text(value, name) {
        check(typeof value === "string" && value.length > 0 && value.length <= 2048,
            "Invalid field " + name)
    }

    function integer(value, min, max, name) {
        check(Number.isSafeInteger(value) && value >= min && value <= max, "Invalid field " +
            name)
    }
    const LONG_SELECTION = "all active source-slice encounters, final substep, pre-kick";
    const LONG_TIMING =
        "field samples are post-solve/pre-kick on the final substep; particles are displayed after the diagonal barrier";
    const LONG_Z_MAPPING =
        "source-specific nominal slice bands under the legacy particle-dump z convention; raw encounter z is retained separately";
    const near = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <=
        1e-10 * Math.max(1, Math.abs(b));
    const keys = (object, expected) => object && typeof object === "object" && !Array.isArray(
            object) &&
        Object.keys(object).length === expected.length && expected.every(k => Object.prototype
            .hasOwnProperty.call(object, k));

    function validateLongitudinal(f) {
        if (!Object.prototype.hasOwnProperty.call(f, "longitudinal")) {
            check(f.frames.every(r => !Object.prototype.hasOwnProperty.call(r, "longitudinal")),
                "Longitudinal samples without geometry");
            return null;
        }
        const m = f.longitudinal,
            g = f.geometry,
            c = f.conventions,
            n = c.n_z,
            dz = m?.delta_z_nm;
        check(m?.schema_version === "feebl.collider.longitudinal/1" && m.coordinate_unit ===
            "nm" && m.frame === "legacy_collider_dump",
            "Unsupported longitudinal field schema/coordinates");
        check(m.n_z === n && Number.isFinite(dz) && dz > 0,
            "Invalid longitudinal slice geometry");
        check(m.selection === LONG_SELECTION && m.timing === LONG_TIMING && m.z_mapping ===
            LONG_Z_MAPPING, "Unsupported longitudinal field conventions");
        check(f.sources.length === 2 && [1, 2].every(i => f.sources.some(s => s.id === "beam" +
                i && s.kind === "beam_beam" && s.source_beam === i)),
            "Unsupported longitudinal native sources");
        check(keys(m.planes, ["zx", "zy"]), "Invalid longitudinal planes");
        const cap = n <= 128 ? Math.min(64, Math.floor(4096 / n)) : 2,
            geometry = {};
        for (const [view, axis, fixed, domainIndex] of [
                ["zx", "x", "y", 0],
                ["zy", "y", "x", 2]
            ]) {
            const q = m.planes[view];
            check(q && JSON.stringify(q.axes) === JSON.stringify(["z", axis]) && q
                .fixed_axis === fixed && q.fixed_coordinate_nm === 0,
                "Invalid longitudinal cross-section plane");
            integer(q.samples, 2, cap, "longitudinal sample count");
            check(near(q.min_nm, g[axis + "_min_nm"]) && near(q.max_nm, g[axis + "_max_nm"]),
                "Longitudinal sample extent mismatch");
            const step = (q.max_nm - q.min_nm) / (q.samples - 1),
                domain = g.native_domain_nm;
            const edges = Array.from({
                length: q.samples + 1
            }, (_, i) => Math.min(domain[domainIndex + 1], Math.max(domain[domainIndex],
                q.min_nm + (i - .5) * step)) / 1000);
            check(edges.every(Number.isFinite), "Nonfinite longitudinal cell edges");
            geometry[view] = {
                axes: ["z", axis],
                transverse_edges_um: edges,
                samples: q.samples
            };
        }
        const expectedTotal = n <= 128 ? f.frames.reduce((a, r) => a + Math.min(r.diagonal, n -
            1) - Math.max(0, r.diagonal - n + 1) + 1, 0) : 0;
        check(2 * (m.planes.zx.samples + m.planes.zy.samples) * expectedTotal <= LIMITS
            .longitudinalScalars, "Longitudinal sample budget");
        const maximum = {
            beam1: {
                zx: 0,
                zy: 0
            },
            beam2: {
                zx: 0,
                zy: 0
            }
        };
        let scalarCount = 0,
            availableFrames = 0,
            availableStrips = 0;
        for (const frame of f.frames) {
            const r = frame.longitudinal,
                d = frame.diagonal,
                first = Math.max(0, d - n + 1),
                last = Math.min(d, n - 1),
                expected = last - first + 1;
            check(r && typeof r.available === "boolean" && r.expected_encounters === expected &&
                Array.isArray(r.strips), "Invalid longitudinal frame coverage");
            if (n > 128) {
                check(!r.available && r.strips.length === 0,
                    "Unsupported longitudinal slice count carries data");
                text(r.reason, "longitudinal unavailability");
                continue
            }
            check(r.strips.length === expected, "Missing longitudinal encounter coverage");
            let anyAvailable = false;
            const ordinals = new Set();
            r.strips.forEach((strip, index) => {
                const i1 = first + index,
                    i2 = d - i1,
                    z = (i2 - i1) * dz / 2;
                check(strip && strip.beam1_slice === i1 && strip.beam2_slice === i2,
                    "Longitudinal encounter order mismatch");
                check(near(strip.nominal_z_nm, z) && Array.isArray(strip
                        .nominal_z_bounds_nm) && strip.nominal_z_bounds_nm
                    .length === 2 && near(strip.nominal_z_bounds_nm[0], z - dz /
                    2) && near(strip.nominal_z_bounds_nm[1], z + dz / 2),
                    "Invalid longitudinal nominal z");
                check(keys(strip.display_z_nm, ["beam1", "beam2"]) && keys(strip
                        .display_z_bounds_nm, ["beam1", "beam2"]),
                    "Missing source-specific longitudinal z");
                for (const [source, sign] of [
                        ["beam1", 1],
                        ["beam2", -1]
                    ]) {
                    const centre = -z + sign * dz,
                        bounds = strip.display_z_bounds_nm[source];
                    check(near(strip.display_z_nm[source], centre) && Array.isArray(
                            bounds) && bounds.length === 2 && near(bounds[0],
                            centre - dz / 2) && near(bounds[1], centre + dz / 2),
                        "Longitudinal display z mapping mismatch");
                }
                check(typeof strip.available === "boolean",
                    "Missing longitudinal strip availability");
                if (!strip.available) {
                    text(strip.reason, "longitudinal strip reason");
                    check(strip.maps === null,
                        "Unavailable longitudinal strip carries data");
                    if (strip.substep_index !== null || strip.encounter_counter !==
                        null) {
                        check(strip.substep_index === c.n_t - 1,
                            "Invalid unavailable longitudinal sampling receipt");
                        integer(strip.encounter_counter, 0, Number.MAX_SAFE_INTEGER,
                            "unavailable longitudinal encounter counter");
                        check(!ordinals.has(strip.encounter_counter),
                            "Duplicate longitudinal encounter counter");
                        ordinals.add(strip.encounter_counter);
                    }
                    return;
                }
                anyAvailable = true;
                availableStrips++;
                check(strip.reason === null && strip.substep_index === c.n_t - 1,
                    "Invalid longitudinal sampling phase");
                integer(strip.encounter_counter, 0, Number.MAX_SAFE_INTEGER,
                    "longitudinal encounter counter");
                check(!ordinals.has(strip.encounter_counter),
                    "Duplicate longitudinal encounter counter");
                ordinals.add(strip.encounter_counter);
                check(c.charge_sign === -1 && c.scale_step.every(v => v === 1),
                    "Unadmitted longitudinal field normalization");
                check(g.x_min_nm <= 0 && g.x_max_nm >= 0 && g.y_min_nm <= 0 && g
                    .y_max_nm >= 0,
                    "Longitudinal cross-section outside sample domain");
                check(keys(strip.maps, ["beam1", "beam2"]),
                    "Invalid longitudinal source maps");
                for (const source of ["beam1", "beam2"]) {
                    check(keys(strip.maps[source], ["zx", "zy"]),
                        "Invalid longitudinal map planes");
                    for (const view of ["zx", "zy"]) {
                        const values = strip.maps[source][view];
                        check(Array.isArray(values) && values.length === m.planes[view]
                            .samples, "Longitudinal map shape mismatch");
                        for (const value of values) {
                            check(Number.isFinite(value) && value >= 0,
                                "Nonfinite or negative longitudinal field");
                            maximum[source][view] = Math.max(maximum[source][view],
                                value)
                        }
                        scalarCount += values.length;
                    }
                }
            });
            check(r.available === anyAvailable, "Longitudinal frame availability mismatch");
            if (anyAvailable) {
                check(r.reason === null, "Available longitudinal frame has reason");
                availableFrames++
            } else text(r.reason, "longitudinal frame reason");
        }
        check(scalarCount <= LIMITS.longitudinalScalars, "Longitudinal value bound");
        return {
            metadata: m,
            geometry,
            maximum,
            scalarCount,
            availableFrames,
            availableStrips
        };
    }
    const SNAPSHOT_SELECTION = "all primary source slices at every saved diagonal";
    const SNAPSHOT_TIMING =
        "post-diagonal; all source copies drift-synchronized with the sampled particle display";
    const SNAPSHOT_MODEL =
        "quasi-static native transverse slice-field reconstruction; primary sources only; not a full 3D electromagnetic field";

    function validateSnapshot(f) {
        if (!Object.prototype.hasOwnProperty.call(f, "snapshot")) {
            check(f.frames.every(r => !Object.prototype.hasOwnProperty.call(r, "snapshot")),
                "Snapshot field samples without geometry");
            return null;
        }
        const m = f.snapshot,
            g = f.geometry,
            c = f.conventions,
            n = c.n_z,
            dz = m?.delta_z_nm;
        check(m?.schema_version === "feebl.collider.snapshot-fields/1" && m.coordinate_unit ===
            "nm" && m.frame === "legacy_collider_dump",
            "Unsupported snapshot field schema/coordinates");
        check(m.n_z === n && Number.isFinite(dz) && dz > 0, "Invalid snapshot slice geometry");
        check(m.selection === SNAPSHOT_SELECTION && m.timing === SNAPSHOT_TIMING && m.model ===
            SNAPSHOT_MODEL, "Unsupported snapshot field conventions");
        check(f.sources.length === 2 && [1, 2].every(i => f.sources.some(s => s.id === "beam" +
                i && s.kind === "beam_beam" && s.source_beam === i)),
            "Unsupported snapshot native sources");
        check(keys(m.planes, ["zx", "zy"]), "Invalid snapshot planes");
        if (f.longitudinal) check(near(dz, f.longitudinal.delta_z_nm),
            "Snapshot/encounter slice spacing mismatch");
        const cap = n <= 128 ? Math.min(64, Math.floor(4096 / n)) : 2,
            geometry = {};
        for (const [view, axis, fixed, di] of [
                ["zx", "x", "y", 0],
                ["zy", "y", "x", 2]
            ]) {
            const q = m.planes[view];
            check(q && JSON.stringify(q.axes) === JSON.stringify(["z", axis]) && q
                .fixed_axis === fixed && q.fixed_coordinate_nm === 0,
                "Invalid snapshot cross-section plane");
            integer(q.samples, 2, cap, "snapshot sample count");
            check(near(q.min_nm, g[axis + "_min_nm"]) && near(q.max_nm, g[axis + "_max_nm"]),
                "Snapshot sample extent mismatch");
            const step = (q.max_nm - q.min_nm) / (q.samples - 1),
                domain = g.native_domain_nm;
            const edges = Array.from({
                length: q.samples + 1
            }, (_, i) => Math.min(domain[di + 1], Math.max(domain[di], q.min_nm + (i -
                .5) * step)) / 1000);
            check(edges.every(Number.isFinite), "Nonfinite snapshot cell edges");
            geometry[view] = {
                axes: ["z", axis],
                transverse_edges_um: edges,
                samples: q.samples
            };
        }
        if (n <= 128) check(f.frames.length * 2 * n * (m.planes.zx.samples + m.planes.zy
            .samples) <= LIMITS.snapshotScalars, "Snapshot sample budget");
        const maximum = {
            beam1: {
                zx: 0,
                zy: 0
            },
            beam2: {
                zx: 0,
                zy: 0
            }
        };
        let scalarCount = 0,
            availableFrames = 0,
            availableStrips = 0;
        for (const frame of f.frames) {
            const r = frame.snapshot,
                d = frame.diagonal;
            check(r && typeof r.available === "boolean" && r.diagonal === d && r.phase ===
                "post-diagonal/display-synchronized" && r.expected_slices === n,
                "Invalid snapshot frame/phase");
            if (!r.available) {
                text(r.reason, "snapshot unavailability");
                check(r.sources === null, "Unavailable snapshot carries sources");
                continue
            }
            check(n <= 128 && r.reason === null && keys(r.sources, ["beam1", "beam2"]),
                "Invalid snapshot availability/sources");
            check(c.charge_sign === -1 && c.scale_step.every(v => v === 1),
                "Unadmitted snapshot normalization");
            check(g.x_min_nm <= 0 && g.x_max_nm >= 0 && g.y_min_nm <= 0 && g.y_max_nm >= 0,
                "Snapshot cross-section outside sample domain");
            availableFrames++;
            for (const [id, sign] of [
                    ["beam1", 1],
                    ["beam2", -1]
                ]) {
                const source = r.sources[id];
                check(source && Array.isArray(source.strips) && source.strips.length === n,
                    "Incomplete snapshot source coverage");
                source.strips.forEach((strip, i) => {
                    const centre = sign * (i + 1 - d / 2) * dz,
                        bounds = strip?.display_z_bounds_nm;
                    check(strip && strip.slice === i && near(strip.display_z_nm,
                        centre) && Array.isArray(bounds) && bounds.length === 2 &&
                        near(bounds[0], centre - dz / 2) && near(bounds[1], centre +
                            dz / 2),
                        "Snapshot source-slice display geometry mismatch");
                    for (const key of ["source_particles", "deposited_particles",
                            "outside_particles"
                        ]) integer(strip[key], 0, Number.MAX_SAFE_INTEGER, "snapshot " +
                        key);
                    check(strip.deposited_particles + strip.outside_particles === strip
                        .source_particles,
                        "Snapshot source/deposition count closure");
                    check(keys(strip.maps, ["zx", "zy"]),
                        "Snapshot map planes mismatch");
                    for (const view of ["zx", "zy"]) {
                        const values = strip.maps[view];
                        check(Array.isArray(values) && values.length === m.planes[view]
                            .samples, "Snapshot map dimension mismatch");
                        check(values.every(v => Number.isFinite(v) && v >= 0),
                            "Negative or nonfinite snapshot magnitude");
                        for (const v of values) maximum[id][view] = Math.max(maximum[id]
                            [view], v);
                        scalarCount += values.length;
                    }
                    availableStrips++;
                });
            }
        }
        check(scalarCount <= LIMITS.snapshotScalars, "Snapshot scalar count bound");
        return {
            metadata: m,
            geometry,
            maximum,
            scalarCount,
            availableFrames,
            availableStrips
        };
    }

    function dataMode(validated, requested = "auto") {
        check(["auto", "snapshot", "encounter"].includes(requested), "Unknown field data mode");
        return requested === "auto" ? (validated?.snapshot?.availableFrames ? "snapshot" :
            "encounter") : requested;
    }

    /** Validate source IDs, field units, bounded meshes and frame alignment; return rendering metadata referencing the original dataset or throw. */
    function validate(f, manifest) {
        check(f?.schema_version === "feebl.collider.fields/1", "Unknown field schema");
        check(f.plane?.coordinate_unit === "nm" && f.plane.frame === "native simulated frame" &&
            JSON.stringify(f.plane.axes) === '["x","y"]',
            "Unsupported field coordinate plane");
        const grid = f.geometry;
        check(grid && grid.storage_order === "row-major-y-x",
        "Unsupported field storage order");
        check(grid.sampling === "uniform CIC gather points; bounds are sample centres",
            "Unsupported field sampling");
        integer(grid.nx, 2, LIMITS.side, "nx");
        integer(grid.ny, 2, LIMITS.side, "ny");
        for (const key of ["x_min_nm", "x_max_nm", "y_min_nm", "y_max_nm"]) check(Number
            .isFinite(grid[key]), "Nonfinite field geometry");
        check(grid.x_max_nm > grid.x_min_nm && grid.y_max_nm > grid.y_min_nm,
            "Empty field extent");
        const dx = (grid.x_max_nm - grid.x_min_nm) / (grid.nx - 1),
            dy = (grid.y_max_nm - grid.y_min_nm) / (grid.ny - 1);
        check(Number.isFinite(dx) && Number.isFinite(dy) && dx > 0 && dy > 0,
            "Invalid field spacing");
        const domain = grid.native_domain_nm;
        check(Array.isArray(domain) && domain.length === 4 && domain.every(Number.isFinite) &&
            domain[0] < domain[1] && domain[2] < domain[3], "Invalid native field domain");
        check(grid.x_min_nm >= domain[0] && grid.x_max_nm <= domain[1] && grid.y_min_nm >=
            domain[2] && grid.y_max_nm <= domain[3],
            "Field sample centres outside native domain");
        const x_edges_um = Array.from({
            length: grid.nx + 1
        }, (_, i) => Math.min(domain[1], Math.max(domain[0], grid.x_min_nm + (i - .5) *
            dx)) / 1000);
        const y_edges_um = Array.from({
            length: grid.ny + 1
        }, (_, i) => Math.min(domain[3], Math.max(domain[2], grid.y_min_nm + (i - .5) *
            dy)) / 1000);
        check([...x_edges_um, ...y_edges_um].every(Number.isFinite),
            "Nonfinite field cell boundary");
        check(Array.isArray(f.sources) && f.sources.length >= 1 && f.sources.length <= LIMITS
            .sources, "Field source count bound");
        const sourceIds = new Set(),
            maximum = Object.create(null);
        for (const s of f.sources) {
            check(typeof s.id === "string" && /^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(s.id) && !
                sourceIds.has(s.id), "Duplicate or invalid field source ID");
            sourceIds.add(s.id);
            maximum[s.id] = 0;
            text(s.label, "source label");
            text(s.quantity, "quantity");
            check(["GV/nm", "V/m"].includes(s.unit), "Unsupported electric field unit");
            if (s.kind === "beam_beam") {
                check(s.quantity === "native transverse electric-equivalent field magnitude" &&
                    s.unit === "GV/nm", "Invalid native field meaning or units");
                check([1, 2].includes(s.source_beam) && [1, 2].includes(s.target_beam) && s
                    .source_beam !== s.target_beam, "Invalid native source/target beam");
            } else check(s.kind === "external" && s.source_beam === null && s.target_beam ===
                null, "Unsupported field source kind or beam references");
        }
        const c = f.conventions;
        check(c && typeof c === "object", "Missing field conventions");
        check(c.selection === "central active slice encounter, final substep, pre-kick" && c
            .scope === "single source slice; not whole-bunch or diagonal-averaged field",
            "Unsupported field encounter convention");
        check([-1, 0, 1].includes(c.charge_sign) && Array.isArray(c.scale_step) && c.scale_step
            .length === 2 && c.scale_step.every(v => Number.isFinite(v) && v > 0),
            "Invalid native field scaling receipt");
        integer(c.n_z, 1, 1000000, "longitudinal slice count");
        integer(c.n_t, 1, 1000000, "substep count");
        check(Array.isArray(f.frames) && f.frames.length === manifest.frames.length && f.frames
            .length <= LIMITS.frames, "Field frame count mismatch");
        let scalarCount = 0,
            previousDiagonal = -1;
        const hasNative = f.sources.some(s => s.kind === "beam_beam");
        f.frames.forEach((r, i) => {
            const expected = manifest.frames[i];
            integer(r.diagonal, 0, 2 * c.n_z - 2, "diagonal");
            check(r.diagonal > previousDiagonal, "Nonmonotone field diagonal");
            previousDiagonal = r.diagonal;
            check(r.id === i && r.diagonal === expected.diagonal && Number.isFinite(r
                    .elapsed_s) && r.elapsed_s === expected.elapsed_s,
                "Field frame/time alignment mismatch");
            check(typeof r.available === "boolean", "Missing field availability flag");
            if (!r.available) {
                text(r.reason, "unavailability reason");
                check(r.encounter === null && r.maps === null,
                    "Unavailable field carries maps");
                return
            }
            check(r.reason === null,
                "Available field has unexpected unavailability reason");
            if (hasNative) check(c.charge_sign === -1 && c.scale_step.every(v => v ===
                1), "Unavailable native field normalization");
            const e = r.encounter;
            check(e && e.phase === "post-solve/pre-kick",
                "Unsupported field evaluation phase");
            integer(e.beam1_slice, 0, c.n_z - 1, "beam1 slice");
            integer(e.beam2_slice, 0, c.n_z - 1, "beam2 slice");
            integer(e.substep_index, 0, c.n_t - 1, "substep index");
            integer(e.encounter_counter, 0, Number.MAX_SAFE_INTEGER,
                "encounter counter");
            check(e.beam1_slice === Math.floor(r.diagonal / 2) && e.beam2_slice === Math
                .ceil(r.diagonal / 2) && e.substep_index === c.n_t - 1,
                "Field is not the declared central encounter/final substep");
            check(Number.isFinite(e.nominal_z_nm), "Nonfinite field encounter plane");
            check(r.maps && typeof r.maps === "object" && !Array.isArray(r.maps) &&
                Object.keys(r.maps).length === sourceIds.size,
                "Missing or extra field maps");
            for (const id of sourceIds) {
                check(Object.prototype.hasOwnProperty.call(r.maps, id),
                    "Missing field source map");
                const values = r.maps[id];
                check(Array.isArray(values) && values.length === grid.nx * grid.ny,
                    "Field map shape mismatch");
                for (const v of values) {
                    check(Number.isFinite(v) && v >= 0,
                        "Negative or nonfinite field magnitude");
                    maximum[id] = Math.max(maximum[id], v)
                }
                scalarCount += values.length;
                check(scalarCount <= LIMITS.scalars, "Field scalar count bound");
            }
        });
        const longitudinal = validateLongitudinal(f),
            snapshot = validateSnapshot(f);
        scalarCount += (longitudinal?.scalarCount || 0) + (snapshot?.scalarCount || 0);
        check(scalarCount <= LIMITS.scalars, "Combined field scalar count bound");
        return {
            data: f,
            maximum,
            geometry: {
                nx: grid.nx,
                ny: grid.ny,
                x_edges_um,
                y_edges_um
            },
            longitudinal,
            snapshot,
            scalarCount
        };
    }

    /** Select an authenticated source/plane/frame view; return null for unrecorded coverage. Coordinates are converted to um for display. */
    function plane(validated, frameIndex, sourceId, view = "xy", requested = "auto") {
        check(Number.isInteger(frameIndex) && frameIndex >= 0 && frameIndex < validated.data
            .frames.length, "Field frame out of range");
        check(Object.prototype.hasOwnProperty.call(validated.maximum, sourceId),
            "Unknown field source");
        check(["xy", "zx", "zy"].includes(view), "Unknown field plane");
        const f = validated.data.frames[frameIndex];
        if (view === "xy") return f.available ? {
            ...validated.geometry,
            axes: ["x", "y"],
            values: f.maps[sourceId]
        } : null;
        if (dataMode(validated, requested) === "snapshot") {
            if (!validated.snapshot || !f.snapshot.available) return null;
            const strips = f.snapshot.sources[sourceId].strips;
            return {
                ...validated.snapshot.geometry[view],
                longitudinal: true,
                snapshot: true,
                coverage: {
                    available: strips.length,
                    expected: f.snapshot.expected_slices
                },
                counts: {
                    source: strips.reduce((a, s) => a + s.source_particles, 0),
                    deposited: strips.reduce((a, s) => a + s.deposited_particles, 0),
                    outside: strips.reduce((a, s) => a + s.outside_particles, 0)
                },
                strips: strips.map(s => ({
                    z_edges_um: s.display_z_bounds_nm.map(v => v / 1000),
                    values: s.maps[view]
                }))
            };
        }
        if (!validated.longitudinal || !f.longitudinal.available) return null;
        return {
            ...validated.longitudinal.geometry[view],
            longitudinal: true,
            coverage: {
                available: f.longitudinal.strips.filter(s => s.available).length,
                expected: f.longitudinal.expected_encounters
            },
            strips: f.longitudinal.strips.filter(s => s.available).map(s => ({
                z_edges_um: s.display_z_bounds_nm[sourceId].map(v => v / 1000),
                values: s.maps[sourceId][view]
            }))
        };
    }

    function scaleMaximum(validated, frameIndex, sourceId, mode, view = "xy", requested =
        "auto") {
        check(["run", "frame"].includes(mode), "Unknown field color scale");
        const p = plane(validated, frameIndex, sourceId, view, requested);
        if (mode === "run") return view === "xy" ? validated.maximum[sourceId] : ((dataMode(
                validated, requested) === "snapshot" ? validated.snapshot :
            validated.longitudinal)?.maximum[sourceId][view] || 0);
        if (!p) return 0;
        return p.longitudinal ? Math.max(0, ...p.strips.map(s => Math.max(...s.values))) : Math
            .max(...p.values);
    }

    const PALETTE = ["#00224e", "#00234f", "#002451", "#002553", "#002554", "#002656",
        "#002758", "#002859", "#00285b", "#00295d", "#002a5f", "#002a61", "#002b62",
        "#002c64", "#002c66", "#002d68", "#002e6a", "#002e6c", "#002f6d", "#00306f",
        "#003070", "#003170", "#003171", "#013271", "#053371", "#053371", "#0c3470",
        "#0f3570", "#123570", "#143670", "#163770", "#18376f", "#1a386f", "#1c396f",
        "#1e3a6f", "#203a6f", "#213b6e", "#233c6e", "#243c6e", "#263d6e", "#273e6e",
        "#293f6e", "#293f6e", "#2b406d", "#2d416d", "#2e416d", "#2f426d", "#31436d",
        "#32436d", "#33446d", "#34456c", "#35456c", "#36466c", "#38476c", "#39486c",
        "#3a486c", "#3b496c", "#3c4a6c", "#3d4a6c", "#3e4b6c", "#3f4c6c", "#404c6c",
        "#414d6c", "#424e6c", "#434e6c", "#444f6c", "#45506c", "#46516c", "#47516c",
        "#48526c", "#49536c", "#4a536c", "#4b546c", "#4c556c", "#4d556c", "#4e566c",
        "#4f576c", "#50576c", "#51586d", "#52596d", "#535a6d", "#545a6d", "#555b6d",
        "#555c6d", "#565c6d", "#575d6d", "#585e6d", "#595e6e", "#5a5f6e", "#5b606e",
        "#5c616e", "#5d616e", "#5e626e", "#5e636f", "#5f636f", "#60646f", "#61656f",
        "#62656f", "#636670", "#646770", "#656870", "#656870", "#666970", "#676a71",
        "#686a71", "#696b71", "#6a6c71", "#6b6d72", "#6c6d72", "#6c6e72", "#6d6f72",
        "#6e6f73", "#6f7073", "#707173", "#717274", "#727274", "#727374", "#737475",
        "#747475", "#757575", "#767676", "#777776", "#777777", "#787877", "#797977",
        "#7a7a78", "#7b7a78", "#7c7b78", "#7d7c78", "#7e7c78", "#7e7d78", "#7f7e78",
        "#807f78", "#817f78", "#828079", "#838179", "#848279", "#858279", "#868379",
        "#878478", "#888578", "#898578", "#8a8678", "#8b8778", "#8c8878", "#8d8878",
        "#8e8978", "#8f8a78", "#908b78", "#918b78", "#928c78", "#928d78", "#938e78",
        "#948e77", "#958f77", "#969077", "#979177", "#989277", "#999277", "#9a9376",
        "#9b9476", "#9c9576", "#9d9576", "#9e9676", "#9f9775", "#a09875", "#a19975",
        "#a29975", "#a39a74", "#a49b74", "#a59c74", "#a69c74", "#a79d73", "#a89e73",
        "#a99f73", "#aaa073", "#aba072", "#aca172", "#ada272", "#aea371", "#afa471",
        "#b0a571", "#b1a570", "#b3a670", "#b4a76f", "#b5a86f", "#b6a96f", "#b7a96e",
        "#b8aa6e", "#b9ab6d", "#baac6d", "#bbad6d", "#bcae6c", "#bdae6c", "#beaf6b",
        "#bfb06b", "#c0b16a", "#c1b26a", "#c2b369", "#c3b369", "#c4b468", "#c5b568",
        "#c6b667", "#c7b767", "#c8b866", "#c9b965", "#cbb965", "#ccba64", "#cdbb63",
        "#cebc63", "#cfbd62", "#d0be62", "#d1bf61", "#d2c060", "#d3c05f", "#d4c15f",
        "#d5c25e", "#d6c35d", "#d7c45c", "#d9c55c", "#dac65b", "#dbc75a", "#dcc859",
        "#ddc858", "#dec958", "#dfca57", "#e0cb56", "#e1cc55", "#e2cd54", "#e4ce53",
        "#e5cf52", "#e6d051", "#e7d150", "#e8d24f", "#e9d34e", "#ead34c", "#ebd44b",
        "#edd54a", "#eed649", "#efd748", "#f0d846", "#f1d945", "#f2da44", "#f3db42",
        "#f5dc41", "#f6dd3f", "#f7de3e", "#f8df3c", "#f9e03a", "#fbe138", "#fce236",
        "#fde334", "#fee434", "#fee535", "#fee636", "#fee838"
    ];

    function color(value, maximum) {
        const f = maximum > 0 ? Math.min(1, Math.max(0, value / maximum)) : 0;
        return PALETTE[Math.round(f * (PALETTE.length - 1))];
    }
    // Only combine the two authenticated canonical beam layers. This selector is
    // outside the allowed native source-ID grammar; no synthetic field is created.
    const BOTH_BEAMS = "__both_beams__";

    function beamLayers(validated) {
        const sources = validated?.data?.sources || [];
        const a = sources.find(s => s.id === "beam1" && s.kind === "beam_beam" && s
            .source_beam === 1 && s.target_beam === 2);
        const b = sources.find(s => s.id === "beam2" && s.kind === "beam_beam" && s
            .source_beam === 2 && s.target_beam === 1);
        return a && b && a.unit === b.unit && a.quantity === b.quantity ? [a, b] : [];
    }

    function layerStyle(panel) {
        const value = panel.trim();
        let rgb;
        if (/^#[0-9a-f]{6}$/i.test(value)) rgb = [1, 3, 5].map(i => parseInt(value.slice(i, i +
            2), 16));
        else if (/^#[0-9a-f]{3}$/i.test(value)) rgb = [1, 2, 3].map(i => parseInt(value[i] +
            value[i], 16));
        else {
            const m = value.match(/^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/);
            if (m) rgb = m.slice(1).map(Number)
        }
        check(rgb && rgb.every(v => v >= 0 && v <= 255),
        "Unsupported opaque field panel color");
        const light = (.2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]) / 255 > .5;
        // Okabe-Ito blue/orange, matching the viewer's light/dark beam colors.
        return {
            background: rgb,
            composite: light ? "multiply" : "screen",
            hues: light ? {
                beam1: [0, 114, 178],
                beam2: [213, 94, 0]
            } : {
                beam1: [86, 180, 233],
                beam2: [230, 159, 0]
            }
        };
    }

    function layerColor(value, maximum, sourceId, style) {
        check(style.hues[sourceId], "Unknown colored beam layer");
        const t = maximum > 0 ? Math.min(1, Math.max(0, value / maximum)) : 0;
        const neutral = style.composite === "multiply" ? 255 : 0;
        return "rgb(" + style.hues[sourceId].map(v => Math.round(neutral + t * (v - neutral)))
            .join(",") + ")";
    }

    function viewKey(horizontal, vertical) {
        if (horizontal === "x" && vertical === "y" || horizontal === "y" && vertical === "x")
            return "xy";
        if (horizontal === "z" && (vertical === "x" || vertical === "y")) return "z" + vertical;
        return null;
    }

    function projection(horizontal, vertical) {
        return viewKey(horizontal, vertical) !== null
    }

    /** Paint only recorded mesh cells on the supplied canvas transform, with explicit opacity and source scale. Does not infer missing fields or mutate native data. */
    function drawPlane(g, plane, view, opacity, maximum, layer = null) {
        const key = viewKey(view.horizontal, view.vertical);
        if (!plane || !key || Boolean(plane.longitudinal) !== (key !== "xy") || plane
            .longitudinal && plane.axes[1] !== view.vertical) return false;
        if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1 || !Number.isFinite(
            maximum) || maximum < 0)
            throw new Error("Invalid field display scale or opacity");
        g.save();
        g.beginPath();
        g.rect(view.L, view.T, view.W, view.H);
        g.clip();
        g.globalAlpha = opacity;
        g.globalCompositeOperation = layer ? layer.style.composite : "source-over";
        const paint = value => layer ? layerColor(value, maximum, layer.sourceId, layer.style) :
            color(value, maximum);
        if (plane.longitudinal) {
            for (const strip of plane.strips)
                for (let j = 0; j < plane.samples; j++) {
                    const left = view.px(strip.z_edges_um[0]),
                        right = view.px(strip.z_edges_um[1]);
                    const top = view.py(plane.transverse_edges_um[j + 1]),
                        bottom = view.py(plane.transverse_edges_um[j]);
                    g.fillStyle = paint(strip.values[j]);
                    g.fillRect(Math.min(left, right), Math.min(top, bottom), Math.abs(right -
                        left), Math.abs(bottom - top));
                }
        } else
            for (let j = 0; j < plane.ny; j++)
                for (let i = 0; i < plane.nx; i++) {
                    const x0 = plane.x_edges_um[i],
                        x1 = plane.x_edges_um[i + 1],
                        y0 = plane.y_edges_um[j],
                        y1 = plane.y_edges_um[j + 1];
                    const horizontal = view.horizontal === "x" ? [x0, x1] : [y0, y1],
                        vertical = view.vertical === "y" ? [y0, y1] : [x0, x1];
                    const left = view.px(horizontal[0]),
                        right = view.px(horizontal[1]),
                        top = view.py(vertical[1]),
                        bottom = view.py(vertical[0]);
                    g.fillStyle = paint(plane.values[j * plane.nx + i]);
                    g.fillRect(Math.min(left, right), Math.min(top, bottom), Math.abs(right -
                        left), Math.abs(bottom - top));
                }
        g.restore();
        return true;
    }

    /** Draw the numerical magnitude scale using the supplied display unit and upper bound; does not convert source data. */
    function legend(canvas, maximum, unit) {
        const g = canvas.getContext("2d"),
            L = 16,
            R = 16,
            T = 8,
            H = 15,
            W = canvas.width - L - R;
        g.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < PALETTE.length; i++) {
            g.fillStyle = maximum > 0 ? PALETTE[i] : PALETTE[0];
            g.fillRect(L + W * i / PALETTE.length, T, W / PALETTE.length + 1, H);
        }
        const text = v => v === 0 ? "0" : v.toExponential(3);
        const cs = getComputedStyle(document.documentElement);
        g.fillStyle = cs.getPropertyValue("--ink");
        g.font = "12px sans-serif";
        g.textAlign = "left";
        g.fillText("0", L, T + H + 18);
        g.textAlign = "right";
        g.fillText(text(maximum) + " " + unit, canvas.width - R, T + H + 18);
    }

    // Each ramp is one source composited over the actual panel at layer opacity.
    // Overlap colors are not an aggregate magnitude scale. Multiplication of the
    // two per-channel factors makes both modes order-independent in real arithmetic
    // (a browser may round intermediate 8-bit channels by one level).
    function layerLegend(canvas, maximum, unit, sourceId, style, opacity) {
        const g = canvas.getContext("2d"),
            L = 16,
            R = 16,
            T = 4,
            H = 12,
            W = canvas.width - L - R;
        g.clearRect(0, 0, canvas.width, canvas.height);
        g.save();
        g.globalAlpha = 1;
        g.globalCompositeOperation = "source-over";
        g.fillStyle = "rgb(" + style.background.join(",") + ")";
        g.fillRect(L, T, W, H);
        g.globalAlpha = opacity;
        g.globalCompositeOperation = style.composite;
        // Non-overlapping strips avoid applying opacity twice at ramp boundaries.
        for (let i = 0; i < W; i++) {
            g.fillStyle = layerColor(maximum * i / Math.max(1, W - 1), maximum, sourceId,
            style);
            g.fillRect(L + i, T, 1, H)
        }
        g.restore();
        g.save();
        g.globalAlpha = 1;
        g.globalCompositeOperation = "source-over";
        g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink");
        g.font = "12px sans-serif";
        g.textAlign = "left";
        g.fillText("0", L, T + H + 17);
        g.textAlign = "right";
        g.fillText((maximum === 0 ? "0" : maximum.toExponential(3)) + " " + unit, canvas.width -
            R, T + H + 17);
        g.restore();
    }

    // Stable coordinate extent across saved encounters and both source beams.
    // This is a sampling window, never a claim of simultaneous field coverage.
    function samplingWindow(validated, view, requested = "auto") {
        if (!validated || !["zx", "zy"].includes(view)) return null;
        const full = dataMode(validated, requested) === "snapshot",
            layer = full ? validated.snapshot : validated.longitudinal;
        if (!layer) return null;
        let low = Infinity,
            high = -Infinity;
        for (const frame of validated.data.frames) {
            if (full) {
                if (!frame.snapshot.available) continue;
                for (const source of validated.data.sources)
                    for (const strip of frame.snapshot.sources[source.id].strips) {
                        low = Math.min(low, strip.display_z_bounds_nm[0] / 1000);
                        high = Math.max(high, strip.display_z_bounds_nm[1] / 1000);
                    }
            } else
                for (const strip of frame.longitudinal.strips.filter(s => s.available))
                    for (const source of validated.data.sources) {
                        const edges = strip.display_z_bounds_nm[source.id];
                        low = Math.min(low, edges[0] / 1000);
                        high = Math.max(high, edges[1] / 1000);
                    }
        }
        if (!Number.isFinite(low) || !(high > low)) return null;
        const edges = layer.geometry[view].transverse_edges_um;
        return {
            horizontal: [low, high],
            vertical: [edges[0], edges[edges.length - 1]],
            sourceSlices: layer.metadata.n_z
        };
    }
    const api = {
        BOTH_BEAMS,
        beamLayers,
        layerStyle,
        layerColor,
        layerLegend,
        LIMITS,
        validate,
        plane,
        scaleMaximum,
        color,
        projection,
        viewKey,
        drawPlane,
        legend,
        samplingWindow,
        dataMode
    };
    root.ColliderFields = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);
