"use strict";
const assert = require("node:assert/strict"),
    fs = require("node:fs"),
    path = require("node:path"),
    f = require("./collider_fields.js");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    "testdata/collider_longitudinal.json"), "utf8"));
const data = fixture.fields,
    manifest = fixture.manifest;
let checks = 0;
const v = f.validate(data, manifest);
assert.equal(v.longitudinal.scalarCount, 32);
checks++;
assert.equal(v.longitudinal.availableStrips, 4);
checks++;
assert.equal(f.scaleMaximum(v, 1, "beam2", "run", "zy"), 15);
checks++;
assert.equal(f.scaleMaximum(v, 0, "beam2", "frame", "zy"), 13);
checks++;
const gap = structuredClone(data),
    missing = gap.frames[1].longitudinal.strips[1];
Object.assign(missing, {
    available: false,
    reason: "uncaptured",
    maps: null,
    substep_index: null,
    encounter_counter: null
});
const vg = f.validate(gap, manifest);
assert.equal(vg.longitudinal.availableStrips, 3);
checks++;
const p = f.plane(vg, 1, "beam1", "zx");
assert.deepEqual(p.coverage, {
    available: 2,
    expected: 3
});
checks++;
assert.deepEqual(p.strips.map(s => s.z_edges_um), [
    [-.5, .5],
    [1.5, 2.5]
]);
checks++;
assert.deepEqual(f.plane(vg, 1, "beam2", "zx").strips.map(s => s.z_edges_um), [
    [-2.5, -1.5],
    [-.5, .5]
]);
checks++;
const rects = [],
    g = {
        save() {},
        restore() {},
        beginPath() {},
        rect() {},
        clip() {},
        fillRect(...r) {
            assert(r.every(Number.isFinite));
            rects.push(r)
        }
    };
const view = {
    horizontal: "z",
    vertical: "x",
    L: 0,
    T: 0,
    W: 400,
    H: 400,
    px: x => 200 + 50 * x,
    py: y => 200 - 50 * y
};
f.drawPlane(g, p, view, .4, 3);
assert.equal(rects.length, 4);
checks++;
assert.deepEqual(rects.map(r => [r[0], r[2]]), [
    [175, 50],
    [175, 50],
    [275, 50],
    [275, 50]
]);
checks++;
assert(rects.every(r => r[0] + r[2] <= 225 || r[0] >= 275));
checks++; // no colored interpolation across the missing band
rects.length = 0;
f.drawPlane(g, f.plane(vg, 1, "beam1", "zy"), {
    ...view,
    vertical: "y"
}, .4, 5);
assert.equal(rects.length, 4);
checks++;
assert.equal(f.drawPlane(g, p, {
    ...view,
    vertical: "y"
}, .4, 3), false);
checks++;
const legacy = structuredClone(data);
delete legacy.longitudinal;
for (const r of legacy.frames) delete r.longitudinal;
assert.equal(f.plane(f.validate(legacy, manifest), 0, "beam1", "zx"), null);
checks++;

function reject(change) {
    const d = structuredClone(data);
    change(d);
    assert.throws(() => f.validate(d, manifest));
    checks++
}
for (const key of ["schema_version", "frame", "coordinate_unit", "selection", "timing",
    "z_mapping"]) reject(d => d.longitudinal[key] = "wrong");
reject(d => d.longitudinal.n_z = 4);
reject(d => d.longitudinal.delta_z_nm = 0);
reject(d => d.longitudinal.delta_z_nm = Infinity);
reject(d => d.longitudinal.planes.zx.samples = 65);
reject(d => d.longitudinal.planes.zx.samples = true);
reject(d => d.longitudinal.planes.zx.min_nm = -999);
reject(d => d.longitudinal.planes.zx.fixed_coordinate_nm = false);
reject(d => d.longitudinal.planes.zx.axes = ["x", "z"]);
reject(d => delete d.longitudinal.planes.zy);
reject(d => delete d.longitudinal);
for (const change of [s => s.beam1_slice = 1, s => s.nominal_z_nm = 1, s => s
        .nominal_z_bounds_nm = [-500, 501],
        s => s.display_z_nm.beam1 = 0, s => s.display_z_bounds_nm.beam2 = [-1500, 500], s => s
        .substep_index = 0,
        s => s.encounter_counter = -1, s => s.available = 1, s => s.reason = "wrong", s => s
        .maps.beam1.zx.pop(),
        s => s.maps.beam1.zx[0] = NaN, s => s.maps.beam1.zx[0] = -1, s => s.maps.beam1.zx[0] =
        true, s => delete s.maps.beam2, s => delete s.maps.beam1.zy
    ])
    reject(d => change(d.frames[0].longitudinal.strips[0]));
reject(d => d.frames[1].longitudinal.strips.pop());
reject(d => d.frames[1].longitudinal.expected_encounters = 2);
reject(d => d.frames[1].longitudinal.available = false);
reject(d => d.frames[1].longitudinal.strips[1].encounter_counter = d.frames[1].longitudinal.strips[
    0].encounter_counter);
for (const key of ["substep_index", "encounter_counter"]) reject(d => Object.assign(d.frames[1]
    .longitudinal.strips[1], {
        available: false,
        reason: "unsupported",
        maps: null,
        [key]: null
    }));
const visited = structuredClone(data);
Object.assign(visited.frames[1].longitudinal.strips[1], {
    available: false,
    reason: "unsupported dispersion",
    maps: null
});
assert.equal(f.validate(visited, manifest).longitudinal.availableStrips, 3);
checks++;
const unavailable = structuredClone(data);
for (const r of unavailable.frames) {
    r.longitudinal.available = false;
    r.longitudinal.reason = "unsupported";
    for (const s of r.longitudinal.strips) Object.assign(s, {
        available: false,
        reason: "unsupported",
        maps: null
    })
}
assert.equal(f.validate(unavailable, manifest).longitudinal.availableFrames, 0);
checks++;
assert.equal(f.plane(f.validate(unavailable, manifest), 0, "beam1", "zx"), null);
checks++;
const unsupported = structuredClone(data);
unsupported.conventions.n_z = 129;
unsupported.longitudinal.n_z = 129;
for (const r of unsupported.frames) Object.assign(r.longitudinal, {
    available: false,
    reason: "slice bound",
    strips: []
});
assert.equal(f.validate(unsupported, manifest).longitudinal.scalarCount, 0);
checks++;
const zero = structuredClone(data);
for (const r of zero.frames)
    for (const s of r.longitudinal.strips)
        for (const views of Object.values(s.maps))
            for (const a of Object.values(views)) a.fill(0);
const zv = f.validate(zero, manifest);
rects.length = 0;
f.drawPlane(g, f.plane(zv, 1, "beam1", "zx"), view, .4, 0);
assert.equal(rects.length, 6);
checks++;
const crypto = require("node:crypto"),
    contract = require("./collider_contract.js");
(async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(data)),
        ref = {
            path: "gui_fields.json",
            bytes: bytes.length,
            sha256: crypto.createHash("sha256").update(bytes).digest("hex")
        };
    await contract.bytesHash(bytes, ref);
    checks++;
    const altered = new TextEncoder().encode(JSON.stringify(gap));
    await assert.rejects(() => contract.bytesHash(altered, ref));
    checks++;
    console.log(JSON.stringify({
        status: "PASS_FOR_STATED_TARGET",
        checks,
        scope: "Synthetic longitudinal geometry, coverage, gaps, source mapping and hash rejection; no physics or browser QA"
    }));
})().catch(e => {
    console.error(e);
    process.exitCode = 1
});
