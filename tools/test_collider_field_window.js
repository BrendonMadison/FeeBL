"use strict";
const assert = require("node:assert/strict"),
    fs = require("node:fs"),
    path = require("node:path"),
    f = require("./collider_fields.js");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    "testdata/collider_longitudinal.json"), "utf8"));
const original = JSON.stringify(fixture.fields),
    v = f.validate(fixture.fields, fixture.manifest);
let checks = 0;
for (const view of ["zx", "zy"]) {
    const w = f.samplingWindow(v, view);
    assert.deepEqual(w.horizontal, [-2.5, 2.5]);
    checks++;
    const e = v.longitudinal.geometry[view].transverse_edges_um;
    assert.deepEqual(w.vertical, [e[0], e[e.length - 1]]);
    checks++;
    assert.equal(w.sourceSlices, 3);
    checks++;
    for (let i = 0; i < fixture.fields.frames.length; i++)
        for (const source of ["beam1", "beam2"]) {
            const p = f.plane(v, i, source, view);
            assert(p.strips.every(s => s.z_edges_um[0] >= w.horizontal[0] && s.z_edges_um[1] <= w
                .horizontal[1]));
            checks++;
        }
}
assert.equal(f.samplingWindow(v, "xy"), null);
checks++;
assert.equal(f.samplingWindow(null, "zx"), null);
checks++;
const missing = structuredClone(fixture.fields);
for (const frame of missing.frames) {
    frame.longitudinal.available = false;
    frame.longitudinal.reason = "uncaptured";
    for (const strip of frame.longitudinal.strips) Object.assign(strip, {
        available: false,
        reason: "uncaptured",
        maps: null,
        substep_index: null,
        encounter_counter: null
    });
}
const vm = f.validate(missing, fixture.manifest);
assert.equal(f.samplingWindow(vm, "zx"), null);
checks++;
assert.equal(f.plane(vm, 0, "beam1", "zx"), null);
checks++;
assert.equal(JSON.stringify(fixture.fields), original);
checks++;
console.log(JSON.stringify({
    status: "PASS_FOR_STATED_TARGET",
    checks,
    scope: "Stable sampling window, both sources and views; unavailable values remain absent; no data rescaling or mutation"
}));
