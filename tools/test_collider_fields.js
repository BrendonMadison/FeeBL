"use strict";
const assert = require("node:assert/strict"),
    f = require("./collider_fields.js");
let checks = 0;
const manifest = {
    frames: [{
        id: 0,
        diagonal: 0,
        elapsed_s: 0
    }, {
        id: 1,
        diagonal: 2,
        elapsed_s: 2e-12
    }]
};
const data = {
    schema_version: "feebl.collider.fields/1",
    plane: {
        axes: ["x", "y"],
        coordinate_unit: "nm",
        frame: "native simulated frame"
    },
    geometry: {
        nx: 3,
        ny: 2,
        x_min_nm: -1000,
        x_max_nm: 1000,
        y_min_nm: 0,
        y_max_nm: 1000,
        native_domain_nm: [-1200, 1200, -200, 1200],
        storage_order: "row-major-y-x",
        sampling: "uniform CIC gather points; bounds are sample centres"
    },
    sources: [{
        id: "beam1",
        kind: "beam_beam",
        label: "Source beam 1 → beam 2",
        quantity: "native transverse electric-equivalent field magnitude",
        unit: "GV/nm",
        source_beam: 1,
        target_beam: 2
    }],
    conventions: {
        selection: "central active slice encounter, final substep, pre-kick",
        scope: "single source slice; not whole-bunch or diagonal-averaged field",
        charge_sign: -1,
        scale_step: [1, 1],
        n_z: 3,
        n_t: 2
    },
    frames: manifest.frames.map((r, i) => ({
        ...r,
        available: true,
        reason: null,
        encounter: {
            beam1_slice: i,
            beam2_slice: i,
            substep_index: 1,
            encounter_counter: i,
            nominal_z_nm: i * 100,
            phase: "post-solve/pre-kick"
        },
        maps: {
            beam1: i ? [0, 0, 0, 0, 0, 0] : [0, 1, 2, 3, 4, 5]
        }
    }))
};
const validated = f.validate(data, manifest);
checks++;
assert.deepEqual(validated.geometry.x_edges_um, [-1.2, -.5, .5, 1.2]);
checks++;
assert.deepEqual(validated.geometry.y_edges_um, [-.2, .5, 1.2]);
checks++;
assert.equal(f.scaleMaximum(validated, 1, "beam1", "run"), 5);
checks++;
assert.equal(f.scaleMaximum(validated, 1, "beam1", "frame"), 0);
checks++;

function reject(mutate) {
    const x = structuredClone(data);
    mutate(x);
    assert.throws(() => f.validate(x, manifest));
    checks++
}
reject(x => x.schema_version = "unknown");
reject(x => x.plane.axes = ["z", "x"]);
reject(x => x.plane.coordinate_unit = "m");
reject(x => x.geometry.nx = 1);
reject(x => x.geometry.ny = 65);
reject(x => x.geometry.x_min_nm = Infinity);
reject(x => x.geometry.x_max_nm = x.geometry.x_min_nm);
reject(x => x.geometry.native_domain_nm[0] = 0);
reject(x => x.geometry.storage_order = "row-major-x-y");
reject(x => x.frames[1].elapsed_s = 3e-12);
reject(x => x.frames[1].diagonal = 1);
reject(x => x.frames[0].maps.beam1.pop());
reject(x => x.frames[0].maps.beam1[0] = -1);
reject(x => x.frames[0].maps.beam1[0] = NaN);
reject(x => x.frames[0].maps.other = [0]);
reject(x => x.frames[0].encounter.phase = "post-kick");
reject(x => x.frames[0].encounter.substep_index = 2);
reject(x => x.sources[0].target_beam = 1);
reject(x => x.sources[0].source_beam = true);
reject(x => x.sources[0].unit = "T");
reject(x => x.sources.push({
    ...x.sources[0]
}));
reject(x => x.sources[0].kind = "unknown");
reject(x => x.frames[1].available = false);
reject(x => x.conventions.selection = "unknown");
reject(x => x.conventions.scope = "whole bunch");
reject(x => x.conventions.charge_sign = 0);
reject(x => x.conventions.scale_step = [1, 2]);
reject(x => x.conventions.scale_step = [0, 1]);
reject(x => x.sources[0].id = "1beam");
reject(x => x.frames[1].encounter.beam1_slice = 0);
reject(x => x.frames[0].encounter.substep_index = 0);
const unavailable = structuredClone(data);
Object.assign(unavailable.frames[1], {
    available: false,
    reason: "not materialized on this backend",
    encounter: null,
    maps: null
});
const uv = f.validate(unavailable, manifest);
assert.equal(f.plane(uv, 1, "beam1"), null);
checks++;
const zeroControl = structuredClone(data);
zeroControl.conventions.charge_sign = 0;
for (const r of zeroControl.frames) Object.assign(r, {
    available: false,
    reason: "native normalization requires charge_sign=-1",
    encounter: null,
    maps: null
});
const zc = f.validate(zeroControl, manifest);
assert.equal(f.plane(zc, 0, "beam1"), null);
checks++;
zeroControl.conventions.charge_sign = .5;
assert.throws(() => f.validate(zeroControl, manifest));
checks++;
const external = structuredClone(data);
Object.assign(external.sources[0], {
    kind: "external",
    source_beam: null,
    target_beam: null,
    quantity: "supplied external electric field magnitude",
    unit: "V/m"
});
f.validate(external, manifest);
checks++; // Schema-only fixture; no external physics is computed.
assert.equal(f.color(-1, 5), f.color(0, 5));
checks++;
assert.equal(f.color(10, 5), f.color(5, 5));
checks++;
assert.equal(f.color(0, 0), f.color(0, 5));
checks++;
const srgb = v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
const luminance = c => [.2126, .7152, .0722].reduce((sum, w, i) => sum + w * srgb(parseInt(c.slice(
    1 + 2 * i, 3 + 2 * i), 16) / 255), 0);
let last = -Infinity;
for (let i = 0; i < 256; i++) {
    const y = luminance(f.color(i, 255));
    assert(y >= last, "Cividis relative luminance must not reverse");
    last = y
}
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
            rects.push({
                rect: r,
                color: this.fillStyle
            })
        }
    };
const view = {
    horizontal: "x",
    vertical: "y",
    L: 0,
    T: 0,
    W: 200,
    H: 200,
    px: x => 100 + 50 * x,
    py: y => 100 - 50 * y
};
f.drawPlane(g, f.plane(validated, 0, "beam1"), view, .4, 5);
assert.equal(rects.length, 6);
checks++;
assert.deepEqual(rects[0].rect, [40, 75, 35, 35]);
assert.equal(rects[0].color, f.color(0, 5));
checks++;
assert.equal(rects[5].color, f.color(5, 5));
checks++;
rects.length = 0;
f.drawPlane(g, f.plane(validated, 0, "beam1"), {
    ...view,
    horizontal: "y",
    vertical: "x"
}, .4, 5);
assert.deepEqual(rects[0].rect, [90, 125, 35, 35]);
checks++;
rects.length = 0;
assert.equal(f.drawPlane(g, f.plane(validated, 0, "beam1"), {
    ...view,
    horizontal: "z"
}, .4, 5), false);
assert.equal(rects.length, 0);
checks++;
f.drawPlane(g, f.plane(validated, 1, "beam1"), view, .4, 0);
assert.equal(rects.length, 6);
assert(rects.every(x => x.color === f.color(0, 0)));
checks++;
assert.throws(() => f.drawPlane(g, f.plane(validated, 0, "beam1"), view, 2, 5));
checks++;
const crypto = require("node:crypto"),
    contract = require("./collider_contract.js");
(async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(data)),
        digest = crypto.createHash('sha256').update(bytes).digest('hex');
    const ref = {
        path: 'gui_fields.json',
        bytes: bytes.byteLength,
        sha256: digest
    };
    await contract.bytesHash(bytes, ref);
    checks++;
    await assert.rejects(() => contract.bytesHash(bytes, {
        ...ref,
        sha256: '0'.repeat(64)
    }));
    checks++;
    const changed = bytes.slice();
    changed[changed.length - 2] ^= 1;
    await assert.rejects(() => contract.bytesHash(changed, ref));
    checks++;
    await assert.rejects(() => contract.bytesHash(bytes.slice(0, -1), ref));
    checks++;
    await assert.rejects(() => contract.bytesHash(bytes, {
        ...ref,
        path: '../gui_fields.json'
    }));
    checks++;
    console.log(JSON.stringify({
        status: "PASS_FOR_STATED_TARGET",
        checks,
        scope: "Field schema, source layers, orientation, domain clipping, zero/unavailable cases and monotone Cividis luminance; synthetic fixtures, authenticated-field tamper rejection, no physics/browser QA"
    }));
})().catch(e => {
    console.error(e);
    process.exitCode = 1
});
