"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path'),
    f = require('./collider_fields.js');
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_snapshot_fields.json'), 'utf8'));
let checks = 0;
const original = JSON.stringify(fixture),
    v = f.validate(fixture.fields, fixture.manifest);
assert.equal(v.snapshot.availableFrames, 2);
assert.equal(v.snapshot.availableStrips, 12);
assert.equal(v.snapshot.scalarCount, 48);
checks += 3;
assert.equal(f.dataMode(v), 'snapshot');
checks++;
for (const view of ['zx', 'zy']) {
    const window = f.samplingWindow(v, view);
    assert.deepEqual(window.horizontal, [-3.5, 3.5]);
    checks++;
    const old = f.samplingWindow(v, view, 'encounter');
    assert.deepEqual(old.horizontal, [-2.5, 2.5]);
    checks++;
    for (const source of ['beam1', 'beam2']) {
        const first = f.plane(v, 0, source, view),
            last = f.plane(v, 1, source, view),
            oldPlane = f.plane(v, 0, source, view, 'encounter');
        assert(first.snapshot && first.coverage.available === 3 && last.coverage.available === 3);
        checks++;
        assert.equal(oldPlane.coverage.available, 1);
        checks++;
        assert.equal(first.strips.length, 3);
        assert.deepEqual(first.strips[0].values, [0, 0]);
        checks += 2;
        assert.equal(last.strips[0].z_edges_um[0] - first.strips[0].z_edges_um[0], source ===
            'beam1' ? -1 : 1);
        checks++;
        for (const p of [first, last]) assert(p.strips.every(s => s.z_edges_um[0] >= window
            .horizontal[0] && s.z_edges_um[1] <= window.horizontal[1]));
        checks++;
        assert.equal(f.scaleMaximum(v, 0, source, 'run', view), f.scaleMaximum(v, 1, source, 'run',
            view));
        checks++;
        assert(f.scaleMaximum(v, 0, source, 'frame', view) < f.scaleMaximum(v, 1, source, 'frame',
            view));
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
        assert(f.drawPlane(g, first, {
            horizontal: 'z',
            vertical: view[1],
            L: 0,
            T: 0,
            W: 100,
            H: 100,
            px: x => x,
            py: y => -y
        }, .4, 300));
        assert.equal(rects.length, 6);
        checks += 2;
    }
}

function bad(change) {
    const d = structuredClone(fixture.fields);
    change(d);
    assert.throws(() => f.validate(d, fixture.manifest));
    checks++
}
for (const change of [d => delete d.snapshot, d => d.snapshot.schema_version = 'old', d => d
        .snapshot.frame = 'lab', d => d.snapshot.n_z = 2,
        d => d.snapshot.delta_z_nm = 0, d => d.snapshot.model = 'full physical E', d => d
        .snapshot.timing = 'pre-kick', d => d.snapshot.planes.zx.samples = 65,
        d => d.snapshot.planes.zx.fixed_coordinate_nm = 1, d => d.snapshot.planes.zx.min_nm = -
        999, d => delete d.frames[0].snapshot,
        d => d.frames[0].snapshot.diagonal = 1, d => d.frames[0].snapshot.phase = 'pre-kick',
        d => d.frames[0].snapshot.expected_slices = 1,
        d => Object.assign(d.frames[0].snapshot, {
            available: false,
            reason: 'no data'
        }), d => d.frames[0].snapshot.sources = null,
        d => d.frames[0].snapshot.sources.beam1.strips.pop()
    ]) bad(change);
for (const change of [s => s.slice = 0, s => s.display_z_nm = 0, s => s.display_z_bounds_nm = [-1,
            1], s => s.source_particles = 12, s => s.outside_particles = -1,
        s => s.deposited_particles = true, s => s.maps.zx.pop(),
        s => s.maps.zx[0] = -1, s => s.maps.zx[0] = NaN, s => s.maps.zx[0] = true, s => delete s
        .maps.zy
    ]) bad(d => change(d.frames[0].snapshot.sources.beam1.strips[1]));
const residue = structuredClone(fixture.fields);
residue.frames[0].snapshot.sources.beam1.strips[0].maps.zx[0] = 1e-25;
assert.equal(f.plane(f.validate(residue, fixture.manifest), 0, 'beam1', 'zx').strips[0].values[0],
    1e-25);
checks++;
const unavailable = structuredClone(fixture.fields);
Object.assign(unavailable.frames[0].snapshot, {
    available: false,
    reason: 'unsupported',
    sources: null
});
const u = f.validate(unavailable, fixture.manifest);
assert.equal(f.plane(u, 0, 'beam1', 'zx'), null);
assert(f.plane(u, 0, 'beam1', 'zx', 'encounter'));
checks += 2;
const previous = f.LIMITS.scalars;
try {
    f.LIMITS.scalars = 80;
    assert.throws(() => f.validate(fixture.fields, fixture.manifest), /Combined/);
    checks++
} finally {
    f.LIMITS.scalars = previous
}
const old = structuredClone(fixture.fields);
delete old.snapshot;
for (const r of old.frames) delete r.snapshot;
assert.equal(f.dataMode(f.validate(old, fixture.manifest)), 'encounter');
checks++;
const contract = require('./collider_contract.js'),
    ref = {
        path: 'gui_fields.json',
        bytes: 40 * 1024 * 1024,
        sha256: 'a'.repeat(64)
    };
assert.throws(() => contract.fileRef(ref));
contract.fileRef(ref, f.LIMITS.fileBytes);
assert.throws(() => contract.fileRef({
    ...ref,
    bytes: 65 * 1024 * 1024
}, f.LIMITS.fileBytes));
checks += 3;
assert.equal(JSON.stringify(fixture), original);
checks++;
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Synthetic snapshot schema, source-slice motion/coverage, exact count receipts, fixed viewport, separate modes and file bounds; no browser or physics claim'
}));
