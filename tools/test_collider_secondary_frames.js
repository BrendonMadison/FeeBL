"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path');
const kin = require('./collider_secondary_kinematics.js'),
    f = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_secondary_frames.json')));
let checks = 0;
kin.validate(f.secondary, f.refs, f.context);
checks++;

function reject(fn) {
    const a = structuredClone(f.secondary);
    fn(a);
    assert.throws(() => kin.validate(a, f.refs, f.context));
    checks++
}
for (const fn of [a => a.frames.pop(), a => a.frames[0].id = 1, a => a.frames[1].diagonal = 0, a =>
        a.frames[1].elapsed_s = 0, a => a.frames[1].phase = 'pre-kick',
        a => a.frame_availability = false, a => a.frames[0].available = false, a => a.frames[0]
        .reason = 'false missing', a => a.frames[2].reason = null,
        a => a.frames[1].tracker_count_verification.admitted_tracker_count++, a => a.frames[1]
        .tracker_count_verification.authoritative_reserve_count++,
        a => a.frames[1].tracker_count_verification.actual_records++, a => a.frames[2]
        .tracker_count_verification.admitted_tracker_count = 0,
        a => a.frames[1].populations.pairs.joint.counts[0]++, a => a.frames[1].populations.pairs
        .joint_folded.samples++,
        a => a.frames[1].populations.pairs.joint.theta_edges_rad[0] /= 2, a => a.frames[1]
        .populations.pairs.normalization.completeness_verified = true,
        a => a.frames[1].populations.pairs.normalization.weighting = kin.WEIGHTING, a => a
        .frames[1].populations.pairs.selection.admitted_tracker_count++,
        a => a.frames[1].populations.pairs.selection.authoritative_reserve_count++, a => a
        .frames[1].populations.pairs.selection.process_counts['0']++,
        a => a.frames[0].populations.pairs.distributions.theta.statistics = {
            mean: 0,
            variance: 0,
            min: 0,
            max: 0,
            mode: 0
        },
        a => a.frames[1].populations.pairs.distributions.theta.weighting = kin.WEIGHTING, a => a
        .frames[1].populations.pairs.distributions.pT.bins[0].weight++,
        a => delete a.source_artifacts.frame_snapshots, a => a.source_artifacts
        .frame_snapshots = a.source_artifacts.summary,
        a => a.frame_definition = null, a => a.frames[2].populations.pairs = structuredClone(a
            .frames[1].populations.pairs)
    ]) reject(fn);
reject(a => a.frames[1].populations.pairs.label = 'Unverified species label');
assert.throws(() => kin.validate(f.secondary, f.refs));
checks++;
const old = structuredClone(f.secondary);
delete old.frames;
delete old.frame_definition;
delete old.source_artifacts.frame_snapshots;
old.frame_availability = false;
kin.validate(old, f.refs);
checks++;
const unavailable = structuredClone(f.secondary);
unavailable.frames = unavailable.frames.map((x, i) => ({
    ...structuredClone(unavailable.frames[2]),
    ...f.context.manifest.frames[i]
}));
unavailable.frame_availability = false;
const unavailableContext = structuredClone(f.context);
unavailableContext.frameSnapshots.frames = unavailableContext.frameSnapshots.frames.map((x, i) => ({
    ...structuredClone(unavailableContext.frameSnapshots.frames[2]),
    id: i,
    diagonal: i
}));
kin.validate(unavailable, f.refs, unavailableContext);
checks++;
for (const mutate of [c => delete c.manifest.secondary_frames_file, c => c.manifest
        .secondary_frames_file = 'out.dat', c => delete c.frameSnapshots, c => c.frameSnapshots
        .frames[1].joint.counts[0]++, c => c.frameSnapshots.frames[1].marginals.theta.statistics
        .mean += .1,
        c => c.frameSnapshots.evaluated_flags.do_pairs++, c => c.frameSnapshots.evaluated_flags
        .pair_ratio = .5, c => c.frameSnapshots.definitions.timing = 'different timing',
        c => c.frameSnapshots.frames[1].selection.authoritative_reserve_count++, c => c
        .frameSnapshots.axes.theta_edges_rad[0] /= 2
    ]) {
    const context = structuredClone(f.context);
    mutate(context);
    assert.throws(() => kin.validate(f.secondary, f.refs, context));
    checks++;
}
const flow = structuredClone(f),
    h = flow.secondary.frames[1].populations.pairs.distributions.theta,
    r = flow.context.frameSnapshots.frames[1].marginals.theta;
h.bins.forEach(b => b.weight = 0);
h.overflow = h.samples;
h.statistics.mode = null;
h.statistics.mode_method = 'unavailable: no in-range marginal records';
r.counts = h.bins.map(b => b.weight);
r.overflow = h.overflow;
r.statistics = structuredClone(h.statistics);
kin.validate(flow.secondary, flow.refs, flow.context);
checks++;
h.statistics.mode = 0;
assert.throws(() => kin.validate(flow.secondary, flow.refs, flow.context));
checks++;
const b = kin.axisBounds(f.secondary, 'pairs');
assert.equal(b.xmin, Math.log10(f.secondary.populations.pairs.joint.theta_edges_rad[0]) - .25);
checks++;
assert.equal(b.xmax, Math.log10(f.secondary.populations.pairs.joint.theta_edges_rad.at(-1)) + .25);
checks++;
const wide = structuredClone(f.secondary);
for (const frame of wide.frames.filter(x => x.available)) frame.populations.pairs.joint
    .theta_edges_rad = [1e-12, 1e-6, 1e6];
wide.frames[1].populations.pairs.joint.counts = [0, 0, 1, 0];
const w = kin.axisBounds(wide, 'pairs');
assert.equal(w.xmin, -12.25);
assert.equal(w.xmax, b.xmax);
checks++;
assert.deepEqual(kin.axisBounds(wide, 'pairs'), w);
checks++;
const none = structuredClone(unavailable);
none.populations.pairs = structuredClone(none.populations.deflected);
assert(Object.values(kin.axisBounds(none, 'pairs')).filter(x => typeof x === 'number').every(Number
    .isFinite));
checks++;
for (const angle of [false, true]) {
    const axes = kin.axisBounds(f.secondary, 'pairs', angle);
    assert(axes.xmin < axes.xmax && axes.ymin < axes.ymax && axes.padding_decades === .25);
    checks++
}
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Synthetic frame timing/count/weight provenance, empty and unavailable snapshots, backward compatibility and fixed padded axis union'
}));
