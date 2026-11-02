"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path');
const mono = require('./collider_monochromatization.js'),
    kin = require('./collider_secondary_kinematics.js');
const f = JSON.parse(fs.readFileSync(path.join(__dirname, 'testdata/collider_new_diagnostics.json'),
    'utf8'));
let checks = 0;
mono.validate(f.mono, f.refs, f.context);
checks++;
kin.validate(f.secondary, f.refs);
checks++;

function rejectMono(fn) {
    const a = structuredClone(f.mono);
    fn(a);
    assert.throws(() => mono.validate(a, f.refs, f.context));
    checks++
}

function rejectKin(fn) {
    const a = structuredClone(f.secondary);
    fn(a);
    assert.throws(() => kin.validate(a, f.refs));
    checks++
}
for (const fn of [a => a.schema_version = 'wrong', a => a.ideal.value = 2, a => a.ideal.available =
        false, a => a.inputs.sigma_delta[0] = .1, a => a.inputs.energy_GeV[0] = -1,
        a => a.inputs.sigma_beta_x_m[0] = 0, a => a.inputs.which_espread[0] = 5, a => a.ideal
        .scope = 'measured', a => a.measured.reference_sigma_W_GeV = 1,
        a => a.measured.total.value *= 2, a => a.measured.total.sigma_W_GeV *= 2, a => a
        .measured.total.estimator = 'core fit', a => a.measured.frames[1].id = 0,
        a => a.measured.frames[0].available = true, a => a.measured.frames[1].sigma_W_GeV *= 2,
        a => a.measured.frames[1].diagonal_start = 0,
        a => a.measured.frames[1].diagonal_end = 9, a => a.matched_control.available = true,
        a => a.source_artifacts.analysis.sha256 = 'd'.repeat(64),
        a => delete a.source_artifacts.analysis, a => delete a.source_artifacts.diagnostics,
        a => a.adapter_sha256 = 'x'
    ]) rejectMono(fn);
assert.throws(() => mono.validate(f.mono, f.refs));
checks++;
const inputs = structuredClone(f.mono.inputs);
inputs.ip_dispersion_m = [.00035, -.00035];
const expected = Math.sqrt(1 + (inputs.ip_dispersion_m[0] * inputs.sigma_delta[0] / inputs
    .sigma_beta_x_m[0]) ** 2);
assert(Math.abs(mono.idealValue(inputs) - expected) < 1e-12);
checks++;
inputs.ip_dispersion_m = [.00035, .00035];
assert.equal(mono.idealValue(inputs), 1);
checks++;
inputs.native_beam_angles_x_rad = [.007, .007];
assert.equal(mono.idealValue(inputs), null);
checks++;
inputs.ip_dispersion_m = [0, 0];
assert.equal(mono.idealValue(inputs), 1);
checks++;
const h = structuredClone(f.context.diagnostics.frames[1].luminosity_spectrum);
assert.equal(mono.frameWidth(h).available, true);
checks++;
h.overflow = 1;
assert.equal(mono.frameWidth(h).available, false);
checks++;
h.overflow = 0;
h.bins[1].weight = 0;
assert.equal(mono.frameWidth(h).available, false);
checks++;
const unresolved = {
    bins: [{
        low: 0,
        high: 1,
        weight: 1
    }, {
        low: 1,
        high: 2,
        weight: 1
    }],
    underflow: 0,
    overflow: 0
};
assert.equal(mono.frameWidth(unresolved).available, false);
checks++;
for (const fn of [a => a.schema_version = 'wrong', a => a.frame_availability = true, a => a.phase =
        'frame', a => a.coordinate_frame = 'detector', a => a.source_artifacts.pairs.sha256 =
        'a'.repeat(64),
        a => a.populations.pairs.normalization.physical_weight_available = true, a => a
        .populations.pairs.selection.origin_separation = true, a => a.populations.pairs
        .nrecords++,
        a => a.populations.pairs.joint.counts[0]++, a => a.populations.pairs.joint.counts[0] =
        .5, a => a.populations.pairs.joint.theta_edges_rad[0] = 0,
        a => a.populations.pairs.joint.pT_edges_GeV[1] = NaN, a => a.populations.pairs.joint
        .storage_order = 'column-major', a => a.populations.pairs.joint.samples++,
        a => a.populations.pairs.joint.excluded.both_nonpositive = 1, a => a.populations.pairs
        .joint.excluded.underflow_records++, a => a.populations.pairs.joint_folded.counts.pop(),
        a => a.populations.pairs.distributions.theta.unit = 'deg', a => a.populations.pairs
        .distributions.pT.unit = 'GeV', a => a.populations.pairs.distributions.theta.weighting =
        'physical yield',
        a => a.populations.pairs.distributions.theta.statistics.variance = -1, a => a
        .populations.pairs.distributions.theta.statistics.mode = .8,
        a => a.populations.pairs.selection.process_counts['0']++, a => a.populations.pairs
        .selection.signed_energy_counts.positive_energy++,
        a => a.record_count_verification.expected_records++, a => a.evaluated_flags.do_pairs =
        0, a => a.populations.deflected.joint = {},
        a => a.populations.pairs.available = false
    ]) rejectKin(fn);
for (const key of ['deflected', 'combined']) {
    const a = structuredClone(f.secondary),
        p = a.populations.pairs,
        empty = a.populations[key];
    a.populations[key] = p;
    a.populations.pairs = empty;
    a.evaluated_flags.load_events = 1;
    a.evaluated_flags.do_pairs = key === 'combined' ? 1 : 0;
    a.evaluated_flags.do_bhabhas = 0;
    p.normalization.completeness_verified = false;
    a.record_count_verification.available = false;
    a.record_count_verification.expected_records = null;
    kin.validate(a, f.refs);
    checks++;
}
const excluded = structuredClone(f.secondary);
for (const name of ['joint', 'joint_folded']) {
    const j = excluded.populations.pairs[name];
    j.counts[1]--;
    j.in_range_records--;
    Object.assign(j.excluded, {
        theta_nonpositive: 1,
        nonpositive_records: 1
    })
}
kin.validate(excluded, f.refs);
checks++;
const empty = structuredClone(f.secondary);
empty.populations.pairs = structuredClone(empty.populations.deflected);
empty.source_artifacts.pairs = {
    ...empty.source_artifacts.pairs,
    bytes: 0
};
empty.record_count_verification.expected_records = 0;
empty.record_count_verification.actual_records = 0;
kin.validate(empty, [f.refs[0], empty.source_artifacts.pairs]);
checks++;
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Synthetic formula, admission, provenance and integer joint/marginal closure tests; no browser or physics validation'
}));
