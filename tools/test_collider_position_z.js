"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path');
const kin = require('./collider_secondary_kinematics.js'),
    native = JSON.parse(fs.readFileSync(path.join(__dirname,
        'testdata/collider_position_z_native.json'), 'utf8'));
const old = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_secondary_frames.json'), 'utf8'));
let checks = 0;
const clone = x => structuredClone(x),
    template = old.secondary.frames.find(f => f.available && f.populations.pairs.nrecords)
    .populations.pairs;

function population(frame, weighting = kin.WEIGHTING) {
    const p = clone(template);
    p.nrecords = frame.nrecords;
    p.position_z = clone(frame.position_z);
    p.normalization = {
        ...p.normalization,
        weighting
    };
    p.selection = {
        ...p.selection,
        ...frame.selection
    };
    for (const name of ['joint', 'joint_folded']) p[name] = {
        ...p[name],
        ...clone(frame[name]),
        theta_edges_rad: clone(native.axes[name === 'joint' ? 'theta_edges_rad' :
            'theta_folded_edges_rad']),
        pT_edges_GeV: clone(native.axes.pT_edges_GeV)
    };
    for (const name of ['theta', 'theta_folded', 'pT']) {
        const h = frame.marginals[name],
            edges = native.axes['marginal_' + (name === 'pT' ? 'pT_edges_GeV' : name +
                '_edges_rad')];
        p.distributions[name] = {
            ...p.distributions[name],
            ...clone(h),
            weighting,
            bins: h.counts.map((weight, i) => ({
                low: edges[i],
                high: edges[i + 1],
                weight
            }))
        }
    }
    return p;
}
const data = () => ({
    position_z_definition: clone(native.position_z_definition),
    populations: {
        pairs: population(native.frames[0])
    },
    frames: []
});
for (const frame of native.frames) {
    const a = data();
    a.populations.pairs = population(frame);
    kin.validateZSelections(a, {});
    checks++;
    for (const key of ['all', 'positive', 'negative']) {
        const p = kin.selectZ(a.populations.pairs, key);
        assert.equal(p.nrecords, key === 'all' ? frame.nrecords : frame.position_z[key].nrecords);
        checks++;
        assert(p.available);
        checks++;
        if (key !== 'all') {
            assert.equal(p.selection.admitted_tracker_count, undefined);
            assert.equal(p.selection.authoritative_reserve_count, undefined);
            checks += 2;
        }
    }
}
const first = population(native.frames[0]),
    later = population(native.frames[1]);
assert.equal(kin.selectZ(first, 'positive').nrecords, 3);
assert.equal(kin.selectZ(first, 'negative').nrecords, 2);
assert.equal(first.position_z.zero_records, 2);
checks += 3;
assert.equal(kin.selectZ(later, 'positive').nrecords, 2);
assert.equal(kin.selectZ(later, 'negative').nrecords, 3);
checks += 2;
// Native fixture includes positive-z particles moving toward -z and sign migration
// without a momentum change. Selection follows stored positions, not directions.
assert(kin.selectZ(first, 'positive').distributions.theta.statistics.max > Math.PI / 2);
checks++;
const original = clone(first);
kin.selectZ(first, 'positive');
kin.selectZ(first, 'negative');
assert.deepEqual(first, original);
checks++;

function invalid(fn) {
    const a = data();
    fn(a);
    assert.throws(() => kin.validateZSelections(a, {}));
    checks++
}
for (const fn of [a => a.position_z_definition.coordinate = 'longitudinal momentum', a => a
        .position_z_definition.direction_not_used = false, a => delete a.position_z_definition,
        a => a.populations.pairs.position_z.zero_records++, a => a.populations.pairs.position_z
        .positive.nrecords++,
        a => a.populations.pairs.position_z.positive.joint.counts_sparse.push(a.populations
            .pairs.position_z.positive.joint.counts_sparse[0]),
        a => a.populations.pairs.position_z.positive.joint.counts_sparse[0][0] = -1, a => a
        .populations.pairs.position_z.positive.joint.counts_sparse[0][0] = 16384,
        a => a.populations.pairs.position_z.positive.joint.counts_sparse[0][1] = 0, a => a
        .populations.pairs.position_z.positive.joint.counts_sparse[0][1] = .5,
        a => a.populations.pairs.position_z.positive.joint.counts = [1], a => a.populations
        .pairs.position_z.positive.joint.theta_edges_rad = [1, 2],
        a => a.populations.pairs.position_z.positive.marginals.theta.counts[0]++, a => a
        .populations.pairs.position_z.positive.marginals.theta.statistics.mean = Infinity,
        a => a.populations.pairs.position_z.positive.marginals.theta.statistics.max = 10, a => a
        .populations.pairs.position_z.positive.selection.admitted_tracker_count = 3,
        a => a.populations.pairs.position_z.positive.selection.process_counts['99'] = 1, a => a
        .populations.pairs.position_z.positive.selection.signed_energy_counts.positive_energy++
    ]) invalid(fn);
invalid(a => {
    const h = a.populations.pairs.position_z.positive.joint;
    h.excluded.theta_nonpositive++;
    h.excluded.nonpositive_records++;
    h.in_range_records--;
    h.counts_sparse[0][1]--;
    if (!h.counts_sparse[0][1]) h.counts_sparse.shift()
});
const missing = clone(first);
delete missing.position_z;
assert.equal(kin.selectZ(missing, 'all'), missing);
assert.equal(kin.selectZ(missing, 'positive').available, false);
checks += 2;
assert.throws(() => kin.selectZ(first, 'momentum-positive'));
checks++;
const adapters = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_position_z_adapters.json'), 'utf8'));
for (const f of Object.values(adapters)) {
    kin.validate(f.data, f.refs, {
        manifest: f.manifest,
        frameSnapshots: f.raw
    });
    checks++;
}
const adapter = adapters.native;

function badAdapter(change) {
    const f = clone(adapter);
    change(f);
    assert.throws(() => kin.validate(f.data, f.refs, {
        manifest: f.manifest,
        frameSnapshots: f.raw
    }));
    checks++;
}
for (const change of [f => delete f.data.position_z_definition, f => delete f.data.frame_definition
        .position_z_definition,
        f => delete f.data.frames[0].populations.pairs.position_z, f => f.data.frames[0]
        .populations.pairs.position_z = null,
        f => f.raw.frames[0].position_z.positive.joint.counts_sparse[0][1]++, f => f.raw.frames[
            0].position_z.positive.marginals.theta.statistics.mean += .1,
        f => f.raw.position_z_definition.direction_not_used = false, f => delete f.raw
        .position_z_definition
    ]) badAdapter(change);
const legacy = adapters.legacy_frames,
    legacyPop = Object.values(legacy.data.frames[0].populations).find(p => p.available);
assert.equal(kin.selectZ(legacyPop, 'positive').available, false);
checks++;
const finalPop = Object.values(legacy.data.populations).find(p => p.available);
assert(kin.selectZ(finalPop, 'positive').available);
checks++;
// Fixed bounds remain those of All for every selection and snapshot.
const view = {
    populations: {
        pairs: first
    },
    frames: [{
        populations: {
            pairs: later
        }
    }, {
        populations: {
            pairs: population(native.frames[2])
        }
    }]
};
for (const folded of [false, true]) {
    const bounds = kin.axisBounds(view, 'pairs', folded);
    for (const z of ['all', 'positive', 'negative']) {
        kin.selectZ(first, z);
        assert.deepEqual(kin.axisBounds(view, 'pairs', folded), bounds);
        checks++;
    }
}
// Existing DOM harness also checks legacy datasets and all earlier UI features.
const base = fs.readFileSync(path.join(__dirname, 'test_collider_contract.js'), 'utf8');
const harness = new Function('require', '__dirname', base + '\nreturn {sandbox,nodes,vm};')(require,
    __dirname);
const {
    sandbox,
    nodes,
    vm
} = harness;
sandbox.zData = {
    position_z_definition: clone(native.position_z_definition),
    populations: {
        pairs: first
    },
    frames: native.frames.map((f, id) => ({
        id,
        diagonal: id,
        available: true,
        populations: {
            pairs: population(f, kin.FRAME_WEIGHTING)
        }
    }))
};
sandbox.ColliderSecondaryKinematics = kin;
vm.runInContext(
    'secondaryKinematics=zData;man={diagnostic_analysis_sources:[]};initializeSecondaryControls()',
    sandbox);
assert.equal(nodes.get('secondaryZ').value, 'all');
assert.equal(nodes.get('secondaryZ').disabled, false);
checks += 2;
const spoken = () => nodes.get('secondaryHeatmapStats').attributes['aria-label'];
for (const scope of ['total', 'frame'])
    for (const z of ['all', 'positive', 'negative'])
        for (const angle of ['global', 'folded']) {
            nodes.get('secondaryScope').value = scope;
            nodes.get('secondaryZ').value = z;
            nodes.get('secondaryAngle').value = angle;
            nodes.get('secondaryPopulation').value = 'pairs';
            nodes.get('fr').value = '0';
            vm.runInContext('drawSecondaryKinematics()', sandbox);
            assert.equal(nodes.get('secondaryHeatmapPlot').hidden, false);
            checks++;
            const expected = kin.selectZ(first, z);
            assert(nodes.get('secondaryContext').textContent.includes(' / ' + expected.nrecords +
                ' records in the log plot'));
            checks++;
            assert(nodes.get('secondaryContext').textContent.includes(
                'not longitudinal momentum or a detector arm'));
            checks++;
            if (z !== 'all') {
                assert(spoken().includes(z === 'positive' ? 'z > 0' : 'z < 0'));
                checks++;
            }
            sandbox.zExpected = expected.distributions[angle === 'folded' ? 'theta_folded' :
                'theta'].statistics.mean;
            assert(spoken().includes(vm.runInContext('fmt(zExpected,4)', sandbox)));
            checks++;
        }
nodes.get('secondaryScope').value = 'frame';
nodes.get('secondaryZ').value = 'positive';
nodes.get('fr').value = '1';
vm.runInContext('drawSecondaryKinematics()', sandbox);
assert(nodes.get('secondaryContext').textContent.includes(' / 2 records in the log plot'));
checks++;
nodes.get('fr').value = '2';
vm.runInContext('drawSecondaryKinematics()', sandbox);
assert.equal(nodes.get('secondaryHeatmapPlot').hidden, false);
assert(spoken().includes('Mean (records): unavailable: unavailable'));
checks += 2;
vm.runInContext('delete secondaryKinematics.frames[0].populations.pairs.position_z;', sandbox);
nodes.get('fr').value = '0';
vm.runInContext('drawSecondaryKinematics()', sandbox);
assert(nodes.get('secondaryHeatmapPlot').hidden);
assert(nodes.get('secondaryUnavailable').textContent.includes('All remains available'));
checks += 2;
nodes.get('secondaryZ').value = 'all';
vm.runInContext('drawSecondaryKinematics()', sandbox);
assert.equal(nodes.get('secondaryHeatmapPlot').hidden, false);
checks++;
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Actual native z-position serializer fixture; sparse/count/geometry validation, zero records, direction distinction, migration, legacy fallback and selected-scope DOM rendering; no browser QA'
}));
