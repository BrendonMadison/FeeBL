const assert = require('assert/strict'),
    fs = require('fs'),
    path = require('path'),
    C = require('./collider_event_deflections.js');
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_event_deflections.json'), 'utf8'));
let checks = 0;

function test(fn) {
    fn();
    checks++
}

function bad(fn) {
    const f = structuredClone(fixture);
    fn(f);
    test(() => assert.throws(() => C.validate(f.data, f.refs, f.context)))
}
test(() => assert.equal(C.validate(fixture.data, fixture.refs, fixture.context), fixture.data));
test(() => assert.equal(fixture.data.plots.length, 4));
test(() => assert.equal(fixture.data.frames[0].nrecords, 0));
test(() => assert.equal(fixture.data.total.event_counts.complete, 2));
bad(f => f.data.source_artifacts.input = {
    ...f.data.source_artifacts.summary
});
bad(f => f.context.manifest.imported_trajectories_file = 'other.json');
bad(f => f.context.manifest.native_beam_angles_x_rad[0] = .007);
bad(f => f.context.identity.events[0].particles[0].pdg_id = -11);
bad(f => f.context.identity.events[1].particles[0].source_record = 1);
bad(f => f.context.identity.events[1].particles[0].event_slot = 'other');
bad(f => f.context.identity.events[1].particles[0].role = 'not the same slot role');
bad(f => f.context.identity.source_artifact.sha256 = '0'.repeat(64));
bad(f => f.context.trajectories.evaluated_flags.do_pairs = 1);
bad(f => f.context.summaryText = f.context.summaryText.replace('charge_sign=-1', 'charge_sign=0')
    .replace('charge_sign = -1', 'charge_sign = 0'));
bad(f => f.context.trajectories.initial_records[0][8] += .0001);
bad(f => f.context.trajectories.frames[1].records.push(f.context.trajectories.frames[1].records[
0]));
bad(f => f.context.trajectories.frames[2].records[0][0] = 999);
bad(f => f.context.trajectories.final.records[0][1] += .0001);
bad(f => f.data.total.event_counts.complete--);
bad(f => f.data.total.field_gathers.main_mesh++);
bad(f => f.data.total.histograms.electron_delta_theta.statistics.mean += .1);
bad(f => f.data.frames[0].histograms.electron_delta_phi.statistics = {
    mean: 0
});
bad(f => f.data.total.histograms.electron_delta_phi.excluded_undefined_azimuth++);
bad(f => f.data.total.histograms.positron_delta_theta.bins[0].weight++);
bad(f => f.data.plots[0].label = 'positron Δθ');
bad(f => f.data.plots[0].edges[0] -= .1);
bad(f => f.data.frames[2].elapsed_s += 1);
test(() => {
    const f = structuredClone(fixture);
    for (const frame of f.context.trajectories.frames) frame.records.reverse();
    f.context.trajectories.final.records.reverse();
    C.validate(f.data, f.refs, f.context)
});
test(() => {
    const initial = [1, 1, 0, 0, 125, 0, 0, 0, 0, 0, 1],
        current = [1, 0, 0, 1, 0, 0, 0];
    assert.equal(C.angles(initial, current).delta_phi, null);
    assert.equal(C.angles(initial, current).delta_theta, 0)
});
test(() => {
    const d = Math.PI / 180,
        initial = [1, 1, 0, 0, 125, 0, 0, 0, Math.cos(179 * d), Math.sin(179 * d), 0],
        current = [1, Math.cos(-179 * d), Math.sin(-179 * d), 0, 0, 0, 0];
    assert(Math.abs(C.angles(initial, current).delta_phi - 2) < 1e-12)
});
test(() => {
    const h = C.histogramValues([-1, 0, 1], [-1, 0, 1]);
    assert.deepEqual(h.counts, [1, 2]);
    assert.equal(h.overflow, 0)
});
// Exercise the production HTML renderer through the existing finite-coordinate DOM harness.
const harness = new Function('require', '__dirname', fs.readFileSync(path.join(__dirname,
    'test_collider_contract.js'), 'utf8') + '\nreturn {sandbox,nodes,vm,canvasTexts};')(require,
    __dirname);
const {
    sandbox,
    nodes,
    vm
} = harness;
sandbox.ColliderEventDeflections = C;
sandbox.eventFixture = fixture;
vm.runInContext(
    'eventDeflections=ColliderEventDeflections.validate(eventFixture.data,eventFixture.refs,eventFixture.context);initializeSecondaryControls();$("secondaryPopulation").value="deflected";$("secondaryScope").value="total";drawSecondaryKinematics()',
    sandbox);
test(() => assert.equal(nodes.get('eventDeflectionPlots').hidden, false));
test(() => assert.equal(nodes.get('eventDeflectionPlots').children.length, 4));
test(() => assert(nodes.get('secondaryHeatmapPlot').hidden && nodes.get('secondaryZControl')
    .hidden && nodes.get('secondaryAngleControl').hidden && nodes.get(
        'secondaryColorControl').hidden));
test(() => assert(nodes.get('secondaryContext').textContent.includes(
    '2 complete / 2 expected events')));
test(() => assert(nodes.get('secondaryContext').textContent.includes('not a sum of frames')));
const getStats = id => vm.runInContext('eventPlotNodes.get(' + JSON.stringify(id) +
    ').stats.attributes["aria-label"]', sandbox);
const total = getStats('electron_delta_theta');
test(() => assert(total.includes('Mean (records)')));
test(() => assert(total.includes('Variance [deg²]')));
for (const k of [0, 1, 2]) {
    nodes.get('secondaryScope').value = 'frame';
    nodes.get('fr').value = String(k);
    vm.runInContext('drawSecondaryKinematics()', sandbox);
    test(() => assert(!nodes.get('eventDeflectionPlots').hidden));
    test(() => assert(getStats('electron_delta_theta').includes('Snapshot ' + k)));
    test(() => assert(nodes.get('secondaryContext').textContent.includes(fixture.data.frames[k]
        .nrecords + ' identified records')));
    if (k === 0) test(() => assert(getStats('electron_delta_theta').includes(
        'statistics unavailable')));
}
nodes.get('secondaryScope').value = 'total';
nodes.get('fr').value = '0';
vm.runInContext('drawSecondaryKinematics()', sandbox);
test(() => assert.equal(getStats('electron_delta_theta'), total));
nodes.get('fr').value = '2';
vm.runInContext('drawSecondaryKinematics()', sandbox);
test(() => assert.equal(getStats('electron_delta_theta'), total));
vm.runInContext('eventDeflections=null;drawSecondaryKinematics()', sandbox);
test(() => assert(nodes.get('eventDeflectionPlots').hidden));
test(() => assert(nodes.get('secondaryUnavailable').textContent.includes(
    'No authenticated event identities')));
nodes.get('secondaryPopulation').value = 'pairs';
vm.runInContext('drawSecondaryKinematics()', sandbox);
test(() => assert(!nodes.get('secondaryZControl').hidden && !nodes.get('secondaryAngleControl')
    .hidden));
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Explicit identity/raw input joins, signed angular histograms, counts/moments, fixed axes, current/final controls, empty/legacy behavior; synthetic fixtures only'
}));
