"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path');
const c = require('./collider_contract.js'),
    mono = require('./collider_monochromatization.js');
const f = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_peak_diagnostics.json'), 'utf8'));
let checks = 0;
const copy = x => structuredClone(x),
    reject = fn => {
        assert.throws(fn);
        checks++
    };
for (const [name, b] of Object.entries(f.fit)) {
    c.peakDiagnostics(b.diagnostics, b.manifest);
    c.bindPeakReference(b.diagnostics, b.manifest, b.analysis);
    mono.validate(b.mono, b.refs, {
        manifest: b.manifest,
        analysis: b.analysis,
        diagnostics: b.diagnostics
    });
    checks += 3;
    assert.equal(b.mono.peak_fit.total.candidate_status, {
        admitted: 'admitted',
        shape_mismatch: 'resolved_shape_mismatch',
        sparse: 'unavailable'
    } [name]);
    checks++;
}
for (const [name, p] of Object.entries(f.native)) {
    const d = {
        frames: [{
            peak: p,
            luminosity_spectrum: {
                total_weight: p.weight_sum
            }
        }],
        total: {
            peak: copy(p),
            luminosity_spectrum: {
                total_weight: p.weight_sum
            }
        }
    };
    c.peakDiagnostics(d, {
        energy_GeV: [125, 125]
    });
    checks++;
    if (name === 'populated') {
        assert(p.luminosity_spectrum.overflow > 0);
        checks++;
    }
    if (name === 'partial_spin') {
        assert.equal(p.polarization, null);
        assert(p.spin_contributions < p.contributions);
        checks += 2;
    }
}
const b = f.fit.admitted;

function invalidPeak(mutate) {
    const d = copy(b.diagnostics);
    mutate(d);
    reject(() => c.peakDiagnostics(d, b.manifest))
}
invalidPeak(d => delete d.frames[0].peak);
invalidPeak(d => d.frames[0].peak.selection.minimum_W_GeV += .1);
invalidPeak(d => {
    for (const r of [...d.frames, d.total]) r.peak.selection.nominal_reference
        .angle_source = 'evaluated crab_full_crossing_angle matched to beam-angle receipt'
});
invalidPeak(d => {
    for (const r of [...d.frames, d.total]) r.peak.luminosity_spectrum.bins[0].low += 1e-3
});
invalidPeak(d => {
    for (const r of [...d.frames, d.total]) r.peak.luminosity_spectrum.bins[20].high += 1e-3
});
invalidPeak(d => d.total.peak.contributions++);
invalidPeak(d => d.total.peak.beam_energy_GeV.beam1.mean += 1);
invalidPeak(d => d.total.peak.W_GeV.rms += 1);
invalidPeak(d => d.total.peak.polarization.beam1 = 2);
invalidPeak(d => {
    for (const r of [...d.frames, d.total]) r.peak.spin_contributions--;
});
invalidPeak(d => {
    for (const r of [...d.frames, d.total]) r.peak.polarization = null;
});
invalidPeak(d => d.total.peak.spin_weight_sum *= 2);
for (const patch of [a => a.definition.nominal_W_GeV++, a => a.definition.full_crossing_angle_rad =
        .014, a => a.definition.angle_receipt_matches_design = false, a => a.definition
        .design_W = 'head-on approximation'
    ]) {
    const a = copy(b.analysis);
    patch(a);
    reject(() => c.bindPeakReference(b.diagnostics, b.manifest, a));
}
const wrongManifest = copy(b.manifest);
wrongManifest.native_beam_angles_x_rad = [.007, .007];
reject(() => c.bindPeakReference(b.diagnostics, wrongManifest, b.analysis));
const old = copy(b.diagnostics);
for (const r of [...old.frames, old.total]) delete r.peak;
c.peakDiagnostics(old, b.manifest);
c.bindPeakReference(old, b.manifest, null);
checks++;

function invalidFit(mutate, which = 'admitted') {
    const v = copy(f.fit[which].mono.peak_fit),
        context = f.fit[which];
    mutate(v);
    reject(() => mono.validatePeakFit(v, context.mono.measured.reference_sigma_W_GeV, context
        .diagnostics, context.mono.source_artifacts.diagnostics))
}
for (const mutate of [v => v.rules.maximum_relative_l2_residual = .9, v => v.source_reference.path =
        'out.dat', v => v.total.selected_luminosity_weight *= 2,
        v => v.total.fit_lower_W_GeV++, v => v.total.selected_bins--, v => v.total
        .relative_l2_residual = .1, v => v.total.candidate.gaussian_area_native *= 1.1,
        v => v.total.candidate.input_reference_ratio *= 2, v => v.total.candidate.sigma_W_GeV *=
        2, v => v.total.solver.parameter_position[0] = 0,
        v => v.total.solver.converged = false, v => v.total.quality.positive_bins_left = 0, v =>
        v.frames.pop(),
        v => v.total.candidate_status = 'resolved_shape_mismatch', v => v.total.mean_W_GeV++
    ]) invalidFit(mutate);
invalidFit(v => {
    v.total.available = true;
    v.total.candidate_status = 'admitted'
}, 'shape_mismatch');
invalidFit(v => {
    v.total.candidate_display_allowed = true;
    v.total.candidate_status = 'resolved_shape_mismatch';
    v.total.candidate = copy(b.mono.peak_fit.total.candidate)
}, 'sparse');
invalidFit(v => v.verification_tolerance.residual_absolute = .1);
const removed = copy(b.mono);
delete removed.peak_fit;
reject(() => mono.validate(removed, b.refs, {
    manifest: b.manifest,
    analysis: b.analysis,
    diagnostics: b.diagnostics
}));
// Contract fixtures prove producer schema; the DOM harness checks product labels,
// scope routing and normalization. It does not replace actual browser review.
const base = fs.readFileSync(path.join(__dirname, 'test_collider_contract.js'), 'utf8');
const harness = new Function('require', '__dirname', base +
    '\nreturn {sandbox,nodes,vm,canvasTexts};')(require, __dirname);
const {
    sandbox,
    nodes,
    vm
} = harness;
sandbox.peakBundle = f.fit;
sandbox.ColliderMonochromatization = mono;
vm.runInContext(
    'man=peakBundle.admitted.manifest;diagnostics=peakBundle.admitted.diagnostics;scientific=peakBundle.admitted.analysis;scientific=derivedSpectrum(diagnostics.total.peak.luminosity_spectrum,"total");monochromatization=peakBundle.admitted.mono;',
    sandbox);
nodes.get('fr').value = '0';
nodes.get('diagScope').value = 'total';
nodes.get('spectrumScope').value = 'total';
nodes.get('spectrumUnits').value = 'native';
vm.runInContext('drawDiagnostics(0);drawSpectra()', sandbox);
checks++;
const rows = () => new Map(nodes.get('diag').children.map(r => r.children.map(v => v.textContent)));
for (const name of ['Peak ⟨E₁⟩L [GeV]', 'Peak ⟨E₂⟩L [GeV]', 'Peak ⟨P₁⟩L / ⟨P₂⟩L [1]',
        'Selected-peak RMS σW [GeV]', 'Gaussian peak σW [GeV]',
        'Gaussian peak width ratio to input [1]', 'Total full-spectrum RMS σW [GeV]'
    ]) {
    assert(rows().has(name));
    assert(!rows().get(name).includes('unavailable'));
    checks += 2;
}
assert(nodes.get('diagPeakNote').textContent.includes('no upper cut'));
assert(nodes.get('diagPeakDetails').textContent.includes('not a Gaussian fit'));
checks += 2;
assert(nodes.get('spectrumPeakResolution').textContent.includes(
    'Exact 98%-cut native peak histogram'));
checks++;
assert.equal(vm.runInContext('peakPlotSpectrum(scientific,"total").bins.length', sandbox), 512);
checks++;
assert.equal(vm.runInContext('peakPlotSpectrum(scientific,"total").plot.peak.x_min_GeV', sandbox), b
    .diagnostics.total.peak.selection.minimum_W_GeV);
checks++;
assert.equal(vm.runInContext('peakPlotSpectrum(scientific,"total").bins[0].weight_m2_per_crossing',
    sandbox), b.diagnostics.total.peak.luminosity_spectrum.bins[0].weight);
checks++;
const valueNative = vm.runInContext(
    'peakCurve("total",0,scientific,"native").at(monochromatization.peak_fit.total.candidate.mean_W_GeV)',
    sandbox);
const valueShape = vm.runInContext(
    'peakCurve("total",0,scientific,"shape").at(monochromatization.peak_fit.total.candidate.mean_W_GeV)',
    sandbox);
assert(Math.abs(valueNative / valueShape - b.diagnostics.total.peak.weight_sum) <= 1e-9 * b
    .diagnostics.total.peak.weight_sum);
checks++;
vm.runInContext('scientific.normalization.crossing_frequency_hz=100;drawSpectra()', sandbox);
const valueSecond = vm.runInContext(
    'peakCurve("total",0,scientific,"persecond").at(monochromatization.peak_fit.total.candidate.mean_W_GeV)',
    sandbox);
assert(Math.abs(valueSecond / valueNative - .01) < 1e-12);
checks++;
vm.runInContext(
    'monochromatization=structuredClone(peakBundle.admitted.mono);monochromatization.peak_fit.frames[0]=structuredClone(peakBundle.shape_mismatch.mono.peak_fit.total)',
    sandbox);
nodes.get('diagScope').value = 'frame';
vm.runInContext('drawDiagnostics(0);drawSpectra()', sandbox);
assert(rows().has('Gaussian σ estimate (shape mismatch) [GeV]'));
assert(rows().has('Gaussian-shape width ratio (estimate) [1]'));
checks += 2;
assert(nodes.get('diagPeakNote').textContent.includes('70.71%'));
assert(nodes.get('spectrumPeakFitNote').textContent.includes('Gaussian peak fit'));
checks += 2;
nodes.get('spectrumScope').value = 'frame';
vm.runInContext('drawSpectra()', sandbox);
assert(nodes.get('spectrumPeakFitNote').textContent.includes(
    'Gaussian shape estimate — tail mismatch'));
assert(nodes.get('spectrumPeakFitNote').textContent.includes('70.71%'));
checks += 2;
vm.runInContext('monochromatization=peakBundle.sparse.mono;drawDiagnostics(0);drawSpectra()',
    sandbox);
assert.equal(rows().get('Gaussian peak σW [GeV]'), 'unavailable');
assert.equal(vm.runInContext('peakCurve("frame",0,scientific,"native")', sandbox), null);
checks += 2;
vm.runInContext(
    'monochromatization=null;for(const r of [...diagnostics.frames,diagnostics.total])delete r.peak;drawDiagnostics(0);drawSpectra()',
    sandbox);
assert(nodes.get('diagPeakNote').textContent.includes('No native peak selection was exported'));
checks++;
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Native peak serializer forms, Gaussian fixture binding/corruptions, qualified candidate display and normalization; no browser QA'
}));
