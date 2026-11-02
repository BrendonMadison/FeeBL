const assert = require('assert/strict'),
    fs = require('fs'),
    path = require('path'),
    B = require('./collider_beta_gaussian.js'),
    M = require('./collider_monochromatization.js');
const viewerText = ['feebl_gui.html', 'collider_viewer.js', 'collider_viewer_science.js'].map(name => fs.readFileSync(path.join(__dirname, name), 'utf8')).join('\n');
const cases = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_beta_gaussian.json'), 'utf8'));
let checks = 0;

function test(fn) {
    fn();
    checks++
}

function validate(f) {
    return M.validate(f.mono, f.refs, {
        manifest: f.manifest,
        analysis: f.analysis,
        diagnostics: f.diagnostics
    })
}
test(() => assert(viewerText.includes('Peak selection and response model')));
test(() => assert(viewerText.includes('spectrum-model shape analysis below')));
for (const f of Object.values(cases)) test(() => validate(f));

function bad(fn, key = 'beta_admitted') {
    const f = structuredClone(cases[key]);
    fn(f, f.mono.peak_fit.total);
    test(() => assert.throws(() => validate(f)))
}
bad(f => f.mono.peak_fit.source_reference = f.mono.source_artifacts.native_summary);
bad(f => f.mono.peak_fit.fit_source.histogram = 'peak.luminosity_spectrum');
bad((f, r) => r.selected_luminosity_weight *= 2);
bad((f, r) => r.candidate.curve.source_bin_stop--);
bad((f, r) => r.candidate.curve.tail_bin_probabilities[0] = -1);
bad((f, r) => {
    const a = r.candidate.curve.tail_bin_probabilities;
    const i = a.indexOf(Math.max(...a));
    a[i] *= .5
});
bad((f, r) => r.candidate.peak_curve.source_bin_stop--);
bad((f, r) => r.candidate.peak_curve.tail_bin_probabilities[0] = Infinity);
bad((f, r) => r.candidate.sigma_fractional *= 2);
bad((f, r) => r.candidate.alpha = 0);
bad((f, r) => r.candidate.peak_fraction = .99);
bad((f, r) => r.candidate.model_area_native *= 2);
bad((f, r) => r.candidate.input_reference_ratio *= 2);
bad((f, r) => r.relative_l2_residual = .8);
bad((f, r) => r.quality.quadrature_relative_l2_error = 1);
bad((f, r) => r.quality.near_optimal_sigma_spread = 1);
bad((f, r) => r.quality.jacobian_condition_number = 1e11);
bad((f, r) => r.quality.sigma_profile_probes[0].objective_increase = 0);
bad((f, r) => r.solver.parameter_position[0] = 0);
bad((f, r) => r.solver.parameter_bounds[0][0] += 1);
bad((f, r) => r.candidate.curve.tail_bin_probabilities = [], 'gaussian_limit');
bad((f, r) => r.candidate.alpha = 1, 'gaussian_limit');
bad((f, r) => {
    r.candidate_display_allowed = true;
    r.candidate_status = 'admitted';
    r.available = true
}, 'sparse');
// Independently known standard-normal bin probabilities and nonnegative profile.
const bins = [{
    low: -10,
    high: 0,
    weight: 5
}, {
    low: 0,
    high: 10,
    weight: 5
}];
test(() => B.gaussianBins(bins, 0, 1).forEach(p => assert(Math.abs(p - .5) < 2e-14)));
test(() => {
    const p = B.profile(bins, [.5, .5]);
    assert(Math.abs(p.area - 1) < 1e-14);
    assert.equal(p.fraction, 1);
    assert(p.residual < 1e-14)
});
test(() => {
    const p = B.profile([{
        low: 0,
        high: 1,
        weight: 1
    }, {
        low: 1,
        high: 2,
        weight: 3
    }], [1, 0], [0, 1]);
    assert(Math.abs(p.fraction - .25) < 1e-14);
    assert(Math.abs(p.area - 1) < 1e-14);
    assert(p.residual < 1e-14)
});
test(() => {
    const p = B.profile([{
        low: 0,
        high: 1,
        weight: 0
    }, {
        low: 1,
        high: 2,
        weight: 2
    }], [1, 0], [.5, .5]);
    assert.equal(p.coefficients[0], 0);
    assert(p.coefficients[1] >= 0)
});
const harness = new Function('require', '__dirname', fs.readFileSync(path.join(__dirname,
    'test_collider_contract.js'), 'utf8') + '\nreturn {sandbox,nodes,vm};')(require, __dirname);
const {
    sandbox,
    nodes,
    vm
} = harness;
sandbox.ColliderBetaGaussian = B;
sandbox.betaCases = cases;
for (const [name, f] of Object.entries(cases)) {
    sandbox.betaFixture = f;
    vm.runInContext(
        'man=betaFixture.manifest;scientific=betaFixture.analysis;diagnostics=betaFixture.diagnostics;monochromatization=ColliderMonochromatization.validate(betaFixture.mono,betaFixture.refs,{manifest:man,analysis:scientific,diagnostics});scientific=derivedSpectrum(diagnostics.total.luminosity_spectrum,"total");$("fr").value="0";$("diagScope").value="total";$("spectrumScope").value="total";$("spectrumUnits").value="native";drawDiagnostics(0);drawSpectra()',
        sandbox);
    const beta = name !== 'legacy_gaussian',
        rows = new Map(nodes.get('diag').children.map(row => row.children.map(c => c.textContent)));
    test(() => assert(rows.has(beta ? 'Beta–Gaussian core σ [GeV]' :
        'Gaussian σ estimate (shape mismatch) [GeV]')));
    test(() => assert(rows.has('Total full-spectrum RMS σW [GeV]')));
    test(() => assert(nodes.get('spectrumPeakFitNote').textContent.includes(beta ? 'Beta–Gaussian' :
        'Gaussian')));
    if (name === 'sparse') {
        test(() => assert.equal(vm.runInContext('peakCurve("total",0,scientific,"native")',
            sandbox), null));
        continue
    }
    if (beta) {
        test(() => assert(rows.has('Model endpoint E₀ [GeV]')));
        test(() => assert(rows.has('Core-width ratio to input [1]')));
        test(() => assert(nodes.get('diagPeakDetails').textContent.includes(
            'not the mixture mean')));
        test(() => assert(nodes.get('spectrumPeakFitNote').textContent.includes(
            'fitted to the full spectrum')));
        const curve = vm.runInContext('peakCurve("total",0,scientific,"native")', sandbox),
            h = f.diagnostics.total.peak.luminosity_spectrum;
        test(() => assert.equal(curve.bins.length, h.bins.length));
        test(() => curve.bins.forEach((b, i) => {
            assert.equal(b.low, h.bins[i].low);
            assert.equal(b.high, h.bins[i].high);
            assert(Number.isFinite(b.density) && b.density >= 0)
        }));
        const area = curve.bins.reduce((s, b) => s + b.density * (b.high - b.low), 0);
        test(() => assert(area > 0 && area <= f.mono.peak_fit.total.candidate.model_area_native * (
            1 + 1e-6)));
        nodes.get('diagScope').value = 'frame';
        nodes.get('spectrumScope').value = 'frame';
        vm.runInContext('drawDiagnostics(0);drawSpectra()', sandbox);
        test(() => assert(nodes.get('diagPeakNote').textContent.includes('Beta–Gaussian')));
    }
}
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    scope: 'Beta/delta-convolution fixtures, independent Gaussian integrals and nonnegative amplitude profiling, corruption rejection, core diagnostics and exact-bin overlay; legacy Gaussian retained'
}));
