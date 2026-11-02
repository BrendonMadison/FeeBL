"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"),
    path = require("node:path"),
    vm = require("node:vm");
const c = require("./collider_contract.js");
const a = {
    path: "gui_b1.0",
    sha256: "a".repeat(64),
    rows: 1,
    columns: 6,
    bytes: 12,
    bounds: [
        [125, 125],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
};
const m = {
    schema_version: "feebl.collider/2",
    units: {
        energy: "GeV",
        position: "um",
        angle: "urad",
        time: "s"
    },
    coordinates: {
        frame: "legacy_collider_dump"
    },
    energy_GeV: [125, 125],
    validation: {
        status: "UNVALIDATED"
    },
    backend_actual_receipt: "serial",
    source: {
        commit: "a".repeat(40)
    },
    frames: [{
        id: 0,
        elapsed_s: 0,
        beam1: a,
        beam2: {
            ...a,
            path: "gui_b2.0"
        }
    }]
};
let n = 0;

function reject(fn) {
    assert.throws(fn);
    n++
}
c.manifest(m);
n++;
assert.equal(c.rows("125 0 0 0 0 0\n", a).length, 1);
n++;
for (const name of ["../a", "https://example.org", "..", "a/b", "a\\b", ""]) reject(() => c
    .safeName(name));
reject(() => c.rows("125 0", a));
reject(() => c.rows("125 NaN 0 0 0 0", a));
reject(() => c.rows("125 0 0 0 0 0\n125 0 0 0 0 0", a));
reject(() => c.rows("126 0 0 0 0 0", a));
for (const patch of [{
        schema_version: "old"
    }, {
        units: {
            energy: "MeV"
        }
    }, {
        energy_GeV: [-1, 2]
    }, {
        frames: []
    }])
    reject(() => c.manifest({
        ...m,
        ...patch
    }));
reject(() => c.manifest({
    ...m,
    frames: [{
        ...m.frames[0],
        beam2: a
    }]
}));
const html = fs.readFileSync(path.join(__dirname, "feebl_gui.html"), "utf8");
const js = ["collider_viewer.js", "collider_viewer_science.js", "collider_viewer_fields.js"].map(name => fs.readFileSync(path.join(__dirname, name), "utf8")).join("\n");
new vm.Script(js, {
    filename: "feebl_gui.html"
});
n++;
const sampleAnalysis = {
    schema_version: "feebl.collider.analysis/1",
    observable: "e+e- luminosity spectrum",
    definition: {
        nominal_W_GeV: 250
    },
    plot: {
        crop_renormalization: false,
        full: {
            y_scale: "log"
        },
        peak: {
            y_scale: "linear",
            x_min_GeV: 245
        }
    },
    normalization: {
        underflow_m2_per_crossing: 0,
        overflow_m2_per_crossing: 0,
        total_spectrum_m2_per_crossing: 2
    },
    bins: [{
        W_low_GeV: 249,
        W_high_GeV: 250,
        dL_dW_m2_per_crossing_GeV: 2,
        weight_m2_per_crossing: 2,
        uncertainty: null,
        dL_dW_cm2_s_GeV: null,
        shape_density_per_GeV: 1
    }]
};
c.analysis(sampleAnalysis);
n++;
c.analysis({
    ...sampleAnalysis,
    native_statistics: null
});
n++;
c.analysis({
    ...sampleAnalysis,
    native_statistics: {
        mean_W_GeV: 249,
        full_RMS_W_GeV: 2
    }
});
n++;
for (const native of [{}, {
            mean_W_GeV: 249
        }, {
            mean_W_GeV: 249,
            full_RMS_W_GeV: -1
        }, {
            mean_W_GeV: NaN,
            full_RMS_W_GeV: 1
        }, {
            mean_W_GeV: 249,
            full_RMS_W_GeV: Infinity
        }, {
            mean_W_GeV: 249,
            full_RMS_W_GeV: 1e200
        },
        []
    ]) reject(() => c.analysis({
    ...sampleAnalysis,
    native_statistics: native
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    schema_version: "unknown"
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    observable: "photons"
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    plot: {
        ...sampleAnalysis.plot,
        crop_renormalization: true
    }
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    plot: {
        ...sampleAnalysis.plot,
        peak: {
            y_scale: "log",
            x_min_GeV: 245
        }
    }
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    bins: [{
        ...sampleAnalysis.bins[0],
        uncertainty: 0
    }]
}));
reject(() => c.analysis({
    ...sampleAnalysis,
    bins: [{
        ...sampleAnalysis.bins[0],
        W_low_GeV: NaN
    }]
}));
reject(() => c.fileRef({
    path: "../out.dat",
    bytes: 100,
    sha256: "a".repeat(64)
}));
// Native scientific fixtures test schema/weight arithmetic only; never physics results.
const beamMoment = {
    weight_sum: 3,
    energy_GeV: {
        mean: 125,
        rms: .1
    },
    x_nm: {
        mean: 0,
        rms: 500
    },
    y_nm: {
        mean: 0,
        rms: 8
    },
    vx: {
        mean: 0,
        rms: 1e-6
    },
    vy: {
        mean: 0,
        rms: 2e-6
    }
};
const hm = w => ({
    quantity: "W",
    unit: "GeV",
    weight_unit: "m^-2 per BX",
    weighting: "native collision pair luminosity",
    normalization: "bin-integrated",
    bins: [{
        low: 249,
        high: 250,
        weight: w
    }, {
        low: 250,
        high: 251,
        weight: 0
    }],
    underflow: 0,
    overflow: 0,
    total_weight: w
});
const nativeRecord = (id, w) => ({
    id,
    diagonal_start: id,
    diagonal_end: id,
    luminosity_spectrum: hm(w),
    polarization: w ? {
        weighting: "native collision pair luminosity",
        weight_sum: w,
        beam1: .2,
        beam2: -.3
    } : null,
    beam_moments: w ? {
        weighting: "native collision pair luminosity",
        beam1: {
            ...structuredClone(beamMoment),
            weight_sum: w
        },
        beam2: {
            ...structuredClone(beamMoment),
            weight_sum: w
        }
    } : null,
    bunch_moments: {
        weighting: "all stored native primary macroparticles at diagonal barrier; no inactive-slice drift synchronization",
        beam1: structuredClone(beamMoment),
        beam2: structuredClone(beamMoment)
    },
    bunch_polarization: [.2, -.3],
    distributions: {
        chi_e: null,
        chi_gamma: null,
        photon_energy: null,
        pair_energy: null
    }
});
const dm = {
    ...m,
    frames: [{
        ...m.frames[0],
        id: 0,
        diagonal: 0
    }, {
        ...m.frames[0],
        id: 1,
        diagonal: 1
    }]
};
const dg = {
    schema_version: "feebl.collider.diagnostics/1",
    frames: [nativeRecord(0, 0), nativeRecord(1, 3)],
    total: {
        ...nativeRecord(0, 3),
        diagonal_end: 1
    },
    nominal_polarization: {
        convention: "configured native reference",
        enabled: true,
        beam1: [0, 0, .2],
        beam2: [0, 0, .3],
        reference_helicity: [.2, -.3]
    },
    unavailable_reasons: {
        chi_e: "not computed",
        chi_gamma: "not computed",
        photon_energy: "not computed",
        pair_energy: "not computed"
    }
};
c.diagnostics(dg, dm);
n++;

function invalidDiagnostics(mutate) {
    const d = structuredClone(dg);
    mutate(d);
    reject(() => c.diagnostics(d, dm))
}
invalidDiagnostics(d => d.schema_version = "unknown");
invalidDiagnostics(d => d.frames.pop());
invalidDiagnostics(d => d.frames[1].id = 0);
invalidDiagnostics(d => d.frames[1].diagonal_start = 0);
invalidDiagnostics(d => d.frames[1].diagonal_end = 2);
invalidDiagnostics(d => d.total.diagonal_end = 0);
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.total_weight = 4);
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.bins[0].weight = -1);
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.bins[0].high = Infinity);
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.bins[0].low = 248);
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.weighting = "display samples");
invalidDiagnostics(d => d.frames[1].luminosity_spectrum.bins = new Array(2049).fill(d.frames[1]
    .luminosity_spectrum.bins[0]));
invalidDiagnostics(d => d.frames[1].polarization.beam1 = 2);
invalidDiagnostics(d => d.frames[1].polarization.weight_sum = 2);
invalidDiagnostics(d => d.total.polarization.beam2 = .3);
invalidDiagnostics(d => d.frames[1].beam_moments.weighting = "display samples");
invalidDiagnostics(d => d.frames[1].beam_moments.beam1.energy_GeV.rms = -1);
invalidDiagnostics(d => d.nominal_polarization.reference_helicity[1] = .3);
invalidDiagnostics(d => d.frames[1].distributions.chi_e = undefined);
invalidDiagnostics(d => d.frames[1].beam_moments.beam1.weight_sum = 2);
invalidDiagnostics(d => d.total.beam_moments.beam1.x_nm.mean = 10);
invalidDiagnostics(d => d.total.beam_moments.beam1.x_nm.rms = 600);
invalidDiagnostics(d => delete d.unavailable_reasons.chi_e);
invalidDiagnostics(d => d.frames[0].luminosity_spectrum.bins = []);
invalidDiagnostics(d => d.nominal_polarization = null);
const noNominal = structuredClone(dg);
noNominal.nominal_polarization = null;
noNominal.unavailable_reasons.nominal_polarization = "spin storage disabled";
c.diagnostics(noNominal, dm);
n++;
invalidDiagnostics(d => delete d.nominal_polarization.enabled);
const dh = structuredClone(dg);
for (const rec of [...dh.frames, dh.total]) rec.distributions.chi_e = {
    ...structuredClone(rec.luminosity_spectrum),
    quantity: "native primary kick chi_e",
    unit: "1",
    weight_unit: "macroparticle-kick evaluations",
    weighting: "one per actual primary field kick; repeated across substeps"
};
c.diagnostics(dh, dm);
n++;
const missing = structuredClone(dg);
for (const r of [...missing.frames, missing.total]) {
    r.luminosity_spectrum = null;
    r.polarization = null;
    r.beam_moments = null
}
c.diagnostics(missing, dm);
n++;
assert(html.includes('id="playback"'));
n++;
assert(html.indexOf('id="playback"') > html.indexOf('id="c"') && html.indexOf('id="playback"') <
    html.indexOf('id="spectrumPanel"'));
n++;
assert(html.includes('#playback{position:fixed;inset:auto 0 0;'));
n++;
for (const id of ['spectrumScope', 'distributionScope', 'polMode', 'polScope', 'diagScope',
        'momentScope'
    ]) {
    assert(html.includes('id="' + id + '"'));
    n++
}

// DOM/canvas mocks exercise data modes and absent-data paths; this is not browser QA.
const nodes = new Map(),
    canvasTexts = [];
const defaults = {
    fr: '0',
    globalScope: 'total',
    distributionRange: 'occupied',
    spectrumScope: 'total',
    spectrumUnits: 'native',
    distributionScope: 'frame',
    polMode: 'nominal',
    polScope: 'frame',
    momentScope: 'frame',
    diagScope: 'frame'
};

function element(id = '') {
    const e = {
        id,
        value: defaults[id] || '',
        hidden: false,
        checked: id === "scopeOverrides",
        selectedIndex: 0,
        disabled: false,
        textContent: '',
        innerHTML: '',
        children: [],
        options: [{textContent: "Frame interval"}, {textContent: "Total"}],
        after() {},
        width: 1100,
        height: 330,
        classList: {
            contains: () => false,
            add() {},
            remove() {},
            toggle() {}
        },
        addEventListener() {},
        setAttribute(k, v) {
            this.attributes ??= {};
            this.attributes[k] = v
        },
        replaceChildren() {
            this.children = []
        },
        append(...x) {
            this.children.push(...x)
        },
        appendChild(x) {
            this.children.push(x)
        }
    };
    e.getContext = () => new Proxy({}, {
        get: (o, k) => o[k] || ((...args) => {
            if (k === "fillText") canvasTexts.push(args[0]);
            for (const a of args)
                if (typeof a === 'number') assert(Number.isFinite(a),
                    'nonfinite canvas coordinate')
        }),
        set: (o, k, v) => (o[k] = v, true)
    });
    return e;
}
const sandbox = {
    structuredClone,
    ColliderContract: c,
    ColliderViewerUI: require("./collider_viewer_ui.js"),
    ColliderFields: require("./collider_fields.js"),
    ColliderStatistics: require("./collider_statistics.js"),
    console,
    URLSearchParams,
    TextDecoder,
    TextEncoder,
    Map,
    Set,
    Math,
    Number,
    Array,
    Object,
    JSON,
    document: {
        documentElement: {},
        getElementById: id => {
            if (!nodes.has(id)) nodes.set(id, element(id));
            return nodes.get(id)
        },
        createElement: () => element(),
        querySelectorAll: () => []
    },
    location: {
        search: '',
        hostname: '127.0.0.1'
    },
    getComputedStyle: () => ({
        getPropertyValue: () => '#14181a'
    }),
    addEventListener() {},
    setInterval() {
        return 1
    },
    clearInterval() {},
    setTimeout() {},
    fixture: dg,
    fixtureManifest: dm,
    fixtureAnalysis: {
        ...sampleAnalysis,
        native_statistics: {
            mean_W_GeV: 249,
            full_RMS_W_GeV: 2
        },
        plot: {
            ...sampleAnalysis.plot,
            full: {
                x_min_GeV: 0,
                y_scale: 'log'
            },
            x_max_GeV: 251.25
        },
        normalization: {
            ...sampleAnalysis.normalization,
            total_summary_m2_per_crossing: 2,
            total_L_cm2_s: null,
            crossing_frequency_hz: null,
            comparison_status: 'TEST_FIXTURE'
        },
        limitations: []
    }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, 'collider_fields.js'), 'utf8'), sandbox);
vm.runInContext(js, sandbox);
n++;
vm.runInContext('man=fixtureManifest;diagnostics=null;scientific=null;drawScientificViews(0)',
    sandbox);
n++;
assert(nodes.get('spectrumFull').hidden && nodes.get('chiECanvas').hidden);
n++;
assert(nodes.get('pol').children[0].children[1].textContent.includes('unavailable'));
n++;
vm.runInContext('scientific=fixtureAnalysis;drawScientificViews(0)', sandbox);
n++;
assert.equal(nodes.get('spectrumFull').hidden, false);
n++;
const fullStats = () => nodes.get('spectrumFullStats').attributes['aria-label'];
assert(fullStats().includes('Mean (native): 249.0000'));
n++;
assert(fullStats().includes('Variance [GeV²]: 4.0000'));
n++;
assert(fullStats().includes('including flows'));
n++;
assert(fullStats().includes('Mode (density bin)'));
n++;
assert(nodes.get('spectrumPeakStatsNotes').textContent.includes('Selected bin weight: 2.0000'));
n++;
assert(nodes.get('spectrumPeakStatsNotes').textContent.includes(
    'cut-straddling bin weight excluded: 0.0000'));
n++;
assert.equal(nodes.get('spectrumPeakStatsDetails').hidden, false);
n++;
assert(nodes.get('spectrumPeakStats').attributes['aria-label'].includes('Mean (bins): 249.5000'));
n++;
assert(nodes.get('spectrumPeakStats').attributes['aria-label'].includes(
    'Whole bins inside the peak window'));
n++;
const initialStats = fullStats();
for (const normalization of ['persecond', 'shape', 'native']) {
    nodes.get('spectrumUnits').value = normalization;
    vm.runInContext('drawSpectra()', sandbox);
    assert.equal(fullStats(), initialStats);
    n++
}
for (const term of ['Mean', 'Mode', 'Variance', 'Min', 'Max']) {
    assert(fullStats().includes(term));
    n++
}
nodes.get('spectrumScope').value = 'frame';
vm.runInContext('drawScientificViews(0)', sandbox);
n++;
assert.equal(nodes.get('spectrumFull').hidden, true);
n++;
vm.runInContext('diagnostics=fixture;drawScientificViews(0)', sandbox);
n++;
assert(nodes.get('spectrumContext').textContent.includes('diagonals 0–0'));
n++;
assert.equal(nodes.get('spectrumFullStats').hidden, false);
n++;
assert(fullStats().includes('Statistics unavailable'));
n++;
nodes.get('fr').value = '1';
nodes.get('polMode').value = 'lumi';
nodes.get('momentScope').value = 'total';
nodes.get('diagScope').value = 'total';
vm.runInContext('drawScientificViews(1)', sandbox);
n++;
assert(nodes.get('momentNote').textContent.includes('native collision pair luminosity'));
n++;
assert(nodes.get('pol').children[0].children[1].textContent.includes('0.200000'));
n++;
assert(fullStats().includes('Mean (bins): 249.5000'));
n++;
assert(!fullStats().includes('Mean (native)'));
n++;
vm.runInContext('scientific=null;drawScientificViews(1)', sandbox);
n++;
assert.equal(nodes.get('spectrumPeak').hidden, true);
n++;
assert.equal(vm.runInContext('selectedSpectrum().bins[0].dL_dW_m2_per_crossing_GeV', sandbox), 3);
n++;
assert.equal(vm.runInContext('selectedSpectrum().bins[0].shape_density_per_GeV', sandbox), 1);
n++;
sandbox.fixture = dh;
vm.runInContext('diagnostics=fixture;drawScientificViews(1)', sandbox);
n++;
assert.equal(nodes.get('chiECanvas').hidden, false);
n++;
const chiStats = () => nodes.get('chiECanvasStats').attributes['aria-label'];
assert(chiStats().includes('Mean (bins): 249.5000'));
n++;
assert.equal(nodes.get('chiECanvasStats').hidden, false);
n++;
assert.equal(nodes.get('chiGammaCanvasStats').hidden, true);
n++;
assert.equal(nodes.get('chiGammaCanvasStatsDetails').hidden, true);
n++;
nodes.get('distributionScope').value = 'total';
vm.runInContext('drawDistributions(1)', sandbox);
assert(chiStats().includes('Total collision'));
n++;
nodes.get('distributionScope').value = 'frame';

canvasTexts.length = 0;
nodes.get('fr').value = '0';
vm.runInContext('drawScientificViews(0)', sandbox);
n++;
assert.equal(nodes.get('chiECanvas').hidden, false);
n++;
assert(canvasTexts.includes('No in-range entries in this interval'));
n++;
assert.equal(nodes.get('chiECanvasStats').hidden, false);
n++;
assert(chiStats().includes('Statistics unavailable'));
n++;
assert.equal(nodes.get('chiGammaCanvas').hidden, true);
n++;
assert(nodes.get('spectrumSummary').textContent.includes('Native interval weights'));
n++;
assert(!nodes.get('spectrumSummary').textContent.includes('AUTHENTICATED_NATIVE_INTERVAL_WEIGHTS'));
n++;
nodes.get('fr').value = '1';
sandbox.fixture = structuredClone(dg);
sandbox.fixture.nominal_polarization.enabled = false;
vm.runInContext('diagnostics=fixture;drawScientificViews(1)', sandbox);
assert(nodes.get('polNote').textContent.includes('disabled'));
n++;
// Optional field UI must preserve older datasets and gate physical projections.
vm.runInContext('nativeFields=null;initializeFieldControls()', sandbox);
n++;
assert.equal(nodes.get('fieldMode').disabled, true);
n++;
assert(nodes.get('fieldNote').textContent.includes('no authenticated field export'));
n++;
sandbox.fieldManifest = {
    frames: [{
        id: 0,
        diagonal: 0,
        elapsed_s: 0
    }, {
        id: 1,
        diagonal: 1,
        elapsed_s: 1e-12
    }]
};
sandbox.fieldFixture = {
    schema_version: 'feebl.collider.fields/1',
    plane: {
        axes: ['x', 'y'],
        coordinate_unit: 'nm',
        frame: 'native simulated frame'
    },
    geometry: {
        nx: 2,
        ny: 2,
        x_min_nm: -500,
        x_max_nm: 500,
        y_min_nm: -500,
        y_max_nm: 500,
        native_domain_nm: [-1000, 1000, -1000, 1000],
        storage_order: 'row-major-y-x',
        sampling: 'uniform CIC gather points; bounds are sample centres'
    },
    sources: [{
        id: 'beam1',
        kind: 'beam_beam',
        label: 'Source beam 1 → beam 2',
        quantity: 'native transverse electric-equivalent field magnitude',
        unit: 'GV/nm',
        source_beam: 1,
        target_beam: 2
    }],
    conventions: {
        selection: 'central active slice encounter, final substep, pre-kick',
        scope: 'single source slice; not whole-bunch or diagonal-averaged field',
        charge_sign: -1,
        scale_step: [1, 1],
        n_z: 2,
        n_t: 1
    },
    frames: [{
        id: 0,
        diagonal: 0,
        elapsed_s: 0,
        available: true,
        reason: null,
        encounter: {
            beam1_slice: 0,
            beam2_slice: 0,
            substep_index: 0,
            encounter_counter: 0,
            nominal_z_nm: 0,
            phase: 'post-solve/pre-kick'
        },
        maps: {
            beam1: [0, 0, 0, 0]
        }
    }, {
        id: 1,
        diagonal: 1,
        elapsed_s: 1e-12,
        available: true,
        reason: null,
        encounter: {
            beam1_slice: 0,
            beam2_slice: 1,
            substep_index: 0,
            encounter_counter: 1,
            nominal_z_nm: 10,
            phase: 'post-solve/pre-kick'
        },
        maps: {
            beam1: [1, 2, 3, 4]
        }
    }]
};
vm.runInContext(
    'nativeFields=ColliderFields.validate(fieldFixture,fieldManifest);initializeFieldControls()',
    sandbox);
n++;
assert.equal(nodes.get('fieldMode').disabled, false);
assert.equal(nodes.get('fieldMode').value, 'off');
n++;
const fieldRects = [];
sandbox.fieldG = {
    save() {},
    restore() {},
    beginPath() {},
    rect() {},
    clip() {},
    setLineDash() {},
    strokeRect(...r) {
        assert(r.every(Number.isFinite))
    },
    fillRect(...r) {
        assert(r.every(Number.isFinite));
        fieldRects.push({
            r,
            alpha: this.globalAlpha,
            color: this.fillStyle,
            composite: this.globalCompositeOperation
        })
    }
};
sandbox.fieldView = {
    horizontal: 'z',
    vertical: 'x',
    L: 0,
    T: 0,
    W: 100,
    H: 100,
    px: v => v + 50,
    py: v => 50 - v
};
nodes.get('fieldMode').value = 'on';
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 0);
assert.equal(nodes.get('fieldPlane').hidden, true);
n++;
assert(nodes.get('fieldNote').textContent.includes('No authenticated longitudinal field export'));
n++;
sandbox.fieldView.horizontal = 'x';
sandbox.fieldView.vertical = 'y';
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 4);
assert.equal(nodes.get('fieldLegend').hidden, false);
n++;
assert(nodes.get('fieldDetails').textContent.includes('post-diagonal display state'));
n++;
assert(nodes.get('fieldNote').textContent.includes(
    'pre-kick field; cloud at the end of the diagonal'));
n++;
assert(nodes.get('fieldLegend').attributes['aria-label'].includes(
    '0 to 4.000e+9 GV/m, fixed run scale'));
n++;
nodes.get('fieldOpacity').value = '.75';
nodes.get('fieldScale').value = 'frame';
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.at(-1).alpha, .75);
assert(nodes.get('fieldNote').textContent.includes('auto frame scale'));
n++;
assert(nodes.get('fieldLegend').attributes['aria-label'].includes('auto frame scale'));
n++;
fieldRects.length = 0;
sandbox.fieldView.horizontal = 'z';
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 0);
assert.equal(nodes.get('fieldLegend').hidden, true);
n++;
sandbox.fieldView.horizontal = 'x';
vm.runInContext(
    'nativeFields.data.frames[1]={...nativeFields.data.frames[1],available:false,reason:"native map unavailable",encounter:null,maps:null};paintFieldOverlay(1,fieldG,fieldView)',
    sandbox);
n++;
assert.equal(fieldRects.length, 0);
assert.equal(nodes.get('fieldLegend').hidden, true);
assert(nodes.get('fieldNote').textContent.includes('native map unavailable'));
n++;
assert(html.indexOf('id="fieldControls"') < html.indexOf('id="c"'));
n++;
const clearCanvasIndex = js.search(/g\.clearRect\(\s*0\s*,\s*0\s*,\s*size\.width\s*,\s*size\.height\s*\)/);
const paintFieldIndex = js.search(/paintFieldOverlay\(\s*k\s*,\s*g\s*,/);
assert(clearCanvasIndex >= 0 && paintFieldIndex > clearCanvasIndex);
n++;

assert.equal((html.match(/class="field-control-group"/g) || []).length, 5);
n++;
assert(html.includes(
    '<details id="fieldDetailsDisclosure"><summary>Field sampling details</summary>'));
n++;
// Dataset name comes from the actual loader path, never a guessed native run name.
vm.runInContext(
    'base="data/run-a";man={...fixtureManifest,receipts:{card:{sha256:"b".repeat(64)}}};updateDatasetIdentity()',
    sandbox);
n++;
assert(nodes.get('datasetIdentity').textContent.includes('data/run-a'));
n++;
assert(nodes.get('datasetIdentity').textContent.includes('bbbbbbbbbbbb'));
n++;
vm.runInContext(
    'base=null;files={"gui_manifest.json":{webkitRelativePath:"selected-run/gui_manifest.json"}};updateDatasetIdentity()',
    sandbox);
n++;
assert(nodes.get('datasetIdentity').textContent.includes('selected-run'));
n++;
vm.runInContext('files={};man=fixtureManifest;updateDatasetIdentity()', sandbox);
n++;
assert(nodes.get('datasetIdentity').textContent.includes('name unavailable'));
n++;
assert(nodes.get('datasetIdentity').textContent.includes('SHA-256: unavailable'));
n++;
// Projection shortcuts preserve frame and field mode; this checks UI state only.
vm.runInContext('drawBusy=true', sandbox);
nodes.get('fr').value = '1';
nodes.get('view').value = 'central';
nodes.get('fieldMode').value = 'on';
vm.runInContext('setParticleProjection("overview")', sandbox);
n++;
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'x');
assert.equal(nodes.get('view').value, 'full');
n++;
assert.equal(nodes.get('fr').value, '1');
assert.equal(nodes.get('fieldMode').value, 'on');
n++;
assert(nodes.get('projectionContext').textContent.includes('longitudinal approach'));
n++;
vm.runInContext('setParticleProjection("zx")', sandbox);
n++;
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'x');
n++;
assert.equal(nodes.get('fr').value, '1');
assert.equal(nodes.get('fieldMode').value, 'off');
n++;
assert(nodes.get('projectionContext').textContent.includes('longitudinal approach'));
n++;
assert(html.indexOf('id="projectionContext"') < html.indexOf('id="c"'));
n++;
assert(/addEventListener\("resize",\s*\(\)\s*=>\s*\{\s*if\s*\(man\)\s*draw\(\);?\s*\}\)/.test(js));
n++;
const fitCanvas = {
    clientWidth: 480,
    clientHeight: 320,
    width: 1100,
    height: 330
};
sandbox.fitCanvas = fitCanvas;
vm.runInContext('fitStatisticsCanvas(fitCanvas)', sandbox);
assert.equal(fitCanvas.width, 480);
assert.equal(fitCanvas.height, 320);
n++;
for (const id of ['spectrumFull', 'spectrumPeak', 'chiECanvas', 'chiGammaCanvas', 'photonCanvas',
        'pairCanvas'
    ]) {
    assert(html.includes('id="' + id + 'Stats" class="plot-statistics" role="group"'));
    n++;
}
vm.runInContext(
    'renderPlotStatistics("spectrumFull",ColliderStatistics.histogram({bins:[],underflow:3,overflow:0},{nativeMoments:{mean:249,rms:2,validated:true}}),"Full · total collision","GeV","m⁻²/crossing")',
    sandbox);
n++;
assert(fullStats().includes('Mean (native): 249.0000'));
n++;
assert(fullStats().includes('Mode (density bin): unavailable'));
n++;
assert(fullStats().includes('Min (bin edge): —'));
n++;
assert(!fullStats().includes('0 tied bins'));
n++;
vm.runInContext(
    'renderPlotStatistics("spectrumPeak",ColliderStatistics.histogram({bins:[{low:244,high:246,weight:10},{low:246,high:247,weight:5}],underflow:2,overflow:1},{window:[245,250]}),"Peak · total collision","GeV","m⁻²/crossing")',
    sandbox);
n++;
assert(nodes.get('spectrumPeakStatsNotes').textContent.includes('Selected bin weight: 5.0000'));
n++;
assert(nodes.get('spectrumPeakStatsNotes').textContent.includes(
    'cut-straddling bin weight excluded: 10.0000'));
n++;
vm.runInContext(
    'renderPlotStatistics("chiECanvas",ColliderStatistics.histogram({bins:[{low:1,high:2,weight:1},{low:2,high:4,weight:2}],underflow:0,overflow:0}),"Total collision","1","evaluations")',
    sandbox);
n++;
assert(chiStats().includes('Mode (density bin): 2 tied bins'));
n++;
// Resolution comes from actual edges, including dense zero bins in interval exports.
vm.runInContext('scientific=fixtureAnalysis;$("spectrumScope").value="total";drawSpectra()',
    sandbox);
n++;
assert(nodes.get('spectrumPeakResolution').textContent.includes('1000 MeV'));
n++;
assert(nodes.get('spectrumPeakResolution').textContent.includes(
    'native total export lists occupied bins only'));
n++;
sandbox.fineBins = [{
    W_low_GeV: 245,
    W_high_GeV: 245.025,
    weight_m2_per_crossing: 3
}, {
    W_low_GeV: 245.05,
    W_high_GeV: 245.075,
    weight_m2_per_crossing: 5
}];
vm.runInContext(
    'scientific={...fixtureAnalysis,bins:fineBins,plot:{...fixtureAnalysis.plot,x_max_GeV:245.1}}',
    sandbox);
assert(vm.runInContext('peakResolutionNote(scientific,"total")', sandbox).includes(
    'width: 25 MeV; 2 occupied / 2 exported'));
n++;
sandbox.densePeak = [{
    low: 244.75,
    high: 245,
    weight: 0
}, {
    low: 245,
    high: 245.025,
    weight: 0
}, {
    low: 245.025,
    high: 245.05,
    weight: 0
}, {
    low: 245.05,
    high: 245.075,
    weight: 3
}, {
    low: 245.075,
    high: 245.1,
    weight: 0
}];
vm.runInContext(
    'diagnostics=structuredClone(fixture);diagnostics.frames[0].luminosity_spectrum.bins=densePeak;$("fr").value="0"',
    sandbox);
assert(vm.runInContext(
    'peakResolutionNote(derivedSpectrum(diagnostics.frames[0].luminosity_spectrum,"frame"),"frame")',
    sandbox).includes('25 MeV; 1 occupied / 4 exported'));
n++;
vm.runInContext(
    'diagnostics.frames[0].luminosity_spectrum.bins.push({low:245.1,high:245.35,weight:0})',
    sandbox);
assert(vm.runInContext(
    'peakResolutionNote(derivedSpectrum(diagnostics.frames[0].luminosity_spectrum,"frame"),"frame")',
    sandbox).includes('width range: 25–250 MeV'));
n++;
vm.runInContext('diagnostics.frames[0].luminosity_spectrum.bins.forEach(b=>b.weight=0)', sandbox);
assert(vm.runInContext(
    'peakResolutionNote(derivedSpectrum(diagnostics.frames[0].luminosity_spectrum,"frame"),"frame")',
    sandbox).includes('0 occupied / 5 exported'));
n++;
vm.runInContext('scientific=null;drawSpectra()', sandbox);
assert.equal(nodes.get('spectrumPeakResolution').hidden, true);
n++;
// The explicit field-view action enables available data; axis selectors alone do not.
vm.runInContext(
    'nativeFields=ColliderFields.validate(fieldFixture,fieldManifest);initializeFieldControls();$("fr").value="1";drawBusy=true;setParticleProjection("zx")',
    sandbox);
n++;
assert.equal(nodes.get('fieldMode').value, 'off');
assert.equal(nodes.get('fr').value, '1');
n++;
nodes.get('fieldMode').value = 'off';
nodes.get('ax').value = 'x';
nodes.get('ay').value = 'y';
vm.runInContext('updateProjectionContext()', sandbox);
assert.equal(nodes.get('fieldMode').value, 'off');
n++;
vm.runInContext('nativeFields=null;setParticleProjection("zx")', sandbox);
n++;
assert.equal(nodes.get('fieldMode').value, 'off');
assert.equal(nodes.get('fr').value, '1');
n++;
assert(nodes.get('fieldNote').textContent.includes('no authenticated longitudinal field export'));
n++;
vm.runInContext(
    'nativeFields={longitudinal:{availableFrames:0},data:{frames:[{longitudinal:{reason:"unsupported backend",strips:[]}},{longitudinal:{reason:"unsupported backend",strips:[]}}]}};setParticleProjection("zx")',
    sandbox);
n++;
assert.equal(nodes.get('fieldMode').value, 'off');
assert(nodes.get('fieldNote').textContent.includes('unsupported backend'));
n++;
// Longitudinal field actions retain z horizontally and expose actual source-band coverage.
sandbox.longFixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_longitudinal.json'), 'utf8'));
vm.runInContext(
    'nativeFields=ColliderFields.validate(longFixture.fields,longFixture.manifest);initializeFieldControls();$("fr").value="1";setParticleProjection("zx")',
    sandbox);
n++;
assert.equal(nodes.get('fieldMode').value, 'on');
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'x');
n++;
assert.equal(nodes.get('fr').value, '1');
n++;
sandbox.fieldView.horizontal = 'z';
sandbox.fieldView.vertical = 'x';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 6);
assert.equal(nodes.get('fieldLegend').hidden, false);
n++;
assert(nodes.get('fieldNote').textContent.includes('3/3 source slices sampled (3 active)'));
assert(nodes.get('fieldDetails').textContent.includes('at y=0'));
n++;
assert(nodes.get('fieldDetails').textContent.includes('not an integral'));
n++;
vm.runInContext('setParticleProjection("zy")', sandbox);
n++;
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'y');
assert.equal(nodes.get('fr').value, '1');
n++;
sandbox.fieldView.vertical = 'y';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 6);
assert(nodes.get('fieldDetails').textContent.includes('at x=0'));
n++;
vm.runInContext(
    'Object.assign(nativeFields.data.frames[1].longitudinal.strips[1],{available:false,reason:"missing",maps:null,substep_index:null,encounter_counter:null});paintFieldOverlay(1,fieldG,fieldView)',
    sandbox);
n++;
assert(nodes.get('fieldNote').textContent.includes('2/3 source slices sampled (3 active)'));
n++;
assert(nodes.get('fieldNote').textContent.includes('Blank regions have no field samples'));
n++;
assert(html.includes('id="fieldZX"') && html.includes('id="fieldZY"') && !html.includes(
    'id="transverseView"'));
n++;
// Additive species diagnostics: canonical metadata, exact count closures and honest legacy fallback.
const sf = JSON.parse(fs.readFileSync(path.join(__dirname, 'testdata/collider_species_chi.json'),
    'utf8'));
c.diagnostics(sf.diagnostics, sf.manifest);
n++;
const speciesKeys = ['chi_e', 'chi_e_electron', 'chi_e_positron'];
const speciesRecords = d => [...d.frames, d.total];

function noSpecies(d) {
    delete d.primary_species;
    for (const r of speciesRecords(d))
        for (const k of speciesKeys.slice(1)) delete r.distributions[k]
}

function unsupportedSpecies(d) {
    Object.assign(d.primary_species, {
        supported: false,
        charge_sign: 0,
        beam1_pdg: null,
        beam2_pdg: null
    });
    for (const r of speciesRecords(d))
        for (const k of speciesKeys.slice(1)) r.distributions[k] = null
}

function invalidSpecies(mutate) {
    const d = structuredClone(sf.diagnostics);
    mutate(d);
    reject(() => c.diagnostics(d, sf.manifest))
}
for (const mutate of [d => delete d.primary_species, d => d.primary_species.beam1_pdg = -11, d => d
        .primary_species.charge_sign = 0,
        d => d.primary_species.basis = 'energy sign', d => delete d.primary_species.supported,
        d => delete d.frames[0].distributions.chi_e_electron,
        d => d.frames[0].distributions.chi_e_electron = null, d => d.total.distributions
        .chi_e_electron.quantity = 'native primary positron kick chi_e',
        d => d.total.distributions.chi_e_electron.weighting = 'display particles', d => d
        .frames[0].distributions.chi_e_electron.bins[0].weight = 1.5,
        d => d.total.distributions.chi_e_electron.samples = 9, d => d.frames[0].distributions
        .chi_e_electron.bins[0].low = .5,
        d => {
            unsupportedSpecies(d);
            delete d.unavailable_reasons.chi_e_electron
        },
        d => Object.assign(d.primary_species, {
            supported: false,
            charge_sign: 0,
            beam1_pdg: null,
            beam2_pdg: null
        })
    ]) invalidSpecies(mutate);
invalidSpecies(d => {
    for (const r of speciesRecords(d)) {
        const h = r.distributions.chi_e_electron;
        h.bins[0].weight++;
        h.total_weight++;
        h.samples++
    }
});
invalidSpecies(d => {
    for (const r of speciesRecords(d))
        for (const k of speciesKeys) {
            const h = r.distributions[k];
            for (const b of h.bins) b.weight *= 1e9;
            for (const f of ['underflow', 'overflow', 'total_weight', 'samples']) h[f] *=
                1e9
        }
    for (const k of ['chi_e', 'chi_e_electron']) {
        const h = d.total.distributions[k];
        h.bins[0].weight++;
        h.bins[1].weight--
    }
});
for (const mutate of [noSpecies, unsupportedSpecies, d => {
        for (const r of speciesRecords(d))
            for (const k of speciesKeys) r.distributions[k] = null
    }]) {
    const d = structuredClone(sf.diagnostics);
    mutate(d);
    c.diagnostics(d, sf.manifest);
    n++;
}
assert(html.includes('Combined primary χe (both beams)'));
n++;
assert(html.includes('Primary electrons χe (beam 1)') && html.includes(
    'Primary positrons χe (beam 2)'));
n++;
assert(!html.includes('Electron strong-field parameter χe'));
n++;
sandbox.speciesFixture = sf.diagnostics;
vm.runInContext(
    'diagnostics=speciesFixture;$("distributionScope").value="frame";drawDistributions(0)',
    sandbox);
n++;
assert(nodes.get('chiCombinedPanel').hidden && !nodes.get('chiElectronPanel').hidden && !nodes.get(
    'chiPositronPanel').hidden);
n++;
assert(!nodes.get('chiElectronCanvas').hidden && !nodes.get('chiPositronCanvas').hidden && nodes
    .get('chiECanvas').hidden);
n++;
const electronStats = () => nodes.get('chiElectronCanvasStats').attributes['aria-label'];
const positronStats = () => nodes.get('chiPositronCanvasStats').attributes['aria-label'];
assert(electronStats().includes('Mean (bins): 1.5000') && positronStats().includes(
    'Mean (bins): 3.0000'));
n++;
assert(nodes.get('distributionContext').textContent.includes('not physical particle yields'));
n++;
vm.runInContext('drawDistributions(1)', sandbox);
n++;
assert(electronStats().includes('Mean (bins): 3.0000') && positronStats().includes(
    'Mean (bins): 1.5000'));
n++;
vm.runInContext('$("distributionScope").value="total";drawDistributions(1)', sandbox);
n++;
assert(electronStats().includes('Mean (bins): 2.5000') && positronStats().includes(
    'Mean (bins): 2.0625'));
n++;
assert(electronStats().includes('Total collision') && electronStats().includes('Variance: 0.5000'));
n++;
assert(nodes.get('chiElectronCanvasStatsNotes').textContent.includes(
'Selected bin weight: 6.0000'));
n++;
assert(nodes.get('chiElectronCanvasStatsNotes').textContent.includes(
    'underflow / overflow excluded from binned statistics: 1.0000 / 1.0000'));
n++;
const zeroSpecies = structuredClone(sf.diagnostics);
for (const r of speciesRecords(zeroSpecies))
    for (const k of speciesKeys) {
        const h = r.distributions[k];
        for (const b of h.bins) b.weight = 0;
        for (const f of ['underflow', 'overflow', 'total_weight', 'samples']) h[f] = 0
    }
c.diagnostics(zeroSpecies, sf.manifest);
n++;
sandbox.zeroSpecies = zeroSpecies;
vm.runInContext('diagnostics=zeroSpecies;drawDistributions(0)', sandbox);
n++;
assert(!nodes.get('chiElectronCanvas').hidden && !nodes.get('chiPositronCanvas').hidden);
n++;
assert(electronStats().includes('Statistics unavailable'));
n++;
const legacySpecies = structuredClone(sf.diagnostics);
noSpecies(legacySpecies);
sandbox.legacySpecies = legacySpecies;
vm.runInContext('diagnostics=legacySpecies;drawDistributions(0)', sandbox);
n++;
assert(!nodes.get('chiCombinedPanel').hidden && nodes.get('chiElectronPanel').hidden && nodes.get(
    'chiPositronPanel').hidden);
n++;
assert(!nodes.get('chiECanvas').hidden && nodes.get('chiElectronCanvas').hidden);
n++;
assert(nodes.get('distributionContext').textContent.includes('species split unavailable'));
n++;
const unavailableSpecies = structuredClone(sf.diagnostics);
unsupportedSpecies(unavailableSpecies);
sandbox.unavailableSpecies = unavailableSpecies;
vm.runInContext('diagnostics=unavailableSpecies;drawDistributions(0)', sandbox);
n++;
assert(!nodes.get('chiECanvas').hidden && nodes.get('chiElectronPanel').hidden);
n++;
vm.runInContext('diagnostics=fixture;$("distributionScope").value="frame"', sandbox);

// Full snapshots default when available; encounter values remain a separate selectable mode.
sandbox.snapshotFixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_snapshot_fields.json'), 'utf8'));
vm.runInContext(
    'nativeFields=ColliderFields.validate(snapshotFixture.fields,snapshotFixture.manifest);initializeFieldControls();$("fr").value="0";drawBusy=true;setParticleProjection("zx")',
    sandbox);
n++;
assert.equal(nodes.get('fieldData').value, 'snapshot');
assert.equal(nodes.get('fieldData').disabled, false);
n++;
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'x');
assert.equal(nodes.get('fr').value, '0');
n++;
assert.equal(nodes.get('fieldMode').value, 'on');
n++;
sandbox.fieldView.horizontal = 'z';
sandbox.fieldView.vertical = 'x';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 6);
n++;
assert(nodes.get('fieldNote').textContent.includes('3/3 source slices synchronized to the frame'));
n++;
assert(nodes.get('fieldNote').textContent.includes('reconstructed native slice field'));
n++;
assert(nodes.get('fieldDetails').textContent.includes('outside that mesh: 2'));
n++;
assert.equal(vm.runInContext(
    'ColliderFields.samplingWindow(nativeFields,"zx",selectedFieldData()).horizontal[1]',
    sandbox), 3.5);
n++;
nodes.get('fieldData').value = 'encounter';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 2);
n++;
assert(nodes.get('fieldNote').textContent.includes('Active-encounter fields'));
n++;
assert.equal(vm.runInContext(
    'ColliderFields.samplingWindow(nativeFields,"zx",selectedFieldData()).horizontal[1]',
    sandbox), 2.5);
n++;
nodes.get('fieldData').value = 'snapshot';
nodes.get('fr').value = '1';
vm.runInContext('setParticleProjection("zy")', sandbox);
n++;
assert.equal(nodes.get('fr').value, '1');
assert.equal(nodes.get('ax').value, 'z');
assert.equal(nodes.get('ay').value, 'y');
n++;
vm.runInContext(
    'Object.assign(nativeFields.data.frames[1].snapshot,{available:false,reason:"reconstruction unavailable",sources:null})',
    sandbox);
sandbox.fieldView.vertical = 'y';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 0);
assert(nodes.get('fieldLegend').hidden);
n++;
assert(nodes.get('fieldNote').textContent.includes('reconstruction unavailable'));
n++;
nodes.get('fieldData').value = 'encounter';
vm.runInContext('paintFieldOverlay(1,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 6);
n++;
assert(html.includes('id="fieldData"') && html.includes('Full bunch snapshot'));
n++;

// Visible source-mesh exclusion appears only when the authenticated count is nonzero.
vm.runInContext(
    'nativeFields=ColliderFields.validate(snapshotFixture.fields,snapshotFixture.manifest);initializeFieldControls();$("fieldMode").value="on";paintFieldOverlay(0,fieldG,fieldView)',
    sandbox);
n++;
assert(nodes.get('fieldNote').textContent.includes('2 source particles outside transverse mesh'));
n++;
vm.runInContext(
    'for(const s of nativeFields.data.frames[0].snapshot.sources.beam1.strips){s.source_particles=s.deposited_particles;s.outside_particles=0};paintFieldOverlay(0,fieldG,fieldView)',
    sandbox);
n++;
assert(!nodes.get('fieldNote').textContent.includes('source particles outside transverse mesh'));
n++;
// Both canonical beams are visual layers, never a synthetic summed field.
sandbox.bothFixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_snapshot_fields.json'), 'utf8'));
vm.runInContext(
    'nativeFields=ColliderFields.validate(bothFixture.fields,bothFixture.manifest);initializeFieldControls();$("fr").value="1"',
    sandbox);
n++;
assert.equal(nodes.get('fieldSource').value, 'beam1');
assert.equal(nodes.get('fieldMode').value, 'off');
n++;
assert(nodes.get('fieldSource').children.some(o => o.value === '__both_beams__' && o.textContent ===
    'Both beams (separate layers)'));
n++;
assert.equal(nodes.get('fieldBothLegends').hidden, true);
n++;
nodes.get('fieldSource').value = '__both_beams__';
nodes.get('fieldMode').value = 'on';
nodes.get('fieldOpacity').value = '.4';
const beforeAxes = [nodes.get('ax').value, nodes.get('ay').value, nodes.get('fr').value];
for (const panel of ['#ffffff', '#1c2024']) {
    sandbox.getComputedStyle = () => ({
        getPropertyValue: k => k === '--panel' ? panel : '#14181a'
    });
    for (const mode of ['snapshot', 'encounter'])
        for (const axis of ['x', 'y']) {
            nodes.get('fieldData').value = mode;
            sandbox.fieldView.vertical = axis;
            nodes.get('fieldScale').value = 'run';
            const labels = [];
            for (const k of [0, 1]) {
                fieldRects.length = 0;
                vm.runInContext('paintFieldOverlay(' + k + ',fieldG,fieldView)', sandbox);
                n++;
                const expected = vm.runInContext('ColliderFields.plane(nativeFields,' + k +
                    ',"beam1","z' + axis + '","' + mode + '").strips.length*4', sandbox);
                assert.equal(fieldRects.length, expected);
                n++;
                assert(fieldRects.every(r => r.composite === (panel === '#ffffff' ? 'multiply' :
                    'screen') && r.alpha === .4));
                n++;
                assert.equal(nodes.get('fieldLegend').hidden, true);
                assert.equal(nodes.get('fieldBothLegends').hidden, false);
                n++;
                assert(nodes.get('fieldNote').textContent.includes(
                    'Visual blend; not a vector sum'));
                n++;
                assert(nodes.get('fieldDetails').textContent.includes(
                    'Each layer retains its own source-slice positions'));
                n++;
                assert(nodes.get('fieldDetails').textContent.includes(
                    'Layer opacity applies separately'));
                n++;
                labels.push(nodes.get('fieldLegendBeam1').attributes['aria-label']);
                assert(labels.at(-1).includes('shared fixed run scale; 40% layer opacity'));
                n++;
                assert(nodes.get('fieldLegendBeam2').attributes['aria-label'].includes(
                    'separate orange layer'));
                n++;
            }
            assert.equal(labels[0], labels[1], 'Run scale must remain fixed across frames');
            n++;
            nodes.get('fieldScale').value = 'frame';
            vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
            n++;
            const maximum = vm.runInContext(
                'Math.max(...ColliderFields.beamLayers(nativeFields).map(s=>ColliderFields.scaleMaximum(nativeFields,0,s.id,"frame","z' +
                axis + '","' + mode + '")))', sandbox);
            assert(nodes.get('fieldLegendBeam1').attributes['aria-label'].includes('0 to ' + vm
                .runInContext('fmt(' + maximum + '*1e9,3)', sandbox) +
                ' GV/m, shared auto frame'));
            n++;
        }
}
assert.deepEqual([nodes.get('ax').value, nodes.get('ay').value, nodes.get('fr').value], beforeAxes);
n++;
nodes.get('fieldData').value = 'snapshot';
nodes.get('fieldSource').value = 'beam1';
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert(nodes.get('fieldBothLegends').hidden && !nodes.get('fieldLegend').hidden);
n++;
assert.equal(fieldRects.at(-1).composite, 'source-over');
assert(fieldRects.at(-1).color.startsWith('#'));
n++;
nodes.get('fieldSource').value = '__both_beams__';
vm.runInContext(
    'Object.assign(nativeFields.data.frames[1].snapshot,{available:false,reason:"missing reconstruction",sources:null});paintFieldOverlay(1,fieldG,fieldView)',
    sandbox);
n++;
assert(nodes.get('fieldBothLegends').hidden && nodes.get('fieldLegend').hidden);
assert(nodes.get('fieldNote').textContent.includes('missing reconstruction'));
n++;
sandbox.fieldView.horizontal = 'E';
fieldRects.length = 0;
vm.runInContext('paintFieldOverlay(0,fieldG,fieldView)', sandbox);
n++;
assert.equal(fieldRects.length, 0);
assert(nodes.get('fieldBothLegends').hidden);
n++;
vm.runInContext(
    'nativeFields=ColliderFields.validate(fieldFixture,fieldManifest);initializeFieldControls()',
    sandbox);
n++;
assert(!nodes.get('fieldSource').children.some(o => o.value === '__both_beams__'));
n++;
assert(html.includes('Layer opacity') && html.includes(
    'grid-template-columns:repeat(2,minmax(0,1fr))'));
n++;
// Diagnostic polarization keeps configured, instantaneous bunch, and L weights distinct.
sandbox.polDiagnosticFixture = structuredClone(dg);
sandbox.polDiagnosticFixture.frames[0].bunch_polarization = [.19, -.28];
sandbox.polDiagnosticFixture.frames[1].bunch_polarization = [.17, -.26];
sandbox.polDiagnosticFixture.total.bunch_polarization = [.99,
.99]; // endpoint total member is never an averaged bunch observable
vm.runInContext('diagnostics=polDiagnosticFixture;$("diagScope").value="frame";drawDiagnostics(0)',
    sandbox);
n++;
const diagRows = () => new Map(nodes.get('diag').children.map(row => row.children.map(cell => cell
    .textContent)));
assert.equal(diagRows().get('Configured P₁ / P₂ [1]'), '0.200000 / -0.300000');
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 0'), '0.190000 / -0.280000');
n++;
assert.equal(diagRows().get('Interval ⟨P₁⟩L / ⟨P₂⟩L [1]'), 'unavailable / no weight');
n++;
assert.equal(diagRows().get('Polarization L weight'), 'unavailable');
n++;
vm.runInContext('drawDiagnostics(1)', sandbox);
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 1'), '0.170000 / -0.260000');
n++;
assert.equal(diagRows().get('Interval ⟨P₁⟩L / ⟨P₂⟩L [1]'), '0.200000 / -0.300000');
n++;
assert.equal(diagRows().get('Polarization L weight'), '3.00000 m⁻² / crossing');
n++;
vm.runInContext('$("diagScope").value="total";drawDiagnostics(0)', sandbox);
n++;
assert.equal(diagRows().get('Total ⟨P₁⟩L / ⟨P₂⟩L [1]'), '0.200000 / -0.300000');
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 0'), '0.190000 / -0.280000');
n++;
assert(nodes.get('diagNote').textContent.includes(
    'Bunch helicity always refers to the displayed snapshot'));
n++;
for (const phrase of ['dimensionless', 'not measurements of loaded beams', 'not time-averaged',
        'not time-averaged or derived from displayed samples', 'own native direction',
        'No design crossing-angle'
    ]) {
    assert(nodes.get('diagPolNote').textContent.includes(phrase));
    n++;
}
vm.runInContext('diagnostics.nominal_polarization.enabled=false;drawDiagnostics(1)', sandbox);
n++;
assert.equal(diagRows().get('Configured P₁ / P₂ [1]'), '0.200000 / -0.300000');
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 1'), 'unavailable / spin not stored');
n++;
assert.equal(diagRows().get('Total ⟨P₁⟩L / ⟨P₂⟩L [1]'), 'unavailable / spin not stored');
n++;
assert(nodes.get('diagPolNote').textContent.includes('configured inputs only'));
n++;
vm.runInContext(
    'diagnostics.nominal_polarization=null;diagnostics.frames[1].bunch_polarization=[null,-.26];drawDiagnostics(1)',
    sandbox);
n++;
assert.equal(diagRows().get('Configured P₁ / P₂ [1]'), 'unavailable');
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 1'), 'unavailable / -0.260000');
n++;
vm.runInContext('diagnostics=null;drawDiagnostics(1)', sandbox);
n++;
assert.equal(diagRows().get('Configured P₁ / P₂ [1]'), 'unavailable');
n++;
assert.equal(diagRows().get('Bunch P₁ / P₂ [1], snapshot 1'), 'unavailable');
n++;
assert.equal(diagRows().get('Total ⟨P₁⟩L / ⟨P₂⟩L [1]'), 'unavailable / no weight');
n++;
assert(html.includes('id="diagPolNote"'));
n++;
// Authenticated static input factor, scope-specific width ratio and final joint particle panel.
const diagnosticAdaptersFixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_new_diagnostics.json'), 'utf8'));
sandbox.ColliderMonochromatization = require('./collider_monochromatization.js');
sandbox.ColliderSecondaryKinematics = require('./collider_secondary_kinematics.js');
sandbox.newDiagnosticFixture = diagnosticAdaptersFixture;
vm.runInContext(
    'monochromatization=ColliderMonochromatization.validate(newDiagnosticFixture.mono,newDiagnosticFixture.refs,newDiagnosticFixture.context);secondaryKinematics=ColliderSecondaryKinematics.validate(newDiagnosticFixture.secondary,newDiagnosticFixture.refs);man=fixtureManifest;man.diagnostic_analysis_sources=newDiagnosticFixture.refs;initializeSecondaryControls();drawMonochromatizationInputs();drawDiagnostics(0)',
    sandbox);
n++;
const monoRows = () => new Map(nodes.get('monoInput').children.map(r => r.children.map(c => c
    .textContent)));
assert.equal(monoRows().get('Ideal monochromatization λ [1]'), '1.000000');
n++;
assert(nodes.get('monoInputNote').textContent.includes('static evaluated-optics reference'));
n++;
assert(diagRows().get('Total full-RMS ratio to input [1]').startsWith('0.353553'));
n++;
assert.equal(diagRows().get('Matched-control monochromatization gain'), 'unavailable');
n++;
nodes.get('diagScope').value = 'frame';
vm.runInContext('drawDiagnostics(0)', sandbox);
n++;
assert.equal(diagRows().get('Interval full-RMS ratio to input [1]'), 'unavailable');
n++;
vm.runInContext('drawDiagnostics(1)', sandbox);
n++;
assert(diagRows().get('Interval full-RMS ratio to input [1]').startsWith('0.353553'));
n++;
assert(nodes.get('diagNote').textContent.includes('not an isolated dispersion gain'));
n++;
assert.equal(nodes.get('secondaryPopulation').value, 'pairs');
n++;
assert.equal(nodes.get('secondaryPopulation').disabled, false);
assert(!nodes.get('secondaryPopulation').children.find(o => o.value === 'deflected').disabled);
assert(nodes.get('secondaryPopulation').children.find(o => o.value === 'deflected').textContent
    .includes('(not available)'));
n++;
assert.equal(nodes.get('diagnosticRawSource').children.length, 4);
n++;
assert.equal(nodes.get('diagnosticRawDownload').disabled, false);
n++;
nodes.get('secondaryAngle').value = 'global';
nodes.get('secondaryColor').value = 'log';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert.equal(nodes.get('secondaryHeatmapPlot').hidden, false);
n++;
const jointStats = () => nodes.get('secondaryHeatmapStats').attributes['aria-label'];
assert(jointStats().includes('Mean (records): 0.4500: 0.3188'));
n++;
assert(jointStats().includes('Min (record): 0.1000: 0.0100'));
n++;
assert(nodes.get('secondaryContext').textContent.includes('8 / 8 records in the log plot'));
n++;
assert(nodes.get('secondaryContext').textContent.includes('independent of collision playback'));
n++;
const finalStats = jointStats(),
    finalContext = nodes.get('secondaryContext').textContent;
nodes.get('fr').value = '1';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert.equal(jointStats(), finalStats);
assert.equal(nodes.get('secondaryContext').textContent, finalContext);
n++;
for (const angle of ['global', 'folded'])
    for (const color of ['log', 'linear']) {
        nodes.get('secondaryAngle').value = angle;
        nodes.get('secondaryColor').value = color;
        vm.runInContext('drawSecondaryKinematics()', sandbox);
        n++;
        assert(nodes.get('secondaryHeatmap').attributes['aria-label'].includes(color +
            ' count color scale'));
        n++;
        assert(nodes.get('secondaryContext').textContent.includes(angle === 'folded' ?
            'Nearest-axis angle' : 'Common +z polar angle'));
        n++;
        assert(jointStats().includes('Physical marginals, not log-coordinate moments'));
        n++;
    }
nodes.get('secondaryPopulation').value = 'deflected';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(nodes.get('secondaryHeatmapPlot').hidden && nodes.get('secondaryHeatmapStats').hidden && !
    nodes.get('secondaryUnavailable').hidden);
n++;
const deflectedFixture = structuredClone(diagnosticAdaptersFixture.secondary);
deflectedFixture.populations.deflected = deflectedFixture.populations.pairs;
deflectedFixture.populations.pairs = structuredClone(deflectedFixture.populations.combined);
Object.assign(deflectedFixture.evaluated_flags, {
    do_pairs: 0,
    load_events: 1,
    do_bhabhas: 0
});
deflectedFixture.record_count_verification.available = false;
deflectedFixture.record_count_verification.expected_records = null;
deflectedFixture.populations.deflected.label =
    'Imported/scattered particles after beam-field tracking';
deflectedFixture.populations.deflected.normalization.completeness_verified = false;
sandbox.deflectedDiagnosticFixture = deflectedFixture;
vm.runInContext(
    'secondaryKinematics=ColliderSecondaryKinematics.validate(deflectedDiagnosticFixture,newDiagnosticFixture.refs);initializeSecondaryControls();drawSecondaryKinematics()',
    sandbox);
n++;
assert.equal(nodes.get('secondaryPopulation').value, 'deflected');
assert(nodes.get('secondaryHeatmapPlot').hidden);
n++;
assert(nodes.get('secondaryUnavailable').textContent.includes('No authenticated event identities'));
n++;
vm.runInContext(
    'monochromatization=null;secondaryKinematics=null;initializeSecondaryControls();drawMonochromatizationInputs();drawDiagnostics(1);drawSecondaryKinematics()',
    sandbox);
n++;
assert.equal(monoRows().get('Ideal monochromatization λ [1]'), 'unavailable');
n++;
assert(nodes.get('secondaryHeatmapPlot').hidden);
assert(nodes.get('secondaryUnavailable').textContent.includes('No authenticated'));
n++;
// Same authenticated raw source shared by adapters counts once; distinct receipts and sizes cannot alias.
sandbox.artifactManifest = {
    frames: dm.frames,
    analysis: null,
    diagnostic_analysis_sources: [diagnosticAdaptersFixture.refs[0], {
        ...diagnosticAdaptersFixture.refs[0]
    }]
};
vm.runInContext('man=artifactManifest', sandbox);
assert.equal(vm.runInContext('scientificArtifactRefs().length', sandbox), 1);
n++;
vm.runInContext('man.diagnostic_analysis_sources[1].bytes++', sandbox);
reject(() => vm.runInContext('scientificArtifactRefs()', sandbox));
vm.runInContext('man.diagnostic_analysis_sources=[{...newDiagnosticFixture.refs[0],bytes:0}]',
    sandbox);
assert.equal(vm.runInContext('scientificArtifactRefs()[0].bytes', sandbox), 0);
n++;
vm.runInContext('man.monochromatization=man.diagnostic_analysis_sources[0]', sandbox);
reject(() => vm.runInContext('scientificArtifactRefs()', sandbox));
vm.runInContext('man=fixtureManifest;delete man.diagnostic_analysis_sources', sandbox);

// Snapshot scope is state, never an interval birth count or sum of final records.
assert(html.includes('id="secondaryScope"') && html.includes('Current frame'));
n++;
assert(html.includes('#secondaryHeatmapStats{width:390px;font-size:10px;top:auto;bottom:60px}'));
n++;
sandbox.particleSnapshotFixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_secondary_frames.json'), 'utf8'));
vm.runInContext(
    'secondaryKinematics=ColliderSecondaryKinematics.validate(particleSnapshotFixture.secondary,particleSnapshotFixture.refs,particleSnapshotFixture.context);initializeSecondaryControls()',
    sandbox);
n++;
assert.equal(nodes.get('secondaryScope').value, 'total');
n++;
nodes.get('secondaryScope').value = 'frame';
nodes.get('fr').value = '0';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(!nodes.get('secondaryHeatmapPlot').hidden && !nodes.get('secondaryHeatmapStats').hidden);
n++;
assert(jointStats().includes('Mean (records): unavailable: unavailable'));
n++;
assert(nodes.get('secondaryContext').textContent.includes('0 / 0 records in the log plot'));
n++;
assert(nodes.get('secondaryContext').textContent.includes('including completed trajectories'));
n++;
assert(nodes.get('secondaryContext').textContent.includes('not a sum of snapshots'));
n++;
nodes.get('fr').value = '1';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(jointStats().includes('Snapshot 1 tracked-record marginal statistics'));
n++;
assert(jointStats().includes('Mean (records): 0.4500: 0.3188'));
n++;
const frameBounds = vm.runInContext(
    'ColliderSecondaryKinematics.axisBounds(secondaryKinematics,"pairs",false)', sandbox);
nodes.get('secondaryScope').value = 'total';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(jointStats().includes('Final stored-record marginal statistics'));
n++;
assert.deepEqual(vm.runInContext(
        'ColliderSecondaryKinematics.axisBounds(secondaryKinematics,"pairs",false)', sandbox),
    frameBounds);
n++;
const totalStats = jointStats();
nodes.get('fr').value = '2';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert.equal(jointStats(), totalStats);
n++;
nodes.get('secondaryScope').value = 'frame';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(nodes.get('secondaryHeatmapPlot').hidden && nodes.get('secondaryUnavailable').textContent
    .includes('Synthetic unsupported snapshot'));
n++;
vm.runInContext(
    'secondaryKinematics=newDiagnosticFixture.secondary;initializeSecondaryControls();$("secondaryScope").value="frame";drawSecondaryKinematics()',
    sandbox);
n++;
assert(nodes.get('secondaryHeatmapPlot').hidden && nodes.get('secondaryUnavailable').textContent
    .includes('No authenticated particle snapshots'));
n++;
nodes.get('secondaryScope').value = 'total';
vm.runInContext('drawSecondaryKinematics()', sandbox);
n++;
assert(!nodes.get('secondaryHeatmapPlot').hidden);
n++;

console.log(JSON.stringify({
    status: "PASS_FOR_STATED_TARGET",
    checks: n,
    scope: "JS contract and viewer syntax; no browser QA"
}));


