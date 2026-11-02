"use strict";
const assert = require('node:assert/strict'),
    fs = require('node:fs'),
    path = require('node:path');
const f = require('./collider_fields.js');
let checks = 0;
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname,
    'testdata/collider_snapshot_fields.json'), 'utf8'));
const v = f.validate(fixture.fields, fixture.manifest),
    original = JSON.stringify(v.data);
assert.deepEqual(f.beamLayers(v).map(s => s.id), ['beam1', 'beam2']);
checks++;
assert.equal(f.BOTH_BEAMS, '__both_beams__');
checks++;
assert(!/^[A-Za-z]/.test(f.BOTH_BEAMS));
checks++;
assert.deepEqual(f.beamLayers(null), []);
checks++;
for (const mutate of [s => s.pop(), s => s[1].unit = 'V/m', s => s[1].quantity = 'other magnitude',
        s => s[1].kind = 'external', s => s[1].source_beam = 1, s => s[1].target_beam = 2, s =>
        s[1].id = 'other'
    ]) {
    const copy = structuredClone(v);
    mutate(copy.data.sources);
    assert.equal(f.beamLayers(copy).length, 0);
    checks++;
}
const reversed = structuredClone(v);
reversed.data.sources.reverse();
assert.deepEqual(f.beamLayers(reversed).map(s => s.id), ['beam1', 'beam2']);
checks++;
const rgb = css => css.startsWith('#') ? [1, 3, 5].map(i => parseInt(css.slice(i, i + 2), 16)) : css
    .match(/\d+/g).map(Number);

function blend(back, front, alpha, operation, quantize = true) {
    return back.map((b, i) => {
        const s = front[i],
            mixed = operation === 'multiply' ? b * s / 255 : operation === 'screen' ? 255 -
            (255 - b) * (255 - s) / 255 : s;
        const c = b * (1 - alpha) + alpha * mixed;
        return quantize ? Math.round(c) : c;
    });
}
// One-pixel reference canvas implements the specified standard blend equations.
// It executes drawPlane, including alpha, palette and source-position mapping.
function pixelContext(background, point = [.5, .5]) {
    const stack = [],
        g = {
            pixel: background.slice(),
            calls: [],
            fillStyle: '#000000',
            globalAlpha: 1,
            globalCompositeOperation: 'source-over',
            save() {
                stack.push({
                    fillStyle: this.fillStyle,
                    globalAlpha: this.globalAlpha,
                    globalCompositeOperation: this.globalCompositeOperation
                })
            },
            restore() {
                Object.assign(this, stack.pop())
            },
            beginPath() {},
            rect() {},
            clip() {},
            fillRect(...r) {
                this.calls.push({
                    r,
                    color: this.fillStyle,
                    alpha: this.globalAlpha,
                    composite: this.globalCompositeOperation
                });
                if (point[0] >= r[0] && point[0] < r[0] + r[2] && point[1] >= r[1] && point[1] < r[
                        1] + r[3]) this.pixel = blend(this.pixel, rgb(this.fillStyle), this
                    .globalAlpha, this.globalCompositeOperation)
            },
            clearRect() {},
            fillText() {}
        };
    return g;
}
const view = {
    L: 0,
    T: 0,
    W: 1,
    H: 1,
    horizontal: 'z',
    vertical: 'x',
    px: x => x,
    py: y => y
};
const plane = value => ({
    longitudinal: true,
    axes: ['z', 'x'],
    samples: 1,
    transverse_edges_um: [0, 1],
    strips: [{
        z_edges_um: [0, 1],
        values: [value]
    }]
});
let testedPixels = 0,
    maximumOrderDifference = 0;
for (const panel of ['#ffffff', '#1c2024']) {
    const style = f.layerStyle(panel),
        expected = panel === '#ffffff' ? 'multiply' : 'screen';
    assert.equal(style.composite, expected);
    checks++;
    for (const id of ['beam1', 'beam2']) {
        const neutral = expected === 'multiply' ? [255, 255, 255] : [0, 0, 0];
        assert.deepEqual(rgb(f.layerColor(0, 10, id, style)), neutral);
        checks++;
        assert.deepEqual(rgb(f.layerColor(0, 0, id, style)), neutral);
        checks++;
        const g = pixelContext(style.background);
        f.drawPlane(g, plane(10), view, 1, 10, {
            sourceId: id,
            style
        });
        assert.notDeepEqual(g.pixel, style.background);
        checks++;
        assert.equal(g.globalCompositeOperation, 'source-over');
        checks++;
    }
    for (const a of [0, .1, .4, .75, 1])
        for (const x of [0, .01, .1, .33, .5, .9, 1])
            for (const y of [0, .01, .1, .33, .5, .9, 1]) {
                const forward = pixelContext(style.background),
                    reverse = pixelContext(style.background);
                f.drawPlane(forward, plane(x), view, a, 1, {
                    sourceId: 'beam1',
                    style
                });
                f.drawPlane(forward, plane(y), view, a, 1, {
                    sourceId: 'beam2',
                    style
                });
                f.drawPlane(reverse, plane(y), view, a, 1, {
                    sourceId: 'beam2',
                    style
                });
                f.drawPlane(reverse, plane(x), view, a, 1, {
                    sourceId: 'beam1',
                    style
                });
                const difference = Math.max(...forward.pixel.map((p, i) => Math.abs(p - reverse
                    .pixel[i])));
                maximumOrderDifference = Math.max(maximumOrderDifference, difference);
                assert(difference <= 1,
                    'Intermediate 8-bit rounding may differ by at most one channel level');
                const ca = rgb(f.layerColor(x, 1, 'beam1', style)),
                    cb = rgb(f.layerColor(y, 1, 'beam2', style));
                const exactA = blend(blend(style.background, ca, a, expected, false), cb, a,
                    expected, false);
                const exactB = blend(blend(style.background, cb, a, expected, false), ca, a,
                    expected, false);
                assert(exactA.every((p, i) => Math.abs(p - exactB[i]) < 1e-10));
                if (a === 0 || x === 0 && y === 0) assert.deepEqual(forward.pixel, style
                .background);
                testedPixels++;
            }
    for (const a of [0, .4, 1]) {
        const missing = pixelContext(style.background),
            zero = pixelContext(style.background);
        assert.equal(f.drawPlane(missing, null, view, a, 1, {
            sourceId: 'beam2',
            style
        }), false);
        f.drawPlane(zero, plane(0), view, a, 1, {
            sourceId: 'beam2',
            style
        });
        assert.deepEqual(zero.pixel, missing.pixel);
        checks++;
    }
    // The legend runs the same renderer composition on the same background, once.
    global.document = {
        documentElement: {}
    };
    global.getComputedStyle = () => ({
        getPropertyValue: () => '#111111'
    });
    for (const id of ['beam1', 'beam2'])
        for (const a of [0, .4, 1]) {
            const g = pixelContext([0, 0, 0], [23.5, 5]),
                canvas = {
                    width: 40,
                    height: 38,
                    getContext: () => g
                };
            f.layerLegend(canvas, 10, 'GV/nm', id, style, a);
            assert.deepEqual(g.pixel, blend(style.background, rgb(f.layerColor(10, 10, id, style)),
                a, expected));
            checks++;
            assert.equal(g.calls.length, 9);
            checks++;
        }
}
for (const mode of ['snapshot', 'encounter'])
    for (const key of ['zx', 'zy'])
        for (let k = 0; k < 2; k++) {
            const style = f.layerStyle('#ffffff'),
                vw = {
                    ...view,
                    vertical: key[1]
                };
            const maximum = Math.max(...f.beamLayers(v).map(s => f.scaleMaximum(v, k, s.id, 'run',
                key, mode)));
            const all = [],
                separate = [];
            for (const source of f.beamLayers(v)) {
                const p = f.plane(v, k, source.id, key, mode),
                    g = pixelContext(style.background),
                    single = pixelContext(style.background);
                f.drawPlane(g, p, vw, .4, maximum, {
                    sourceId: source.id,
                    style
                });
                f.drawPlane(single, p, vw, .4, maximum);
                all.push(...g.calls.map(c => c.r));
                separate.push(...single.calls.map(c => c.r));
                assert(single.calls.every(c => c.composite === 'source-over' && c.color.startsWith(
                    '#')));
                checks++;
            }
            assert.deepEqual(all, separate,
            'Both-layer painting must retain exact source geometry');
            checks++;
        }
assert.notEqual(f.scaleMaximum(v, 0, 'beam1', 'run', 'zx', 'snapshot'), f.scaleMaximum(v, 0,
    'beam2', 'run', 'zx', 'snapshot'));
checks++;
assert.equal(JSON.stringify(v.data), original, 'Rendering must not modify native samples');
checks++;
assert.throws(() => f.layerStyle('transparent'));
checks++;
console.log(JSON.stringify({
    status: 'PASS_FOR_STATED_TARGET',
    checks,
    pixel_cases: testedPixels,
    maximum_order_difference_8bit: maximumOrderDifference,
    scope: 'Blend equations and actual drawPlane/legend calls with quantized reference pixels; no browser rasterization claim'
}));
