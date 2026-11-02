"use strict";
const assert = require("node:assert/strict");
const ui = require("./collider_viewer_ui.js");
const ticks = ui.linearTicks(-2413.78, 2398.91, 300);
assert(ticks.values.length <= 4);
assert(ticks.values.every(v => Number.isInteger(v / ticks.step)));
assert.deepEqual(ui.logTicks(1e-10, 1e3, 500).values, [1e-9, 1e-6, 1e-3, 1, 1000]);
assert.equal(ui.tickLabel(.048, .002), "0.048");
assert.equal(ui.tickLabel(-0, 1), "0");
assert.equal(ui.tickLabel(.025, .025), "0.025");
assert.equal(ui.acceptsFrameShortcut({key:"ArrowRight",target:{closest:()=>null}}), true);
for (const control of ["INPUT", "SELECT", "TEXTAREA", "BUTTON"]) {
    assert.equal(ui.acceptsFrameShortcut({key:"ArrowRight",target:{closest:()=>control}}), false);
}
assert.equal(ui.acceptsFrameShortcut({key:"ArrowRight",ctrlKey:true}), false);
assert.deepEqual(ui.fieldDisplay(1.166e-7,"GV/nm"),{maximum:116.6,unit:"GV/m"});
assert.throws(()=>ui.linearTicks(0,0));
console.log("PASS: round ticks, field display units and focused-control shortcuts");


