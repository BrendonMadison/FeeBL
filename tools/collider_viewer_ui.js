"use strict";
/** Pure display helpers. Values retain the caller's physical units. */
(function(root) {
    /** Round linear major ticks, bounded by pixel width; invalid ranges throw. */
    function linearTicks(low, high, pixels = 500) {
        if (![low, high, pixels].every(Number.isFinite) || high <= low) throw Error("Invalid tick range");
        const count = Math.max(2, Math.min(8, Math.floor(pixels / 88)));
        const rough = (high - low) / count, decade = 10 ** Math.floor(Math.log10(rough));
        const step = [1, 2, 2.5, 5, 10].find(v => v * decade >= rough) * decade;
        const ticks = [];
        for (let i = Math.ceil(low / step); i <= Math.floor(high / step); i++) ticks.push(i * step);
        return {values: ticks, step};
    }
    /** Major log ticks lie at powers of ten; narrow intervals use 1, 2, 5 ticks. */
    function logTicks(low, high, pixels = 500) {
        if (!(low > 0) || !(high > low)) throw Error("Invalid logarithmic tick range");
        const span = Math.log10(high / low), stride = Math.max(1, Math.ceil(span / Math.max(2, pixels / 88)));
        const values = [];
        for (let exponent = Math.floor(Math.log10(low)); exponent <= Math.ceil(Math.log10(high)); exponent++) {
            if (span >= 2 && ((exponent % stride) + stride) % stride !== 0) continue;
            for (const factor of span < 2 ? [1, 2, 5] : [1]) {
                const value = factor * 10 ** exponent;
                if (value >= low && value <= high) values.push(value);
            }
        }
        return {values, step: null};
    }
    /** Tick precision follows spacing, so labels do not round to another position. */
    function tickLabel(value, step) {
        if (value === 0 || Object.is(value, -0)) return "0";
        if (Math.abs(value) < .001 || Math.abs(value) >= 1e5) return value.toExponential(1).replace(".0e", "e");
        const exponent = step ? Math.floor(Math.log10(step)) : 0;
        const halfStep = step && Math.abs(step / 10 ** exponent - 2.5) < 1e-10;
        const digits = step ? Math.max(0, Math.min(10, -exponent + (halfStep ? 1 : 0))) : 3;
        return value.toFixed(digits).replace(/(\.[0-9]*?)0+$/, "$1").replace(/\.$/, "");
    }
    /** Page shortcuts leave focused controls and editable content to the browser. */
    function acceptsFrameShortcut(event) {
        return ["ArrowLeft", "ArrowRight"].includes(event.key) && !event.defaultPrevented &&
            !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey &&
            !event.target?.closest?.("input, select, textarea, button, [contenteditable=true]");
    }
    /** Convert native field labels only; the source values and color mapping stay native. */
    function fieldDisplay(maximum, unit) {
        return unit === "GV/nm" ? {maximum: maximum * 1e9, unit: "GV/m"} :
            unit === "V/m" ? {maximum: maximum * 1e-9, unit: "GV/m"} : {maximum, unit};
    }
    const api = {linearTicks, logTicks, tickLabel, acceptsFrameShortcut, fieldDisplay};
    root.ColliderViewerUI = api;
    if (typeof module !== "undefined") module.exports = api;
})(globalThis);

