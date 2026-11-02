/** Native scientific panels and their local display state.
 * Consumes validated frame/total diagnostics, authenticated spectrum models,
 * and secondary trajectories. Rendering changes neither selection nor weights.
 * Missing computations remain visibly absent; malformed artifacts fail loading.
 */
let monochromatization = null,
    secondaryKinematics = null,
    eventDeflections = null,
    lastSecondaryRender = null;
const eventPlotNodes = new Map();

function scientificFileLimit(ref) {
    return ref === man.fields ? ColliderFields.LIMITS.fileBytes : ref === man.monochromatization ?
        ColliderMonochromatization.LIMITS.fileBytes : ref === man.secondary_kinematics ?
        ColliderSecondaryKinematics.LIMITS.fileBytes : ref === man.event_deflections ?
        ColliderEventDeflections.LIMITS.fileBytes : 32 * 1024 * 1024
}

function rawDiagnosticRef(ref) {
    return (man.diagnostic_analysis_sources || []).includes(ref) && ![man.analysis, man
        .analysis_csv, man.diagnostics, man.fields, man.monochromatization, man
        .secondary_kinematics, man.event_deflections
    ].includes(ref)
}

function scientificArtifactRefs() {
    const refs = man.analysis ? [man.analysis, man.analysis_csv, ...(man.analysis_sources || [])] :
        [];
    for (const key of ["diagnostics", "fields", "monochromatization", "secondary_kinematics",
            "event_deflections"
        ])
        if (man[key]) refs.push(man[key]);
    refs.push(...(man.diagnostic_analysis_sources || []));
    const names = new Map(),
        frameNames = new Set(man.frames.flatMap(f => [f.beam1.path, f.beam2.path]));
    const primary = [man.analysis, man.analysis_csv, man.diagnostics, man.fields, man
        .monochromatization, man.secondary_kinematics, man.event_deflections
    ].filter(Boolean).map(r => r.path);
    if (new Set(primary).size !== primary.length) throw new Error(
        "Scientific roles require distinct artifacts");
    let total = man.frames.reduce((sum, f) => sum + f.beam1.bytes + f.beam2.bytes, 0);
    for (const ref of refs) {
        const limit = scientificFileLimit(ref);
        ColliderContract.fileRef(ref, limit, rawDiagnosticRef(ref));
        if (frameNames.has(ref.path)) throw new Error(
        "Scientific artifact aliases a display frame");
        const previous = names.get(ref.path);
        if (previous) {
            if (previous.sha256 !== ref.sha256 || previous.bytes !== ref.bytes) throw new Error(
                "Conflicting scientific artifact receipt");
            continue
        }
        names.set(ref.path, ref);
        total += ref.bytes;
    }
    if (total > ColliderContract.LIMITS.totalBytes) throw new Error(
        "Dataset plus scientific observables exceeds byte bound");
    return [...names.values()];
}
/** Verify every optional scientific artifact before binding its renderer; errors reject the dataset. */
async function loadScientific() {
    scientific = null;
    diagnostics = null;
    nativeFields = null;
    monochromatization = null;
    secondaryKinematics = null;
    eventDeflections = null;
    lastSecondaryRender = null;
    const refs = scientificArtifactRefs();
    let monoRaw = null,
        secondaryRaw = null,
        frameSnapshotsRaw = null,
        eventRaw = null,
        trajectoryRaw = null,
        identityRaw = null,
        inputText = null,
        summaryText = null;
    for (const ref of refs) {
        const limit = scientificFileLimit(ref);
        const bytes = await ColliderContract.bytesHash(await readBytes(ref.path, limit), ref,
            limit, rawDiagnosticRef(ref));
        const text = () => new TextDecoder("utf-8", {
            fatal: true
        }).decode(bytes);
        if (ref === man.analysis) scientific = ColliderContract.analysis(JSON.parse(text()));
        if (ref === man.diagnostics) diagnostics = ColliderContract.diagnostics(JSON.parse(
        text()), man);
        if (ref === man.fields) nativeFields = ColliderFields.validate(JSON.parse(text()), man);
        if (ref === man.monochromatization) monoRaw = JSON.parse(text());
        if (ref === man.secondary_kinematics) secondaryRaw = JSON.parse(text());
        if (ref === man.event_deflections) eventRaw = JSON.parse(text());
        if (ref.path === "gui_imported_trajectories.json") trajectoryRaw = JSON.parse(text());
        if (ref.path === "event_identity.json") identityRaw = JSON.parse(text());
        if (ref.path === "event.ini") inputText = text();
        if (ref.path === "out.dat") summaryText = text();
        if (ref.path === "gui_secondary_frames.json") frameSnapshotsRaw = JSON.parse(text());
    }
    if (diagnostics) ColliderContract.bindPeakReference(diagnostics, man, scientific);
    if (monoRaw) monochromatization = ColliderMonochromatization.validate(monoRaw, refs, {
        manifest: man,
        analysis: scientific,
        diagnostics
    });
    if (secondaryRaw) secondaryKinematics = ColliderSecondaryKinematics.validate(secondaryRaw,
        refs, {
            manifest: man,
            frameSnapshots: frameSnapshotsRaw
        });
    if (eventRaw) eventDeflections = ColliderEventDeflections.validate(eventRaw, refs, {
        manifest: man,
        trajectories: trajectoryRaw,
        identity: identityRaw,
        inputText,
        summaryText
    });
    initializeSecondaryControls();
    $("monoDownload").disabled = !monochromatization;
    initializeFieldControls();
    const frequency = scientific?.normalization.crossing_frequency_hz ?? null;
    $("spectrumUnits").options[1].disabled = frequency === null;
    $("spectrumUnits").value = frequency === null ? "native" : "persecond";
    $("analysisDownload").disabled = !scientific && !diagnostics;
    $("spectrumPanel").hidden = false;
}
// Statistics use the shared native-weight adapter, independent of the displayed y normalization.
function fitStatisticsCanvas(canvas) {
    const width = Math.round(canvas.clientWidth),
        height = Math.round(canvas.clientHeight);
    if (width > 0 && height > 0) {
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height
    }
}

function statisticNumber(value) {
    return Number.isFinite(value) ? fmt(value, 4) : "—"
}

function renderPlotStatistics(canvasID, stats, title, unit, weightUnit) {
    const box = $(canvasID + "Stats");
    box.replaceChildren();
    box.hidden = false;
    $(canvasID + "StatsDetails").hidden = false;
    $(canvasID + "StatsNotes").textContent = "Selected bin weight: " + statisticNumber(stats
            .selected_weight) +
        "; underflow / overflow excluded from binned statistics: " + statisticNumber(stats.excluded
            .underflow) + " / " + statisticNumber(stats.excluded.overflow) +
        "; outside-window bin weight: " + statisticNumber(stats.excluded.outside_window_weight) +
        "; cut-straddling bin weight excluded: " + statisticNumber(stats.excluded
            .cut_straddling_weight) + " [" + weightUnit + "]. " + stats.notes.join(" ");
    const heading = document.createElement("strong");
    heading.textContent = title + (unit === "1" ? "" : " [" + unit + "]");
    box.append(heading);
    if (!stats.available) {
        const text = document.createElement("p");
        text.textContent = "Statistics unavailable: " + stats.reason;
        box.append(text);
        box.setAttribute("aria-label", heading.textContent + ". " + text.textContent);
        return;
    }
    const native = stats.provenance.moments === "native-all-pairs",
        dl = document.createElement("dl");
    const mode = !stats.mode_intervals.length ? "unavailable" : stats.mode_intervals.length === 1 ?
        statisticNumber(stats.mode_intervals[0].low) + "–" + statisticNumber(stats.mode_intervals[0]
            .high) : stats.mode_intervals.length + " tied bins";
    const varianceUnit = unit === "1" ? "" : " [" + unit + "²]";
    const rows = [
        ["Mean (" + (native ? "native" : "bins") + ")", statisticNumber(stats.mean)],
        ["Mode (density bin)", mode],
        ["Variance" + varianceUnit, statisticNumber(stats.variance)],
        ["Min (bin edge)", statisticNumber(stats.minimum)],
        ["Max (bin edge)", statisticNumber(stats.maximum)]
    ];
    for (const [label, value] of rows) {
        const key = document.createElement("dt"),
            val = document.createElement("dd");
        key.textContent = label;
        val.textContent = value;
        dl.append(key, val)
    }
    box.append(dl);
    const note = document.createElement("p"),
        cropped = stats.provenance.scope === "whole-contained-window-bins";
    note.textContent = (cropped ? "Whole bins inside the peak window. " : "Full histogram. ") +
        (native ? "Mean/variance: native, including flows. Mode/edges: in-range bins." :
            "Mean/variance: bin-centre estimates; flows excluded.") +
        (cropped ? " Cut-straddling bins excluded." : "");
    if (stats.native_moments_status === "rejected") note.textContent +=
        " Native moments unavailable; binned fallback.";
    box.append(note);
    box.setAttribute("aria-label", heading.textContent + ". " + rows.map(r => r.join(": ")).join(
        "; ") + ". " + note.textContent);
    const selection = " Selected bin weight: " + statisticNumber(stats.selected_weight) +
        "; cut-straddling bin weight excluded: " + statisticNumber(stats.excluded
            .cut_straddling_weight) + ".";
    if (stats.mode_intervals.length > 1) box.setAttribute("title", "No unique modal bin; " + stats
        .mode_intervals.length + " bins share the maximum weight density." + selection);
    else box.setAttribute("title",
        "Mode is the maximum weight per unit-x bin; min and max are occupied-bin bounds, not raw extrema." +
        selection);
}

function spectrumStatistics(a, scope, view) {
    const h = {
        bins: a.bins.map(b => ({
            low: b.W_low_GeV,
            high: b.W_high_GeV,
            weight: b.weight_m2_per_crossing
        })),
        underflow: a.normalization.underflow_m2_per_crossing,
        overflow: a.normalization.overflow_m2_per_crossing
    };
    const options = {};
    if (view === "peak") options.window = [a.plot.peak.x_min_GeV, a.plot.x_max_GeV];
    const native = scope === "total" && view === "full" && a === scientific ? a.native_statistics :
        null;
    if (native) options.nativeMoments = {
        mean: native.mean_W_GeV,
        rms: native.full_RMS_W_GeV,
        validated: true
    };
    return ColliderStatistics.histogram(h, options);
}

function peakPlotSpectrum(a, scope, k = +$("fr").value) {
    const p = diagnosticRecord(scope, k)?.peak,
        h = p?.available ? p.luminosity_spectrum : null;
    if (!a || !h) return a;
    const total = a.normalization.total_spectrum_m2_per_crossing,
        frequency = a.normalization.crossing_frequency_hz,
        factor = frequency === null ? null : frequency * 1e-4;
    return {
        ...a,
        _nativePeakHistogram: h,
        bins: h.bins.map(b => ({
            W_low_GeV: b.low,
            W_high_GeV: b.high,
            weight_m2_per_crossing: b.weight,
            dL_dW_m2_per_crossing_GeV: b.weight / (b.high - b.low),
            dL_dW_cm2_s_GeV: factor === null ? null : b.weight / (b.high - b.low) *
                factor,
            shape_density_per_GeV: total > 0 ? b.weight / (b.high - b.low) / total :
                null
        })),
        plot: {
            ...a.plot,
            peak: {
                ...a.plot.peak,
                x_min_GeV: p.selection.minimum_W_GeV
            },
            x_max_GeV: h.bins.at(-1).high
        },
        normalization: {
            ...a.normalization,
            underflow_m2_per_crossing: h.underflow,
            overflow_m2_per_crossing: h.overflow
        }
    };
}

function peakResolutionNote(a, scope) {
    const sparse = scope === "total" && a === scientific;
    const bins = a._nativePeakHistogram?.bins || (sparse ? a.bins.map(b => ({
            low: b.W_low_GeV,
            high: b.W_high_GeV,
            weight: b.weight_m2_per_crossing
        })) :
        diagnosticRecord(scope)?.luminosity_spectrum?.bins || []);
    const low = a.plot.peak.x_min_GeV,
        high = a.plot.x_max_GeV;
    const windowBins = bins.filter(b => b.low >= low && b.high <= high);
    if (!windowBins.length)
    return "Peak resolution unavailable: no whole exported bins lie inside this window.";
    const widths = windowBins.map(b => b.high - b.low),
        minimum = Math.min(...widths),
        maximum = Math.max(...widths);
    const same = Math.abs(maximum - minimum) <= 1e-9 * Math.max(maximum, 1e-12);
    const mev = v => Number((v * 1000).toPrecision(6)).toString();
    const occupied = windowBins.filter(b => b.weight > 0).length;
    return (a._nativePeakHistogram ? "Exact 98%-cut native peak histogram. " : "") + "Peak bin " + (
            same ? "width: " : "width range: ") + mev(minimum) + (same ? "" : "–" + mev(maximum)) +
        " MeV; " +
        occupied + " occupied / " + windowBins.length + " exported whole-window bins. " +
        (sparse ? "The native total export lists occupied bins only." :
            "Counts include exported zero-weight bins.");
}

/** Draw authenticated integrated or interval luminosity bins with the selected normalization. */
function drawSpectra() {
    const a = selectedSpectrum(),
        scope = $("spectrumScope").value,
        mode = $("spectrumUnits").value,
        peakA = peakPlotSpectrum(a, scope);
    $("spectrumContext").textContent = intervalLabel(scope) + ". " + (scope === "frame" ?
            "Interval contributions, not cumulative snapshots. " : "") +
        "Native luminosity weights only.";
    $("spectrumUnavailable").hidden = Boolean(a);
    $("spectrumFull").hidden = !a;
    $("spectrumPeak").hidden = !a || a.definition.nominal_W_GeV === null;
    $("spectrumPeakResolution").hidden = $("spectrumPeak").hidden;
    $("spectrumPeakResolution").textContent = $("spectrumPeak").hidden ? "" : peakResolutionNote(
        peakA, scope);
    const peakFit = peakFitRecord(scope);
    $("spectrumPeakFitNote").hidden = $("spectrumPeak").hidden;
    $("spectrumPeakFitNote").textContent = peakFitStatus(peakFit) + (peakFit
        ?.candidate_display_allowed ? " Green dashed curve: " + (isBetaFit() ?
            "beta-plus-delta convolved with Gaussian, fitted to the full spectrum and evaluated on the exact peak-bin edges" :
            (peakFit.available ? "Gaussian peak fit" : "Gaussian shape estimate") +
            " from the exact-selected native peak histogram") + "; Gaussian response σ = " +
        fmt(peakFit.candidate.sigma_W_GeV, 6) + " GeV." : "");
    $("spectrumPeakFitNote").classList.toggle("shape-mismatch", peakFit?.candidate_status ===
        "resolved_shape_mismatch");
    for (const id of ["spectrumFull", "spectrumPeak"]) {
        $(id + "Plot").hidden = $(id).hidden;
        $(id + "Stats").hidden = $(id).hidden;
        $(id + "StatsDetails").hidden = $(id).hidden
    }
    if (!a) {
        $("spectrumUnavailable").textContent = "Unavailable: " + (scope === "frame" ? absentReason(
            "luminosity_spectrum") : "No authenticated native total spectrum was exported.");
        $("spectrumSummary").textContent = "";
        $("spectrumLimits").textContent = "";
        return;
    }
    const key = mode === "persecond" ? "dL_dW_cm2_s_GeV" : mode === "shape" ?
        "shape_density_per_GeV" : "dL_dW_m2_per_crossing_GeV";
    const unit = mode === "persecond" ? "cm⁻² s⁻¹ GeV⁻¹" : mode === "shape" ?
        "GeV⁻¹ (all flow in denominator)" : "m⁻² crossing⁻¹ GeV⁻¹";
    for (const [id, view] of [
            ["spectrumFull", "full"],
            ["spectrumPeak", "peak"]
        ]) {
        const c = $(id);
        if (c.hidden) continue;
        fitStatisticsCanvas(c);
        const plotted = view === "peak" ? peakA : a;
        renderPlotStatistics(id, spectrumStatistics(plotted, scope, view), (view === "full" ?
            "Full" : "Peak") + " · " + (scope === "frame" ? "frame interval" :
            "total collision"), "GeV", "m⁻²/crossing");
        const g = c.getContext("2d"),
            L = 105,
            R = 30,
            T = 45,
            B = 48,
            W = c.width - L - R,
            H = c.height - T - B;
        const low = plotted.plot[view].x_min_GeV,
            high = plotted.plot.x_max_GeV;
        const rows = plotted.bins.filter(b => b.W_high_GeV > low && b.W_low_GeV < high && b[key] >
            0);
        const log = view === "full",
            curve = view === "peak" ? peakCurve(scope, +$("fr").value, a, mode) : null;
        let yl = log && rows.length ? Math.log10(Math.min(...rows.map(b => b[key]))) - .15 : 0;
        let yh = rows.length ? (log ? Math.log10(Math.max(...rows.map(b => b[key]))) + .15 : Math
            .max(...rows.map(b => b[key])) * 1.1) : 1;
        if (curve) yh = Math.max(yh, curve.maximum * 1.1);
        if (yh <= yl) yh = yl + 1;
        const px = v => L + (v - low) / (high - low) * W,
            py = v => T + H - ((log ? Math.log10(v) : v) - yl) / (yh - yl) * H;
        const cs = getComputedStyle(document.documentElement),
            ink = cs.getPropertyValue("--ink"),
            bg = cs.getPropertyValue("--panel");
        g.fillStyle = bg;
        g.fillRect(0, 0, c.width, c.height);
        g.strokeStyle = cs.getPropertyValue("--line");
        g.fillStyle = ink;
        g.font = "12px sans-serif";
        g.textAlign = "center";
        const xTicks = ColliderViewerUI.linearTicks(low, high, W);
        for (const x of xTicks.values) {
            const X = px(x);
            g.beginPath();
            g.moveTo(X, T);
            g.lineTo(X, T + H);
            g.stroke();
            g.fillText(ColliderViewerUI.tickLabel(x, xTicks.step), X, T + H + 19)
        }
        g.textAlign = "right";
        const yTicks = log ? ColliderViewerUI.logTicks(10 ** yl, 10 ** yh, H) :
            ColliderViewerUI.linearTicks(yl, yh, H);
        for (const y of yTicks.values) {
            const Y = py(y);
            g.beginPath();
            g.moveTo(L, Y);
            g.lineTo(L + W, Y);
            g.stroke();
            g.fillText(ColliderViewerUI.tickLabel(y, yTicks.step), L - 8, Y + 4)
        }
        g.save();
        g.beginPath();
        g.rect(L, T, W, H);
        g.clip();
        g.strokeStyle = "#0072b2";
        g.lineWidth = 2;
        for (const b of rows) {
            g.beginPath();
            g.moveTo(px(b.W_low_GeV), py(b[key]));
            g.lineTo(px(b.W_high_GeV), py(b[key]));
            g.stroke()
        }
        if (curve) {
            g.strokeStyle = "#009e73";
            g.lineWidth = 2;
            g.setLineDash([7, 5]);
            g.beginPath();
            if (curve.bins) {
                for (const b of curve.bins)
                    if (b.high > low && b.low < high) {
                        g.moveTo(px(b.low), py(b.density));
                        g.lineTo(px(b.high), py(b.density))
                    }
            } else {
                const start = Math.max(low, curve.record.fit_lower_W_GeV),
                    stop = Math.min(high, curve.record.fit_upper_W_GeV);
                for (let j = 0; j <= 256; j++) {
                    const x = start + (stop - start) * j / 256,
                        y = curve.at(x);
                    if (j === 0) g.moveTo(px(x), py(y));
                    else g.lineTo(px(x), py(y))
                }
            }
            g.stroke();
            g.setLineDash([])
        }
        if (a.definition.nominal_W_GeV !== null) {
            g.strokeStyle = "#d55e00";
            g.setLineDash([5, 4]);
            g.beginPath();
            g.moveTo(px(a.definition.nominal_W_GeV), T);
            g.lineTo(px(a.definition.nominal_W_GeV), T + H);
            g.stroke()
        }
        g.restore();
        g.textAlign = "left";
        g.fillText((log ? "Full / log y" : "Peak / linear y / W ≥ 0.98 Wnom") + " · dL/dW [" +
            unit + "]", L, 24);
        g.textAlign = "center";
        g.fillText("Native W [GeV]" + (a.definition.nominal_W_GeV === null ?
                " · design Wnom unavailable" : " — dashed line: design Wnom"), L + W / 2, c
            .height - 8);
        if (!rows.length) g.fillText("No sampled records in this view", L + W / 2, T + H / 2);
    }
    const n = a.normalization;
    $("spectrumSummary").textContent = (scope === "frame" ? "Interval" : "Spectrum") +
        " luminosity = " + fmt(n.total_spectrum_m2_per_crossing, 5) +
        " m⁻²/crossing; underflow / overflow = " + fmt(n.underflow_m2_per_crossing, 3) + " / " +
        fmt(n.overflow_m2_per_crossing, 3) + " m⁻²/crossing. " +
        (n.total_summary_m2_per_crossing === null ? "" : ("Native out.dat total = " + fmt(n
            .total_summary_m2_per_crossing, 5) + "; ")) +
        (n.comparison_status === "AUTHENTICATED_NATIVE_INTERVAL_WEIGHTS" ?
            "Native interval weights" : n.comparison_status) +
        (n.total_L_cm2_s === null ?
            ". Per-second normalization unavailable: collision frequency was not supplied." : ". " +
            (scope === "frame" ? "Interval contribution to " : "") + (n.allocation_scope ===
                "machine_total_unallocated" ? "machine-total L (IP allocation unknown)" : "per-IP L"
                ) +
            " = " + fmt(n.total_L_cm2_s, 5) + " cm⁻² s⁻¹ at supplied " + n.crossing_frequency_hz +
            " crossings/s.");
    $("spectrumLimits").textContent = a.limitations.map(t => monochromatization?.peak_fit && t
        .includes("Gaussian-core fit is unavailable") ? (isBetaFit() ?
            "The beta–Gaussian response fit is supplied separately; full RMS includes the tail and is not the response core width." :
            "Gaussian shape fits are supplied separately; native full RMS remains tail-sensitive and is not a Gaussian width."
            ) : t).join(" ");
}
$("spectrumUnits").addEventListener("change", drawSpectra);
$("analysisDownload").addEventListener("click", () => {
    const useDiagnostics = $("spectrumScope").value === "frame" || !scientific;
    const data = useDiagnostics ? diagnostics : scientific;
    if (data) download(useDiagnostics ? "gui_diagnostics.json" : "collider_analysis.json",
        JSON.stringify(data, null, 2), "application/json");
});

function diagnosticRecord(scope, k = +$("fr").value) {
    return diagnostics ? (scope === "frame" ? diagnostics.frames[k] : diagnostics.total) : null
}

function intervalLabel(scope, k = +$("fr").value) {
    const r = diagnosticRecord(scope, k);
    return scope === "total" ? "Total collision" : r ? "Frame interval · diagonals " + r
        .diagonal_start + "–" + r.diagonal_end : "Frame interval";
}

function absentReason(key) {
    return diagnostics?.unavailable_reasons[key] || "Not computed or exported in this dataset."
}

function tableRows(id, rows) {
    $(id).replaceChildren();
    for (const [label, value] of rows) {
        const tr = document.createElement("tr"),
            a = document.createElement("td"),
            b = document.createElement("td");
        a.className = "k";
        a.textContent = label;
        b.textContent = value;
        tr.append(a, b);
        $(id).append(tr);
    }
}

function derivedSpectrum(h, scope) {
    if (!h) return null;
    const n = scientific?.normalization,
        nominal = scientific?.definition.nominal_W_GeV || null;
    const freq = n?.crossing_frequency_hz ?? null,
        factor = freq === null ? null : freq * 1e-4;
    return {
        definition: {
            nominal_W_GeV: nominal
        },
        bins: h.bins.filter(b => b.weight > 0).map(b => ({
            W_low_GeV: b.low,
            W_high_GeV: b.high,
            dL_dW_m2_per_crossing_GeV: b.weight / (b.high - b.low),
            weight_m2_per_crossing: b.weight,
            dL_dW_cm2_s_GeV: factor === null ? null : b.weight / (b.high - b.low) *
                factor,
            shape_density_per_GeV: h.total_weight > 0 ? b.weight / (b.high - b.low) / h
                .total_weight : null
        })),
        plot: {
            full: {
                x_min_GeV: 0
            },
            peak: {
                x_min_GeV: nominal === null ? null : .98 * nominal
            },
            x_max_GeV: Math.max(1, ...h.bins.map(b => b.high), nominal === null ? 0 : nominal *
                1.005)
        },
        normalization: {
            total_spectrum_m2_per_crossing: h.total_weight,
            total_summary_m2_per_crossing: null,
            underflow_m2_per_crossing: h.underflow,
            overflow_m2_per_crossing: h.overflow,
            comparison_status: "AUTHENTICATED_NATIVE_INTERVAL_WEIGHTS",
            crossing_frequency_hz: freq,
            allocation_scope: n?.allocation_scope ?? null,
            total_L_cm2_s: factor === null ? null : h.total_weight * factor
        },
        limitations: [
            "Native collision-pair luminosity weights; frame intervals combine all diagonals since the preceding snapshot. No display-sample pairing or inferred uncertainty.",
            ...(scientific?.limitations || [
                "Design Wnom unavailable: explicit design-angle analysis is required for the peak crop."
            ])
        ]
    };
}

function selectedSpectrum() {
    const scope = $("spectrumScope").value;
    return scope === "total" && scientific ? scientific : derivedSpectrum(diagnosticRecord(scope)
        ?.luminosity_spectrum, scope);
}

/** Render bin-integrated diagnostic weights; range selection changes only the visible axes. */
function drawHistogram(canvas, h) {
    fitStatisticsCanvas(canvas);
    const g = canvas.getContext("2d"),
        L = 92,
        R = 20,
        T = 43,
        B = 50,
        W = canvas.width - L - R,
        H = canvas.height - T - B;
    const cs = getComputedStyle(document.documentElement),
        ink = cs.getPropertyValue("--ink");
    g.fillStyle = cs.getPropertyValue("--panel");
    g.fillRect(0, 0, canvas.width, canvas.height);
    const rows = h.bins.filter(b => b.weight > 0);
    const occupied = !h.forceLinear && $("distributionRange").value !== "full" && rows.length > 0;
    const low = occupied ? rows[0].low : h.bins[0].low,
        high = occupied ? rows[rows.length - 1].high : h.bins[h.bins.length - 1].high;
    const ymax = rows.length ? Math.max(...rows.map(b => b.weight)) * 1.12 : 1;
    const logX = !h.forceLinear && low > 0 && high / low >= 100,
        tx = v => logX ? Math.log10(v) : v;
    const px = v => L + (tx(v) - tx(low)) / (tx(high) - tx(low)) * W,
        py = v => T + H - v / ymax * H;
    g.font = "12px sans-serif";
    g.fillStyle = ink;
    g.strokeStyle = cs.getPropertyValue("--line");
    g.textAlign = "center";
    const xTicks = logX ? ColliderViewerUI.logTicks(low, high, W) : ColliderViewerUI.linearTicks(low, high, W);
    for (const v of xTicks.values) {
        const x = px(v);
        g.beginPath();
        g.moveTo(x, T);
        g.lineTo(x, T + H);
        g.stroke();
        g.fillText(ColliderViewerUI.tickLabel(v, xTicks.step), x, T + H + 20)
    }
    g.textAlign = "right";
    const yTicks = ColliderViewerUI.linearTicks(0, ymax, H);
    for (const v of yTicks.values) {
        const y = py(v);
        g.beginPath();
        g.moveTo(L, y);
        g.lineTo(L + W, y);
        g.stroke();
        g.fillText(ColliderViewerUI.tickLabel(v, yTicks.step), L - 8, y + 4)
    }
    g.strokeStyle = cs.getPropertyValue("--e1");
    g.lineWidth = 2;
    for (const b of rows) {
        g.beginPath();
        g.moveTo(px(b.low), py(b.weight));
        g.lineTo(px(b.high), py(b.weight));
        g.stroke()
    }
    g.fillStyle = ink;
    g.textAlign = "left";
    g.fillText("Bin weight [" + h.weight_unit + "]", L, 22);
    g.textAlign = "center";
    g.fillText(h.quantity + " [" + h.unit + "]" + (logX ? " · log x" : ""), L + W / 2, canvas
        .height - 8);
    if (!rows.length) g.fillText(h.emptyLabel || "No in-range entries in this interval", L + W / 2,
        T + H / 2);
}

function drawDistributions(k) {
    const scope = $("distributionScope").value,
        r = diagnosticRecord(scope, k);
    const split = diagnostics?.primary_species?.supported === true && !!r?.distributions
        .chi_e_electron && !!r?.distributions.chi_e_positron;
    $("chiCombinedPanel").hidden = split;
    $("chiElectronPanel").hidden = !split;
    $("chiPositronPanel").hidden = !split;
    $("distributionContext").textContent = intervalLabel(scope, k) +
        ". Only computed native distributions are drawn. " +
        (split ?
            "Primary χe uses the canonical e⁺e⁻ assignment: beam 1 electrons, beam 2 positrons. Counts are macroparticle kick evaluations before radiation, repeated across substeps; they are not physical particle yields." :
            "Primary species split unavailable; χe, when computed, combines both primary beams.");
    for (const [key, canvasID, noteID] of [
            ["chi_e", "chiECanvas", "chiENote"],
            ["chi_e_electron", "chiElectronCanvas", "chiElectronNote"],
            ["chi_e_positron", "chiPositronCanvas", "chiPositronNote"],
            ["chi_gamma", "chiGammaCanvas", "chiGammaNote"],
            ["photon_energy", "photonCanvas", "photonNote"],
            ["pair_energy", "pairCanvas", "pairNote"]
        ]) {
        const h = r?.distributions[key],
            canvas = $(canvasID),
            note = $(noteID),
            shown = !!h && (key === "chi_e" ? !split : !key.startsWith("chi_e_") || split);
        canvas.hidden = !shown;
        $(canvasID + "Plot").hidden = !shown;
        $(canvasID + "Stats").hidden = !shown;
        $(canvasID + "StatsDetails").hidden = !shown;
        if (!shown) {
            note.className = "unavailable";
            note.textContent = "Unavailable: " + absentReason(key);
            continue
        }
        note.className = "hint";
        note.textContent = h.weighting + "; total weight " + fmt(h.total_weight, 4) +
            "; underflow / overflow " + fmt(h.underflow, 3) + " / " + fmt(h.overflow, 3) + " " + h
            .weight_unit + ". No inferred uncertainty.";
        drawHistogram(canvas, h);
        renderPlotStatistics(canvasID, ColliderStatistics.histogram(h), scope === "frame" ?
            "Frame interval" : "Total collision", h.unit, h.weight_unit);
    }
}

/** Display nominal or luminosity-weighted native helicities using the selected interval. */
function drawPolarization(k) {
    const nominal = $("polMode").value === "nominal";
    $("polScope").disabled = nominal;
    if (nominal) {
        const p = diagnostics?.nominal_polarization;
        tableRows("pol", p ? [
            ["Configured spin 1", p.beam1.map(v => fmt(v, 4)).join(" / ")],
            ["Configured spin 2", p.beam2.map(v => fmt(v, 4)).join(" / ")],
            ["Reference helicity 1 / 2", p.reference_helicity.map(v => fmt(v, 5)).join(
                " / ")]
        ] : [
            ["Nominal polarization", "unavailable"]
        ]);
        $("polNote").textContent = p ? (p.enabled ? "" :
                "Spin transport/storage disabled; configured inputs only. ") + p.convention :
            absentReason("nominal_polarization");
        return;
    }
    const scope = $("polScope").value,
        disabled = diagnostics?.nominal_polarization?.enabled === false;
    const p = disabled ? null : diagnosticRecord(scope, k)?.polarization;
    tableRows("pol", p ? [
        ["⟨helicity⟩ beam 1", fmt(p.beam1, 6)],
        ["⟨helicity⟩ beam 2", fmt(p.beam2, 6)],
        ["Luminosity weight", fmt(p.weight_sum, 5) + " m⁻² / crossing"]
    ] : [
        ["Luminosity polarization", "unavailable / no weight"]
    ]);
    $("polNote").textContent = intervalLabel(scope, k) + ". " + (p ?
        "Native collision-pair luminosity weighted helicity ⟨s·p̂⟩; beam directions use +z and −z reference conventions." :
        disabled ? "Spin transport/storage disabled; configured inputs only." : absentReason(
            "polarization"));
}

/** Display native centered RMS and mean with frame-state or total luminosity weighting. */
function drawMoments(k) {
    const scope = $("momentScope").value,
        r = diagnosticRecord(scope, k),
        v = scope === "frame" ? r?.bunch_moments : r?.beam_moments;
    const rows = [];
    if (v) {
        for (const [beam, label] of [
                ["beam1", "Beam 1"],
                ["beam2", "Beam 2"]
            ]) {
            if (!v[beam]) {
                rows.push([label, "no native particles"]);
                continue
            }
            rows.push([label, "mean / RMS"]);
            for (const [key, name, unit] of [
                    ["energy_GeV", "E", "GeV"],
                    ["x_nm", "x", "µm"],
                    ["y_nm", "y", "µm"],
                    ["vx", "vx direction", "µrad"],
                    ["vy", "vy direction", "µrad"]
                ]) {
                const p = v[beam][key];
                const factor = unit === "µm" ? 1e-3 : unit === "µrad" ? 1e6 : 1;
                rows.push([name + " [" + unit + "]", fmt(p.mean * factor, 4) + " / " + fmt(p.rms * factor, 4)]);
            }
        }
    } else rows.push(["Native moments", "unavailable / no weight"]);
    tableRows("stats", rows);
    $("momentNote").textContent = (scope === "frame" ?
            "Stored states at the diagonal barrier; inactive slices are not drift-synchronized with the displayed cloud. " :
            "Total collision. ") +
        (v ? v.weighting +
            ". RMS is centered standard deviation. vx/vy are transverse direction components displayed in µrad." :
            absentReason(scope === "frame" ? "bunch_moments" : "beam_moments"));
}

function drawMonochromatizationInputs() {
    const ideal = monochromatization?.ideal;
    tableRows("monoInput", [
        ["Ideal monochromatization λ [1]", ideal?.available ? fmt(ideal.value, 6) :
            "unavailable"
        ]
    ]);
    $("monoInputNote").textContent = ideal ? (ideal.available ? ideal.scope : ideal.reason) :
        "No authenticated monochromatization analysis in this dataset.";
    const d = monochromatization?.definition;
    $("monoDefinition").textContent = d ? [d.model, d.formula, d.meaning, ...(d.limitations || [])]
        .join(" ") :
        "An ideal input model is distinct from the observed spectrum-width ratio and a matched-control gain.";
}

function isBetaFit() {
    return monochromatization?.peak_fit?.schema_version ===
        "feebl.collider.beta-gaussian-peak-fit/1"
}

function peakFitRecord(scope, k = +$("fr").value) {
    const f = monochromatization?.peak_fit;
    return scope === "frame" ? f?.frames?.[k] : f?.total
}

function peakFitStatus(r) {
    const beta = isBetaFit(),
        name = beta ? "Beta–Gaussian response fit" : "Gaussian peak fit";
    if (!r?.candidate_display_allowed) return name + " unavailable: " + (r?.reason ||
        "No authenticated peak fit was exported.");
    const mismatch = r.candidate_status === "resolved_shape_mismatch";
    return (mismatch ? (beta ? "Beta–Gaussian shape estimate — mismatch" :
            "Gaussian shape estimate — tail mismatch") : name) + "; density-shape residual " +
        fmt(100 * r.relative_l2_residual, 2) + "% (admission limit 25%); window " + fmt(r
            .fit_lower_W_GeV, 4) + "–" + fmt(r.fit_upper_W_GeV, 4) + " GeV.";
}

function peakCurve(scope, k, a, mode) {
    const r = peakFitRecord(scope, k);
    if (!r?.candidate_display_allowed) return null;
    const factor = mode === "shape" ? (a.normalization.total_spectrum_m2_per_crossing > 0 ? 1 / a
            .normalization.total_spectrum_m2_per_crossing : null) :
        mode === "persecond" ? (Number.isFinite(a.normalization.crossing_frequency_hz) ? a
            .normalization.crossing_frequency_hz * 1e-4 : null) : 1;
    if (factor === null) return null;
    if (isBetaFit()) {
        const h = diagnosticRecord(scope, k)?.peak?.luminosity_spectrum;
        if (!h) return null;
        const bins = ColliderBetaGaussian.curveBins(r.candidate, h, true).map(b => ({
            ...b,
            density: b.density * factor
        }));
        return {
            record: r,
            bins,
            maximum: Math.max(0, ...bins.map(b => b.density))
        };
    }
    return {
        record: r,
        maximum: ColliderMonochromatization.gaussianDensity(r.candidate, r.candidate.mean_W_GeV) *
            factor,
        at: x => ColliderMonochromatization.gaussianDensity(r.candidate, x) * factor
    };
}

function peakDiagnosticRows(scope, k) {
    const p = diagnosticRecord(scope, k)?.peak,
        r = peakFitRecord(scope, k),
        positive = p?.available && p.weight_sum > 0,
        c = r?.candidate_display_allowed ? r.candidate : null;
    const mismatch = r?.candidate_status === "resolved_shape_mismatch",
        beta = isBetaFit(),
        rows = [];
    rows.push(["Peak selection", p?.selection ? "W ≥ " + fmt(p.selection.minimum_W_GeV, 5) +
        " GeV (98% design Wnom)" : "unavailable"
    ]);
    rows.push(["Peak L weight", p?.available ? fmt(p.weight_sum, 5) + " m⁻² / crossing" :
        "unavailable"
    ]);
    for (const [beam, label] of [
            ["beam1", "Peak ⟨E₁⟩L [GeV]"],
            ["beam2", "Peak ⟨E₂⟩L [GeV]"]
        ]) rows.push([label, positive ? fmt(p.beam_energy_GeV[beam].mean, 6) :
        "unavailable / no selected weight"
    ]);
    rows.push(["Peak ⟨P₁⟩L / ⟨P₂⟩L [1]", p?.polarization ? [p.polarization.beam1, p.polarization
        .beam2
    ].map(v => fmt(v, 6)).join(" / ") : "unavailable"]);
    rows.push(["Selected-peak RMS σW [GeV]", positive ? fmt(p.W_GeV.rms, 6) : "unavailable"]);
    rows.push([beta ? (mismatch ? "Core σ estimate (shape mismatch) [GeV]" :
            "Beta–Gaussian core σ [GeV]") : mismatch ?
        "Gaussian σ estimate (shape mismatch) [GeV]" : "Gaussian peak σW [GeV]", c ? fmt(c
            .sigma_W_GeV, 6) : "unavailable"
    ]);
    rows.push([beta ? (mismatch ? "Core-width ratio estimate [1]" :
        "Core-width ratio to input [1]") : mismatch ?
        "Gaussian-shape width ratio (estimate) [1]" :
        "Gaussian peak width ratio to input [1]", Number.isFinite(c
        ?.input_reference_ratio) ? fmt(c.input_reference_ratio, 6) : "unavailable"
    ]);
    if (beta) {
        rows.push(["Model endpoint E₀ [GeV]", c ? fmt(c.endpoint_W_GeV, 6) : "unavailable"]);
        rows.push(["Beta α / β", c ? (c.alpha === null ? "pure delta limit" : fmt(c.alpha, 5) +
            " / " + fmt(c.beta, 5)) : "unavailable"]);
        rows.push(["Peak component fraction f [1]", c ? fmt(c.peak_fraction, 6) : "unavailable"])
    }
    rows.push([beta ? "Beta–Gaussian density-shape residual" : "Gaussian density-shape residual",
        c ? fmt(100 * r.relative_l2_residual, 2) + "%" + (mismatch ? " — above 25% limit" :
            "") : "unavailable"
    ]);
    let note = p?.available ?
        "Peak means use the same exact W ≥ 0.98 Wnom luminosity-weighted population, with no upper cut. " :
        "Peak diagnostics unavailable: " + (p?.reason || "No native peak selection was exported.") +
        " ";
    if (p?.polarization === null && p.polarization_unavailable_reason) note += "Polarization: " + p
        .polarization_unavailable_reason + ". ";
    return {
        rows,
        note: note + peakFitStatus(r),
        mismatch
    };
}

function monochromatizationRows(scope, k) {
    const measured = monochromatization?.measured,
        record = scope === "frame" ? measured?.frames?.[k] : measured?.total;
    const rows = [
        ["Input-reference σW [GeV]", Number.isFinite(measured?.reference_sigma_W_GeV) ? fmt(
            measured.reference_sigma_W_GeV, 6) : "unavailable"],
        [scope === "frame" ? "Interval full-RMS ratio to input [1]" :
            "Total full-RMS ratio to input [1]", record?.available ? fmt(record.value, 6) :
            "unavailable"
        ],
        [scope === "frame" ? "Interval full-spectrum RMS σW [GeV]" :
            "Total full-spectrum RMS σW [GeV]", Number.isFinite(record?.sigma_W_GeV) ? fmt(
                record.sigma_W_GeV, 6) : "unavailable"
        ],
        ["Matched-control monochromatization gain", "unavailable"]
    ];
    return {
        rows,
        note: record ? (record.available ? record.estimator : record.reason) :
            "No authenticated width-ratio analysis."
    };
}

/** Render derived and native scalar diagnostics with their weighting and fit qualifications. */
function drawDiagnostics(k) {
    const scope = $("diagScope").value,
        r = diagnosticRecord(scope, k),
        h = r?.luminosity_spectrum,
        rows = [];
    if (h) {
        rows.push([scope === "frame" ? "Interval L (integrated)" : "Total L (integrated)", fmt(h
            .total_weight, 5) + " m⁻² / crossing"]);
        if (scope === "frame" && diagnostics.total.luminosity_spectrum?.total_weight > 0)
            rows.push(["Fraction of total L", fmt(h.total_weight / diagnostics.total
                .luminosity_spectrum.total_weight, 5)]);
        rows.push(["L below / above W bins", fmt(h.underflow, 3) + " / " + fmt(h.overflow, 3)]);
    } else if (scope === "total" && scientific) rows.push(["Total native L", fmt(scientific
        .normalization.total_spectrum_m2_per_crossing, 5) + " m⁻² / crossing"]);
    else rows.push(["Native luminosity", "unavailable"]);
    const b = r?.beam_moments;
    for (const [key, label] of [
            ["beam1", "Beam 1 ⟨E⟩L"],
            ["beam2", "Beam 2 ⟨E⟩L"]
        ])
        rows.push([label, b ? fmt(b[key].energy_GeV.mean, 6) + " GeV" : "unavailable / no weight"]);
    const nominal = diagnostics?.nominal_polarization,
        spinDisabled = nominal?.enabled === false;
    const bunch = spinDisabled ? null : diagnosticRecord("frame", k)?.bunch_polarization;
    const lumi = spinDisabled ? null : r?.polarization;
    const helicities = values => Array.isArray(values) ? values.map(v => v === null ?
        "unavailable" : fmt(v, 6)).join(" / ") : "unavailable";
    rows.push(["Configured P₁ / P₂ [1]", helicities(nominal?.reference_helicity)]);
    rows.push(["Bunch P₁ / P₂ [1], snapshot " + k, spinDisabled ? "unavailable / spin not stored" :
        helicities(bunch)
    ]);
    rows.push([scope === "frame" ? "Interval ⟨P₁⟩L / ⟨P₂⟩L [1]" : "Total ⟨P₁⟩L / ⟨P₂⟩L [1]",
        lumi ? helicities([lumi.beam1, lumi.beam2]) : spinDisabled ?
        "unavailable / spin not stored" : "unavailable / no weight"
    ]);
    rows.push(["Polarization L weight", lumi ? fmt(lumi.weight_sum, 5) + " m⁻² / crossing" :
        "unavailable"
    ]);
    const peak = peakDiagnosticRows(scope, k);
    rows.push(...peak.rows);
    const width = monochromatizationRows(scope, k);
    rows.push(...width.rows);
    $("diagPeakNote").textContent = peak.note;
    $("diagPeakNote").classList.toggle("shape-mismatch", peak.mismatch);
    const f = monochromatization?.peak_fit;
    $("diagPeakDetails").textContent = (r?.peak?.selection?.nominal_reference?.native_W || "") +
        ". " +
        "Peak energy means use beam energies before optional ISR sampling; polarization requires valid spin for every selected contribution. Selected-peak RMS includes all selected high-side luminosity, including overflow; it is not a Gaussian fit. " +
        (f ? [f.estimator, f.scope, f.normalization, f.statistical_interpretation].join(". ") :
            "Gaussian fit unavailable.") + ". " +
        (isBetaFit() ?
            "Wilson Appendix A.1 beta-plus-delta convolution uses the full exported spectrum. Core σW = E₀ × fractional Gaussian response width; E₀ is the endpoint parameter, not the mixture mean. " :
            "") +
        "Core width ratio = input-reference σW / Gaussian response σ. Shape-mismatch estimates are descriptive candidates, not admitted fits. Neither width ratio isolates dispersion gain or replaces a matched D=0 control.";
    tableRows("diag", rows);
    $("diagNote").textContent = intervalLabel(scope, k) +
        ". Mean energies and luminosity helicities use native collision-pair luminosity weights; L is the integrated normalization. Bunch helicity always refers to the displayed snapshot, including when Total is selected. " +
        width.note +
        " Full-RMS width ratio = input-reference σW / observed full-spectrum RMS; values below 1 indicate broadening. It is not an isolated dispersion gain or a matched D=0 control comparison.";
    $("diagPolNote").textContent =
        "P is dimensionless. Configured values are reference helicities (+Sz for beam 1, −Sz for beam 2), not measurements of loaded beams. " +
        "Bunch values are unweighted native means over stored primary slices at snapshot " + k +
        ", using valid spin/direction records; they are not time-averaged or derived from displayed samples. " +
        "Luminosity helicity is ⟨s·p̂⟩ in each particle's own native direction, with its selected interval/total weight shown above. No design crossing-angle or detector-frame transformation is applied. " +
        (spinDisabled ? "Spin transport/storage is disabled; configured inputs only." :
            "Missing spin or zero collision weight remains unavailable; no polarization uncertainty is inferred."
            );
}

function drawScientificViews(k) {
    drawSpectra();
    drawDistributions(k);
    drawPolarization(k);
    drawMoments(k);
    drawDiagnostics(k);
    drawMonochromatizationInputs();
    drawSecondaryKinematics()
}
const scopeIds = ["spectrumScope", "distributionScope", "secondaryScope", "polScope", "diagScope", "momentScope"];
/** Apply global scope unless overrides are enabled; update visible per-panel scope badges. */
function synchronizeScopes() {
    const overrides = $("scopeOverrides").checked;
    for (const id of scopeIds) {
        const selector = $(id);
        selector.disabled = !overrides;
        if (!overrides) selector.value = $("globalScope").value;
        let badge = $(id + "Badge");
        if (!badge) { badge = document.createElement("span"); badge.id = id + "Badge";
            badge.className = "scope-badge"; selector.after(badge); }
        badge.textContent = selector.options[selector.selectedIndex].textContent;
    }
}
for (const id of ["globalScope", "scopeOverrides", ...scopeIds]) $(id).addEventListener("change", () => {
    synchronizeScopes();
    if (man) draw();
});
synchronizeScopes();

for (const id of ["spectrumScope", "distributionScope", "distributionRange", "polMode", "polScope", "momentScope",
        "diagScope"
    ])
    $(id).addEventListener("change", () => drawScientificViews(+$("fr").value));

function initializeSecondaryControls() {
    const select = $("secondaryPopulation");
    select.replaceChildren();
    const available = [];
    for (const key of ["pairs", "deflected", "combined"]) {
        const pop = secondaryKinematics?.populations[key],
            option = document.createElement("option");
        option.value = key;
        option.textContent = pop?.label || ({
            pairs: "Produced pairs",
            deflected: "Deflected particles",
            combined: "Mixed stored population"
        })[key];
        const any = key === "deflected" ? eventDeflections?.available : pop?.available ||
            secondaryKinematics?.frames?.some(f => f.populations[key]?.available);
        option.disabled = false;
        if (!any) option.textContent += " (not available)";
        select.append(option);
        if (any) available.push(key);
    }
    select.value = available[0] || (secondaryKinematics?.populations.deflected?.available ?
        "deflected" : "pairs");
    select.disabled = !secondaryKinematics && !eventDeflections;
    $("secondaryScope").value = "total";
    $("secondaryScope").disabled = !secondaryKinematics && !eventDeflections;
    $("secondaryZ").value = "all";
    $("secondaryZ").disabled = !secondaryKinematics;
    $("secondaryDownload").disabled = !secondaryKinematics && !eventDeflections;
    eventPlotNodes.clear();
    $("eventDeflectionPlots").replaceChildren();
    $("diagnosticRawSource").replaceChildren();
    for (const ref of man.diagnostic_analysis_sources || []) {
        const option = document.createElement("option");
        option.value = ref.path;
        option.textContent = ref.path;
        $("diagnosticRawSource").append(option)
    }
    const first = man.diagnostic_analysis_sources?.[0];
    $("diagnosticRawSource").value = first?.path || "";
    $("diagnosticRawSource").disabled = !first;
    $("diagnosticRawDownload").disabled = !first;
    lastSecondaryRender = null;
}

function renderSecondaryStatistics(pop, folded, scopeLabel = "Final stored-record") {
    const thetah = pop.distributions[folded ? "theta_folded" : "theta"],
        pth = pop.distributions.pT;
    const fromRecords = h => ({
        ...ColliderStatistics.histogram(h),
        available: h.samples > 0,
        mean: h.statistics?.mean ?? null,
        variance: h.statistics?.variance ?? null,
        minimum: h.statistics?.min ?? null,
        maximum: h.statistics?.max ?? null
    });
    const theta = fromRecords(thetah),
        pt = fromRecords(pth);
    const box = $("secondaryHeatmapStats");
    box.replaceChildren();
    box.hidden = false;
    const heading = document.createElement("strong");
    heading.textContent = scopeLabel + " marginal statistics";
    box.append(heading);
    const table = document.createElement("table"),
        head = document.createElement("tr");
    for (const label of ["Physical units", folded ? "θ nearest ±z [rad]" : "θ from +z [rad]",
            "pT [GeV/c]"
        ]) {
        const td = document.createElement("th");
        td.textContent = label;
        head.append(td)
    }
    table.append(head);
    const mode = s => !s.available || !s.mode_intervals.length ? "unavailable" : s.mode_intervals
        .length === 1 ? statisticNumber(s.mode_intervals[0].low) + "–" + statisticNumber(s
            .mode_intervals[0].high) : s.mode_intervals.length + " tied bins";
    const values = [s => s.available ? statisticNumber(s.mean) : "unavailable", mode, s => s
        .available ? statisticNumber(s.variance) : "unavailable", s => statisticNumber(s
            .minimum), s => statisticNumber(s.maximum)
    ];
    const labels = ["Mean (records)", "Mode (density bin)", "Variance (units²)", "Min (record)",
        "Max (record)"
    ];
    const spoken = [];
    values.forEach((fn, i) => {
        const tr = document.createElement("tr"),
            row = [labels[i], fn(theta), fn(pt)];
        for (const value of row) {
            const td = document.createElement("td");
            td.textContent = value;
            tr.append(td)
        }
        table.append(tr);
        spoken.push(row.join(": "))
    });
    box.append(table);
    const note = document.createElement("p");
    note.textContent =
        "Physical marginals, not log-coordinate moments. Record means/variance/extrema; mode is a bin estimate.";
    box.append(note);
    box.setAttribute("aria-label", heading.textContent + ". θ [rad]; pT [GeV/c]. " + spoken.join(
        "; ") + ". " + note.textContent);
    return "Theta selected weight " + statisticNumber(theta.selected_weight) + ", flow " +
        statisticNumber(theta.excluded.underflow) + " / " + statisticNumber(theta.excluded
        .overflow) +
        "; pT selected weight " + statisticNumber(pt.selected_weight) + ", flow " + statisticNumber(
            pt.excluded.underflow) + " / " + statisticNumber(pt.excluded.overflow) +
        " stored records. Means, variance and extrema include all selected stored records; mode uses occupied marginal bins only. No seed/model uncertainty is inferred.";
}

function selectedSecondaryRecords() {
    const scope = $("secondaryScope").value || "total",
        k = +$("fr").value || 0,
        key = $("secondaryPopulation").value || "pairs",
        z = $("secondaryZ").value || "all";
    const frame = scope === "frame" ? secondaryKinematics?.frames?.[k] || null : null;
    const inclusive = scope === "total" ? secondaryKinematics?.populations[key] : frame
        ?.populations[key];
    const reason = scope === "frame" ? (frame ? (frame.available ? null : frame.reason) :
        "No authenticated particle snapshots in this dataset; Total shows final stored records."
        ) : null;
    const pop = inclusive ? ColliderSecondaryKinematics.selectZ(inclusive, z, scope === "frame" ?
            ColliderSecondaryKinematics.FRAME_WEIGHTING : ColliderSecondaryKinematics.WEIGHTING) :
        inclusive;
    return {
        scope,
        k: scope === "frame" ? k : null,
        key,
        z,
        frame,
        pop,
        reason
    };
}

/** Plot each event slot angular change from its admitted direction; degrees, final or frame records. */
function drawEventDeflections(scope, k) {
    const data = eventDeflections,
        r = data ? (scope === "frame" ? data.frames[k] : data.total) : null,
        shown = Boolean(data?.available && r?.available);
    $("secondaryHeatmapPlot").hidden = true;
    $("secondaryHeatmapStats").hidden = true;
    $("secondaryStatsDetails").hidden = true;
    $("eventDeflectionPlots").hidden = !shown;
    $("secondaryUnavailable").hidden = shown;
    const scopeLabel = scope === "frame" ? "Snapshot " + k : "Final tracked records";
    const explanation =
        "Signed changes from each identified particle's actual native admission direction: Δθ = θcurrent − θinitial about common +z; Δφ is wrapped to [−180°, 180°). Each event slot has two distributions across events. Current frames contain the tracked population admitted so far; Total is the final state, not a sum of frames. ";
    $("secondaryContext").textContent = scopeLabel + ". " + explanation;
    if (!shown) {
        $("secondaryUnavailable").textContent = "Unavailable: " + (r?.reason || data?.reason ||
            "No authenticated event identities and initial-to-current trajectories are available. Final particle rows alone do not determine event deflections."
            );
        return
    }
    const count = r.event_counts,
        g = r.field_gathers;
    $("secondaryContext").textContent += r.nrecords + " identified records; " + count.complete +
        " complete / " + count.expected + " expected events, " + count.incomplete +
        " incomplete and " + count.not_yet_admitted +
        " not admitted in this state. Native field gathers: " + g.main_mesh + " main mesh, " + g
        .outer_mesh + " outer mesh, " + g.monopole + " monopole fallback. " + (g.monopole > 0 ?
            "Some tracked states used the native analytic monopole outside the meshes; this is not a mesh-only field history. " :
            "") + "Each count is one stored record; no event or luminosity reweighting.";
    for (const descriptor of data.plots) {
        let nodes = eventPlotNodes.get(descriptor.id);
        if (!nodes) {
            const section = document.createElement("section"),
                heading = document.createElement("h3"),
                plot = document.createElement("div"),
                canvas = document.createElement("canvas"),
                stats = document.createElement("div"),
                note = document.createElement("p");
            heading.textContent = descriptor.label;
            plot.className = "statistics-plot distribution-plot event-deflection-plot";
            canvas.width = 760;
            canvas.height = 320;
            stats.className = "plot-statistics";
            stats.setAttribute("role", "group");
            note.className = "hint";
            plot.append(canvas, stats);
            section.append(heading, plot, note);
            $("eventDeflectionPlots").append(section);
            nodes = {
                section,
                canvas,
                stats,
                note
            };
            eventPlotNodes.set(descriptor.id, nodes);
        }
        const h = r.histograms[descriptor.id],
            a = h.statistics;
        drawHistogram(nodes.canvas, {
            ...h,
            quantity: descriptor.label,
            forceLinear: true,
            emptyLabel: "No admitted records in this scope"
        });
        nodes.stats.replaceChildren();
        const title = document.createElement("strong");
        title.textContent = scopeLabel + " [deg]";
        nodes.stats.append(title);
        const rows = [
            ["Mean (records)", a?.mean],
            ["Mode (density bin)", a?.mode],
            ["Variance [deg²]", a?.variance],
            ["Min (record)", a?.min],
            ["Max (record)", a?.max]
        ];
        const dl = document.createElement("dl");
        for (const [label, value] of rows) {
            const dt = document.createElement("dt"),
                dd = document.createElement("dd");
            dt.textContent = label;
            dd.textContent = statisticNumber(value);
            dl.append(dt, dd)
        }
        nodes.stats.append(dl);
        const note = document.createElement("p");
        note.textContent = a ? "Exact record moments; modal-bin estimate." :
            "No records; statistics unavailable.";
        nodes.stats.append(note);
        nodes.stats.setAttribute("aria-label", descriptor.label + ". " + title.textContent + ". " +
            rows.map(([label, value]) => label + ": " + statisticNumber(value)).join("; ") +
            ". " + note.textContent);
        nodes.note.textContent = h.samples + " defined records; " + h.excluded_undefined_azimuth +
            " undefined-azimuth records excluded; " + h.underflow + " below / " + h.overflow +
            " above range. Fixed signed degree bins across all frames and final state. Δφ statistics use the wrapped interval; they are not circular means.";
        nodes.canvas.setAttribute("aria-label", descriptor.label + "; " + scopeLabel + "; " + h
            .samples + " identified records; signed degrees, fixed bin edges.");
    }
}

/** Render stored secondary count density on log(theta), log(pT) axes with explicit population cuts. */
function drawSecondaryKinematics() {
    const deflected = $("secondaryPopulation").value === "deflected";
    for (const id of ["secondaryZControl", "secondaryAngleControl", "secondaryColorControl"]) $(id)
        .hidden = deflected;
    if (deflected) {
        lastSecondaryRender = null;
        drawEventDeflections($("secondaryScope").value || "total", +$("fr").value || 0);
        return
    }
    $("eventDeflectionPlots").hidden = true;
    const selected = selectedSecondaryRecords(),
        {
            scope,
            k,
            key,
            z,
            pop
        } = selected,
        folded = $("secondaryAngle").value === "folded";
    const shown = !!pop?.available,
        joint = shown ? pop[folded ? "joint_folded" : "joint"] : null;
    $("secondaryHeatmapPlot").hidden = !shown;
    $("secondaryHeatmapStats").hidden = !shown;
    $("secondaryStatsDetails").hidden = !shown;
    $("secondaryUnavailable").hidden = shown;
    const scopeText = scope === "frame" ? "Snapshot " + k + (selected.frame ? ", after diagonal " +
            selected.frame.diagonal : "") +
        ". Produced/imported records tracked so far, including completed trajectories. Records can persist across frames; Total is final stored records, not a sum of snapshots. " :
        "Final stored records only; independent of collision playback. ";
    const zLabel = z === "all" ? "All stored z positions" : z === "positive" ? "Stored z > 0" :
        "Stored z < 0";
    const context = scopeText + zLabel +
        ". Selection uses stored native z position, not longitudinal momentum or a detector arm; z = 0 records are included only in All. " +
        (folded ?
            "Nearest-axis angle = min(θ, π−θ); backward tracks use π−θ. This folds the angle without selecting a hemisphere or species." :
            "Common +z polar angle θ ranges from 0 to π. Backward tracks cluster near θ = π, producing a narrow band in log(θ); this is an angular-coordinate effect."
            ) +
        " Native tracking coordinates; no design crossing-angle or detector-frame transformation.";
    if (!shown) {
        $("secondaryContext").textContent = context;
        $("secondaryUnavailable").textContent = "Unavailable: " + (selected.reason || pop?.reason ||
            "No authenticated final particle-kinematics export in this dataset.");
        lastSecondaryRender = null;
        return
    }
    const excluded = joint.excluded;
    $("secondaryContext").textContent = context + " " + pop.label + ": " + joint.in_range_records +
        " / " + pop.nrecords + " records in the log plot; " + excluded.nonpositive_records +
        " nonpositive, " + excluded.underflow_records + " below range, " + excluded
        .overflow_records +
        " above range. Blank cells: zero stored records; colors are counts per joint bin. Axes stay fixed across z selections, frames and Total, with 0.25-decade padding beyond the final-map bounds and occupied snapshot bins.";
    const canvas = $("secondaryHeatmap");
    fitStatisticsCanvas(canvas);
    const cs = getComputedStyle(document.documentElement),
        bg = cs.getPropertyValue("--panel"),
        colorMode = $("secondaryColor").value || "log";
    const signature = [scope, k, key, z, folded, colorMode, canvas.width, canvas.height, bg].join(
        "|");
    if (lastSecondaryRender?.data === secondaryKinematics && lastSecondaryRender.signature ===
        signature) return;
    lastSecondaryRender = {
        data: secondaryKinematics,
        signature
    };
    const g = canvas.getContext("2d"),
        L = 92,
        R = 92,
        T = 43,
        B = 54,
        W = Math.max(1, canvas.width - L - R),
        H = Math.max(1, canvas.height - T - B);
    const xe = joint.theta_edges_rad.map(Math.log10),
        ye = joint.pT_edges_GeV.map(Math.log10),
        nx = xe.length - 1,
        ny = ye.length - 1;
    const {
        xmin,
        xmax,
        ymin,
        ymax
    } = ColliderSecondaryKinematics.axisBounds(secondaryKinematics, key, folded);
    const px = x => L + (x - xmin) / (xmax - xmin) * W,
        py = y => T + H - (y - ymin) / (ymax - ymin) * H;
    const maximum = Math.max(0, ...joint.counts),
        transform = v => colorMode === "log" ? Math.log1p(v) : v;
    g.fillStyle = bg;
    g.fillRect(0, 0, canvas.width, canvas.height);
    for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
            const count = joint.counts[j * nx + i];
            if (!count) continue;
            g.fillStyle = ColliderFields.color(transform(count), transform(maximum));
            g.fillRect(px(xe[i]), py(ye[j + 1]), px(xe[i + 1]) - px(xe[i]), py(ye[j]) - py(ye[j +
                1]))
        }
    g.fillStyle = cs.getPropertyValue("--ink");
    g.strokeStyle = cs.getPropertyValue("--line");
    g.font = "12px sans-serif";
    const xTicks = ColliderViewerUI.linearTicks(xmin, xmax, W);
    const yTicks = ColliderViewerUI.linearTicks(ymin, ymax, H);
    g.textAlign = "center";
    for (const x of xTicks.values)
        g.fillText(ColliderViewerUI.tickLabel(x, xTicks.step), px(x), T + H + 20);
    g.textAlign = "right";
    for (const y of yTicks.values)
        g.fillText(ColliderViewerUI.tickLabel(y, yTicks.step), L - 8, py(y) + 4);
    g.strokeRect(L, T, W, H);
    g.textAlign = "left";
    g.fillText("log₁₀(pT / (GeV/c))", L, 24);
    g.textAlign = "center";
    g.fillText(folded ? "log₁₀(θ nearest ±z / rad)" : "log₁₀(θ from +z / rad)", L + W / 2, canvas
        .height - 8);
    const legendX = L + W + 16,
        legendW = 14;
    for (let i = 0; i < 100; i++) {
        g.fillStyle = maximum ? ColliderFields.color(transform(1) + (transform(maximum) - transform(
            1)) * i / 99, transform(maximum)) : bg;
        g.fillRect(legendX, T + (99 - i) * H / 100, legendW, H / 100 + 1)
    }
    g.fillStyle = cs.getPropertyValue("--ink");
    g.textAlign = "left";
    g.fillText(String(maximum), legendX + 18, T + 9);
    g.fillText(maximum ? "1" : "0", legendX + 18, T + H);
    if (!maximum) {
        g.textAlign = "center";
        g.fillText("No positive in-range joint records", L + W / 2, T + H / 2)
    }
    const stats = renderSecondaryStatistics(pop, folded, (scope === "frame" ? "Snapshot " + k +
        " tracked-record" : "Final stored-record") + (z === "all" ? "" : " · " + (z ===
        "positive" ? "z > 0" : "z < 0")));
    $("secondaryStatsNotes").textContent = stats + " " + pop.normalization.weighting +
        ". No physical-yield or pair-ratio reweighting. Selection: " + JSON.stringify(pop
        .selection);
    canvas.setAttribute("aria-label", pop.label + (scope === "frame" ? " snapshot " + k :
        " final") + "; " + zLabel + "; joint log angle and log pT; " + joint.in_range_records +
        " in-range records; " + colorMode + " count color scale, maximum " + maximum + ". " + (
            folded ? "Nearest ±z-axis angle, min(θ, π−θ)." : "Common +z polar angle, 0 to π."));
}
for (const id of ["secondaryScope", "secondaryPopulation", "secondaryZ", "secondaryAngle",
        "secondaryColor"
    ]) $(id).addEventListener("change", drawSecondaryKinematics);
$("monoDownload").addEventListener("click", () => {
    if (monochromatization) download("collider_monochromatization.json", JSON.stringify(
        monochromatization, null, 2), "application/json")
});
$("secondaryDownload").addEventListener("click", () => {
    const events = $("secondaryPopulation").value === "deflected",
        data = events ? eventDeflections : secondaryKinematics;
    if (data) download(events ? "collider_event_deflections.json" :
        "collider_secondary_kinematics.json", JSON.stringify(data, null, 2),
        "application/json")
});
$("diagnosticRawDownload").addEventListener("click", async () => {
    try {
        const ref = (man.diagnostic_analysis_sources || []).find(r => r.path === $(
            "diagnosticRawSource").value);
        if (!ref) return;
        const bytes = await ColliderContract.bytesHash(await readBytes(ref.path, 32 *
            1024 * 1024), ref, 32 * 1024 * 1024, true);
        download(ref.path, bytes, "application/octet-stream");
    } catch (e) {
        fail(e)
    }
});
