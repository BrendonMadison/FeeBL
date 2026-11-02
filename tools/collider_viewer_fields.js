/** Native field-layer rendering and final viewer startup.
 * Reads source-specific electric-equivalent transverse magnitudes, converts
 * labels to GV/m, and paints recorded coverage in native longitudinal planes.
 * The two-layer display is not a net vector field. Gaps stay transparent.
 */
function selectedFieldData() {
    return $("fieldData").value || "auto"
}

function selectedFieldLayer() {
    return ColliderFields.dataMode(nativeFields, selectedFieldData()) === "snapshot" ? nativeFields
        ?.snapshot : nativeFields?.longitudinal
}

function initializeFieldControls() {
    const full = Boolean(nativeFields?.snapshot?.availableFrames),
        encounters = Boolean(nativeFields?.longitudinal?.availableFrames);
    const hasMaps = Boolean(nativeFields && (nativeFields.data.frames.some(f => f.available) ||
        full || encounters));
    for (const id of ["fieldMode", "fieldSource", "fieldOpacity", "fieldScale"]) $(id).disabled = !
        hasMaps;
    $("fieldData").options[0].disabled = !full;
    $("fieldData").options[1].disabled = !encounters;
    $("fieldData").value = full ? "snapshot" : "encounter";
    $("fieldData").disabled = !full || !encounters;
    $("fitField").disabled = !full && !encounters;
    $("fieldSource").replaceChildren();
    if (nativeFields)
        for (const source of nativeFields.data.sources) {
            const option = document.createElement("option");
            option.value = source.id;
            option.textContent = source.label;
            $("fieldSource").append(option);
        }
    if (ColliderFields.beamLayers(nativeFields).length === 2) {
        const option = document.createElement("option");
        option.value = ColliderFields.BOTH_BEAMS;
        option.textContent = "Both beams (separate layers)";
        $("fieldSource").append(option);
    }
    if (nativeFields) $("fieldSource").value = nativeFields.data.sources[0].id;
    $("fieldMode").value = "off";
    $("fieldOpacity").value = "0.65";
    $("fieldScale").value = "run";
    $("fieldOpacityValue").textContent = "65%";
    $("fieldLegend").hidden = true;
    $("fieldBothLegends").hidden = true;
    $("fieldPlane").hidden = true;
    $("fieldDetailsDisclosure").hidden = !nativeFields;
    $("fieldDetails").textContent = "";
    $("fieldNote").textContent = full ?
        "Full bunch snapshots available: reconstructed native slice fields synchronized to each frame. Choose Field z–x or Field z–y." :
        encounters ?
        "Active-encounter maps available. Field z–x and Field z–y preserve the longitudinal particle view." :
        hasMaps ?
        "This dataset has legacy x–y maps only; longitudinal field export is unavailable. x–y remains a manual axis choice." :
        nativeFields ? "Field overlay unavailable: " + nativeFields.data.frames[0].reason :
        "Field overlay unavailable: this dataset has no authenticated field export.";
}

function paintFieldWindow(g, view, key, mode) {
    const window = ColliderFields.samplingWindow(nativeFields, key, mode);
    if (!window) return;
    g.save();
    g.beginPath();
    g.rect(view.L, view.T, view.W, view.H);
    g.clip();
    g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--mid").trim();
    g.lineWidth = 1;
    g.setLineDash([5, 5]);
    const left = view.px(window.horizontal[0]),
        right = view.px(window.horizontal[1]);
    const top = view.py(window.vertical[1]),
        bottom = view.py(window.vertical[0]);
    g.strokeRect(left, top, right - left, bottom - top);
    g.restore();
}

/** Composite two separately normalized magnitude layers; preserve source-specific coverage. */
function paintBothFieldLayers(k, g, view) {
    const sources = ColliderFields.beamLayers(nativeFields),
        enabled = $("fieldMode").value === "on";
    const key = ColliderFields.viewKey(view.horizontal, view.vertical),
        mode = selectedFieldData();
    const opacity = Number($("fieldOpacity").value),
        scaleMode = $("fieldScale").value;
    $("fieldOpacityValue").textContent = Math.round(opacity * 100) + "%";
    $("fieldLegend").hidden = true;
    $("fieldBothLegends").hidden = true;
    $("fieldPlane").hidden = !enabled || Boolean(key);
    const qualification =
        "Visual blend; not a vector sum. Overlap colors are a mixture; read each source separately for magnitude.";
    $("fieldDetails").textContent = qualification +
        " Layer opacity applies separately to each source; both use the same linear magnitude scale.";
    if (sources.length !== 2) {
        $("fieldNote").textContent =
            "Both-source display unavailable: compatible canonical beam layers are required.";
        return
    }
    if (!enabled) {
        $("fieldNote").textContent =
            "Field overlay off. Both beams are selected as separate layers.";
        return
    }
    if (!key) {
        $("fieldNote").textContent = "Choose Field z–x or Field z–y to display both source layers.";
        return
    }
    const records = sources.map(source => ({
        source,
        plane: ColliderFields.plane(nativeFields, k, source.id, key, mode)
    }));
    if (!records.some(r => r.plane)) {
        const frame = nativeFields.data.frames[k],
            full = ColliderFields.dataMode(nativeFields, mode) === "snapshot";
        const reason = (key === "xy" ? frame : full ? frame.snapshot : frame.longitudinal)?.reason;
        $("fieldNote").textContent = "Field unavailable in this projection/frame: " + (reason ||
            "No authenticated maps.");
        return;
    }
    const maximum = Math.max(...sources.map(s => ColliderFields.scaleMaximum(nativeFields, k, s.id,
        scaleMode, key, mode)));
    const style = ColliderFields.layerStyle(getComputedStyle(document.documentElement)
        .getPropertyValue("--panel"));
    for (const {
            source,
            plane
        }
        of records) {
        if (plane) ColliderFields.drawPlane(g, plane, view, opacity, maximum, {
            sourceId: source.id,
            style
        });
        const legend = $(source.id === "beam1" ? "fieldLegendBeam1" : "fieldLegendBeam2");
        const displayed = ColliderViewerUI.fieldDisplay(maximum, source.unit);
        ColliderFields.layerLegend(legend, displayed.maximum, displayed.unit, source.id, style, opacity);
        legend.setAttribute("aria-label", source.label + " separate " + (source.id === "beam1" ?
                "blue" : "orange") + " layer: 0 to " + fmt(displayed.maximum, 3) + " " + displayed.unit +
            ", shared " + (scaleMode === "run" ? "fixed run" : "auto frame") + " scale; " + Math
            .round(opacity * 100) +
            "% layer opacity; overlap colors are not net-field magnitudes");
    }
    $("fieldBothLegends").hidden = false;
    paintFieldWindow(g, view, key, mode);
    const full = key !== "xy" && ColliderFields.dataMode(nativeFields, mode) === "snapshot",
        metadata = selectedFieldLayer()?.metadata;
    const coverage = key === "xy" ?
        "central-encounter pre-kick fields; cloud at the end of the diagonal" : records.map(({
                source,
                plane
            }) =>
            "beam " + source.source_beam + ": " + (plane ? plane.coverage.available + "/" + metadata
                .n_z + " slices" : "unavailable")).join("; ");
    const phase = key === "xy" ? "" : full ? " synchronized to the frame" :
        " at active encounters, pre-kick; cloud at the end of the diagonal";
    const outside = records.filter(r => r.plane?.counts?.outside > 0).map(r => "Beam " + r.source
        .source_beam + ": " + r.plane.counts.outside +
        " source particles outside transverse mesh.").join(" ");
    const displayed = ColliderViewerUI.fieldDisplay(maximum, sources[0].unit);
    $("fieldNote").textContent = "Both beams · Frame " + k + ": " + coverage + phase +
        ". Shared 0–" + fmt(displayed.maximum, 3) + " " + displayed.unit + " (" + (scaleMode === "run" ?
            "fixed run scale" : "auto frame scale") +
        "). Visual blend; not a vector sum. Overlap colors are a mixture." + (outside ? " " +
            outside : "");
    $("fieldDetails").textContent += (key === "xy" ? " " + nativeFields.data.conventions.selection +
        "; " + nativeFields.data.conventions.scope :
        " " + (full ? metadata.model + ". " : "") + metadata.selection + ". " + metadata
        .timing + ". " + view.horizontal + "–" + view.vertical + " cross-section at " + metadata
        .planes[key].fixed_axis +
        "=0 nm. Each layer retains its own source-slice positions; missing regions remain transparent. The dashed outline is the stable run sampling window."
        );
    for (const {
            source,
            plane
        }
        of records)
        if (plane?.counts) $("fieldDetails").textContent += " Beam " + source.source_beam +
            " source particles: " + plane.counts.source + "; deposited: " + plane.counts.deposited +
            "; outside: " + plane.counts.outside + ".";
}

/** Paint one source or both layers without extrapolating outside recorded mesh coverage. */
function paintFieldOverlay(k, g, view) {
    $("fieldBothLegends").hidden = true;
    if (!nativeFields) return;
    if ($("fieldSource").value === ColliderFields.BOTH_BEAMS) {
        paintBothFieldLayers(k, g, view);
        return
    }
    const enabled = $("fieldMode").value === "on",
        key = ColliderFields.viewKey(view.horizontal, view.vertical);
    const id = $("fieldSource").value,
        source = nativeFields.data.sources.find(s => s.id === id),
        record = nativeFields.data.frames[k];
    const opacity = Number($("fieldOpacity").value);
    $("fieldOpacityValue").textContent = Math.round(opacity * 100) + "%";
    const longitudinal = key === "zx" || key === "zy",
        mode = selectedFieldData(),
        full = ColliderFields.dataMode(nativeFields, mode) === "snapshot";
    const p = key ? ColliderFields.plane(nativeFields, k, id, key, mode) : null;
    const reason = longitudinal ? ((full ? record.snapshot : record.longitudinal)?.reason ||
        "No authenticated longitudinal field export in this dataset.") : record.reason;
    $("fieldLegend").hidden = !enabled || !p;
    $("fieldPlane").hidden = !enabled || Boolean(key);
    $("fieldDetails").textContent = source.label + " · " + source.quantity + " [" + source.unit +
        "]. " +
        (longitudinal ? (selectedFieldLayer()?.metadata.timing || reason) : nativeFields.data
            .conventions.selection + "; " + nativeFields.data.conventions.scope);
    if (!enabled) {
        $("fieldNote").textContent = "Field overlay off. " + (p ? source.label +
            " is available in this projection." : reason || "Choose Field z–x or Field z–y.");
        return
    }
    if (!key) {
        $("fieldNote").textContent =
            "Select Field z–x or Field z–y for a longitudinal cross-section. Legacy x–y maps remain available through manual axis selection.";
        return
    }
    if (!p) {
        $("fieldNote").textContent = "Field unavailable in this projection/frame: " + reason;
        return
    }
    const scaleMode = $("fieldScale").value,
        maximum = ColliderFields.scaleMaximum(nativeFields, k, id, scaleMode, key, mode);
    ColliderFields.drawPlane(g, p, view, opacity, maximum);
    const displayed = ColliderViewerUI.fieldDisplay(maximum, source.unit);
    ColliderFields.legend($("fieldLegend"), displayed.maximum, displayed.unit);
    if (longitudinal) paintFieldWindow(g, view, key, mode);
    const scaleLabel = scaleMode === "run" ? "fixed run scale" : "auto frame scale";
    const magnitude = source.kind === "beam_beam" ? "native electric-equivalent magnitude" :
        "declared electric-field magnitude";
    $("fieldLegend").setAttribute("aria-label", source.label + " " + view.horizontal + "–" + view
        .vertical + " field magnitude: 0 to " + fmt(displayed.maximum, 3) + " " + displayed.unit + ", " +
        scaleLabel + ", Cividis-derived palette");
    if (longitudinal) {
        const metadata = selectedFieldLayer().metadata,
            plane = metadata.planes[key];
        if (full) {
            $("fieldNote").textContent = source.label + " · Frame " + k +
                ": reconstructed native slice field. " + p.coverage.available + "/" + metadata.n_z +
                " source slices synchronized to the frame; 0–" + fmt(displayed.maximum, 3) + " " + displayed.unit + " (" + scaleLabel + ")." +
                (p.counts.outside > 0 ? " " + p.counts.outside +
                    " source particles outside transverse mesh." : "");
            $("fieldDetails").textContent = metadata.model + ". " + metadata.selection + ". " +
                metadata.timing + ". " +
                view.horizontal + "–" + view.vertical + " cross-section at " + plane.fixed_axis +
                "=0 nm. Source particles: " + p.counts.source +
                "; deposited on the native mesh: " + p.counts.deposited + "; outside that mesh: " +
                p.counts.outside + ". " +
                "Every source slice is represented at its display position; colors show the transverse electric-equivalent magnitude, not Ez or a sum of both source fields. " +
                "The dashed outline covers the selected snapshot sampling window over all saved frames. Outside the transverse mesh remains unsampled.";
            return;
        }
        $("fieldNote").textContent = source.label + " · Frame " + k + ": " + p.coverage.available +
            "/" + metadata.n_z +
            " source slices sampled (" + p.coverage.expected +
            " active). Active-encounter fields. Blank regions have no field samples. Colors: " +
            scaleLabel + ".";
        $("fieldDetails").textContent = source.label + " · " + source.quantity + " [" + source
            .unit + "], " + view.horizontal + "–" + view.vertical + " cross-section at " + plane
            .fixed_axis + "=0 nm. " +
            metadata.selection + ". " + metadata.timing + ". " + metadata.z_mapping + ". " +
            "Color scale is " + (scaleMode === "run" ?
                "fixed across this source and projection's run" : "automatic for this frame") +
            ". " +
            "The dashed outline is the sampling extent across both sources and the entire run, not simultaneous field coverage. Each colored band has the native slice width; unavailable and inactive bands remain transparent. A narrowing colored footprint describes recorded coverage, not collapse of the complete bunch field. This is a direct transverse-field cross-section, not an integral over the other transverse coordinate, a longitudinal Ez component, or a common laboratory E(z) field.";
    } else {
        const e = record.encounter,
            conventions = nativeFields.data.conventions;
        $("fieldNote").textContent = source.label + ": " + magnitude + ", 0–" + fmt(displayed.maximum, 3) +
            " " + displayed.unit + " (" + scaleLabel + "). " +
            "Frame " + k + ", slices " + e.beam1_slice + "/" + e.beam2_slice +
            ": pre-kick field; cloud at the end of the diagonal.";
        $("fieldDetails").textContent = source.label + " · " + source.quantity + " [" + source
            .unit + "] · " +
            (scaleMode === "run" ? "fixed scale across this source's run" :
                "automatic scale for this frame") + " · Cividis-derived palette. " +
            "Native x–y samples; nominal encounter z = " + fmt(e.nominal_z_nm / 1000, 4) +
            " µm; slices " + e.beam1_slice + " / " + e.beam2_slice +
            "; substep " + e.substep_index + "; " + e.phase + ". " + conventions.selection + "; " +
            conventions.scope +
            ". The cloud is the post-diagonal display state; the field is evaluated before the kick. " +
            (source.kind === "beam_beam" ?
                "This is an electric-equivalent native slice field, not a complete physical E/B field." :
                "This declared external layer is supplied by the dataset; the viewer does not calculate laser fields."
                );
    }
}
for (const id of ["fieldMode", "fieldSource", "fieldScale", "fieldData"]) $(id).addEventListener(
    "change", draw);
$("fieldOpacity").addEventListener("input", () => {
    $("fieldOpacityValue").textContent = Math.round(Number($("fieldOpacity").value) * 100) +
        "%";
    draw()
});
$("fieldPlane").addEventListener("click", () => setParticleProjection("zx"));
addEventListener("resize", () => {
    if (man) draw()
});
// Overlay controls can change height after toggling or wrapping. Queue a final
// render after that layout settles, using the same guard as frame playback.
if (typeof ResizeObserver === "function") {
    const particleLayoutObserver = new ResizeObserver(() => {
        if (man) draw()
    });
    for (const id of ["fieldControls", "projectionContext", "playback"]) particleLayoutObserver
        .observe($(id));
}
// ?run=<served directory> for a scriptable path in
const q = new URLSearchParams(location.search);
if (q.get("run")) {
    try {
        base = safeBase(q.get("run"));
        load().catch(fail)
    } catch (e) {
        fail(e)
    }
}
