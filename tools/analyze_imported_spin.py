#!/usr/bin/env python3
"""Compare matched radiation-free imported-spin runs and plot their changes.

The inputs are run directories containing imported_spin.json or
imported_spin.json.gz, or those files directly. Initial source-record
identities must match exactly. This
analysis assumes unit, initially longitudinal spins, as in the manufactured
Bhabha fixture. Its SEM describes variation across the supplied events at a
fixed beam realization; it does not estimate beam, mesh or model uncertainty.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


RUNS = ("field_on", "field_off", "refined")
SPECIES = ((11, "electron"), (-11, "positron"))


def read_input(path):
    """Read native JSON or a lossless gzip copy, retaining both byte identities.

    Compression changes the archive hash, but the decoded SHA256 still binds
    the exact original native output. Scientific values are never rounded or
    transformed during replay. A directory prefers its uncompressed original.
    """
    path = Path(path)
    if path.is_dir():
        path /= "imported_spin.json"
        if not path.is_file():
            path = path.with_suffix(".json.gz")
    archived = path.read_bytes()
    compressed = path.suffix.lower() == ".gz"
    content = gzip.decompress(archived) if compressed else archived
    provenance = dict(file=path.name, bytes=len(archived),
                      sha256=hashlib.sha256(archived).hexdigest(),
                      compression="gzip" if compressed else "none",
                      native_bytes=len(content),
                      native_sha256=hashlib.sha256(content).hexdigest())
    return json.loads(content), provenance


def number(value, label):
    """Require a real finite JSON number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def vector(value, label):
    """Read one finite three-vector without inferring missing components."""
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must have three components")
    return tuple(number(x, label) for x in value)


def dot(left, right):
    return math.fsum(a*b for a, b in zip(left, right))


def read_state(state, label):
    """Validate serialized helicity against its spin and momentum definition."""
    energy = number(state["signed_energy_GeV"], label + " energy")
    position = vector(state["position_nm"], label + " position")
    beta = vector(state["beta"], label + " beta")
    spin = vector(state["spin"], label + " spin")
    speed = math.sqrt(dot(beta, beta))
    if abs(energy) <= .00051099895 or not 0 < speed <= 1 + 1e-6:
        raise ValueError(f"{label}: invalid electron/positron energy or velocity")
    direction = tuple(x/speed for x in beta)
    helicity = dot(spin, direction)
    if abs(number(state["helicity"], label + " helicity") - helicity) > 4e-12:
        raise ValueError(f"{label}: serialized helicity does not close")
    norm_error = dot(spin, spin) - 1
    if abs(norm_error) > 2e-9:
        raise ValueError(f"{label}: this analysis requires unit spins")
    return dict(energy=energy, position=position, beta=beta, spin=spin,
                direction=direction, helicity=helicity, norm_error=norm_error)


def validate_run(payload, label):
    """Index complete initial/final records by source identity, never row order."""
    if payload.get("schema_version") != "feebl.imported-spin/1" or payload.get("mode") != 1:
        raise ValueError(f"{label}: unsupported imported-spin schema or mode")
    if payload.get("radiative_spin_flips") is not False:
        raise ValueError(f"{label}: expected explicitly radiation-free BMT data")
    records, labels = {}, set()
    for track in payload["tracks"]:
        source, pdg, native_label = track["source_record"], track["pdg_id"], track["label"]
        if type(source) is not int or source <= 0 or source in records:
            raise ValueError(f"{label}: invalid or duplicate source_record")
        if type(native_label) is not int or native_label in labels or type(pdg) is not int or pdg not in (11, -11):
            raise ValueError(f"{label}: invalid/duplicate label or species")
        labels.add(native_label)
        initial = read_state(track["initial"], f"{label} source {source} initial")
        final = read_state(track["final"], f"{label} source {source} final")
        if any((state["energy"] > 0) != (pdg == 11) for state in (initial, final)):
            raise ValueError(f"{label}: signed energy and PDG identity disagree")
        sign = 1 if initial["helicity"] >= 0 else -1
        initial_loss = .5*math.fsum((sign*s-n)**2 for s, n in zip(initial["spin"], initial["direction"]))
        if initial_loss > 5e-15:
            raise ValueError(f"{label}: initial spin must follow or oppose its actual momentum")
        # Equal to 1-h0*h_final for unit spins, but stable near zero loss.
        # Retain and report norm errors independently rather than renormalizing.
        loss = .5*math.fsum((sign*s-n)**2 for s, n in zip(final["spin"], final["direction"]))
        chi = number(track["max_chi"], f"{label} max_chi")
        if chi < 0:
            raise ValueError(f"{label}: negative max_chi")
        records[source] = dict(pdg=pdg, initial=initial, final=final, loss=loss,
                               initial_loss=initial_loss, chi=chi,
                               delta_h=final["helicity"]-initial["helicity"])
    if not records:
        raise ValueError(f"{label}: no imported tracks")
    return records


def estimate(values, scale=1):
    """Mean and sample SEM using centered arithmetic for small paired shifts."""
    if len(values) < 2:
        raise ValueError("At least two events per species are required for an SEM")
    mean = math.fsum(values)/len(values)
    sem = math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)*(len(values)-1)))
    return dict(mean=scale*mean, sem=abs(scale)*sem, n=len(values))


def relative_change(initial, changes):
    """Paired ratio uncertainty includes covariance with sampled initial P."""
    p0, delta = estimate(initial)["mean"], estimate(changes)["mean"]
    if p0 == 0:
        return dict(mean=None, sem=None, reason="relative change is undefined at zero sampled initial polarization")
    ratio = delta/p0
    influence = [(change-ratio*h0)/p0 for change, h0 in zip(changes, initial)]
    return dict(mean=100*ratio, sem=100*estimate(influence)["sem"], n=len(initial))


def analyze(payloads):
    """Produce matched species estimates and retain arrays used in the figure."""
    runs = {label: validate_run(payloads[label], label) for label in RUNS}
    baseline = runs["field_on"]
    for label in RUNS[1:]:
        if set(runs[label]) != set(baseline):
            raise ValueError(f"{label}: source_record population does not match field_on")
        for source, reference in baseline.items():
            other = runs[label][source]
            # Timings and runtime labels may differ with n_t. Initial phase
            # space, spin and species must be exactly the same serialized data.
            if other["pdg"] != reference["pdg"] or other["initial"] != reference["initial"]:
                raise ValueError(f"{label}: initial identity mismatch at source_record {source}")
    result = dict(schema_version="feebl.imported-spin-analysis/1", species={},
                  uncertainty="sample SEM of paired event changes conditional on one fixed beam realization; excludes beam-seed, mesh and model uncertainty",
                  sampling_unit="one equal-weight track per species per manufactured event; no joint two-track observable is estimated",
                  delta_P_definition="100*(P_final-P_initial), percent of unit polarization; negative means a decrease",
                  relative_delta_definition="100*(P_final-P_initial)/P_initial, percent; paired delta-method SEM",
                  retention_loss_definition="0.5*|sign(h_initial)*S_final-n_final|^2; equals 1-h0*h_final for unit initially longitudinal spins",
                  refinement_definition="refined minus field_on on exactly matched initial records; a numerical sensitivity, not an independent statistical sample",
                  cuts="all imported records; no energy, angle or luminosity selection")
    arrays = {}
    for pdg, species in SPECIES:
        sources = sorted(source for source in baseline if baseline[source]["pdg"] == pdg)
        if len(sources) < 2:
            raise ValueError(f"{species}: at least two matched events required")
        per_run, values = {}, {}
        for label in RUNS:
            records = [runs[label][source] for source in sources]
            initial = [r["initial"]["helicity"] for r in records]
            final = [r["final"]["helicity"] for r in records]
            changes = [r["delta_h"] for r in records]
            losses = [r["loss"] for r in records]
            values[label] = dict(changes=changes, losses=losses)
            per_run[label] = dict(P_initial=estimate(initial), P_final=estimate(final),
                                  delta_P_pp=estimate(changes, 100),
                                  relative_delta_percent=relative_change(initial, changes),
                                  retention_loss=estimate(losses),
                                  initial_geometric_loss=estimate([r["initial_loss"] for r in records]),
                                  max_spin_norm2_error=max(abs(r[state]["norm_error"]) for r in records for state in ("initial", "final")),
                                  max_chi=max(r["chi"] for r in records))
        difference = lambda key, a, b: [x-y for x, y in zip(values[a][key], values[b][key])]
        refinement = difference("changes", "refined", "field_on")
        result["species"][species] = dict(pdg_id=pdg, matched_events=len(sources), runs=per_run,
            control_subtracted_delta_P_pp=estimate(difference("changes", "field_on", "field_off"), 100),
            control_subtracted_retention_loss=estimate(difference("losses", "field_on", "field_off")),
            paired_refinement_delta_P_pp=estimate(refinement, 100),
            paired_refinement_retention_loss=estimate(difference("losses", "refined", "field_on")))
        arrays[species] = dict(values=values, refinement=refinement)
    return result, arrays


def plot_results(result, arrays, output, figure_width=10, stem="imported_spin_comparison"):
    """Draw capped conditional uncertainties and complete signed distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"electron": "#0072B2", "positron": "#D55E00"}
    markers = {"electron": "o", "positron": "s"}
    names = {"electron": r"$e^-$", "positron": r"$e^+$"}
    paper_size = figure_width <= 7
    fig, axes = plt.subplots(2, 2, figsize=(figure_width, 5.5 if paper_size else 7.1), constrained_layout=True)
    run_names = ["Field on", "Field off", "Refined"]
    combined = [x for species in arrays for label in ("field_on", "refined")
                for x in arrays[species]["values"][label]["changes"]]
    low, high = min(combined), max(combined)
    padding = max((high-low)*.06, abs(low)*.02, 1e-15)
    edges = [low-padding+(high-low+2*padding)*i/40 for i in range(41)]
    for species, offset in (("electron", -.07), ("positron", .07)):
        entry = result["species"][species]
        for axis, key, scale in ((axes[0, 0], "delta_P_pp", 1), (axes[0, 1], "retention_loss", 1)):
            values = [entry["runs"][label][key] for label in RUNS]
            axis.errorbar([i+offset for i in range(3)], [scale*v["mean"] for v in values],
                          yerr=[scale*v["sem"] for v in values], fmt=markers[species], markersize=4.5,
                          color=colors[species], capsize=4, elinewidth=1, label=names[species])
            axis.set_xticks(range(3), run_names)
        changes = arrays[species]["values"]
        for label, style in (("field_on", "-"), ("refined", "--")):
            axes[1, 0].hist(changes[label]["changes"], bins=edges,
                            histtype="step", linewidth=1.2, linestyle=style,
                            color=colors[species], label=names[species]+(" on" if label == "field_on" else " refined"))
        shift = entry["paired_refinement_delta_P_pp"]
        index = 0 if species == "electron" else 1
        axes[1, 1].errorbar(index, shift["mean"], yerr=shift["sem"], fmt=markers[species], markersize=5,
                          color=colors[species], capsize=5, elinewidth=1)
    axes[0, 0].set_ylabel(r"$P_{\rm final}-P_{\rm initial}$ [%]")
    axes[0, 1].set_ylabel("Mean unit-spin retention loss")
    axes[1, 0].set_xlabel(r"Signed per-track $h_{\rm final}-h_{\rm initial}$")
    axes[1, 0].ticklabel_format(axis="x", style="sci", scilimits=(-3, 3), useOffset=False)
    axes[1, 0].set_ylabel("Tracks / bin")
    axes[1, 1].set_xticks([0, 1], [r"$e^-$", r"$e^+$"])
    axes[1, 1].set_xlim(-.6, 1.6)
    axes[1, 1].set_ylabel(r"Refinement shift in $\Delta P$ [%]")
    titles = ("Polarization change", "Geometric retention", "Signed helicity changes", "Time-step sensitivity")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, fontsize=9 if paper_size else 11)
        axis.grid(alpha=.18)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        axis.tick_params(direction="in", top=True, right=True)
    for axis in (axes[0, 0], axes[1, 1]):
        axis.axhline(0, color=".45", linewidth=.7, zorder=0)
    axes[0, 0].legend(frameon=True, framealpha=.95, fontsize=8 if paper_size else 9, loc="upper right")
    axes[1, 0].legend(frameon=True, framealpha=.95, fontsize=8, loc="upper right")
    fig.suptitle("Imported-particle BMT transport\nConditional event SEM; one fixed beam realization", fontsize=10 if paper_size else 12)
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"{stem}.{suffix}", dpi=200, metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in RUNS:
        parser.add_argument("--"+label.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads, provenance = {}, {}
    for label in RUNS:
        payloads[label], provenance[label] = read_input(getattr(args, label))
    result, arrays = analyze(payloads)
    result["inputs"] = provenance
    args.output.mkdir(parents=True, exist_ok=True)
    plot_results(result, arrays, args.output)
    (args.output / "imported_spin_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "matched_events_per_species": {
        name: data["matched_events"] for name, data in result["species"].items()}}, indent=2))


if __name__ == "__main__":
    main()
