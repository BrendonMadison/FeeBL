"""Export manuscript tables from a verified independent-collision reduction.

No numerical or physics qualification is assigned here. The complete-collision
vectors define the SEM, including covariance between subprocess yields.
"""
from pathlib import Path

import numpy as np

from reduce_figures import digest, read_csv, read_json, save_json
from render_figures import verify_reduction


def mean_and_sem(values):
    """Return the mean and its sampling error for independent complete runs."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("A finite vector of independent runs is required")
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values)))


def sum_mean_and_sem(vectors):
    """Sum channels within each run before estimating the total uncertainty."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Expected a run-by-channel matrix")
    return mean_and_sem(vectors.sum(axis=1))


def value_with_sem(mean, sem, places):
    if not np.isfinite(mean) or not np.isfinite(sem) or sem < 0:
        raise ValueError("Invalid table value or standard uncertainty")
    return rf"${mean:,.{places}f}\pm{sem:,.{places}f}$".replace(",", r"\,")


def lep_table(reduction):
    """Match the three tabulated LEP observables to their published definitions."""
    scalars = {r["observable"]: r for r in read_csv(reduction/"lep_boundary_scalars.csv")}
    theta_inner, theta_outer = .0313, .0516  # radians
    focus_inner, focus_outer = 12.81, 11.19  # microradians
    bias = (-2/(theta_inner**-2-theta_outer**-2)
            *(focus_inner*1e-6/theta_inner**3-focus_outer*1e-6/theta_outer**3)*100)
    definitions = [
        ("Inner focusing [$\\mu$rad]", "inner_electron_mean_urad", focus_inner, 3),
        ("Outer focusing [$\\mu$rad]", "outer_electron_mean_urad", focus_outer, 3),
        ("Bias [\\%]", "electron_boundary_bias_percent", bias, 5),
    ]
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Observable & Reference~\cite{lep} & \FeeBL & Difference\\", r"\midrule"]
    records = []
    for label, name, reference, places in definitions:
        row = scalars[name]
        mean, sem = float(row["mean"]), float(row["standard_uncertainty"])
        difference = 100*(mean/reference-1)
        reference_text = f"{reference:.{5 if name.endswith('percent') else 2}f}"
        lines.append(f"{label} & ${reference_text}$ & {value_with_sem(mean, sem, places)}"
                     +rf" & ${difference:+.2f}\%$\\")
        records.append(dict(observable=name, mean=mean, standard_uncertainty=sem,
                            reference=reference, relative_difference_percent=difference))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)+"\n", records


def ilc_table(reduction):
    """Report luminosity-normalized channels, retaining their shared covariance."""
    models, lumis = {}, {}
    summary = read_json(reduction/"ilc_pair_summary.json")["comparisons"]
    for model in (1, 2):
        normalized = summary[f"model{model}_effective_cross_section"]
        mean = np.array(normalized["mean_mb"])
        covariance = np.array(normalized["covariance_of_mean_mb2"])
        if mean.shape != (3,) or covariance.shape != (3, 3) or np.any(mean < 0):
            raise ValueError("Expected three effective cross sections and their joint covariance")
        models[model] = list(zip(mean, np.sqrt(np.maximum(0, np.diag(covariance)))))
        models[model].append((mean.sum(), np.sqrt(max(0, covariance.sum()))))
        luminosity = summary[f"model{model}_luminosity"]
        index = luminosity["columns"].index("lumi_ee")
        # One inverse square metre is 10^-34 inverse microbarn.
        lumis[model] = (float(luminosity["mean"][index])*1e-34,
                        float(luminosity["standard_uncertainty"][index])*1e-34)
    refs = {}
    for code in ("gp", "cain"):
        rows = read_csv(reduction/f"ilc_model1_{code}_pull.csv")
        if [r["process"] for r in rows] != ["Breit-Wheeler", "Bethe-Heitler", "Landau-Lifshitz"]:
            raise ValueError("Reference subprocess order differs from the native channel vector")
        refs[code] = [float(row["reference"]) for row in rows]
        refs[code].append(sum(refs[code]))
    labels = ["Breit--Wheeler", "Bethe--Heitler", "Landau--Lifshitz", "Total"]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Quantity & \GP & \CAIN & \FeeBL\ model 1 & \FeeBL\ model 2\\", r"\midrule"]
    records = []
    for i, label in enumerate(labels):
        ref_text = [f"{refs[code][i]:.2f}".replace(",", r"\,") for code in ("gp", "cain")]
        lines.append(label+" & "+" & ".join(ref_text+[value_with_sem(*models[m][i], 3) for m in (1, 2)])+r"\\")
        records.append(dict(observable=label, units="mb", reference_gp=refs["gp"][i], reference_cain=refs["cain"][i],
            **{f"model{m}": dict(mean=models[m][i][0], standard_uncertainty=models[m][i][1]) for m in (1, 2)}))
    lines.append(r"$\mathcal L_{\rm BX}$ [$\mu\mathrm b^{-1}$] & 1.82 & 1.82 & "
                 +" & ".join(value_with_sem(*lumis[m], 4) for m in (1, 2))+r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    records.append(dict(observable="luminosity_per_crossing", units="inverse microbarn", reference_gp=1.82,
        reference_cain=1.82, **{f"model{m}": dict(mean=lumis[m][0], standard_uncertainty=lumis[m][1]) for m in (1, 2)}))
    return "\n".join(lines)+"\n", records


def export_tables(source, reduction, output):
    receipt = verify_reduction(reduction, source)
    lep_text, lep_values = lep_table(reduction)
    ilc_text, ilc_values = ilc_table(reduction)
    output.mkdir(parents=True, exist_ok=False)
    (output/"lep_focusing_ensemble.tex").write_text(lep_text, encoding="utf-8")
    (output/"ilc_pair_ensemble.tex").write_text(ilc_text, encoding="utf-8")
    save_json(output/"table_values.json", dict(lep=lep_values, ilc=ilc_values,
        uncertainty="Complete-collision joint count/luminosity covariance; shared normalization and interchannel correlations retained",
        qualification="Numerical export only; source-model and convergence interpretation is separate"))
    save_json(output/"table_receipt.json", dict(status="EXPORTED_FROM_VERIFIED_REDUCTION",
        analysis_receipt_sha256=digest(reduction/"analysis_receipt.json"),
        source_native_identity=receipt["source_native_identity"],
        producer_sha256=receipt["producer_sha256"],
        table_exporter_sha256=digest(Path(__file__)),
        output_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_tables(args.source, args.reduction, args.output)
