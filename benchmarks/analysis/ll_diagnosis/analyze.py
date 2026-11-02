"""Reduce the completed 36-collision controls and produce the pair-process figure.

No simulator is launched. The output directory must be new, and an incomplete
campaign cannot produce a publication summary.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from contracts import digest, load_campaign
from definitions import ARMS, CHANNELS, REFERENCE_TARGETS
from ensemble_statistics import moments, summarize
from plotting import plot_controls


def save_json(path, value):
    """Reject nonfinite JSON numbers rather than serializing invalid results."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save_csv(path, records):
    """Write named units and columns with stable ordering."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def references(summary):
    """Attach source targets only to controls with corresponding beam-size flags."""
    lookup = {(row["arm"], row["channel"]): row for row in summary["records"]}
    rows = []
    for arm, code, table, values in REFERENCE_TARGETS:
        for channel, reference in zip(CHANNELS, values):
            if reference is None:
                continue
            measured = lookup[arm, channel]
            rows.append(dict(arm=arm, channel=channel, source_code=code, source_table=table,
                reference_mb=reference, simulation_mb=measured["cross_section_mb"],
                simulation_sem_mb=measured["sem_mb"],
                difference_percent=100*(measured["cross_section_mb"]/reference-1),
                source="https://doi.org/10.1103/PhysRevSTAB.9.034402",
                qualification="Matching suppression/cut control, not an identical implementation. BDK includes electroweak diagrams; reference statistical errors are not supplied."))
    return rows


def profile_interaction(summary):
    """Test whether the spatial-profile change depends on the hard-process model."""
    arms = summary["arms"]
    selection = np.zeros((3, 3*len(arms)))
    for arm, coefficient in (("exact_profile", 1), ("exact_ring", -1),
                             ("finite_mass_epa_profile", -1), ("finite_mass_epa_ring", 1)):
        for channel in range(3):
            selection[channel, 3*arms.index(arm)+channel] = coefficient
    covariance = selection@np.array(summary["covariance_of_mean_mb2"])@selection.T
    return dict(label="Model/profile interaction", definition="(full profile - full ring) - (EPA profile - EPA ring)",
        change_mb=(selection@summary["mean_mb"]).tolist(), covariance_mb2=covariance.tolist(),
        sem_mb=np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
        qualification="Difference-in-differences with measured paired covariance; controls need not be additive.")


def latex_changes(summary, output):
    """Write a compact table of paired differences with absolute mb units."""
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Control change & $\Delta\sigma_{\rm BW}$ [mb] & $\Delta\sigma_{\rm BH}$ [mb] & $\Delta\sigma_{\rm LL}$ [mb]\\",
             r"\midrule"]
    for row in [*summary["changes"], summary["profile_interaction"]]:
        values = [f"${mean:+.4f}\\pm{sem:.4f}$" for mean, sem in zip(row["change_mb"], row["sem_mb"])]
        lines.append(row["label"]+" & "+" & ".join(values)+r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output/"paired_control_changes.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def reduce(campaign, inputs, output):
    """Verify native evidence, independently check runner statistics, then render."""
    if output.exists():
        raise ValueError("Use a new output directory")
    vectors, arms, saved, records, provenance = load_campaign(campaign, inputs)
    summary = summarize(vectors, arms)
    np.testing.assert_allclose(summary["mean_mb"], np.array(saved["ratio_of_means"])/1000, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(summary["covariance_of_mean_mb2"],
        np.array(saved["delta_method_covariance_of_ratio_of_means"])/1e6, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(summary["jackknife_covariance_mb2"],
        np.array(saved["delete_one_jackknife_covariance_of_ratio_of_means"])/1e6, rtol=1e-10, atol=1e-11)
    summary["profile_interaction"] = profile_interaction(summary)
    summary["model_scope"] = "All arms use unpolarized real photons. Legacy EPA already has corrected Born normalization. Full-QED BH/LL do not use the EPA hard pair_q2 veto."
    summary["population_scope"] = "Admitted electron/positron daughters only; 5 MeV production cut, pair_ratio=1, no final recut and no scattered spectators. Exact daughter assignment follows the minimum-virtuality spectator rule."
    summary["domain_scope"] = "Expanded extents change field/source coverage and virtual-photon acceptance together, at fixed transverse cell spacing. beam_size=0 places photons locally; it does not model infinite beams."
    diagnostics = []
    for (arm, seed), (vector, receipt) in sorted(records.items()):
        exact = receipt["output_check"]["exact_kernel_diagnostics"]
        diagnostics.append(dict(arm=arm, seed=seed, N_BW=vector[0], N_BH=vector[1], N_LL=vector[2],
            L_ee_inverse_microbarn=vector[3], main_grid_visit_rejection_beam1=receipt["main_grid_rejection_fractions"][0],
            main_grid_visit_rejection_beam2=receipt["main_grid_rejection_fractions"][1],
            maximum_BH_kernel_contribution=exact.get("pairs_1", {}).get("max_contribution", "not_applicable"),
            maximum_LL_kernel_contribution=exact.get("pairs_2", {}).get("max_contribution", "not_applicable"),
            wall_seconds=receipt["wall_seconds"], peak_rss_MiB=receipt["resources"]["peak_rss_MiB"]))
    output.mkdir(parents=True)
    save_json(output/"control_summary.json", summary)
    save_csv(output/"effective_cross_sections.csv", summary["records"])
    paired_rows = [dict(label=row["label"], arm=row["arm"], baseline=row["baseline"], channel=channel,
        change_mb=row["change_mb"][i], sem_mb=row["sem_mb"][i], definition=row["definition"])
        for row in summary["changes"] for i, channel in enumerate(CHANNELS)]
    save_csv(output/"paired_control_changes.csv", paired_rows)
    save_csv(output/"reference_control_targets.csv", references(summary))
    save_csv(output/"collision_diagnostics.csv", diagnostics)
    mean, covariance = moments(vectors)
    np.savez_compressed(output/"joint_collision_vectors.npz", vectors=vectors, mean=mean,
                        covariance_of_mean=covariance, arms=np.array(arms), seeds=np.array(saved["seeds"]))
    latex_changes(summary, output)
    plot_controls(summary, output)
    producers = {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    save_json(output/"analysis_receipt.json", dict(status="COMPLETE_VERIFIED_REDUCTION", input_provenance=provenance,
        producer_sha256=producers, output_sha256={path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
        qualification="Statistical reduction of all 36 approved cases; this status is not physics convergence or equivalence certification."))
    return summary


def main():
    """Require explicit paths so a historical campaign is never substituted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reduce(args.campaign, args.inputs, args.output)
    print("Verified all 36 collisions and wrote the three-channel controls figure and paired-change table.")


if __name__ == "__main__":
    main()
