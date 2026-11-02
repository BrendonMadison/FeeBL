"""Render the selected spectrum, sampler, polarization and peak figures.

The default inputs combine the completed ILC seed ensemble with explicitly
historical HALHF comparisons. This producer performs no native simulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import regenerate_paper as figures
from polarization_figure import generate as polarization
from qed_publication import DEFAULT_REDUCTION, export_products, copy_figures


def generate(output, comparisons, qed_reduction=DEFAULT_REDUCTION):
    """Replay the saved data and record every generated numerical artifact."""
    output.mkdir(parents=True, exist_ok=False)
    qed_products = output / "qed_products/figures"
    export_products("figures", qed_reduction, qed_products)
    qed_provenance = copy_figures(qed_products, output, qed_reduction)
    numerical = output / "numerical_data"
    figures.regenerate_studies(numerical)
    figures.plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    figures.collider_spectrum_figure(
        comparisons, output, "ilc_1000", "ILC 1 TeV", "ilc_spectrum"
    )
    figures.collider_spectrum_figure(
        comparisons, output, "halhf_250", "HALHF 250 GeV", "halhf_spectra_250"
    )
    figures.sampler_figure(
        output, figures.evaluate_sampler(figures.INPUTS / "sampler_records")
    )
    figures.confirmation_figures(output, numerical)
    polarization(output)
    receipt = {
        "scope": "No simulation. Focusing and pair comparisons use the verified complete QED ensembles. HALHF luminosity spectrum retains historical proxy covariance; ILC spectrum retains its completed twelve-seed covariance.",
        "qed_ensemble": qed_provenance,
        "producer_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "render_selected_figures.py",
                "regenerate_paper.py",
                "display_rebin.py",
                "polarization_figure.py",
                "qed_publication.py",
            )
        },
        "inputs": {
            path.relative_to(figures.BENCHMARKS)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(comparisons.iterdir())
            if path.is_file()
        },
        "auxiliary_inputs": {
            path.relative_to(figures.BENCHMARKS)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(
                [
                    figures.INPUTS / "manifest.json",
                    figures.DATA / "binned_energy_cuts.csv",
                    figures.DATA / "intercode_comparison.csv",
                ]
                + list((figures.INPUTS / "sampler_records").glob("*"))
            )
            if path.is_file()
        },
        "outputs": {
            path.relative_to(output)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "generation.json"
        },
    }
    (output / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pdfs": [p.name for p in output.glob("*.pdf")],
                "output_files": len(receipt["outputs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qed-reduction", type=Path, default=DEFAULT_REDUCTION)
    parser.add_argument(
        "--comparisons", type=Path, default=figures.DATA / "selected_comparison_inputs"
    )
    arguments = parser.parse_args()
    generate(arguments.output, arguments.comparisons, arguments.qed_reduction)
