"""Write tabular-only manuscript fragments and case-to-result traceability.

Numerical tables are derived from the same CSV records as their plots. Physics
definitions, installation examples and reference quotations remain authored
LaTeX. Historical results retain their originating cards and source revisions.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from comparison_statistics import BENCHMARKS, read_rows, write_rows
from qed_publication import DEFAULT_REDUCTION, campaign_cases, digest, export_products, verified_products

DATA = BENCHMARKS / "example_output"


def latex(value):
    """Escape plain labels without interpreting them as LaTeX instructions."""
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def tabular(path, headers, rows, alignment=None):
    """Write one booktabs table body; captions and float placement stay in the paper."""
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    text = [
        r"\begin{tabular}{@{}" + alignment + r"@{}}",
        r"\toprule",
        " & ".join(headers) + r"\\",
        r"\midrule",
    ]
    text.extend(" & ".join(str(value) for value in row) + r"\\" for row in rows)
    text += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def mean_sem(mean, sem, digits=3, scale=1):
    """Keep two significant uncertainty digits and match the mean's decimal place."""
    value = float(mean) / scale
    result = f"{value:.{digits}f}"
    if sem != "" and sem is not None:
        error = float(sem) / scale
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid seed SEM")
        places = max(0, 1 - math.floor(math.log10(error))) if error else digits
        result = "$" + f"{value:.{places}f}" + r"\pm" + f"{error:.{places}f}" + "$"
    return result


def write_confirmation(output, scan_data=DATA / "scans"):
    """Generate selected-optics means and ratios, with three-seed SEMs."""
    rows = read_rows(scan_data / "confirmation_metrics.csv")
    indexed = {(r["kind"], r["name"], r["metric"]): r for r in rows}
    # Locate the paired group by metric and family rather than relying on a
    # display name inherited from an earlier analysis directory.
    result = []
    for label, kind, suffix, metric, digits in (
        (r"Baseline $\sigma_G$ [MeV]", "group", "baseline", "core_sigma", 2),
        (r"Selected $\sigma_G$ [MeV]", "group", "selected", "core_sigma", 2),
        ("Paired width ratio", "paired", "", "core_sigma", 4),
        ("Paired luminosity ratio", "paired", "", "luminosity_per_crossing", 4),
        ("Pre-ISR resonance-proxy gain", "paired", "", "resonance_luminosity", 3),
    ):
        values = []
        for machine in ("ilc", "halhf"):
            if kind == "group":
                row = indexed[(kind, machine + "_" + suffix, metric)]
            else:
                choices = [
                    r
                    for r in rows
                    if r["kind"] != "group"
                    and r["name"].startswith(machine)
                    and (r["metric"] == metric or r["metric"] == metric + "_ratio")
                ]
                if len(choices) != 1:
                    # Metric spelling is an input contract, not a fuzzy match.
                    raise ValueError(f"Expected one paired {machine}/{metric} record")
                row = choices[0]
            values.append(mean_sem(row["mean"], row["sem"], digits))
        if len(values) == 2:
            result.append([label, *values])
    tabular(output / "confirmation.tex", ["Quantity", "ILC/LCF", "HALHF"], result)
    intervals = [
        [
            latex(r["name"]),
            latex(r["metric"]),
            r["n"],
            f"{float(r['mean']):.6g}",
            f"{float(r['sem']):.3g}",
            f"[{float(r['t95_low']):.6g}, {float(r['t95_high']):.6g}]",
        ]
        for r in rows
        if r["mean"] and r["sem"]
    ]
    write_rows(output / "confirmation_statistics.csv", rows)
    tabular(
        output / "confirmation_intervals.tex",
        ["Group", "Metric", "$n$", "Mean", "SEM", r"95\% $t$ interval"],
        intervals,
    )


def write_high_dispersion(output, scan_data=DATA / "scans"):
    """Generate observed widths with separate analysis-resolution sensitivity."""
    source = read_rows(scan_data / "high_dispersion.csv")
    rows = []
    for machine in ("ilc", "halhf"):
        for item in sorted(
            [r for r in source if r["machine"] == machine and r["role"] == "scan"],
            key=lambda r: float(r["dispersion_mm"]),
        ):
            rows.append(
                [
                    "ILC/LCF" if machine == "ilc" else "HALHF",
                    f"{float(item['dispersion_mm']):.3f}",
                    item["n"],
                    mean_sem(
                        item["luminosity_cm2_per_second_mean"],
                        item["luminosity_cm2_per_second_SEM"],
                        3,
                        1e32,
                    ),
                    mean_sem(
                        item["observed_peak_FWHM_MeV_mean"],
                        item["observed_peak_FWHM_MeV_SEM"],
                        3,
                    ),
                    f"{100*float(item['maximum_per_seed_representation_change']):.2f}",
                ]
            )
    tabular(
        output / "high_dispersion.tex",
        [
            "Family",
            "$|D|$ [mm]",
            "Seeds",
            r"$\mathcal L/10^{32}$",
            "FWHM [MeV]",
            r"Resolution [\%]",
        ],
        rows,
    )
    fcc = json.loads((DATA / "scans/fcc_crabwaist.json").read_text())
    widths = json.loads((scan_data / "fcc_resolution_summary.json").read_text())
    by_name = {row["case"]: row["resolution"] for row in widths["cases"]}
    rows = [
        [f"{r['dispersion_mm']:.0f}", f"{r['luminosity_cm2_s']/1e34:.3f}",
         f"{by_name[r['name']]['fwhm_MeV']:.2f}",
         f"[{by_name[r['name']]['resolution_low_MeV']:.2f}, "
         f"{by_name[r['name']]['resolution_high_MeV']:.2f}]"]
        for r in fcc["rows"]
        if r["role"] == "single_seed_pilot"
    ]
    tabular(
        output / "fcc_luminosity.tex",
        [
            "$|D|$ [mm]",
            r"$\mathcal L$ [$10^{34}$ cm$^{-2}$ s$^{-1}$]",
            "FWHM [MeV]",
            "Resolution range [MeV]",
        ],
        rows,
        "rrrr",
    )


def write_polarization(output):
    """Generate later-code comparisons without averaging different simulators."""
    rows = read_rows(DATA / "intercode_comparison.csv")
    result = [
        [
            latex(row["case"].replace("_", " ")),
            f"{float(row['GPpp_pp']):.2f}",
            f"{float(row['CAIN_pp']):.2f}",
            f"{float(row['local_axis_loss_pp']):.4f}",
            row["local_particles"],
        ]
        for row in rows
    ]
    tabular(
        output / "intercode_depolarization.tex",
        ["Optics", r"GUINEA-PIG++ [\%]", r"CAIN [\%]", r"FeeBL [\%]", "$N_m$"],
        result,
    )


def card_assignments(path):
    """Read explicit native card assignments, keeping expressions unevaluated."""
    text = "\n".join(
        line.split("#", 1)[0] for line in Path(path).read_text().splitlines()
    )
    return {
        key: value.strip()
        for key, value in re.findall(r"([A-Za-z_][A-Za-z_0-9.]*)\s*=\s*([^;]+);", text)
    }


def input_tables(output, manifest, qed_cases, qed_receipt):
    """Export numerical settings for every case plus a compact collider table."""
    parameters = []
    current_cases = [dict(id=case["name"], accelerator=case["accelerator"],
        parameters=case["parameters"], group=case["group"],
        inputs={"acc.dat": "cases/qed_figure_campaign/"+case["name"]+"/acc.dat"},
        origin_source_revision=qed_receipt["source_native_identity"]) for case in qed_cases]
    # Both model arms share the complete card except for seed and pair_model.
    # Check this before presenting one representative numerical input row.
    for group in ("ilc_pairs", "halhf_pairs"):
        cards = [card_assignments(BENCHMARKS/case["inputs"]["acc.dat"])
                 for case in current_cases if case["group"] == group]
        common = [{key:value for key,value in card.items() if key not in ("rndm_seed", "pair_model")}
                  for card in cards]
        if len(common) != 24 or any(card != common[0] for card in common[1:]):
            raise ValueError("Pair-model arms do not share the stated input parameters: "+group)
    for case in [*manifest["cases"], *current_cases]:
        card = card_assignments(BENCHMARKS / case["inputs"]["acc.dat"])
        row = {
            "case_id": case["id"],
            "accelerator": case["accelerator"],
            "parameters": case["parameters"],
            "card": case["inputs"]["acc.dat"],
            "card_sha256": digest(BENCHMARKS/case["inputs"]["acc.dat"]),
            "case_group": case.get("group", "historical"),
            "origin_source_revision": case.get("origin_source_revision")
            or "current source",
        }
        for key in (
            "energy",
            "particles",
            "beta_x",
            "beta_y",
            "emitt_x",
            "emitt_y",
            "sigma_z",
            "espread",
        ):
            for beam in (1, 2):
                row[f"{key}.{beam}"] = card.get(
                    f"{key}.{beam}", card.get(key, "parser default")
                )
        for key in (
            "n_m",
            "n_x",
            "n_y",
            "n_z",
            "n_t",
            "grids",
            "cut_x",
            "cut_y",
            "cut_z",
            "rndm_seed",
            "crab_full_crossing_angle",
            "crab_mode",
            "do_isr",
            "f_rep",
            "n_b",
            "pair_model", "beam_pair", "pair_ratio", "pair_ecut", "beam_size",
            "do_photon_lumi", "electron_ratio", "do_photons", "photon_timing",
            "photon_angle", "photon_polarisation", "do_coherent", "coherent_model",
            "do_compt", "store_pairs", "track_pairs", "load_events", "do_bhabhas",
            "store_photons",
        ):
            row[key] = card.get(key, "parser default")
        row["cut_z_expression"] = row["cut_z"]
        if case in current_cases:
            # A fixed cut or a scalar multiple of the first bunch's rms length
            # is supported. Preserve the original expression in the next column.
            expression = re.fullmatch(r"\s*([0-9.]+)\s*\*\s*sigma_z\.1\s*", row["cut_z"])
            cut_z = (float(expression[1])*float(row["sigma_z.1"]) if expression
                     else float(row["cut_z"]))
            if not math.isfinite(cut_z) or cut_z <= 0:
                raise ValueError("Invalid QED longitudinal domain")
            row["cut_z"] = f"{cut_z:.12g}"
        parameters.append(row)
    write_rows(output / "input_parameters.csv", parameters)
    labels = {
        "lep_field01": "LEP focusing",
        "ilc_m1_seed01": "ILC pairs",
        "halhf_m1_seed01": "HALHF 250 pairs",
        "ilc1000_tdr_b1b_gaussian_proxy": "ILC 1 TeV",
        "halhf250_wide_domain": "HALHF 250 spectrum",
        "halhf380": "HALHF 380",
        "halhf550": "HALHF 550",
    }
    table = []
    for row in parameters:
        if row["case_id"] not in labels:
            continue
        paired = (
            lambda key: row[key + ".1"]
            if row[key + ".1"] == row[key + ".2"]
            else row[key + ".1"] + ", " + row[key + ".2"]
        )
        table.append(
            [
                labels[row["case_id"]],
                paired("energy"),
                paired("particles"),
                paired("beta_x"),
                paired("beta_y"),
                row["n_m"],
                row["n_x"] + r"$\times$" + row["n_y"] + r"$\times$" + row["n_z"],
            ]
        )
    tabular(
        output / "collider_inputs.tex",
        [
            "Case",
            "$E_{1,2}$ [GeV]",
            "$N_{1,2}/10^{10}$",
            r"$\beta_x$ [mm]",
            r"$\beta_y$ [mm]",
            "$N_m$",
            "Grid",
        ],
        table,
    )
    return parameters


def polarization_input_table(output):
    """Evaluate the cards used for the four 2009 luminosity-weighted spin cases."""
    choices = (
        ("Nominal", "ilc_depol_beta20_40k", 639, 5.7),
        ("Low N", "rimbault2009_low_n", 474, 3.5),
        ("Large Y", "rimbault2009_large_y_40k", 474, 9.9),
        ("Low P", "rimbault2009_low_p_40k", 474, 3.8),
    )
    mass_path = BENCHMARKS.parent / "src/physconst.h"
    mass = float(re.search(r"#define\s+EMASS\s+([^\s]+)", mass_path.read_text())[1])
    rows, table = [], []
    for label, case, published_x, published_y in choices:
        path = BENCHMARKS / "cases" / case / "acc.dat"
        card = card_assignments(path)
        scalar = lambda key: float(card.get(key + ".1", card.get(key)))
        energy, beta_x, beta_y = (scalar(key) for key in ("energy", "beta_x", "beta_y"))
        emitt_x, emitt_y = scalar("emitt_x"), scalar("emitt_y")
        gamma = energy / mass
        sigma_x = math.sqrt(emitt_x * 1e-6 * beta_x * 1e-3 / gamma) * 1e9
        sigma_y = math.sqrt(emitt_y * 1e-6 * beta_y * 1e-3 / gamma) * 1e9
        row = {
            "case": label,
            "card": path.relative_to(BENCHMARKS).as_posix(),
            "card_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "energy_per_beam_GeV": energy,
            "particles_per_beam": scalar("particles") * 1e10,
            "normalized_emittance_x_mm_mrad": emitt_x,
            "normalized_emittance_y_mm_mrad": emitt_y,
            "beta_x_mm": beta_x,
            "beta_y_mm": beta_y,
            "sigma_x_nm_evaluated": sigma_x,
            "sigma_y_nm_evaluated": sigma_y,
            "source_sigma_x_nm_rounded": published_x,
            "source_sigma_y_nm_rounded": published_y,
            "sigma_z_um": scalar("sigma_z"),
            "cut_x_nm": scalar("cut_x"),
            "cut_y_nm": scalar("cut_y"),
            "cut_z_um_evaluated": 3.5 * scalar("sigma_z"),
            "macroparticles_per_beam": int(scalar("n_m")),
            "n_x": int(scalar("n_x")),
            "n_y": int(scalar("n_y")),
            "n_z": int(scalar("n_z")),
            "substeps": int(scalar("n_t")),
            "initial_fixed_axis_spin_z_1": float(card["polar_z.1"]),
            "initial_fixed_axis_spin_z_2": float(card["polar_z.2"]),
            "intrinsic_energy_spread": scalar("espread"),
            "full_crossing_angle_rad": scalar("crab_full_crossing_angle"),
            "ecm_high_counter_threshold_GeV": scalar("ecm_min"),
            "comparison_selection": "inclusive luminosity-weighted fixed-axis spin product; ecm_min labels a separate high-energy counter",
            "primary_source": "https://doi.org/10.1088/1742-6596/198/1/012006",
            "source_locator": "Table 1 (printed page 2); Section 4 and Table 5 (printed page 5)",
        }
        if card["cut_z"].replace(" ", "") != "3.5*sigma_z.1":
            raise ValueError("Unexpected longitudinal-domain expression")
        rows.append(row)
        table.append(
            [
                label,
                f"{scalar('particles'):g}",
                f"{emitt_x:g}/{emitt_y:g}",
                f"{beta_x:g}/{beta_y:g}",
                f"{sigma_x:.1f}/{sigma_y:.2f}",
                f"{scalar('sigma_z'):g}",
                f"{int(scalar('n_m')):,}",
            ]
        )
    write_rows(output / "polarization_inputs.csv", rows)
    tabular(
        output / "polarization_inputs.tex",
        [
            "Case",
            r"$N/10^{10}$",
            r"$\epsilon_{nx}/\epsilon_{ny}$",
            r"$\beta_x/\beta_y$",
            r"$\sigma_x/\sigma_y$",
            r"$\sigma_z$",
            r"$N_m$",
        ],
        table,
    )
    (output / "polarization_inputs_provenance.json").write_text(
        json.dumps(
            {
                "source_url": rows[0]["primary_source"],
                "source_locator": rows[0]["source_locator"],
                "definitions": {
                    "N": "physical particles per bunch",
                    "epsilon_n": "normalized rms emittance in mm mrad",
                    "beta": "IP beta function in mm",
                    "sigma_xy": "evaluated rms transverse size in nm",
                    "sigma_z": "rms bunch length in micrometres",
                    "N_m": "simulation macroparticles per beam",
                },
                "evaluation": "sigma=sqrt(epsilon_n*beta/gamma); gamma=E/EMASS with native mass convention and explicit SI conversion",
                "native_electron_mass_GeV": mass,
                "native_constants_sha256": hashlib.sha256(
                    mass_path.read_bytes()
                ).hexdigest(),
                "common_settings": "250 GeV per beam; head-on; zero incoming energy spread; fixed-axis spins +1 and -1; 128x128x32 cells and two substeps",
                "nominal_scope": "2009 nominal beta_x=20 mm; the separate 2006 nominal card uses 21 mm and must retain its own label",
                "name_scope": "Low N is the 2009 parameter set in Table 1. Equivalence with the earlier Low Q set is not established; the names are not substituted.",
                "statistics_scope": "Low N has 10000 macroparticles; nominal, Large Y and Low P shown here use 40000. The Large Y/Low P sensitivity to 10000 particles remains separate.",
                "producer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def write_imported_spin(output, result):
    """Export event-paired polarization changes without discarding covariance."""
    rows, records = [], []
    labels = {
        "field_on": r"Field on, $N_t=2$",
        "field_off": r"Zero field, $N_t=2$",
        "refined": r"Field on, $N_t=4$",
    }
    for run, label in labels.items():
        values = []
        for species in ("electron", "positron"):
            data = result["species"][species]["runs"][run]
            change = data["delta_P_pp"]
            values.append(
                "$0$"
                if change["mean"] == 0 and change["sem"] == 0
                else mean_sem(change["mean"], change["sem"])
            )
            records.append(
                {
                    "run": run,
                    "species": species,
                    "events": result["species"][species]["matched_events"],
                    "initial_polarization": data["P_initial"]["mean"],
                    "final_polarization": data["P_final"]["mean"],
                    "delta_p_percentage_points": change["mean"],
                    "conditional_paired_event_sem_pp": change["sem"],
                    "retention_loss": data["retention_loss"]["mean"],
                    "conditional_retention_sem": data["retention_loss"]["sem"],
                }
            )
        rows.append([label, *values])
    tabular(
        output / "imported_spin.tex",
        ["Calculation", r"Electron $\Delta P$ [\%]", r"Positron $\Delta P$ [\%]"],
        rows,
    )
    write_rows(output / "imported_spin_statistics.csv", records)


def build_tables(
    output, scan_data=DATA / "scans", imported_spin=None, include_imported_spin=False,
    qed_reduction=DEFAULT_REDUCTION, qed_tables=None
):
    """Generate manuscript table bodies and row-level numerical source maps."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if Path(scan_data) == DATA / "scans":
        # Standalone exports must use the same current estimator as the figures.
        from reduce_study_sources import regenerate
        scan_data = output / "numerical_data"
        regenerate(scan_data)
    if qed_tables is None:
        qed_tables = output / "qed_products"
        export_products("tables", qed_reduction, qed_tables)
    else:
        verified_products(qed_tables, qed_reduction, "tables")
    qed_cases, qed_receipt = campaign_cases(qed_reduction)
    for name in ("lep_focusing_ensemble.tex", "ilc_pair_ensemble.tex", "table_values.json", "table_receipt.json"):
        shutil.copyfile(Path(qed_tables)/name, output/name)
    manifest = json.loads((BENCHMARKS / "manifest.json").read_text())
    input_tables(output, manifest, qed_cases, qed_receipt)
    polarization_input_table(output)
    write_confirmation(output, scan_data)
    write_high_dispersion(output, scan_data)
    write_polarization(output)
    selected_imported_spin = include_imported_spin or imported_spin is not None
    if selected_imported_spin:
        if imported_spin is None:
            imported_spin = json.loads(
                (DATA / "imported_spin/imported_spin_comparison.json").read_text()
            )
        write_imported_spin(output, imported_spin)
    mapping = {
        "lep_focusing.pdf": {
            "cases": [case["name"] for case in qed_cases if case["group"] == "lep_on"],
            "controls": [case["name"] for case in qed_cases if case["group"] == "lep_zero"],
            "source": "verified QED ensemble reduction",
            "analysis_receipt_sha256": digest(Path(qed_reduction)/"analysis_receipt.json"),
            "reproduction": "ten independent field realizations with paired event legs and azimuth strata",
        },
        "ilc_pair_yields.pdf": {
            "cases": [case["name"] for case in qed_cases if case["group"] == "ilc_pairs"],
            "source": "verified QED ensemble reduction; pair models 1 and 2",
            "analysis_receipt_sha256": digest(Path(qed_reduction)/"analysis_receipt.json"),
            "reproduction": "independent-seed statistical",
        },
        "ilc_spectrum.pdf": {
            "cases": ["ilc1000_tdr_b1b_gaussian_proxy"],
            "source": "generated comparisons/ilc_1000_*",
            "reproduction": "independent-seed statistical",
        },
        "halhf_spectra_250.pdf": {
            "cases": ["halhf250_wide_domain"],
            "source": "generated comparisons/halhf_250_*",
            "reproduction": "independent-seed statistical",
        },
        "halhf_pair_populations.pdf": {
            "cases": [case["name"] for case in qed_cases if case["group"] == "halhf_pairs"],
            "source": "verified QED ensemble reduction; pair models 1 and 2",
            "analysis_receipt_sha256": digest(Path(qed_reduction)/"analysis_receipt.json"),
            "reproduction": "independent-seed final pair density and occupied support; published raster is support only",
        },
        "sampler_pulls.pdf": {
            "cases": ["analysis/sampler_probe.cc"],
            "source": "analysis_inputs/sampler_records",
            "reproduction": "sampler_oracle.py regenerates exact-Bessel target and sampling errors; native trials statistical",
        },
        "imported_spin_transport.pdf": {
            "cases": [
                "bhabha_spin_field_on",
                "bhabha_spin_field_off",
                "bhabha_spin_time_refined",
            ],
            "source": "imported_spin/{field_on,field_off,time_refined}.json.gz",
            "reproduction": "tools/analyze_imported_spin.py validates identity and recomputes conditional paired-event statistics",
        },
        "higgs_ilc_selected_vs_baseline_peak_spectrum.pdf": {
            "cases": [
                c["id"] for c in manifest["cases"] if c["id"].startswith("ilc_confirm_")
            ],
            "source": "scans/ilc_selected_vs_baseline_peak_spectrum.csv",
            "reproduction": "paired independent-seed study",
        },
        "higgs_halhf_selected_vs_baseline_peak_spectrum.pdf": {
            "cases": [
                c["id"]
                for c in manifest["cases"]
                if c["id"].startswith("halhf_confirm_")
            ],
            "source": "scans/halhf_selected_vs_baseline_peak_spectrum.csv",
            "reproduction": "paired independent-seed study",
        },
        "three_collider_dispersion_summary.pdf": {
            "cases": [
                c["id"]
                for c in manifest["cases"]
                if c["id"].startswith(("high_", "fcc_cw_"))
            ],
            "source": "scans/high_dispersion.csv; scans/fcc_crabwaist.json",
            "reproduction": "historical native extraction; mesh and representation sensitivities retained",
        },
        "fcc_peak_fragmentation.pdf": {
            "cases": ["fcc_cw_d000_pilot_s1", "fcc_cw_d100_pilot_s1"],
            "source": "analysis_inputs/native/fcc_cw_*.fne.gz",
            "reproduction": "fixed-bin regrouping of retained native spectra",
        },
        "common_finite_angle_extrapolations.pdf": {
            "cases": ["analysis/gaussian_overlap.py"],
            "source": "analysis_inputs/gaussian_overlap_inputs.json",
            "reproduction": "deterministic model; no native continuation claim",
        },
    }
    for energy in (380, 550):
        mapping[f"halhf_spectra_{energy}.pdf"] = {
            "cases": [f"halhf{energy}"],
            "source": f"generated comparisons/halhf_{energy}_*",
            "reproduction": "independent-seed statistical",
        }
    if not selected_imported_spin:
        mapping.pop("imported_spin_transport.pdf")
    (output / "paper_traceability.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )
    table_sources = {
        "collider_inputs.tex": "manifest.json and every cases/*/acc.dat; complete parameters in input_parameters.csv",
        "polarization_inputs.tex": "four exact depolarization cards; evaluated rms sizes using src/physconst.h; Rimbault2009 Table1 and Table5 definitions",
        "confirmation.tex": "regenerated numerical_data/confirmation_metrics.csv; retained native spectra and qualified fit records",
        "confirmation_intervals.tex": "regenerated numerical_data/confirmation_metrics.csv; paired-seed Student-t intervals",
        "high_dispersion.tex": "regenerated numerical_data/high_dispersion.csv; scale-specific observed FWHM and all-phase bin-resolution checks",
        "fcc_luminosity.tex": "example_output/scans/fcc_crabwaist.json and regenerated numerical_data/fcc_resolution_summary.json; luminosity, observed FWHM and numerical resolution envelope",
        "intercode_depolarization.tex": "example_output/intercode_comparison.csv; distinct published definitions retained",
        "imported_spin.tex": "regenerated numerical_data/imported_spin_comparison.json; matched native record reduction",
    }
    if not selected_imported_spin:
        table_sources.pop("imported_spin.tex")
    traceability = {
                name: {
                    "producer": "benchmarks/analysis/paper_tables.py",
                    "source": source,
                }
                for name, source in table_sources.items()
            }
    for name in ("lep_focusing_ensemble.tex", "ilc_pair_ensemble.tex", "table_values.json", "table_receipt.json"):
        traceability[name] = dict(producer="benchmarks/analysis/qed_figures/render_tables.py",
            source="verified QED ensemble reduction",
            analysis_receipt_sha256=digest(Path(qed_reduction)/"analysis_receipt.json"),
            table_receipt_sha256=digest(output/"table_receipt.json"), output_sha256=digest(output/name))
    (output / "table_traceability.json").write_text(json.dumps(traceability, indent=2)+"\n")
    return mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qed-reduction", type=Path, default=DEFAULT_REDUCTION)
    args = parser.parse_args()
    build_tables(args.output, qed_reduction=args.qed_reduction)
