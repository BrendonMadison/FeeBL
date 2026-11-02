"""Construct comparison ordinates and uncertainty from native/reference inputs.

Luminosity bins are correlated through reused particles and collective fields.
Independent complete collisions, rather than luminosity encounters, are the
sampling units for the empirical spectrum covariance. Reference-curve errors
use their own local-trend estimate. The two sources are independent here.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

import reference_uncertainty as reference

BENCHMARKS = Path(__file__).resolve().parents[1]
INPUTS = BENCHMARKS / "example_output/analysis_inputs"
PANELS = BENCHMARKS / "example_output/comparisons"


def read_rows(path):
    """Read a UTF-8 table, retaining its published column names."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    """Write numerical records, preserving all explicitly supplied fields."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column(rows, name):
    """Return a finite numerical column; malformed inputs are an error."""
    values = np.asarray([float(row[name]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {name}")
    return values


def panel_case(name):
    """Map a comparison to the exact packaged native configuration."""
    if name.startswith("ilc_1000_"):
        return "ilc1000_tdr_b1b_gaussian_proxy"
    if name.startswith("halhf_"):
        energy = name.split("_")[1]
        return "halhf250_wide_domain" if energy == "250" else "halhf" + energy
    if name.startswith("monochromatization_"):
        return "geometry_" + name.split("_")[1]
    if name.startswith("ilc_pair_yields_"):
        return "ilc500_ussc_pairs_2006"
    return "lep1994_focusing"


def read_native_spectrum(path):
    """Read W_low/W_high [GeV] and integrated luminosity weight per bin.

    The native file may omit empty bins. Those represent zero weight, not a
    change in the comparison's fixed retained support.
    """
    records, metadata = [], {}
    active = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("#BLOCK pair_spectrum"):
            active = True
        elif line.startswith("#END"):
            active = False
        elif line.startswith("#n_eff_total"):
            metadata["encounter_kish_count"] = float(line.split()[1])
        elif line.startswith("#rndm_seed"):
            metadata["seed"] = int(line.split()[1])
        elif active and line.strip() and not line.startswith("#"):
            values = line.split()
            records.append([float(values[i]) for i in (0, 1, 3)])
    bins = np.asarray(records)
    if bins.ndim != 2 or bins.shape[1] != 3 or not np.isfinite(bins).all():
        raise ValueError(f"Malformed spectrum: {path}")
    if np.any(bins[:, 1] <= bins[:, 0]) or np.any(bins[:, 2] < 0):
        raise ValueError(f"Invalid native bin width or weight: {path}")
    return bins, metadata


def rebin_mass(native, low, high):
    """Integrate piecewise-constant native density into fixed retained bins."""
    result = np.array(
        [
            np.sum(
                np.maximum(0, np.minimum(b, native[:, 1]) - np.maximum(a, native[:, 0]))
                / (native[:, 1] - native[:, 0])
                * native[:, 2]
            )
            for a, b in zip(low, high)
        ]
    )
    if result.sum() <= 0:
        raise ValueError("The selected reference support contains no luminosity.")
    return result / result.sum()


def reference_estimate(name, rows):
    """Reconstruct the published/analytic ordinates and their covariance.

    A reference table with three different subprocesses has no smooth trend.
    Its count-cluster exposure scenario is therefore declared separately.
    """
    count = len(rows)
    if name.startswith("monochromatization_"):
        low, high = column(rows, "bin_low"), column(rows, "bin_high")
        sigma = 0.18577664838424576 if "opposite" in name else 0.26516504294495535
        mass = np.array(
            [
                0.5
                * (
                    math.erf((b - 250) / (math.sqrt(2) * sigma))
                    - math.erf((a - 250) / (math.sqrt(2) * sigma))
                )
                for a, b in zip(low, high)
            ]
        )
        values = mass / mass.sum() / (high - low)
        return (
            values,
            np.zeros((count, count)),
            {
                "method": "fixed Gaussian bin integral",
                "mean_GeV": 250,
                "sigma_GeV": sigma,
                "sampling_variance": 0,
            },
        )
    if name.startswith("ilc_pair_yields_"):
        values = column(rows, "reference_count")
        return (
            values,
            np.diag(2 * values),
            {
                "method": "one-equivalent-crossing paired-daughter count scenario",
                "variance": "2Y",
                "alternative_variances": ["Y", "4Y"],
                "limitation": "The source's equivalent exposure is unknown; this is an explicit scenario.",
                "source_url": "https://doi.org/10.1103/PhysRevSTAB.9.034402",
                "locator": "Section II.A.1 and Table II",
                "observable": "Total individual electrons plus positrons per bunch crossing, with 5 MeV produced-lepton energy cut; the paper does not state a both-daughters event requirement.",
            },
        )
    source = reference.curve_panel(name)
    linear = reference.linear_three_five_noise(
        source["x"], source["y"], source["mode"], source["low"], source["high"]
    )
    floor = source["floor"] ** 2
    variance = np.maximum(linear["central"], floor)
    companion = np.maximum(linear["five_point_companion"], floor)
    if source["widths"] is None:
        matrix = reference.interpolation_matrix(source["x"], column(rows, "phi_rad"))
        values = matrix @ source["y"]
        covariance = (matrix * variance) @ matrix.T
        companion_covariance = (matrix * companion) @ matrix.T
    else:
        mass = source.get("original_mass", source["y"] * source["widths"])
        values = mass / mass.sum()
        covariance = reference.normalize_covariance(
            mass, np.diag(variance * source["widths"] ** 2)
        )
        companion_covariance = reference.normalize_covariance(
            mass, np.diag(companion * source["widths"] ** 2)
        )
    return (
        values,
        covariance,
        {
            "method": "three-point linear contrast; 15 local probes; graphical floor",
            "source": source["path"].name,
            "source_url": source["source_url"],
            "locator": source["source_locator"],
            "floor": source["floor_description"],
            "five_point_variance": np.diag(companion_covariance).tolist(),
            "limitation": "Estimated unresolved fluctuations include digitization and curvature; not source-certified errors.",
        },
    )


def sample_covariance(samples):
    """Return the mean and covariance of independent complete-run estimates.

    Centering before multiplication avoids subtracting two nearly equal raw
    second moments. The returned covariance is that of the sample mean.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("At least two finite independent sample vectors are required.")
    differences = samples - samples[0]
    centered = differences - differences.mean(axis=0)
    return (
        samples[0] + differences.mean(axis=0),
        centered.T @ centered / (len(samples) * (len(samples) - 1)),
    )


def conditional_probe_estimate(name, rows):
    """Recompute distinct LEP-probe sampling errors conditional on one field.

    The azimuthal means use unbiased sample variances divided by their counts.
    Histogram covariance uses actual counts and the fixed-total constraint;
    a Jeffreys posterior variance marks empty bins without asserting agreement.
    """
    probes = [
        row
        for row in read_rows(INPUTS / "lep_probe_records.csv")
        if row["role"] == "electron" and row["cohort"] == "inner"
    ]
    angle, focus = column(probes, "phi_rad"), column(probes, "focus_urad")
    if "phi_rad" in rows[0]:
        means, variances, counts = [], [], []
        for center in column(rows, "phi_rad"):
            selected = focus[
                (angle >= center - math.pi / 48) & (angle < center + math.pi / 48)
            ]
            if len(selected) < 2:
                raise ValueError("Insufficient distinct probes in azimuth bin")
            means.append(selected.mean())
            variances.append(selected.var(ddof=1) / len(selected))
            counts.append(len(selected))
        return np.asarray(means), np.diag(variances), np.asarray(counts) == 0
    counts = np.array(
        [
            np.count_nonzero((focus >= a) & (focus < b))
            for a, b in zip(column(rows, "bin_low"), column(rows, "bin_high"))
        ]
    )
    values = counts / counts.sum()
    alpha = counts + 0.5
    posterior = alpha / alpha.sum()
    covariance = (np.diag(posterior) - np.outer(posterior, posterior)) / (
        alpha.sum() + 1
    )
    return values, covariance, counts == 0


def paired_counts(path):
    """Count final admitted individual leptons by native process code 0/1/2.

    These rows passed the birth-energy cut separately for each daughter. The
    native n_BW/n_BH/n_LL and n_pairs production counters precede that cut and
    are not read here. No new cut is applied to the transported final energies.
    """
    counts = np.zeros(3)
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() and not line.startswith("#"):
                process = int(line.split()[7])
                if not 0 <= process < 3:
                    raise ValueError("Unexpected incoherent-pair process code")
                counts[process] += 1
    return counts


def pair_input_contract(path):
    """Require the unthinned, per-daughter sample used by this comparison."""
    text = "\n".join(
        line.split("#", 1)[0] for line in Path(path).read_text().splitlines()
    )
    expected = {
        "pair_ecut": 0.005,
        "pair_ratio": 1,
        "track_pairs": 1,
        "store_pairs": 1,
        "beam_pair": 0,
        "load_events": 0,
        "do_bhabhas": 0,
        "do_pairs": 1,
        "do_compt": 0,
    }
    for key, value in expected.items():
        found = re.findall(rf"(?<![\w.]){re.escape(key)}\s*=\s*([^;]+);", text)
        if len(found) != 1 or float(found[0]) != value:
            raise ValueError(f"The pair comparison requires {key}={value}")
    return expected


def campaign_inventory(campaign):
    """Identify complete requested ensembles without replacing failed seeds."""
    campaign = Path(campaign)
    manifest_path = campaign / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    cases = []
    for case in sorted({job["case"] for job in manifest["jobs"]}):
        jobs = [job for job in manifest["jobs"] if job["case"] == case]
        records = []
        for job in jobs:
            path = campaign / job["directory"] / "execution.json"
            receipt = json.loads(path.read_text()) if path.is_file() else {}
            records.append(
                {
                    "seed": job["seed"],
                    "directory": job["directory"],
                    "status": receipt.get("status", "NOT_EXECUTED"),
                    "binary_sha256": receipt.get("binary_sha256"),
                    "error": receipt.get("error"),
                    "execution_receipt_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    if path.is_file()
                    else None,
                }
            )
        binaries = {
            record["binary_sha256"]
            for record in records
            if record["status"] == "EXECUTED"
        }
        complete = (
            len(records) == manifest["seed_count"]
            and len({record["seed"] for record in records}) == len(records)
            and all(record["status"] == "EXECUTED" for record in records)
            and len(binaries) == 1
            and None not in binaries
        )
        cases.append(
            {
                "case": case,
                "complete_requested_ensemble": complete,
                "executed": sum(record["status"] == "EXECUTED" for record in records),
                "requested": len(records),
                "records": records,
            }
        )
    return {
        "schema": "feebl.campaign-inventory/1",
        "campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "declared_seed_count": manifest["seed_count"],
        "cases": cases,
        "all_requested_cases_complete": all(
            case["complete_requested_ensemble"] for case in cases
        ),
        "qualification": "Execution inventory only. An incomplete family is excluded as a whole; no failed seed is replaced or silently discarded.",
    }


def reduce_campaign(campaign, output, complete_families_only=False):
    """Reduce completed replicas to portable panel vectors and native receipts.

    Large particle files stay at the execution site. Failed/missing replicas
    are explicit errors; a partial campaign cannot masquerade as full coverage.
    """
    campaign = Path(campaign)
    manifest = json.loads((campaign / "campaign.json").read_text())
    inventory = campaign_inventory(campaign)
    complete = {
        case["case"]
        for case in inventory["cases"]
        if case["complete_requested_ensemble"]
    }
    if not complete_families_only and not inventory["all_requested_cases_complete"]:
        raise ValueError(
            "Campaign incomplete; use --complete-families-only for an explicitly partial analysis"
        )
    panels = {p.stem: read_rows(p) for p in sorted(PANELS.glob("*_pull.csv"))}
    records, matrices = [], {}
    for name, rows in panels.items():
        if name.startswith("lep_"):
            continue
        case = panel_case(name)
        if case not in complete:
            continue
        selected = [job for job in manifest["jobs"] if job["case"] == case]
        if not selected:
            continue
        if len(selected) != manifest["seed_count"]:
            raise ValueError(f"Incomplete declared seed ensemble for {case}")
        values, seeds, effective, native_runs = [], [], [], []
        binary_hashes = set()
        for job in selected:
            directory = campaign / job["directory"]
            receipt = json.loads((directory / "execution.json").read_text())
            if receipt["status"] != "EXECUTED":
                raise ValueError(f"Incomplete replica: {job['directory']}")
            filename = (
                "pairs.dat"
                if name.startswith("ilc_pair_yields_")
                else "lumi.ee.out.dLdW.fne"
            )
            hasher = hashlib.sha256()
            with (directory / filename).open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    hasher.update(block)
            native_hash = hasher.hexdigest()
            if native_hash != receipt["outputs"][filename]["sha256"]:
                raise ValueError(
                    f"Native output changed after execution: {job['directory']}/{filename}"
                )
            binary_hashes.add(receipt["binary_sha256"])
            native_runs.append(
                {
                    "seed": job["seed"],
                    "input_sha256": job["input_sha256"],
                    "native_file": filename,
                    "native_sha256": native_hash,
                    "execution_receipt_sha256": hashlib.sha256(
                        (directory / "execution.json").read_bytes()
                    ).hexdigest(),
                }
            )
            if name.startswith("ilc_pair_yields_"):
                native_runs[-1]["particle_selection"] = pair_input_contract(
                    directory / "acc.dat"
                )
                value = paired_counts(directory / "pairs.dat")
                effective.append(0)
            else:
                native, metadata = read_native_spectrum(
                    directory / "lumi.ee.out.dLdW.fne"
                )
                value = rebin_mass(
                    native, column(rows, "bin_low"), column(rows, "bin_high")
                )
                if name.startswith("monochromatization_"):
                    value /= column(rows, "bin_high") - column(rows, "bin_low")
                effective.append(metadata.get("encounter_kish_count", 0))
            values.append(value)
            seeds.append(job["seed"])
        if len(seeds) != len(set(seeds)):
            raise ValueError("Replica seed reused in one case")
        if len(binary_hashes) != 1:
            raise ValueError(
                f"Different native binaries cannot define one sampling ensemble: {case}"
            )
        matrices[name] = np.asarray(values)
        matrices[name + "__seeds"] = np.asarray(seeds)
        matrices[name + "__encounter_neff"] = np.asarray(effective)
        records.append(
            {
                "panel": name,
                "case": case,
                "seeds": seeds,
                "binary_sha256": sorted(binary_hashes),
                "native_runs": native_runs,
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not matrices:
        raise ValueError("No complete requested family can be reduced")
    (output / "campaign_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    np.savez_compressed(output / "seed_vectors.npz", **matrices)
    (output / "seed_vectors.json").write_text(
        json.dumps(
            {
                "panels": records,
                "sampling_unit": "independent full native collision",
                "bin_support": "packaged comparison bins",
                "purpose": "Explicit partial campaign: complete requested families only"
                if not inventory["all_requested_cases_complete"]
                else "Complete requested campaign reduction",
                "all_requested_cases_complete": inventory[
                    "all_requested_cases_complete"
                ],
                "inventory": "campaign_inventory.json",
            },
            indent=2,
        )
        + "\n"
    )


def build_comparisons(output, ensemble=None, allow_partial=False):
    """Generate all comparison values, both covariance terms, and capped pulls.

    Spectrum and pair panels require independent replicas. The conditional LEP
    probe study remains separate. Singular covariance is retained, not inverted
    into an unjustified global chi-square with one degree of freedom per bin.
    """
    if ensemble is None:
        raise ValueError(
            "Independent seed vectors are required for spectral comparisons."
        )
    samples = np.load(ensemble)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries, omitted = [], []
    for path in sorted(PANELS.glob("*_pull.csv")):
        name, rows = path.stem, read_rows(path)
        if not name.startswith("lep_") and name not in samples:
            if not allow_partial:
                raise ValueError(
                    f"Missing ensemble for {name}; --allow-partial is required to omit a panel"
                )
            omitted.append(name)
            continue
        target, ref_covariance, ref_details = reference_estimate(name, rows)
        if name.startswith("lep_"):
            measured, covariance, empty = conditional_probe_estimate(name, rows)
            method, count = "distinct probes conditional on one simulated field", 1
        else:
            vectors = samples[name]
            measured, covariance = sample_covariance(vectors)
            empty = np.all(vectors == 0, axis=0)
            method, count = (
                "mean and full sample covariance of independent native collisions",
                len(vectors),
            )
            # Empty sampled bins need a declared finite variance to display a
            # pull. Use a half-observation at the summed encounter exposure,
            # projected onto the fixed-total subspace. This is a regularizer,
            # never evidence of good agreement in an unpopulated bin.
            if np.any(empty):
                if name.startswith("ilc_pair_yields_"):
                    # Pair yields are unnormalized counts per crossing, so
                    # their exposure is the number of independent crossings.
                    covariance += np.diag(empty * 0.5 / count ** 2)
                else:
                    neff = float(np.sum(samples[name + "__encounter_neff"]))
                    if neff <= 0:
                        raise ValueError(
                            "No exposure for empty-bin variance regularization"
                        )
                    conversion = (
                        1 / (column(rows, "bin_high") - column(rows, "bin_low"))
                        if name.startswith("monochromatization_")
                        else np.ones(len(rows))
                    )
                    mass = measured / conversion
                    projector = np.eye(len(rows)) - mass[:, None] / mass.sum()
                    regularizer = (projector * (empty * 0.5 / neff ** 2)) @ projector.T
                    covariance += regularizer * np.outer(conversion, conversion)
        variance = np.diag(covariance + ref_covariance)
        if np.any(variance <= 0) or not np.isfinite(variance).all():
            raise ValueError(f"Unresolved combined variance in {name}")
        pulls = (measured - target) / np.sqrt(variance)
        if "simulation_mass" in rows[0]:
            sim_key, ref_key = "simulation_mass", "reference_mass"
        elif "simulation_density_per_GeV" in rows[0]:
            sim_key, ref_key = (
                "simulation_density_per_GeV",
                "reference_gaussian_density_per_GeV",
            )
        elif "simulation_mean_urad" in rows[0]:
            sim_key, ref_key = "simulation_mean_urad", "reference_mean_urad"
        else:
            sim_key, ref_key = "simulation_count", "reference_count"
        result = []
        for index, row in enumerate(rows):
            coordinates = {
                key: row[key]
                for key in ("bin_low", "bin_high", "phi_rad", "process", "case")
                if key in row
            }
            result.append(
                {
                    **coordinates,
                    sim_key: measured[index],
                    ref_key: target[index],
                    "simulation_variance": covariance[index, index],
                    "reference_variance": ref_covariance[index, index],
                    "simulation_reference_covariance": 0.0,
                    "pull": pulls[index],
                    "pull_standard_uncertainty": 1.0,
                    "empty_bin_regularized": bool(empty[index]),
                    "independent_seed_count": count,
                    "simulation_method": method,
                }
            )
        write_rows(output / path.name, result)
        np.savez_compressed(
            output / (name + "_covariance.npz"),
            simulation=covariance,
            reference=ref_covariance,
            simulation_reference=np.zeros_like(covariance),
        )
        summaries.append(
            {
                "panel": name,
                "case": panel_case(name),
                "bins": len(rows),
                "independent_seed_count": count,
                "sampling_method": method,
                "pull_rms": float(np.sqrt(np.mean(pulls ** 2))),
                "maximum_absolute_pull": float(np.max(np.abs(pulls))),
                "regularized_empty_bins": int(empty.sum()),
                "reference": ref_details,
                "global_chi_square": "not inferred; finite seed rank and estimated reference covariance",
            }
        )
        if name.startswith("ilc_pair_yields_"):
            summaries[-1]["simulation_observable"] = {
                "record": "pairs.dat, one row per admitted daughter; process codes 0=BW, 1=BH, 2=LL",
                "selection": "Each daughter independently requires absolute lab-frame total energy at production >= 0.005 GeV; both daughters need not pass.",
                "storage": "pair_ratio=1, track_pairs=1, store_pairs=1, beam_pair=0; no final-energy recut or detector acceptance",
                "excluded_counters": "out.dat n_BW/n_BH/n_LL and n_pairs are accumulated before the daughter-energy admission cut and are not comparison yields.",
                "reference_scope": "Consistent with the paper's individual produced-lepton wording; an unpublished source-code event-selection audit is not claimed.",
            }
        alternatives = {}
        if "five_point_variance" in ref_details:
            alternatives["five_point_linear_reference"] = np.asarray(
                ref_details["five_point_variance"]
            )
        if name.startswith("ilc_pair_yields_"):
            alternatives.update(
                one_independent_lepton_per_count=target,
                four_lepton_variance_per_count=4 * target,
            )
        summaries[-1]["reference_uncertainty_sensitivity"] = {
            label: {
                "pull_rms": float(
                    np.sqrt(
                        np.mean(
                            (measured - target) ** 2 / (np.diag(covariance) + other)
                        )
                    )
                ),
                "maximum_absolute_pull": float(
                    np.max(
                        abs(measured - target) / np.sqrt(np.diag(covariance) + other)
                    )
                ),
            }
            for label, other in alternatives.items()
        }
    (output / "comparison_methods.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    (output / "comparison_coverage.json").write_text(
        json.dumps(
            {
                "purpose": "Explicit partial comparison; incomplete families omitted"
                if omitted
                else "All comparison panels present",
                "allow_partial": allow_partial,
                "omitted_panels": omitted,
                "conditional_LEP_panels": [
                    row["panel"] for row in summaries if row["panel"].startswith("lep_")
                ],
                "independent_seed_panels": [
                    row["panel"]
                    for row in summaries
                    if not row["panel"].startswith("lep_")
                ],
                "qualification": "Omitted panels are not replaced by selected successful seeds or historical uncertainty proxies.",
            },
            indent=2,
        )
        + "\n"
    )
    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    reduce = actions.add_parser("reduce")
    reduce.add_argument("--campaign", type=Path, required=True)
    reduce.add_argument("--output", type=Path, required=True)
    reduce.add_argument("--complete-families-only", action="store_true")
    build = actions.add_parser("build")
    build.add_argument("--ensemble", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allow-partial", action="store_true")
    inventory = actions.add_parser("inventory")
    inventory.add_argument("--campaign", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "reduce":
        reduce_campaign(
            arguments.campaign, arguments.output, arguments.complete_families_only
        )
    elif arguments.action == "build":
        build_comparisons(arguments.output, arguments.ensemble, arguments.allow_partial)
    else:
        with arguments.output.open("x") as stream:
            json.dump(campaign_inventory(arguments.campaign), stream, indent=2)
            stream.write("\n")
