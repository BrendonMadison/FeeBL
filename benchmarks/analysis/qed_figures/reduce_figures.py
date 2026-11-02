#!/usr/bin/env python3
"""Reduce the prescribed independent-seed campaign without running collisions.

Outputs are written to a new directory. Reference tables, publication figures
and native records are read-only. A completed family is the statistical
unit; correlated pair copies and two legs of an event are not independent runs.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np

import lep_support as lep

LEP_PANELS = (
    "lep_final_focusing_vs_phi_pull",
    "lep_final_focusing_numerical_vs_phi_pull",
    "lep_focusing_distribution_pull",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def card_values(path):
    text = "\n".join(line.split("#", 1)[0] for line in Path(path).read_text().splitlines())
    return {key: float(value) for key, value in re.findall(
        r"(?<![\w.])([A-Za-z_]\w*)\s*=\s*([-+\d.eE]+)\s*;", text)}


def pair_contract(path, model, cut, thinning=1):
    """Require a pure daughter population and the explicitly declared thinning."""
    if not 0 < thinning <= 1:
        raise ValueError("Expected a declared Bernoulli storage fraction in (0,1]")
    values = card_values(path)
    expected = dict(pair_model=model, pair_ecut=cut, pair_ratio=thinning, track_pairs=1,
                    beam_pair=0, load_events=0, do_bhabhas=0, do_pairs=1,
                    do_compt=0, do_coherent=0, do_trident=0,
                    do_photon_lumi=1, do_photons=1, electron_ratio=1)
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value}, got {values.get(key)}")
    if values.get("store_pairs") not in (1, 2):
        raise ValueError("Final admitted secondary records must be stored")
    return expected


def combined_rows(coordinates, measured, covariance, reference, reference_covariance, n):
    """Use both covariance diagonals; pull uncertainties are exactly one."""
    variance = np.diag(covariance+reference_covariance)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Nonpositive or nonfinite combined comparison variance")
    rows = []
    for i, coordinate in enumerate(coordinates):
        rows.append(dict(coordinate, simulation=float(measured[i]), reference=float(reference[i]),
            simulation_variance=float(covariance[i, i]),
            reference_variance=float(reference_covariance[i, i]),
            simulation_reference_covariance=0.0,
            combined_standard_uncertainty=float(np.sqrt(variance[i])),
            pull=float((measured[i]-reference[i])/np.sqrt(variance[i])),
            pull_standard_uncertainty=1.0, independent_seeds=n,
            variance_degrees_of_freedom=n-1,
            zero_empirical_variance=bool(covariance[i, i] == 0)))
    return rows


def covariance_summary(measured, reference, simulation, reference_covariance):
    """Descriptive Mahalanobis distance on the supported covariance subspace.

    Estimated reference uncertainties and a finite seed covariance do not imply
    a calibrated chi-square null distribution, so no significance is assigned.
    """
    total = simulation+reference_covariance
    eigenvalues, eigenvectors = np.linalg.eigh(total)
    tolerance = np.finfo(float).eps*len(total)*max(float(eigenvalues.max()), 1e-300)*20
    if eigenvalues.min() < -tolerance:
        raise ValueError("Comparison covariance is not positive semidefinite")
    supported = eigenvalues > tolerance
    residual = eigenvectors.T @ (measured-reference)
    score = float(np.sum(residual[supported]**2/eigenvalues[supported]))
    pull = (measured-reference)/np.sqrt(np.diag(total))
    ratios = np.sqrt(np.diag(simulation)/np.diag(reference_covariance))
    return dict(covariance_subspace_distance_squared=score,
        covariance_rank=int(supported.sum()),
        residual_norm_outside_supported_subspace=float(np.linalg.norm(residual[~supported])),
        pull_rms=float(np.sqrt(np.mean(pull**2))), max_absolute_pull=float(np.abs(pull).max()),
        simulation_to_reference_sem_ratio_min=float(ratios.min()),
        simulation_to_reference_sem_ratio_median=float(np.median(ratios)),
        simulation_to_reference_sem_ratio_max=float(ratios.max()),
        interpretation="Descriptive distance with estimated reference covariance; no calibrated p-value")


def validate_receipt(case, inputs, campaign):
    """Bind analysis to the frozen input and complete native output identities."""
    directory = campaign/case["name"]
    receipt_path = directory/"receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("manifest_sha256") != digest(inputs/"manifest.json") or
            receipt.get("case") != case["name"] or
            receipt.get("native_seed") != case["native_seed"] or
            receipt.get("sampler_seed") != case["sampler_seed"]):
        raise ValueError("Execution receipt and frozen case identity disagree")
    if receipt.get("status") not in ("PASSED_OUTPUT_CONTRACT", "PASS_EXECUTION_AND_RECORD_CONTRACT_ONLY"):
        raise ValueError("Case has not passed output admission: "+case["name"])
    if receipt.get("returncode") != 0 or receipt.get("timed_out"):
        raise ValueError("Case did not finish successfully: "+case["name"])
    hashes = receipt.get("output_sha256", receipt.get("outputs", {}))
    hashes = {name: item["sha256"] if isinstance(item, dict) else item for name, item in hashes.items()}
    group = case.get("group", case.get("kind"))
    names = ["acc.dat", "out.dat"]
    names += (["event.ini", "event_identity.json", "event_sampling.csv",
               "gui_imported_trajectories.json", "probe_records.csv"]
              if group.startswith("lep_") else ["pairs.dat"])
    names = sorted(set(names+case.get("expected_outputs", [])))
    verified = {}
    for name in names:
        path = directory/name
        actual = digest(path)
        if name in case.get("files", {}):
            frozen = case["files"][name]
            frozen = frozen["sha256"] if isinstance(frozen, dict) else frozen
            if actual != frozen or digest(inputs/case["name"]/name) != frozen:
                raise ValueError("Changed frozen input: "+str(path))
        elif name == "acc.dat" and "card_sha256" in case:
            if actual != case["card_sha256"]:
                raise ValueError("Changed frozen card: "+str(path))
        elif hashes.get(name) != actual:
            raise ValueError("Missing or changed output hash: "+str(path))
        verified[name] = actual
    binary = receipt.get("binary_sha256")
    identity = receipt.get("source_native_identity")
    if not isinstance(binary, str) or not re.fullmatch(r"[0-9a-f]{64}", binary) or not identity:
        raise ValueError("Missing binary/source identity in receipt")
    return dict(case=case, directory=directory, receipt=receipt,
        binary=binary, source_identity=identity, hashes=verified,
        receipt_sha256=digest(receipt_path))


def group_records(records, name, count, model=None):
    selected = [r for r in records if r["case"].get("group", r["case"].get("kind")) == name
                and (model is None or r["case"].get("pair_model") == model)]
    if len(selected) != count:
        raise ValueError(f"Expected {count} complete records for {name}, model {model}")
    seeds = [r["case"]["native_seed"] for r in selected]
    if len(set(seeds)) != count or len({r["binary"] for r in selected}) != 1:
        raise ValueError("Repeated seeds or mixed binaries in one ensemble")
    if len({r["source_identity"] for r in selected}) != 1:
        raise ValueError("Mixed native source identities in one ensemble")
    return sorted(selected, key=lambda r: r["case"]["native_seed"])


def checked_probes(record):
    directory, case = record["directory"], record["case"]
    identity = read_json(directory/"event_identity.json")
    if identity["source_artifact"]["sha256"] != digest(directory/"event.ini"):
        raise ValueError("Event identity does not bind the actual input")
    return lep.validate_probe_records(read_json(directory/"gui_imported_trajectories.json"),
        identity, read_csv(directory/"event_sampling.csv"),
        np.loadtxt(directory/"event.ini", ndmin=2), read_csv(directory/"probe_records.csv"), case)


def reduce_lep(records, source, output, statistics):
    fields = group_records(records, "lep_on", 10)
    zero = group_records(records, "lep_zero", 2)
    templates, refs, ref_covs, notes = {}, {}, {}, {}
    for name in LEP_PANELS:
        templates[name] = read_csv(source/"benchmarks/example_output/comparisons"/(name+".csv"))
        refs[name], ref_covs[name], notes[name] = statistics.reference_estimate(name, templates[name])
    centers = np.array([float(r["phi_rad"]) for r in templates[LEP_PANELS[0]]])
    expected = -np.pi+(np.arange(48)+.5)*2*np.pi/48
    if not np.allclose(centers, expected, rtol=0, atol=1e-12):
        raise ValueError("Angular reference grid is not the prescribed 48 strata")
    low = np.array([float(r["bin_low"]) for r in templates[LEP_PANELS[2]]])
    high = np.array([float(r["bin_high"]) for r in templates[LEP_PANELS[2]]])
    checked = {r["case"]["metadata"]["field_index"]: checked_probes(r) for r in fields}
    if set(checked) != set(range(10)):
        raise ValueError("Field identities must cover the ten declared realizations")
    estimates = [lep.estimate_field(checked[i], low, high) for i in range(10)]
    vectors = np.array([e["vector"] for e in estimates])
    mean, covariance = statistics.sample_covariance(vectors)
    np.savez_compressed(output/"lep_field_covariance.npz", vectors=vectors, mean=mean,
        covariance_of_mean=covariance, histogram_counts=np.array([e["histogram_counts"] for e in estimates]),
        stratum_counts=np.array([e["stratum_counts"] for e in estimates]))
    summaries = {}
    for name in LEP_PANELS:
        angular = name != LEP_PANELS[2]
        indices = np.arange(48) if angular else np.arange(48, 48+len(low))
        sim, cov = mean[indices], covariance[np.ix_(indices, indices)]
        coordinates = ([dict(phi_rad=x) for x in centers] if angular else
                       [dict(bin_low=a, bin_high=b) for a, b in zip(low, high)])
        rows = combined_rows(coordinates, sim, cov, refs[name], ref_covs[name], 10)
        write_csv(output/(name+".csv"), rows)
        np.savez_compressed(output/(name+"_covariance.npz"), simulation=cov,
            reference=ref_covs[name], simulation_reference=np.zeros_like(cov))
        summaries[name] = covariance_summary(sim, refs[name], cov, ref_covs[name])
    controls = []
    for record in zero:
        index = record["case"]["metadata"]["field_index"]
        zero_rows = checked_probes(record)
        delta = lep.pair_zero_control(checked[index], zero_rows)
        controls.append(dict(field_index=index,
            max_absolute_focus_urad=max(abs(float(r["focus_urad"])) for r in zero_rows),
            paired_difference_scalars=lep.estimate_field(delta, low, high)["vector"][-6:].tolist()))
    scalar_names = [f"{a}_{b}_mean_urad" for a, b in lep.POPULATIONS]
    scalar_names += ["electron_boundary_bias_percent", "positron_boundary_bias_percent"]
    write_csv(output/"lep_boundary_scalars.csv", [dict(observable=name, mean=mean[-6+i],
        standard_uncertainty=np.sqrt(covariance[-6+i, -6+i])) for i, name in enumerate(scalar_names)])
    save_json(output/"lep_summary.json", dict(independent_fields=10, variance_degrees_of_freedom=9,
        sampling="Equal weight for 48 planned azimuth strata; paired event legs and inner/outer boundaries; field means averaged",
        covariance_rule="Sample covariance of ten complete field vectors divided by ten; no extra row-Poisson term",
        reference_models=notes, comparisons=summaries, zero_controls=controls,
        retained_histogram_probabilities=[e["retained_probability"] for e in estimates],
        reference_claim="Reconstructed LEP input; statistical precision alone does not establish physical reproduction"))


def luminosity_normalized_estimate(counts, luminosity_m2):
    """Estimate daughter-production cross sections and their joint seed error.

    The ratio of ensemble means is also the ratio of summed counts to summed
    luminosity. A Jacobian transfers the complete count/luminosity covariance;
    a leave-one-collision-out jackknife checks its first-order approximation.
    Luminosity is scaled to inverse millibarns before forming the covariance.
    """
    counts = np.asarray(counts, dtype=float)
    luminosity = np.asarray(luminosity_m2, dtype=float)*1e-31
    if (counts.ndim != 2 or len(counts) < 3 or
            luminosity.shape != (len(counts),) or
            not np.isfinite(counts).all() or not np.isfinite(luminosity).all() or
            np.any(counts < 0) or np.any(luminosity <= 0)):
        raise ValueError("Require nonnegative counts and positive luminosity for complete seeds")
    joint = np.c_[counts, luminosity]
    mean = joint.mean(axis=0)
    delta = joint-mean
    covariance = delta.T@delta/(len(joint)*(len(joint)-1))
    jacobian = np.c_[np.eye(counts.shape[1])/mean[-1], -mean[:-1]/mean[-1]**2]
    cross_sections = mean[:-1]/mean[-1]
    transferred = jacobian@covariance@jacobian.T
    # Roundoff can make an exactly cancelling variance very slightly negative.
    scale = np.max(np.abs(covariance))*np.max(np.abs(jacobian))**2
    if np.min(np.linalg.eigvalsh(transferred)) < -100*np.finfo(float).eps*max(scale, 1):
        raise ValueError("Transferred covariance is not positive semidefinite")
    omitted_means = (joint.sum(axis=0)-joint)/(len(joint)-1)
    omitted_ratios = omitted_means[:, :-1]/omitted_means[:, -1, None]
    centered = omitted_ratios-omitted_ratios.mean(axis=0)
    jackknife = (len(joint)-1)/len(joint)*(centered.T@centered)
    omitted_sem = []
    for index, omitted_mean in enumerate(omitted_means):
        sample = np.delete(joint, index, axis=0)
        residual = sample-omitted_mean
        sample_covariance = residual.T@residual/(len(sample)*(len(sample)-1))
        derivative = np.c_[np.eye(counts.shape[1])/omitted_mean[-1],
                           -omitted_mean[:-1]/omitted_mean[-1]**2]
        omitted_sem.append(np.sqrt(np.maximum(0, np.diag(derivative@sample_covariance@derivative.T))))
    return dict(mean_mb=cross_sections.tolist(), covariance_of_mean_mb2=transferred.tolist(),
        sem_mb=np.sqrt(np.maximum(0, np.diag(transferred))).tolist(),
        jackknife_covariance_mb2=jackknife.tolist(),
        jackknife_sem_mb=np.sqrt(np.diag(jackknife)).tolist(),
        leave_one_out_mean_mb=omitted_ratios.tolist(),
        leave_one_out_sem_mb=np.asarray(omitted_sem).tolist(),
        luminosity_mean_inverse_mb=float(mean[-1]),
        estimator="Ratio of summed daughter counts to summed electron-positron luminosity",
        covariance="Jacobian of the complete joint seed covariance of the mean")


def reduce_ilc(records, source, output, statistics):
    summary = {}
    references = read_json(source/"benchmarks/reference/ilc_rimbault2006_effective_cross_sections.json")
    for model in (1, 2):
        family = group_records(records, "ilc_pairs", 12, model)
        for r in family:
            pair_contract(r["directory"]/"acc.dat", model, .005)
        vectors = np.array([statistics.paired_counts(r["directory"]/"pairs.dat") for r in family])
        mean, covariance = statistics.sample_covariance(vectors)
        luminosities = [collision_luminosity(r["directory"]/"out.dat") for r in family]
        lumi_names = list(luminosities[0])
        lumi_vectors = np.array([[row[key] for key in lumi_names] for row in luminosities])
        lumi_mean, lumi_covariance = statistics.sample_covariance(lumi_vectors)
        joint_mean, joint_covariance = statistics.sample_covariance(np.c_[vectors, lumi_vectors])
        write_csv(output/f"ilc_model{model}_collision_luminosity.csv", [dict(seed=r["case"]["native_seed"], **luminosity)
            for r, luminosity in zip(family, luminosities)])
        np.savez_compressed(output/f"ilc_model{model}_joint_yield_luminosity.npz",
            vectors=np.c_[vectors, lumi_vectors], mean=joint_mean, covariance_of_mean=joint_covariance,
            columns=np.array(["BW", "BH", "LL", *lumi_names]))
        summary[f"model{model}_luminosity"] = dict(columns=lumi_names,
            mean=lumi_mean.tolist(), standard_uncertainty=np.sqrt(np.diag(lumi_covariance)).tolist(),
            units="m^-2 per crossing; native printed precision retained",
            role="Electron-positron luminosity is the common denominator of all three effective cross sections")
        np.savez_compressed(output/f"ilc_model{model}_seed_vectors.npz", vectors=vectors,
            mean=mean, covariance_of_mean=covariance,
            seeds=np.array([r["case"]["native_seed"] for r in family]))
        summary[f"model{model}_seed_diagnostics"] = leave_one_out(vectors)
        normalized = luminosity_normalized_estimate(vectors, lumi_vectors[:, lumi_names.index("lumi_ee")])
        summary[f"model{model}_effective_cross_section"] = normalized
        cross_sections = np.array(normalized["mean_mb"])
        cross_covariance = np.array(normalized["covariance_of_mean_mb2"])
        for code in ("gp", "cain"):
            ref = np.array(references["codes"][code]["effective_cross_sections_mb"])
            # This is a declared sampling scenario, not the authors' MC error.
            exposure = references["luminosity_per_crossing_inverse_microbarn"]*1000
            equivalent_counts = ref*exposure
            variance_one_count = equivalent_counts/exposure**2
            ref_cov = np.diag(2*variance_one_count)
            note = dict(method="ASSUMED_ONE_CROSSING_PAIRED_DAUGHTER_COUNT_PROXY",
                source=references["source"], normalization="Common electron-positron luminosity",
                units="mb", luminosity_uncertainty="Not supplied by the reference; fixed denominator scenario",
                reference_count_variance="2N; N and 4N sensitivity retained",
                printed_rounding="Primary printed cross sections used; rounded counts are not substituted")
            rows = combined_rows([dict(process=p, units="mb") for p in references["processes"]],
                cross_sections, cross_covariance, ref, ref_cov, 12)
            for i, row in enumerate(rows):
                row["pull_reference_variance_Y"] = float((cross_sections[i]-ref[i])/
                    np.sqrt(cross_covariance[i, i]+variance_one_count[i]))
                row["pull_reference_variance_4Y"] = float((cross_sections[i]-ref[i])/
                    np.sqrt(cross_covariance[i, i]+4*variance_one_count[i]))
            write_csv(output/f"ilc_model{model}_{code}_pull.csv", rows)
            np.savez_compressed(output/f"ilc_model{model}_{code}_covariance.npz",
                simulation=cross_covariance, reference=ref_cov,
                simulation_reference=np.zeros_like(cross_covariance))
            summary[f"model{model}_{code}"] = dict(reference_model=note,
                **covariance_summary(cross_sections, ref, cross_covariance, ref_cov))
    save_json(output/"ilc_pair_summary.json", dict(comparisons=summary,
        observable="Effective individual-daughter production cross sections in mb: admitted electrons plus positrons divided by common electron-positron luminosity",
        selection="5 MeV production admission per daughter; no final-energy recut; pair_ratio=1; beam_pair=0; pure incoherent channels",
        simulation_uncertainty="Complete joint count/luminosity covariance of the mean, including field, event and split-copy fluctuations; no additional Poisson variance",
        reference_uncertainty="Explicit Var(N)=2N equivalent-crossing scenario at fixed published luminosity; N and 4N sensitivity; not source-certified statistics",
        raw_count_products="Preserved separately as per-crossing detector-background diagnostics",
        model_comparison="Disjoint seed banks; no assumed inter-model covariance"))


def collision_luminosity(path):
    text = Path(path).read_text(errors="strict")
    values = {}
    for name in ("lumi_ee", "lumi_fine", "lumi_eg", "lumi_ge", "lumi_gg"):
        matches = re.findall(r"^"+name+r"\s*=\s*([-+\d.eE]+)", text, re.MULTILINE)
        if not matches or (name != "lumi_ee" and len(matches) != 1) or len(matches) > 2:
            raise ValueError("Missing or unexpected repeated native luminosity: "+name)
        parsed = [float(match) for match in matches]
        value = parsed[0]
        # RESULTS prints lumi_ee in both the descriptive and machine blocks,
        # using the same precision. Both copies must agree after parsing.
        if not np.isfinite(parsed).all() or value < 0 or any(other != value for other in parsed[1:]):
            raise ValueError("Invalid native collision luminosity")
        values[name] = value
    return values


def leave_one_out(vectors):
    """Expose sensitivity to rare seed weights without removing any realization."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Stability diagnostics require at least four finite complete seeds")
    mean = values.mean(axis=0)
    omitted = [np.delete(values, i, axis=0) for i in range(len(values))]
    means = np.array([sample.mean(axis=0) for sample in omitted])
    sems = np.array([sample.std(axis=0, ddof=1)/np.sqrt(len(sample)) for sample in omitted])
    sem = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    influence = np.divide(np.max(np.abs(means-mean), axis=0), sem,
                          out=np.zeros_like(sem), where=sem > 0)
    return dict(minimum_seed_value=values.min(axis=0).tolist(),
        maximum_seed_value=values.max(axis=0).tolist(),
        leave_one_out_mean_min=means.min(axis=0).tolist(),
        leave_one_out_mean_max=means.max(axis=0).tolist(),
        leave_one_out_sem_min=sems.min(axis=0).tolist(),
        leave_one_out_sem_max=sems.max(axis=0).tolist(),
        full_sample_sem=sem.tolist(),
        first_half=dict(n=len(values)//2, mean=values[:len(values)//2].mean(axis=0).tolist(),
            sem=(values[:len(values)//2].std(axis=0, ddof=1)/np.sqrt(len(values)//2)).tolist()),
        second_half=dict(n=len(values)-len(values)//2, mean=values[len(values)//2:].mean(axis=0).tolist(),
            sem=(values[len(values)//2:].std(axis=0, ddof=1)/np.sqrt(len(values)-len(values)//2)).tolist()),
        maximum_leave_one_out_shift_in_full_sem=influence.tolist(),
        approximate_relative_sem_estimation_error_if_gaussian=float(1/np.sqrt(2*(len(values)-1))),
        qualification="All seeds retained; Gaussian SEM-estimation estimate is descriptive and may fail for heavy weights")


def kernel_weight_diagnostics(records, output):
    """Retain exact-kernel tail diagnostics without calling chunks independent events."""
    rows = []
    for record in records:
        if record["case"].get("pair_model") != 2:
            continue
        diagnostics = record["receipt"]["output_check"]["exact_kernel_diagnostics"]
        for key in ("pairs_1", "pairs_2"):
            item = diagnostics[key]
            total = float(item["sum_contributions"])
            rows.append(dict(case=record["case"]["name"], group=record["case"]["group"],
                seed=record["case"]["native_seed"], pair_model=2,
                process=item["process"], kernel_evaluations=item["kernel_evaluations"],
                sum_contributions=total, sum_squared_contributions=item["sum_squared_contributions"],
                max_contribution=item["max_contribution"],
                max_contribution_fraction_of_total=float(item["max_contribution"]/total) if total else 0.0,
                accepted_chunks=item["accepted_chunks"], stored_records=item["stored_records"]))
    if len(rows) != 48:
        raise ValueError("The two twelve-seed exact arms require all BH and LL weight diagnostics")
    write_csv(output/"exact_kernel_weight_diagnostics.csv", rows)
    summaries = {}
    for group in ("ilc_pairs", "halhf_pairs"):
        summaries[group] = {}
        for process in (1, 2):
            selected = [r for r in rows if r["group"] == group and r["process"] == process]
            summaries[group][str(process)] = dict(
                max_kernel_contribution=max(r["max_contribution"] for r in selected),
                max_single_contribution_fraction=max(r["max_contribution_fraction_of_total"] for r in selected),
                expectation_seed_stability=leave_one_out([[2*r["sum_contributions"]] for r in selected]))
    save_json(output/"exact_kernel_weight_summary.json", dict(groups=summaries,
        observable="Uncut production-kernel contributions; separate from admitted daughter yields",
        qualification="Kernel evaluations and split chunks are correlated diagnostics, not independent collision samples; no effective-count significance inferred"))


def reference_grid(rows):
    x = np.unique([float(r[k]) for r in rows for k in ("log10_theta_low", "log10_theta_high")])
    y = np.unique([float(r[k]) for r in rows for k in ("log10_pT_low", "log10_pT_high")])
    valid = np.zeros((len(x)-1, len(y)-1), dtype=bool)
    occupied = {s: np.zeros_like(valid) for s in ("electron", "positron")}
    for row in rows:
        i = int(np.searchsorted(x, float(row["log10_theta_low"])))
        j = int(np.searchsorted(y, float(row["log10_pT_low"])))
        valid[i, j] = float(row["valid_image_pixels"]) > 0
        for species, key in (("electron", "electron_red_ink_fraction"), ("positron", "positron_blue_ink_fraction")):
            occupied[species][i, j] = float(row[key]) > 0
    return x, y, valid, occupied


def secondary_chunks(path, size=50000):
    """Stream native ten-column records; never subsample the stored population."""
    batch = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) != 10 or not np.isfinite(values).all():
                raise ValueError("Malformed secondary row")
            if values[0] == 0 or values[7] not in (0, 1, 2):
                raise ValueError("Unexpected species/process in pure incoherent record")
            batch.append(values)
            if len(batch) == size:
                yield np.asarray(batch)
                batch = []
    if batch:
        yield np.asarray(batch)


def reduce_secondary(path, grids):
    counts = {f"{side}_{species}": np.zeros((len(grid[0])-1, len(grid[1])-1))
              for side, grid in grids.items() for species in ("electron", "positron")}
    diagnostics = dict(rows=0, z_zero=0, pz_zero=0, position_momentum_sign_disagree=0,
                       zero_theta_or_pt=0, under_or_overflow={key: 0 for key in counts}, selected={key: 0 for key in counts})
    for data in secondary_chunks(path):
        energy, vx, vy, vz, z = (data[:, i] for i in (0, 1, 2, 3, 6))
        transverse = np.hypot(vx, vy)
        theta, pt = np.arctan2(transverse, np.abs(vz)), np.abs(energy)*transverse
        positive = (theta > 0) & (pt > 0)
        diagnostics["rows"] += len(data)
        diagnostics["z_zero"] += int(np.count_nonzero(z == 0))
        diagnostics["pz_zero"] += int(np.count_nonzero(vz == 0))
        diagnostics["position_momentum_sign_disagree"] += int(np.count_nonzero(z*vz < 0))
        diagnostics["zero_theta_or_pt"] += int(np.count_nonzero(~positive))
        for side, sign in (("forward", 1), ("backward", -1)):
            for species, charge in (("electron", 1), ("positron", -1)):
                key = f"{side}_{species}"
                selected = positive & (z*sign > 0) & (energy*charge > 0)
                histogram = np.histogram2d(np.log10(theta[selected]), np.log10(pt[selected]), bins=grids[side][:2])[0]
                counts[key] += histogram
                diagnostics["selected"][key] += int(selected.sum())
                diagnostics["under_or_overflow"][key] += int(selected.sum()-histogram.sum())
    return counts, diagnostics


def low_rank_covariance(vectors):
    """Store C_mean=L.T@L without allocating a dense angular-grid covariance."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) < 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid complete-seed histogram vectors")
    difference = vectors-vectors[0]
    mean = vectors[0]+difference.mean(axis=0)
    factor = (difference-difference.mean(axis=0))/np.sqrt(len(vectors)*(len(vectors)-1))
    return mean, factor, np.sum(factor*factor, axis=0)


def reduce_halhf(records, source, output):
    grids = {side: reference_grid(read_csv(source/"benchmarks/reference"/f"halhf_pairs_250_{side}_ink.csv"))
             for side in ("forward", "backward")}
    arrays, summaries = {}, {}
    for side, (x, y, valid, support) in grids.items():
        arrays[side+"_xedges"], arrays[side+"_yedges"] = x, y
        arrays[side+"_reference_valid"] = valid
        for species in support:
            arrays[f"{side}_{species}_reference_support"] = support[species]
    for model in (1, 2):
        family = group_records(records, "halhf_pairs", 12, model)
        values, notes = [], []
        for r in family:
            thinning = float(r["case"]["metadata"]["pair_ratio"])
            pair_contract(r["directory"]/"acc.dat", model, .0005, thinning)
            counts, note = reduce_secondary(r["directory"]/"pairs.dat", grids)
            values.append({key: count/thinning for key, count in counts.items()})
            notes.append(dict(seed=r["case"]["native_seed"], pair_ratio=thinning,
                              per_row_crossing_weight=1/thinning, **note))
        metrics = {}
        for key in values[0]:
            side, species = key.split("_")
            shape = values[0][key].shape
            vectors = np.array([v[key].ravel() for v in values])
            mean, factor, variance = low_rank_covariance(vectors)
            prefix = f"model{model}_{key}"
            arrays[prefix+"_mean_count"] = mean.reshape(shape)
            arrays[prefix+"_mean_variance"] = variance.reshape(shape)
            arrays[prefix+"_covariance_factor"] = factor
            arrays[prefix+"_seed_counts"] = vectors
            arrays[prefix+"_occupied_seed_fraction"] = (vectors > 0).mean(axis=0).reshape(shape)
            valid, ref = grids[side][2], grids[side][3][species]
            native = mean.reshape(shape) > 0
            intersection, union = ((native & ref & valid).sum(), ((native | ref) & valid).sum())
            eligible = float(mean.reshape(shape)[valid].sum())
            metrics[key] = dict(occupied_support_jaccard=float(intersection/union) if union else 1.0,
                reference_support_coverage=float(intersection/(ref & valid).sum()) if (ref & valid).any() else 0.0,
                simulated_count_fraction_in_reference_support=float(mean.reshape(shape)[ref & valid].sum()/eligible) if eligible else 0.0,
                interpretation="Descriptive support comparison; raster occupancy is not event count or probability")
        summaries[f"model{model}"] = dict(runs=notes, support_metrics=metrics)
        totals = np.array([[v[key].sum() for key in values[0]] for v in values])
        summaries[f"model{model}"]["support_yield_seed_diagnostics"] = dict(
            columns=list(values[0]), **leave_one_out(totals))
    np.savez_compressed(output/"halhf_angular_ensemble.npz", **arrays)
    save_json(output/"halhf_angular_summary.json", dict(models=summaries,
        hemisphere="Final position z>0/z<0; position-versus-momentum sign disagreements retained",
        angle="atan2(hypot(beta_x,beta_y),abs(beta_z)); nearest-beam folding is a declared reconstruction",
        transverse_momentum="abs(E_GeV)*hypot(beta_x,beta_y); beta is p/E, not a unit direction",
        covariance="Per-bin covariance of complete-collision mean = factor.T@factor; no independent-copy Poisson term",
        normalization="Average per-seed counts divided by declared pair_ratio; not unit normalized. Occupancy frequencies describe the stored sample, so remain statistics dependent",
        reference="Published occupied pixels only; no artificial count uncertainty or density pull map",
        masking="Reference cells without valid image pixels are excluded from overlap metrics",
        selection="Pure incoherent daughters; 0.5 MeV native birth admission; no final-energy recut"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    sys.path.insert(0, str(args.source/"benchmarks/analysis"))
    statistics = importlib.import_module("comparison_statistics")
    manifest = read_json(args.inputs/"manifest.json")
    if len(manifest["cases"]) != 60 or len({c["name"] for c in manifest["cases"]}) != 60:
        raise ValueError("Expected exactly the prescribed sixty distinct cases")
    records = [validate_receipt(c, args.inputs, args.campaign) for c in manifest["cases"]]
    progress = read_json(args.campaign/"progress.json")
    if (progress.get("status") != "COMPLETE_EXECUTION_ONLY" or
            progress.get("manifest_sha256") != digest(args.inputs/"manifest.json")):
        raise ValueError("Campaign has no complete matching execution inventory")
    receipts = {r["case"]: r for r in progress["receipts"]}
    if len(receipts) != 60 or len(progress["receipts"]) != 60:
        raise ValueError("Execution inventory repeats or omits cases")
    for record in records:
        if receipts[record["case"]["name"]]["receipt_sha256"] != record["receipt_sha256"]:
            raise ValueError("Case receipt changed after inventory completion")
    if len({r["binary"] for r in records}) != 1:
        raise ValueError("All model arms must use the same qualified binary")
    pair_seeds = [r["case"]["native_seed"] for r in records if not r["case"]["group"].startswith("lep_")]
    if len(pair_seeds) != len(set(pair_seeds)):
        raise ValueError("Pair-model arms require disjoint declared seed banks")
    fig6 = manifest["figures"]["6"]
    if (fig6["hemisphere_coordinate"] != "final_position_z" or
            fig6["angular_convention"] != "nearest_beam_axis" or fig6["final_energy_recut"]):
        raise ValueError("The declared HALHF observable differs from this reduction")
    field_cases = [r["case"] for r in records if r["case"]["group"] == "lep_on"]
    if len({c["sampler_seed"] for c in field_cases}) != 10:
        raise ValueError("LEP fields must use distinct planned vertex sampler seeds")
    identities = {r["source_identity"] for r in records}
    if identities != {(args.source/"SOURCE_REVISION").read_text().strip()}:
        raise ValueError("Analysis and native source identities differ")
    # The frozen physics constant must match the admission reconstruction.
    constant = re.search(r"#define\s+EMASS\s+([\d.eE+-]+)", (args.source/"src/physconst.h").read_text())
    if not constant or float(constant.group(1)) != lep.ELECTRON_MASS_GEV:
        raise ValueError("Update and test the explicit imported-admission mass contract")
    args.output.mkdir(parents=True)
    reduce_lep(records, args.source, args.output, statistics)
    reduce_ilc(records, args.source, args.output, statistics)
    reduce_halhf(records, args.source, args.output)
    kernel_weight_diagnostics(records, args.output)
    if not args.no_render:
        import render_figures
        render_figures.render(args.output, args.source)
    save_json(args.output/"analysis_receipt.json", dict(status="REDUCED_NOT_PHYSICS_QUALIFIED",
        input_manifest_sha256=digest(args.inputs/"manifest.json"),
        execution_inventory_sha256=digest(args.campaign/"progress.json"),
        source_native_identity=next(iter(identities)),
        producer_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        reused_analysis_sha256={name: digest(args.source/"benchmarks/analysis"/name) for name in
            ("comparison_statistics.py", "reference_uncertainty.py", "regenerate_paper.py",
             "qed_publication.py", "paper_tables.py", "render_selected_figures.py")},
        reference_data_sha256={p.relative_to(args.source).as_posix(): digest(p) for p in
            [args.source/"benchmarks/reference/ilc_rimbault2006_effective_cross_sections.json",
             *sorted((args.source/"benchmarks/reference").glob("lep_final_focusing*.csv")),
             *sorted((args.source/"benchmarks/reference").glob("halhf_pairs_250_*_ink.csv")),
             *[args.source/"benchmarks/example_output/comparisons"/(name+".csv") for name in
               (*LEP_PANELS, "ilc_pair_yields_gp_pull", "ilc_pair_yields_cain_pull")]]},
        cases=[dict(case=r["case"], binary_sha256=r["binary"], receipt_sha256=r["receipt_sha256"],
                    source_hashes=r["hashes"], native_output_check=r["receipt"].get("output_check")) for r in records],
        qualification="No numerical convergence or reference precision certified; old and new native ensembles are not pooled",
        output_sha256={p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}))


if __name__ == "__main__":
    main()
