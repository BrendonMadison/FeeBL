"""Verify analysis input contracts without requiring or executing native results."""
import argparse
from collections import Counter
from pathlib import Path

import lep_support as lep
from reduce_figures import card_values, digest, pair_contract, read_csv, read_json, save_json


def check(inputs):
    manifest = read_json(inputs/"manifest.json")
    cases = manifest["cases"]
    expected = {("lep_on", None): 10, ("lep_zero", None): 2,
                ("ilc_pairs", 1): 12, ("ilc_pairs", 2): 12,
                ("halhf_pairs", 1): 12, ("halhf_pairs", 2): 12}
    if Counter((c["group"], c["pair_model"]) for c in cases) != expected:
        raise ValueError("Prepared family counts differ from the analysis design")
    results = []
    for case in cases:
        directory = inputs/case["name"]
        for name, receipt in case["files"].items():
            path = directory/name
            if digest(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
                raise ValueError("Changed frozen input: "+str(path))
        if card_values(directory/"acc.dat").get("rndm_seed") != case["native_seed"]:
            raise ValueError("Card and declared native seed differ")
        row = dict(case=case["name"], group=case["group"], pair_model=case["pair_model"])
        if case["group"].startswith("lep_"):
            sampling = read_csv(directory/"event_sampling.csv")
            inner = [r for r in sampling if r["cohort"] == "inner"]
            if len(inner) != 1024 or len(sampling) != 2048:
                raise ValueError("Incorrect number of LEP vertices")
            _, _, occupancy = lep.uniform_azimuth_weights([float(r["phi_rad"]) for r in inner])
            row.update(vertices_per_boundary=1024, stratum_occupancy=occupancy.tolist())
        else:
            contract = pair_contract(directory/"acc.dat", case["pair_model"],
                case["metadata"]["birth_energy_cut_GeV"], case["metadata"]["pair_ratio"])
            row["verified_population_contract"] = contract
        results.append(row)
    return dict(status="PASS_PREPARED_INPUT_CONTRACT_ONLY",
        input_manifest_sha256=digest(inputs/"manifest.json"), native_cases=60,
        qualification="No simulation output, numerical convergence or physical reproduction checked", cases=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.receipt, check(args.inputs))
