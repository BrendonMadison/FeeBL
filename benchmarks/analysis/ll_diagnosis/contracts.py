"""Admit complete, source-bound native results for the prescribed LL controls."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from definitions import ARMS, SEEDS

SOURCE_ID = "sha256:238eda4954de06bada9c27a56436281d0ddac7137712e8e4bf07be37358f2598"
BINARY_HASH = "cf872ffbfb2d91dfd8286c9f0282329040de8b0899e1cd7133e4c2fd27924d8b"
PASS = "PASSED_OUTPUT_CONTRACT"


def digest(path):
    """Hash files without loading large native records into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    """Use an explicit encoding for portable saved metadata."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def populations(path, birth=False):
    """Count admitted daughters by channel and species, without a final recut."""
    result = np.zeros((3, 2), dtype=np.int64)
    with path.open(encoding="ascii") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = list(map(float, line.split()))
            if len(row) != 10 or not all(map(math.isfinite, row)):
                raise ValueError("Invalid native pair row")
            if row[0] == 0 or row[7] not in (0, 1, 2) or math.hypot(*row[1:4]) > 1.00002:
                raise ValueError("Unexpected particle population or velocity")
            if abs(row[0]) < .51099906e-3*(1-1e-5):
                raise ValueError("Saved total energy is below the host electron rest mass")
            if birth and abs(row[0]) < .005*(1-1e-5):
                raise ValueError("Produced daughter below the declared cut")
            result[int(row[7]), int(row[0] < 0)] += 1
    if result.sum() == 0:
        raise ValueError("Empty admitted daughter population")
    return result


def luminosities(text):
    """Read inclusive native luminosities; duplicated ee entries must agree."""
    values = []
    for name in ("lumi_ee", "lumi_eg", "lumi_ge", "lumi_gg"):
        raw = re.findall(r"^"+name+r"\s*=\s*([^\s;]+)", text, re.MULTILINE)
        if not raw or len(raw) > (2 if name == "lumi_ee" else 1):
            raise ValueError("Missing or repeated luminosity: "+name)
        parsed = list(map(float, raw))
        if not all(math.isfinite(value) and value >= 0 for value in parsed) or any(value != parsed[0] for value in parsed):
            raise ValueError("Nonfinite or inconsistent luminosity")
        values.append(parsed[0])
    if values[0] <= 0:
        raise ValueError("Nonpositive inclusive electron-positron luminosity")
    return values


def validate_card(path, case):
    """Check the frozen card's defining model, population and common switches."""
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    found = re.findall(r"(?<![\w.])([A-Za-z_]\w*(?:\.[12])?)\s*=\s*([-+\d.eE]+)\s*;", text)
    if len({key for key, _ in found}) != len(found):
        raise ValueError("Duplicate numerical input setting")
    values = {key: float(value) for key, value in found}
    definition = {arm: (model, size, virtuality) for arm, _, model, size, virtuality in ARMS}[case["arm"]]
    if tuple(case["changes"][key] for key in ("pair_model", "beam_size", "pair_q2")) != definition:
        raise ValueError("Manifest arm labels disagree with the native model settings")
    required = dict(pair_model=definition[0], beam_size=definition[1], pair_q2=definition[2],
        photon_polarisation=0, photon_timing=3, photon_angle=2, coherent_model=1,
        pair_ratio=1, pair_ecut=.005, beam_pair=0, do_pairs=1, do_compt=0, do_coherent=0,
        do_trident=0, load_events=0, do_bhabhas=0, do_photons=1, do_photon_lumi=1,
        track_pairs=1, store_pairs=2, rndm_seed=case["seed"], n_m=10000, n_z=32, n_t=2)
    expanded = case["arm"] == "exact_profile_expanded_domain"
    required.update(n_x=256 if expanded else 128, n_y=256 if expanded else 128,
        cut_x=6510.32233524 if expanded else 3255.16116762,
        cut_y=137.249645875 if expanded else 68.6248229375)
    if any(values.get(key) != value for key, value in required.items()):
        raise ValueError("Card differs from the declared matched control")


def load_campaign(campaign, inputs):
    """Verify all 36 receipts and native populations, then rebuild seed vectors."""
    campaign, inputs = Path(campaign), Path(inputs)
    manifest = read_json(inputs/"manifest.json")
    progress = read_json(campaign/"progress.json")
    manifest_hash = digest(inputs/"manifest.json")
    if (progress["status"] != "COMPLETE_EXECUTION_ONLY_NOT_PHYSICS_QUALIFIED" or progress["not_started"]
            or progress["source_native_identity"] != SOURCE_ID or progress["binary_sha256"] != BINARY_HASH
            or progress["manifest_sha256"] != manifest_hash):
        raise ValueError("Campaign is incomplete or has a different source/input identity")
    if manifest["source_native_identity"] != SOURCE_ID or manifest["case_count"] != 36:
        raise ValueError("Unexpected source identity or size in the input manifest")
    cases = manifest["cases"]
    expected = {(arm, seed) for arm, *_ in ARMS for seed in SEEDS}
    if len(cases) != 36 or {(case["arm"], case["seed"]) for case in cases} != expected:
        raise ValueError("Expected all nine arms and the same four seeds")
    announcements = {record["case"]: record for record in progress["receipts"]}
    if len(progress["receipts"]) != 36 or set(announcements) != {case["case"] for case in cases}:
        raise ValueError("Receipt identities do not match the prescribed cases")
    records, provenance, job_ids = {}, [], set()
    for case in cases:
        name = case["case"]
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError("Unsafe case name")
        folder = campaign/name
        receipt_path = folder/"receipt.json"
        receipt = read_json(receipt_path)
        if not re.fullmatch(r"[0-9]+", str(receipt.get("job_id", ""))):
            raise ValueError("Missing native job identity")
        job_ids.add(str(receipt["job_id"]))
        announcement = announcements[name]
        if (announcement["status"] != PASS or announcement["receipt_sha256"] != digest(receipt_path)
                or receipt["status"] != PASS or receipt["returncode"] != 0
                or receipt["case"] != name or receipt["arm"] != case["arm"] or receipt["native_seed"] != case["seed"]
                or receipt["source_native_identity"] != SOURCE_ID or receipt["binary_sha256"] != BINARY_HASH
                or receipt["manifest_sha256"] != manifest_hash
                or receipt["execution_binding_sha256"] != progress["execution_binding_sha256"]):
            raise ValueError("Failed or mismatched native receipt: "+name)
        card_hash = digest(inputs/name/"acc.dat")
        if card_hash != case["input_sha256"] or card_hash != receipt["input_sha256"] or card_hash != digest(folder/"acc.dat"):
            raise ValueError("Changed input card: "+name)
        validate_card(folder/"acc.dat", case)
        required = {"acc.dat", "out.dat", "pairs.dat", "pairs0.dat", "stdout.txt", "stderr.txt", "resources.txt"}
        if not required.issubset(receipt["output_sha256"]):
            raise ValueError("Incomplete native hash inventory")
        verified = {}
        for filename, expected_hash in receipt["output_sha256"].items():
            if Path(filename).name != filename:
                raise ValueError("Unsafe native output path")
            path = folder/filename
            if filename in required and not path.is_file():
                raise ValueError("Missing required native file: "+filename)
            if path.is_file():
                if digest(path) != expected_hash:
                    raise ValueError("Changed native output: "+filename)
                verified[filename] = expected_hash
        birth, final = populations(folder/"pairs0.dat", True), populations(folder/"pairs.dat")
        if not np.array_equal(birth, final) or not np.array_equal(final, receipt["pair_final"]["process_species_stored_daughters"]):
            raise ValueError("Birth/final/receipt populations do not close")
        counts = final.sum(axis=1)
        if not np.array_equal(counts, receipt["pair_final"]["process_stored_daughters"]):
            raise ValueError("Receipt process totals differ from native daughters")
        if case["changes"]["pair_model"] == 2:
            exact = receipt["output_check"]["exact_kernel_diagnostics"]
            for process in (1, 2):
                if exact["pairs_"+str(process)]["stored_records"] != counts[process]:
                    raise ValueError("Exact native counters do not close")
        light = luminosities((folder/"out.dat").read_text(encoding="utf-8"))
        if light != [receipt["luminosity_m_minus2"][key] for key in ("lumi_ee", "lumi_eg", "lumi_ge", "lumi_gg")]:
            raise ValueError("Receipt luminosities differ from native output")
        records[case["arm"], case["seed"]] = (counts.tolist()+[value*1e-34 for value in light], receipt)
        provenance.append(dict(case=name, receipt_sha256=digest(receipt_path), verified_native_sha256=verified))
    if len(job_ids) != 1:
        raise ValueError("The completed campaign mixes multiple execution jobs")
    arm_names = sorted(arm for arm, *_ in ARMS)
    vectors = np.asarray([sum((records[arm, seed][0] for arm in arm_names), []) for seed in SEEDS])
    saved = read_json(campaign/"joint_seed_covariance.json")
    if saved["arms"] != arm_names or saved["seeds"] != list(SEEDS) or saved["ratio_units"] != "microbarn":
        raise ValueError("Unexpected saved joint-covariance layout or units")
    np.testing.assert_allclose(vectors, saved["vectors"], rtol=1e-13, atol=0)
    return vectors, arm_names, saved, records, dict(manifest_sha256=manifest_hash,
        progress_sha256=digest(campaign/"progress.json"), joint_seed_covariance_sha256=digest(campaign/"joint_seed_covariance.json"),
        source_native_identity=SOURCE_ID, binary_sha256=BINARY_HASH, job_id=next(iter(job_ids)),
        execution_binding_sha256=progress["execution_binding_sha256"], cases=provenance,
        scope="Every required numerical file and every locally present listed native output was verified; omitted ancillary outputs are not claimed as reverified.")
