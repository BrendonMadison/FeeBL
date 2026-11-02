"""Verify the compact fixed-grid evidence package without executing FeeBL."""

import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(root):
    """Check retained bytes and native receipt bindings; omitted data stay explicit."""
    manifest = json.loads((root/"manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Packaged validation artifact changed: "+name)
    identities = json.loads((root/"identities.json").read_text())
    progress = json.loads((root/"cases/progress.json").read_text())
    if (progress["status"] != "PASSED_OUTPUT_CONTRACT" or len(progress["cases"]) != 7
            or progress["binary_sha256"] != identities["binary_sha256"]
            or (root/"SOURCE_REVISION").read_text().strip() != identities["source_native_identity"]):
        raise ValueError("Validation inventory identity or completeness changed")
    seen = set()
    for saved in progress["cases"]:
        name = saved["case"]
        if Path(name).name != name or name in seen:
            raise ValueError("Invalid or repeated validation case")
        seen.add(name)
        directory = root/"cases"/name
        receipt = json.loads((directory/"receipt.json").read_text())
        if receipt != saved or receipt["status"] != "PASSED_OUTPUT_CONTRACT" or receipt["returncode"] != 0:
            raise ValueError("Run receipt differs from its completed inventory: "+name)
        if digest(directory/"acc.dat") != receipt["card_sha256"]:
            raise ValueError("Native input card changed: "+name)
        output_hashes = receipt["output_check"]["output_sha256"]
        if digest(directory/"out.dat") != output_hashes["out.dat"]:
            raise ValueError("Native diagnostic output changed: "+name)
        omitted = manifest["omitted_native_outputs"][name]
        if omitted != {filename: sha for filename, sha in output_hashes.items() if filename != "out.dat"}:
            raise ValueError("Omitted native-output inventory changed: "+name)
    summary = json.loads((root/"validation_evidence.json").read_text())
    if summary["adaptive_strict_exit"] != 8 or not summary["required_targets_are_subset"]:
        raise ValueError("Qualification scope changed")
    return dict(status="PASS_PACKAGED_EVIDENCE_INTEGRITY", packaged_files=len(manifest["files"]),
                collision_receipts=len(seen), omitted_pair_files=len(manifest["omitted_native_outputs"]),
                scope="Retained evidence only; adaptive mode 4 remains unqualified")


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
