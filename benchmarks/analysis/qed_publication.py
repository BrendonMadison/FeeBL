"""Route current focusing/pair figures and tables through one verified reduction."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_REDUCTION = SOURCE / "benchmarks/example_output/qed_figure_campaign/reduction"
FIGURE_ALIASES = {
    f"{producer}.{extension}": f"{published}.{extension}"
    for producer, published in (("figure2_lep_focusing", "lep_focusing"),
        ("figure3_ilc_pair_yields", "ilc_pair_yields"),
        ("figure6_halhf_angular_support", "halhf_pair_populations"))
    for extension in ("pdf", "png")
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def reduction_receipt(reduction):
    """Read a completed numerical receipt; missing or incomplete data are errors."""
    path = Path(reduction) / "analysis_receipt.json"
    if not path.is_file():
        raise FileNotFoundError("Current publication requires a completed QED ensemble reduction: "
                                + str(path) + "; supply --qed-reduction after reducing all 60 collisions")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "REDUCED_NOT_PHYSICS_QUALIFIED":
        raise ValueError("QED numerical reduction is incomplete")
    return receipt


def export_products(kind, reduction, output, source=SOURCE):
    """Use a fresh interpreter so a cached module cannot bind another repository."""
    reduction_receipt(reduction)
    script = Path(source) / "benchmarks/analysis/qed_figures" / ("render_" + kind + ".py")
    subprocess.run([sys.executable, "-B", str(script), "--source", str(Path(source).resolve()),
                    "--reduction", str(Path(reduction).resolve()), "--output", str(Path(output).resolve())],
                   check=True, timeout=300)
    receipt = verified_products(output, reduction, kind, source)
    shutil.copyfile(Path(reduction)/"analysis_receipt.json", Path(output)/"analysis_receipt.json")
    return receipt


def verified_products(products, reduction, kind, source=SOURCE):
    """Validate an already exported product set before copying publication aliases."""
    products = Path(products)
    analysis = reduction_receipt(reduction)
    if analysis["source_native_identity"] != (Path(source)/"SOURCE_REVISION").read_text().strip():
        raise ValueError("Publication source identity differs from the reduction")
    if not analysis.get("output_sha256"):
        raise ValueError("Reduction does not identify its numerical outputs")
    for name, expected_hash in analysis["output_sha256"].items():
        if Path(name).name != name or digest(Path(reduction)/name) != expected_hash:
            raise ValueError("Numerical QED reduction changed: "+name)
    for group, directory in (("reused_analysis_sha256", Path(source)/"benchmarks/analysis"),
                             ("producer_sha256", Path(source)/"benchmarks/analysis/qed_figures")):
        if not analysis.get(group):
            raise ValueError("Reduction does not identify its analysis producers")
        for name, expected_hash in analysis[group].items():
            if Path(name).name != name or digest(directory/name) != expected_hash:
                raise ValueError("Publication analysis producer changed: "+name)
    filename = "table_receipt.json" if kind == "tables" else "rerender_receipt.json"
    receipt = json.loads((products / filename).read_text())
    expected = "EXPORTED_FROM_VERIFIED_REDUCTION" if kind == "tables" else "RENDERED_FROM_VERIFIED_REDUCTION"
    if receipt.get("status") != expected or receipt.get("analysis_receipt_sha256") != digest(Path(reduction)/"analysis_receipt.json"):
        raise ValueError("Export is not bound to the selected reduction")
    retained = products / "analysis_receipt.json"
    if retained.exists() and digest(retained) != digest(Path(reduction)/"analysis_receipt.json"):
        raise ValueError("Retained reduction receipt changed")
    if kind == "tables" and receipt.get("table_exporter_sha256") != digest(Path(source)/"benchmarks/analysis/qed_figures/render_tables.py"):
        raise ValueError("Table exporter changed after export")
    outputs = receipt.get("output_sha256", {})
    if not outputs:
        raise ValueError("Export receipt contains no outputs")
    for name, expected_hash in outputs.items():
        if Path(name).name != name or digest(products/name) != expected_hash:
            raise ValueError("Exported QED artifact changed: " + name)
    return receipt


def copy_figures(products, destination, reduction):
    """Copy verified PDF/PNG products to stable manuscript names and bind hashes."""
    receipt = verified_products(products, reduction, "figures")
    for producer, published in FIGURE_ALIASES.items():
        if producer not in receipt["output_sha256"]:
            raise ValueError("Required QED figure missing from export: " + producer)
        shutil.copyfile(Path(products)/producer, Path(destination)/published)
    return dict(analysis_receipt_sha256=digest(Path(reduction)/"analysis_receipt.json"),
                rerender_receipt_sha256=digest(Path(products)/"rerender_receipt.json"),
                aliases={source: dict(publication=target, sha256=digest(Path(destination)/target))
                         for source, target in FIGURE_ALIASES.items()})


def campaign_cases(reduction, source=SOURCE):
    """Bind the public input cards to the exact cases admitted by the reduction."""
    receipt = reduction_receipt(reduction)
    inputs = Path(source) / "benchmarks/cases/qed_figure_campaign"
    manifest = json.loads((inputs/"manifest.json").read_text())
    if digest(inputs/"manifest.json") != receipt["input_manifest_sha256"]:
        raise ValueError("Packaged QED input manifest differs from the admitted reduction")
    records = {record["case"]["name"]: record for record in receipt["cases"]}
    if len(records) != 60 or len(receipt["cases"]) != 60 or len(manifest["cases"]) != 60:
        raise ValueError("Current QED input table requires all 60 admitted cases")
    for case in manifest["cases"]:
        record = records[case["name"]]
        if record["case"] != case:
            raise ValueError("Packaged QED case differs from the admitted result: "+case["name"])
        for name, expected in case["files"].items():
            path = inputs/case["name"]/name
            if (Path(name).name != name or record["source_hashes"].get(name) != expected["sha256"]
                    or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]):
                raise ValueError("Packaged QED input differs from its numerical result: "+case["name"]+"/"+name)
    return manifest["cases"], receipt
