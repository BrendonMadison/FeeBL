"""Reviewed, immutable, sequential plans. Execution is native Linux only."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time

from .common import Invalid, child, digest, integer, keys, read_json, require, sha256, write_json

ENV_KEYS = {
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "CUDA_VISIBLE_DEVICES",
    "OPENBLAS_NUM_THREADS",
}


def selected_seed(card_text, accelerator, parameters):
    """Parse a deliberately small resolved-card subset, never FeeBL expressions.

    Comments are lexical, not regex substitutions. Non-RNG quoted scalar values
    are bounded; includes, inheritance and nested blocks are refused.
    """
    clean = []
    index = 0
    while index < len(card_text):
        if card_text.startswith("/*", index):
            end = card_text.find("*/", index + 2)
            require(end >= 0, "Unterminated block comment")
            clean.append(" " * (end + 2 - index))
            index = end + 2
        elif card_text.startswith("//", index) or card_text[index] == "#":
            end = card_text.find("\n", index)
            if end < 0:
                end = len(card_text)
            clean.append(" " * (end - index))
            index = end
        elif card_text[index] in "\"'":
            quote = card_text[index]
            end = card_text.find(quote, index + 1)
            require(end >= 0, "Unterminated quoted scalar")
            value = card_text[index + 1 : end]
            require(
                re.fullmatch(r"[A-Za-z0-9_.:/+ -]+", value) is not None
                and not any(marker in value for marker in ("//", "/*", "*/")),
                "Quoted scalar contains unsupported escapes, comments or control/assignment delimiters",
            )
            clean.append(card_text[index : end + 1])
            index = end + 1
        else:
            clean.append(card_text[index])
            index += 1
    text = "".join(clean)
    blocks = {}
    position = 0
    while position < len(text):
        if not text[position:].strip():
            break
        header = re.match(
            r"\s*\$(ACCELERATOR|PARAMETERS)\s*::\s*([A-Za-z0-9_]+)\s*\{", text[position:]
        )
        require(
            header is not None,
            "Unsupported card header/include/inheritance or text outside a resolved block",
        )
        kind, name = header.groups()
        begin = position + header.end()
        end = text.find("}", begin)
        require(end >= 0 and "{" not in text[begin:end], "Unbalanced or nested card braces")
        body = text[begin:end]
        key = (kind, name)
        require(key not in blocks, "Duplicate block declaration is ambiguous")
        assignments = {}
        for statement in body.split(";")[:-1]:
            match = re.fullmatch(
                r"\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[0-9]+)?)\s*=\s*([^;]+?)\s*", statement, re.DOTALL
            )
            require(match is not None, "Unsupported or incomplete simple-card assignment")
            field, value = match.groups()
            require(field not in assignments, "Duplicate active assignment is ambiguous")
            if '"' in value or "'" in value:
                require(
                    re.fullmatch(r"""(["'])[A-Za-z0-9_.:/+ -]+\1""", value.strip()) is not None,
                    "Quoted values must be complete non-RNG scalar tokens, not expressions",
                )
                require(
                    not field.startswith(("rndm_seed", "rndm_load")),
                    "RNG seed/load cannot be quoted strings",
                )
            assignments[field] = value.strip()
        require(
            not body.split(";")[-1].strip(),
            "Every simple-card assignment must end with a semicolon",
        )
        blocks[key] = assignments
        position = end + 1
    require(("ACCELERATOR", accelerator) in blocks, "Selected accelerator block is absent")
    require(("PARAMETERS", parameters) in blocks, "Selected parameter block is absent")
    active = blocks[("PARAMETERS", parameters)]
    require(
        "rndm_seed" in active and re.fullmatch(r"[0-9]+", active["rndm_seed"]) is not None,
        "Selected active rndm_seed must be one unsigned integer literal, never an expression or comment",
    )
    require(
        active.get("rndm_load") == "0",
        "Simple-card seed authentication requires explicit rndm_load=0 (no loaded RNG state)",
    )
    require(
        not any(
            field.startswith("rndm_seed.") or field.startswith("rndm_load.") for field in active
        ),
        "Indexed RNG overrides are not supported",
    )
    return int(active["rndm_seed"])


def verify_seed_receipt(text, receipt, seed):
    keys(receipt, ("path", "field", "evidence_kind"))
    require(
        receipt["evidence_kind"] == "runtime_rng_initialization",
        "Seed evidence must come from actual RNG initialization, not a requested-input echo",
    )
    require(
        receipt["field"] == "the random number seed is"
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,95}", receipt["field"]) is not None,
        "Invalid seed receipt field; only an identifier or the source-audited FeeBL initialization label is accepted",
    )
    matches = re.findall(
        r"^\s*" + re.escape(receipt["field"]) + r"\s*[:=]\s*([0-9]+)\s*;?\s*$", text, re.MULTILINE
    )
    require(
        len(matches) == 1 and int(matches[0]) == seed,
        "Missing, duplicate or mismatching executed-seed receipt",
    )
    return int(matches[0])


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command_for(config):
    options = config.get("options", [])
    require(
        config["schema"] in ("feebl.run_config.v1", "feebl.run_config.v2"),
        "Unknown run configuration schema",
    )
    require(
        config["schema"] == "feebl.run_config.v2" or "options" not in config,
        "Options require v2 config",
    )
    require(
        isinstance(options, list) and options in ([], ["--make_gui_files"]),
        "Only the bounded GUI exporter flag is supported",
    )
    return [config["binary"], *options, config["accelerator"], config["parameters"], "out.dat"]


def make_plan(config_path):
    config = read_json(config_path)
    keys(
        config,
        (
            "schema",
            "binary",
            "card",
            "binary_sha256",
            "card_sha256",
            "source_revision",
            "output_root",
            "run_name",
            "accelerator",
            "parameters",
            "seed",
            "timeout_seconds",
            "environment",
            "receipts",
            "expected_outputs",
        ),
        ("options",),
    )
    command_for(config)
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", config["run_name"]) is not None, "Unsafe run name")
    for name in ("accelerator", "parameters"):
        require(re.fullmatch(r"[a-zA-Z0-9_]{1,96}", config[name]) is not None, f"Unsafe {name}")
    require(
        re.fullmatch(r"[0-9a-f]{40}", config["source_revision"]) is not None,
        "A full source commit SHA is required",
    )
    for name in ("binary", "card"):
        path = Path(config[name]).resolve(strict=True)
        require(path.is_file(), f"{name} is not a file")
        require(sha256(path) == config[f"{name}_sha256"], f"{name} SHA mismatch")
        config[name] = str(path)
    integer(config["seed"], "seed", 0)
    seed = selected_seed(
        Path(config["card"]).read_text(encoding="utf-8"),
        config["accelerator"],
        config["parameters"],
    )
    require(seed == config["seed"], "Selected active parameter-block seed does not match the plan")
    root = Path(config["output_root"]).resolve(strict=True)
    require(str(root) == config["output_root"], "Output root was retargeted after plan review")
    require(root.is_dir(), "Output root must already exist")
    config["output_root"] = str(root)
    require(not child(root, config["run_name"]).exists(), "Run output already exists")
    integer(config["timeout_seconds"], "timeout_seconds", 1)
    require(config["timeout_seconds"] <= 86400, "Timeout exceeds 24 hours")
    require(
        isinstance(config["environment"], dict) and set(config["environment"]) <= ENV_KEYS,
        "Unapproved environment variable",
    )
    require(
        all(isinstance(v, str) and "\x00" not in v for v in config["environment"].values()),
        "Environment values must be strings",
    )
    require(
        isinstance(config["expected_outputs"], list) and config["expected_outputs"],
        "Expected output list must not be empty",
    )
    for output in config["expected_outputs"]:
        keys(output, ("path", "min_bytes"))
        child(root, output["path"])
        integer(output["min_bytes"], "min_bytes", 1)
    require(
        set(config["receipts"]) == {"source", "backend", "seed"},
        "Source, actual backend and executed-seed receipt contracts are mandatory",
    )
    for role, receipt in config["receipts"].items():
        if role == "seed":
            # Validate shape/field with a manufactured line, not a real execution claim.
            verify_seed_receipt(
                f"{receipt.get('field')} = {config['seed']}", receipt, config["seed"]
            )
            child(root, receipt["path"])
            continue
        keys(receipt, ("path", "contains"))
        child(root, receipt["path"])
        require(
            isinstance(receipt["contains"], list)
            and receipt["contains"]
            and all(isinstance(x, str) and x for x in receipt["contains"]),
            "Receipts require nonempty literal substrings",
        )
        if role == "source":
            require(
                any(config["source_revision"] in x for x in receipt["contains"]),
                "Source receipt must contain the full reviewed commit",
            )
    payload = {
        "schema": "feebl.run_plan.v1",
        "created_utc": utc(),
        "config_sha256": sha256(config_path),
        "config": config,
        "sequential_jobs": 1,
        "execution_platform": "native-linux-only",
        "command": command_for(config),
    }
    return {"payload": payload, "review_token": digest(payload)}


def validate_plan(plan, token):
    keys(plan, ("payload", "review_token"))
    require(
        digest(plan["payload"]) == plan["review_token"] == token, "Plan hash/review token mismatch"
    )
    payload = plan["payload"]
    require(
        payload.get("schema") == "feebl.run_plan.v1" and payload.get("sequential_jobs") == 1,
        "Invalid execution plan",
    )
    config = payload["config"]
    require(
        payload["command"] == command_for(config),
        "Command differs from the reviewed FeeBL interface",
    )
    for name in ("binary", "card"):
        require(sha256(config[name]) == config[f"{name}_sha256"], f"{name} changed after review")
    return config


def _stop_group(process):
    """Kill the complete original process group; refuse a success if it survives."""
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return True
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            process.poll()  # reap the direct child
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.02)
    return False


def run_plan(plan_path, token, execute=False):
    plan = read_json(plan_path)
    config = validate_plan(plan, token)
    root = Path(config["output_root"]).resolve(strict=True)
    require(str(root) == config["output_root"], "Output root was retargeted after plan review")
    output = child(root, config["run_name"])
    require(not output.exists(), "Refusing an existing run directory")
    if not execute:
        return {
            "status": "REVIEWED_NOT_EXECUTED",
            "review_token": token,
            "output": str(output),
            "platform_note": "Native Linux execution only. Windows/WSL invocation must be reviewed and performed in a native Linux shell; the harness does not pretend Windows can kill a WSL process tree.",
        }
    require(
        platform.system() == "Linux",
        "Execution refused: run the harness in native Linux (including inside WSL), never through a Windows WSL wrapper",
    )
    with open(config["binary"], "rb") as binary_stream:
        require(
            binary_stream.read(4) == b"\x7fELF",
            "Execution requires a reviewed native ELF executable, not a shell/WSL launcher",
        )
    lock = root / ".feebl-harness-active.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    result = {
        "schema": "feebl.run_result.v1",
        "status": "FAILED",
        "started_utc": utc(),
        "review_token": token,
        "source_revision_expected": config["source_revision"],
        "seed": config["seed"],
        "outputs": {},
        "no_fallback_permitted": True,
    }
    try:
        output.mkdir()
        shutil.copyfile(config["card"], output / "acc.dat")
        require(sha256(output / "acc.dat") == config["card_sha256"], "Copied card hash mismatch")
        write_json(output / "reviewed_plan.json", plan)
        env = os.environ.copy()
        env.update(config["environment"])
        result["thread_environment"] = {key: env.get(key) for key in sorted(ENV_KEYS)}
        started = time.monotonic()
        with open(output / "stdout.txt", "xb") as stdout, open(output / "log.txt", "xb") as stderr:
            process = subprocess.Popen(
                plan["payload"]["command"],
                cwd=output,
                env=env,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=True,
            )
            try:
                result["returncode"] = process.wait(timeout=config["timeout_seconds"])
            except subprocess.TimeoutExpired:
                result["status"] = "TIMEOUT" if _stop_group(process) else "TIMEOUT_CLEANUP_FAILED"
                result["wall_seconds"] = time.monotonic() - started
                return result
            except BaseException:
                _stop_group(process)
                raise
        result["wall_seconds"] = time.monotonic() - started
        # A successful direct child may not leave background descendants running.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            require(_stop_group(process), "Descendant process cleanup failed")
            raise Invalid("Executable left background descendants; run rejected")
        require(result["returncode"] == 0, "FeeBL exited nonzero")
        for item in config["expected_outputs"]:
            path = child(output, item["path"])
            require(
                path.is_file() and path.stat().st_size >= item["min_bytes"],
                f"Missing/truncated output: {path}",
            )
            result["outputs"][item["path"]] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        for role, receipt in config["receipts"].items():
            text = child(output, receipt["path"]).read_text(encoding="utf-8", errors="strict")
            if role == "seed":
                actual_seed = verify_seed_receipt(text, receipt, config["seed"])
                result["executed_seed_receipt"] = {
                    "path": receipt["path"],
                    "sha256": sha256(child(output, receipt["path"])),
                    "seed": actual_seed,
                    "field": receipt["field"],
                }
                continue
            require(
                all(value in text for value in receipt["contains"]),
                f"Unauthenticated {role} receipt",
            )
            # Conservative: explicit fallback indicators are never treated as success.
            require(
                not re.search(
                    r"fallback\s*[:=]\s*(?:true|1|yes)|fallback reason\s*[:=]\s*\S",
                    text,
                    re.IGNORECASE,
                ),
                "Backend fallback receipt present",
            )
            result[f"{role}_receipt"] = {
                "path": receipt["path"],
                "sha256": sha256(child(output, receipt["path"])),
                "matched": receipt["contains"],
            }
        result["status"] = "COMPLETED_RECEIPTS_VERIFIED_NOT_PHYSICS_VALIDATED"
        return result
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["finished_utc"] = utc()
        if output.is_dir():
            write_json(output / "harness_result.json", result)
        os.close(lock_fd)
        lock.unlink()


def doctor(inventory_path=None, wsl=False):
    result = {
        "schema": "feebl.doctor.v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "read_only": True,
        "tools": {},
        "paths": [],
        "wsl": {"requested": wsl},
    }
    for name in (
        "cmake",
        "ninja",
        "gcc",
        "g++",
        "nvcc",
        "nvidia-smi",
        "git",
        "root-config",
        "python",
        "wsl.exe",
    ):
        result["tools"][name] = shutil.which(name)
    if inventory_path:
        inventory = read_json(inventory_path)
        keys(inventory, ("schema", "paths"))
        require(inventory["schema"] == "feebl.inventory.v1", "Unknown inventory schema")
        for item in inventory["paths"]:
            keys(item, ("label", "path"))
            path = Path(item["path"])
            record = {**item, "exists_on_this_platform": path.exists()}
            if path.is_file():
                record["sha256"] = sha256(path)
            result["paths"].append(record)
    if wsl:
        require(platform.system() == "Windows", "--wsl inventory is only for Windows")
        command = ["wsl.exe", "--list", "--verbose"]
        completed = subprocess.run(command, capture_output=True, timeout=15, shell=False)
        result["wsl"].update(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout.decode("utf-16-le", errors="replace"),
                "stderr": completed.stderr.decode("utf-16-le", errors="replace"),
            }
        )
    return result
