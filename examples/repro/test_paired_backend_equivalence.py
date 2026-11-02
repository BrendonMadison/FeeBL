#!/usr/bin/env python3
"""Focused regression tests for the paired backend-equivalence harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "paired_backend_equivalence", ROOT / "tools" / "paired_backend_equivalence.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedBackendEquivalenceTest(unittest.TestCase):
    def test_injection_targets_only_the_named_parameter_block(self) -> None:
        source = (
            "$PARAMETERS:: first\n{hist_ee_bins=1010;}\n"
            "$PARAMETERS:: chosen\n{rndm_seed=99;hist_ee_bins=1010;}\n"
        )
        result = MODULE.inject_fragment(
            source, 'compute_backend="gpu";', 17, "chosen"
        )
        first, chosen = result.split("$PARAMETERS:: chosen", 1)
        self.assertNotIn("compute_backend", first)
        self.assertIn('compute_backend="gpu";', chosen)
        self.assertTrue(
            chosen.index("rndm_seed=99") < chosen.index("rndm_seed=17"),
            "the enforced seed must be the final seed assignment",
        )

    def test_injection_rejects_rng_override_and_unknown_block(self) -> None:
        source = "$PARAMETERS:: selected\n{x=1;}\n"
        with self.assertRaisesRegex(ValueError, "RNG controls"):
            MODULE.inject_fragment(source, "rndm_seed=8;", 7, "selected")
        with self.assertRaisesRegex(ValueError, "found 0"):
            MODULE.inject_fragment(source, "", 7, "missing")

    def test_parser_collects_preregistered_observables(self) -> None:
        lines = [
            "lumi_fine = 1", "lumi_ee = 2", "lumi_ee_high = 3",
            "lumi_eg = 4", "lumi_ge = 5", "lumi_gg = 6", "upsmax=7",
            "average energy loss of the beam particles (GeV) : de = 8",
            "average energy loss of the beam particles (GeV) : de = 9",
            "beam 1 : miss = 10 beam 2 : miss = 11",
        ]
        lines.extend(
            f"lumi[{row}][{column}] = {12 + 3 * row + column}"
            for row in range(3) for column in range(3)
        )
        parsed = MODULE.parse_metrics("\n".join(lines))
        self.assertEqual(set(parsed), MODULE.REQUIRED_METRICS)
        self.assertEqual(parsed["beam2_energy_loss"], 9.0)
        self.assertEqual(parsed["lumi_2_2"], 20.0)

    def test_relative_margin_does_not_expand_near_zero_scale(self) -> None:
        analysis = MODULE.analyze_metric(
            "near_zero", [0.0] * 30, [0.0] * 30, 5.0e-6, 1.0e-12
        )
        self.assertEqual(analysis["equivalence_margin_absolute"], 1.0e-12)
        self.assertTrue(analysis["tost_equivalent"])


if __name__ == "__main__":
    unittest.main()
