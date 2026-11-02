"""Check that recovery/subset plans preserve the originally declared RNG seeds."""
import json
from pathlib import Path
import unittest
import uuid

import prepare_domain_campaign as domains
import seed_campaign as seeds


class CanonicalSeedTests(unittest.TestCase):
    def test_original_eight_family_assignment_is_unchanged(self):
        for index, case in enumerate(seeds.DEFAULT_CASES):
            for replica in range(12):
                self.assertEqual(
                    seeds.seed_for_case(case, replica),
                    729000000 + 10000 * index + 101 * replica,
                )

    def test_recovery_plan_preserves_failed_seed_and_domain_pairing(self):
        root = Path(__file__).resolve().parent / ("test_seeds_" + uuid.uuid4().hex)
        root.mkdir()
        try:
            selected = tuple(reversed(seeds.DEFAULT_CASES[4:]))
            plan = seeds.prepare(root / "recovery", selected, 12)
            by_case = {
                case: [job for job in plan["jobs"] if job["case"] == case]
                for case in selected
            }
            self.assertEqual(by_case["halhf250_wide_domain"][2]["seed"], 729040202)
            for case in selected:
                self.assertEqual(
                    [job["seed"] for job in by_case[case]],
                    [seeds.seed_for_case(case, replica) for replica in range(12)],
                )
            domain_plan = domains.prepare(root / "domains")
            for job in domain_plan["jobs"]:
                control = next(
                    item
                    for item in by_case[job["baseline_case"]]
                    if item["seed"] == job["seed"]
                )
                self.assertEqual(job["seed"], control["seed"])
                self.assertIn(
                    job["seed"],
                    [
                        seeds.seed_for_case(job["baseline_case"], replica)
                        for replica in range(3)
                    ],
                )
        finally:
            # This newly created test tree contains only the copied card fixtures.
            self.assertEqual(root.parent.resolve(), Path(__file__).resolve().parent)
            for path in sorted(
                root.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()

    def test_seed_block_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            seeds.seed_for_case("halhf380", 100)
        with self.assertRaises(ValueError):
            seeds.seed_for_case("unassigned_case", 0)


if __name__ == "__main__":
    unittest.main()
