"""Integration tests for authenticated derived diagnostics; no simulation runs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import finalize_gui_dataset as finalizer
from test_collider_monochromatization import SUMMARY, MANIFEST


def ref(path):
    return dict(path=path.name, bytes=path.stat().st_size, sha256=finalizer.digest(path))


def fixture(root, empty_pairs=False):
    flags = dict(
        do_pairs=1,
        load_events=0,
        track_pairs=1,
        store_pairs=1,
        pair_ratio=1,
        pair_ecut=0.0005,
        beam_pair=0,
        do_comp=0,
        do_prod=0,
        do_bhabhas=0,
    )
    count = 0 if empty_pairs else 2
    (root / "out.dat").write_text(
        SUMMARY
        + "\n"
        + "\n".join(f"{k} = {v}" for k, v in flags.items())
        + f"\nn_pairs = {count}\n"
    )
    (root / "pairs.dat").write_text(
        ""
        if empty_pairs
        else "2 .3 .4 .866025403784 0 0 0 0 -1 -1\n-2 .3 .4 -.866025403784 0 0 0 1 -1 -1\n"
    )
    (root / "collider_analysis.json").write_text(
        json.dumps(
            dict(
                schema_version="feebl.collider.analysis/1",
                native_statistics=dict(full_RMS_W_GeV=0.25),
            )
        )
    )
    h = dict(
        quantity="invariant mass",
        unit="GeV",
        weight_unit="m^-2 per crossing",
        weighting="native luminosity",
        normalization="bin-integrated",
        bins=[dict(low=249, high=249.1, weight=1), dict(low=249.9, high=250, weight=1)],
        underflow=0,
        overflow=0,
        total_weight=2,
    )
    distributions = {k: None for k in ("chi_e", "chi_gamma", "photon_energy", "pair_energy")}
    record = dict(
        diagonal_start=0,
        diagonal_end=0,
        luminosity_spectrum=h,
        polarization=None,
        beam_moments=None,
        distributions=distributions,
    )
    diag = dict(
        schema_version="feebl.collider.diagnostics/1",
        frames=[dict(record, id=0)],
        total=record,
        nominal_polarization=None,
        unavailable_reasons={
            k: "synthetic fixture omitted" for k in (*distributions, "nominal_polarization")
        },
    )
    (root / "gui_diagnostics.json").write_text(json.dumps(diag))
    arts = {}
    for beam in ("beam1", "beam2"):
        path = root / (beam + ".0")
        path.write_text("125 .2 -.3 1 .04 .05\n")
        arts[beam] = finalizer.describe_frame(root, path.name)
    man = copy.deepcopy(MANIFEST)
    man.update(
        schema_version=finalizer.SCHEMA,
        units=dict(energy="GeV", position="um", angle="urad", time="s"),
        coordinates=dict(frame="legacy_collider_dump"),
        frames=[dict(id=0, diagonal=0, elapsed_s=0, **arts)],
        source=dict(commit="original-native-source", dirty=False),
        binary=dict(sha256="a" * 64),
        backend_actual_receipt="original completed native run",
        analysis=ref(root / "collider_analysis.json"),
        diagnostics=ref(root / "gui_diagnostics.json"),
    )
    (root / "gui_manifest.json").write_text(json.dumps(man))
    return man


class DiagnosticAnalysisBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.man = fixture(self.root)

    def bind(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        return finalizer.verify(self.root, self.man)

    def test_binding_authenticates_all_read_sources_and_preserves_native_bytes(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        identity = copy.deepcopy(
            {k: self.man[k] for k in ("source", "binary", "backend_actual_receipt", "frames")}
        )
        self.assertEqual(self.bind()["status"], "PASS_FOR_STATED_TARGET")
        self.assertEqual({k: self.man[k] for k in identity}, identity)
        self.assertTrue(all((self.root / n).read_bytes() == v for n, v in before.items()))
        bound = {r["path"]: r for r in self.man["diagnostic_analysis_sources"]}
        self.assertEqual(
            set(bound), {"out.dat", "pairs.dat", "collider_analysis.json", "gui_diagnostics.json"}
        )
        for key in ("monochromatization", "secondary_kinematics"):
            data = json.loads((self.root / self.man[key]["path"]).read_text())
            for source in data["source_artifacts"].values():
                if source:
                    self.assertEqual(source, bound[source["path"]])
        self.assertEqual(
            set(self.man["diagnostic_analysis_provenance"]["adapter_sha256"]),
            {
                "collider_monochromatization",
                "collider_secondary_kinematics",
                "collider_secondary_positions",
                "collider_event_deflections",
            },
        )

    def test_augment_preserves_run_identity_and_existing_outputs(self):
        before = {
            p.name: p.read_bytes() for p in self.root.iterdir() if p.name != "gui_manifest.json"
        }
        old = copy.deepcopy(self.man)
        finalizer.augment_diagnostic_analysis(self.root)
        man = json.loads((self.root / "gui_manifest.json").read_text())
        for key in old:
            self.assertEqual(man[key], old[key])
        self.assertTrue(all((self.root / n).read_bytes() == v for n, v in before.items()))
        finalizer.augment_diagnostic_analysis(self.root)
        self.assertTrue(all((self.root / n).read_bytes() == v for n, v in before.items()))

    def test_missing_native_summary_rejected(self):
        (self.root / "out.dat").unlink()
        with self.assertRaises(ValueError):
            self.bind()

    def test_missing_bound_source_rejected(self):
        self.bind()
        (self.root / "pairs.dat").unlink()
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_tampered_native_source_rejected(self):
        self.bind()
        with (self.root / "out.dat").open("a") as stream:
            stream.write("changed\n")
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_predeclared_source_hash_mismatch_rejected_before_augmentation(self):
        self.man["analysis"]["sha256"] = "0" * 64
        (self.root / "gui_manifest.json").write_text(json.dumps(self.man))
        with self.assertRaises(ValueError):
            finalizer.augment_diagnostic_analysis(self.root)
        self.assertFalse((self.root / "collider_monochromatization.json").exists())

    def test_sidecar_source_must_be_bound_by_manifest(self):
        self.bind()
        self.man["diagnostic_analysis_sources"] = [
            r for r in self.man["diagnostic_analysis_sources"] if r["path"] != "pairs.dat"
        ]
        with self.assertRaisesRegex(ValueError, "not bound"):
            finalizer.verify(self.root, self.man)

    def test_sidecar_hash_and_semantic_corruption_rejected(self):
        self.bind()
        path = self.root / self.man["monochromatization"]["path"]
        data = json.loads(path.read_text())
        data["ideal"]["value"] = 99
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            finalizer.verify(self.root, self.man)
        self.man["monochromatization"] = ref(path)
        with self.assertRaisesRegex(ValueError, "formula/admission"):
            finalizer.verify(self.root, self.man)

    def test_empty_native_pair_source_is_valid_but_empty_sidecars_are_not(self):
        self.man = fixture(self.root, empty_pairs=True)
        self.bind()
        pair_ref = next(
            r for r in self.man["diagnostic_analysis_sources"] if r["path"] == "pairs.dat"
        )
        self.assertEqual(pair_ref["bytes"], 0)
        for key in ("monochromatization", "secondary_kinematics"):
            saved = self.man[key]
            path = self.root / saved["path"]
            content = path.read_bytes()
            path.write_bytes(b"")
            self.man[key] = ref(path)
            with self.assertRaises(ValueError):
                finalizer.verify(self.root, self.man)
            # Even duplicating the ref as a permissive raw source cannot make empty JSON valid.
            self.man["diagnostic_analysis_sources"].append(self.man[key])
            with self.assertRaises(ValueError):
                finalizer.verify(self.root, self.man)
            self.man["diagnostic_analysis_sources"].pop()
            path.write_bytes(content)
            self.man[key] = saved

    def test_native_frame_identity_and_weight_closure_rejected(self):
        path = self.root / "gui_diagnostics.json"
        original = path.read_text()
        for mutation in ("identity", "negative_flow", "closure"):
            data = json.loads(original)
            if mutation == "identity":
                data["frames"][0]["id"] = 5
            elif mutation == "negative_flow":
                data["frames"][0]["luminosity_spectrum"]["underflow"] = -1
            else:
                data["frames"][0]["luminosity_spectrum"]["total_weight"] = 3
            path.write_text(json.dumps(data))
            self.man["diagnostics"] = ref(path)
            with self.assertRaises(ValueError):
                finalizer.bind_diagnostic_analysis(self.root, self.man)

    def test_valid_flow_is_explicitly_unavailable_for_frame_width(self):
        path = self.root / "gui_diagnostics.json"
        data = json.loads(path.read_text())
        for record in [*data["frames"], data["total"]]:
            record["luminosity_spectrum"].update(underflow=1, total_weight=3)
        path.write_text(json.dumps(data))
        self.man["diagnostics"] = ref(path)
        self.bind()
        mono = json.loads((self.root / self.man["monochromatization"]["path"]).read_text())
        self.assertFalse(mono["measured"]["frames"][0]["available"])
        self.assertIn("no exported energy coordinates", mono["measured"]["frames"][0]["reason"])

    def test_coupled_metadata_mutation_cannot_replace_actual_run_inputs(self):
        import collider_monochromatization as mono

        self.bind()
        path = self.root / self.man["monochromatization"]["path"]
        data = json.loads(path.read_text())
        data["inputs"]["sigma_delta"] = [v * 2 for v in data["inputs"]["sigma_delta"]]
        data["inputs"]["requested_espread"] = [v * 2 for v in data["inputs"]["requested_espread"]]
        data["native_manifest_parameters"]["espread"] = [
            v * 2 for v in data["native_manifest_parameters"]["espread"]
        ]
        data["ideal"].update(mono.ideal_reference(data["inputs"]))
        data["measured"]["reference_sigma_W_GeV"] *= 2
        for record in [data["measured"]["total"], *data["measured"]["frames"]]:
            if record["available"]:
                record["value"] *= 2
        mono.validate(data)  # Internally consistent; actual run metadata must still reject it.
        path.write_text(json.dumps(data))
        self.man["monochromatization"] = ref(path)
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_factor_source_roles_must_match_actual_analysis_and_diagnostics(self):
        self.bind()
        path = self.root / self.man["monochromatization"]["path"]
        original = path.read_text()
        for key in ("analysis", "diagnostics"):
            data = json.loads(original)
            data["source_artifacts"][key] = data["source_artifacts"]["native_summary"]
            path.write_text(json.dumps(data))
            self.man["monochromatization"] = ref(path)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    finalizer.verify(self.root, self.man)


def imported_event_fixture(root):
    # Synthetic integration fixture only: four explicitly identified records
    # already admitted in one saved frame. Reuse the adapter's independent
    # fixture builder, then supply complete finalizer metadata/artifacts.
    import importlib.util

    source = Path(__file__).resolve().parents[1] / "examples/gui_fields/test_event_deflections.py"
    spec = importlib.util.spec_from_file_location("event_binding_fixture", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    data, identity, records, input_ref, raw, display, summary = helper.fixture()
    man = fixture(root)
    data = b"\n".join(b"0 " + line.split(b" ", 1)[1] for line in data.splitlines()) + b"\n"
    (root / "event.ini").write_bytes(data)
    identity["source_artifact"] = ref(root / "event.ini")
    for row in raw["initial_records"]:
        row[2] = row[3] = 0
    raw["frames"] = [dict(id=0, diagonal=0, records=copy.deepcopy(raw["final"]["records"]))]
    (root / "event_identity.json").write_text(json.dumps(identity))
    (root / "gui_imported_trajectories.json").write_text(json.dumps(raw))
    initial = {r[0]: r for r in raw["initial_records"]}
    lines = []
    for row in raw["final"]["records"]:
        lines.append(" ".join(map(str, [initial[row[0]][4], *row[1:4], 0, 0, 0, -99, -1, -1])))
    (root / "pairs.dat").write_text("\n".join(lines) + "\n")
    (root / "out.dat").write_text(SUMMARY + "\n" + summary + "n_pairs = 4\n")
    man["imported_trajectories_file"] = "gui_imported_trajectories.json"
    (root / "gui_manifest.json").write_text(json.dumps(man))
    return man


class EventDeflectionBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.man = imported_event_fixture(self.root)

    def bind(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        return finalizer.verify(self.root, self.man)

    def test_real_sources_are_bound_and_native_identity_is_preserved(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        metadata = copy.deepcopy(
            {k: self.man[k] for k in ("source", "binary", "backend_actual_receipt", "frames")}
        )
        self.assertEqual(self.bind()["status"], "PASS_FOR_STATED_TARGET")
        sidecar = json.loads((self.root / "collider_event_deflections.json").read_text())
        self.assertTrue(sidecar["available"])
        self.assertEqual(len(sidecar["plots"]), 4)
        self.assertEqual(sidecar["total"]["nrecords"], 4)
        bound = {r["path"]: r for r in self.man["diagnostic_analysis_sources"]}
        self.assertTrue(
            {"event.ini", "event_identity.json", "gui_imported_trajectories.json", "out.dat"}
            <= set(bound)
        )
        for r in sidecar["source_artifacts"].values():
            self.assertEqual(r, bound[r["path"]])
        self.assertTrue(all((self.root / k).read_bytes() == v for k, v in before.items()))
        self.assertEqual({k: self.man[k] for k in metadata}, metadata)

    def test_each_bound_event_source_tampering_is_rejected(self):
        self.bind()
        for name in (
            "event.ini",
            "event_identity.json",
            "gui_imported_trajectories.json",
            "out.dat",
        ):
            p = self.root / name
            old = p.read_bytes()
            p.write_bytes(old + b" ")
            with self.subTest(name=name), self.assertRaises(ValueError):
                finalizer.verify(self.root, self.man)
            p.write_bytes(old)

    def test_missing_identity_or_native_trajectory_is_rejected(self):
        self.bind()
        for name in ("event.ini", "event_identity.json", "gui_imported_trajectories.json"):
            p = self.root / name
            old = p.read_bytes()
            p.unlink()
            with self.subTest(name=name), self.assertRaises(ValueError):
                finalizer.verify(self.root, self.man)
            p.write_bytes(old)

    def test_rehashed_histogram_moment_tamper_rejected_by_native_rederivation(self):
        import collider_event_deflections as events

        self.bind()
        p = self.root / "collider_event_deflections.json"
        data = json.loads(p.read_text())
        stats = data["total"]["histograms"]["electron_delta_theta"]["statistics"]
        stats["mean"] = (2 * stats["min"] + stats["max"]) / 3
        events.validate(data)  # Internally plausible moments are insufficient.
        p.write_text(json.dumps(data))
        self.man["event_deflections"] = ref(p)
        with self.assertRaisesRegex(ValueError, "does not match native trajectories"):
            finalizer.verify(self.root, self.man)

    def test_unbound_native_reference_rejected(self):
        self.bind()
        self.man["diagnostic_analysis_sources"] = [
            r
            for r in self.man["diagnostic_analysis_sources"]
            if r["path"] != "gui_imported_trajectories.json"
        ]
        with self.assertRaisesRegex(ValueError, "not bound"):
            finalizer.verify(self.root, self.man)

    def test_wrong_event_source_role_rejected_even_when_hash_is_bound(self):
        self.bind()
        p = self.root / "collider_event_deflections.json"
        data = json.loads(p.read_text())
        data["source_artifacts"]["identity"] = data["source_artifacts"]["summary"]
        p.write_text(json.dumps(data))
        self.man["event_deflections"] = ref(p)
        with self.assertRaisesRegex(ValueError, "does not match native trajectories"):
            finalizer.verify(self.root, self.man)

    def test_missing_native_declaration_rejected(self):
        self.man.pop("imported_trajectories_file")
        with self.assertRaises(ValueError):
            self.bind()

    def test_alternate_native_declaration_rejected(self):
        self.man["imported_trajectories_file"] = "other.json"
        with self.assertRaises(ValueError):
            self.bind()

    def test_added_identity_cannot_silently_change_bound_unavailable_result(self):
        # Absence is intentional and explicit. Adding an unbound identity later
        # cannot make the existing unavailable sidecar appear authenticated.
        (self.root / "event_identity.json").unlink()
        self.bind()
        unavailable = json.loads((self.root / "collider_event_deflections.json").read_text())
        self.assertFalse(unavailable["available"])
        (self.root / "event_identity.json").write_text("{}")
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_unavailable_identity_still_binds_declared_raw_bytes(self):
        (self.root / "event_identity.json").unlink()
        self.bind()
        bound = {r["path"]: r for r in self.man["diagnostic_analysis_sources"]}
        self.assertIn("gui_imported_trajectories.json", bound)
        p = self.root / "gui_imported_trajectories.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)

    def test_deterministic_rebind_keeps_all_deflection_bytes_identical(self):
        self.bind()
        before = (self.root / "collider_event_deflections.json").read_bytes()
        self.bind()
        self.assertEqual((self.root / "collider_event_deflections.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
