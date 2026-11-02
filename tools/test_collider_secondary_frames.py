"""Current-state native secondary snapshots and authenticated adapter binding."""

import bisect
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collider_secondary_frames as frames
import collider_secondary_kinematics as adapter
import finalize_gui_dataset as finalizer
from test_diagnostic_analysis_binding import fixture, ref

FLAGS = dict(
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


def make_axes():
    axes = {}
    for key, end in [("theta", math.pi), ("theta_folded", math.pi / 2), ("pT", 1e6)]:
        axes[frames.AXIS_KEYS[key]] = [
            10 ** (-12 + (math.log10(end) + 12) * i / 128) for i in range(129)
        ]
        axes[frames.MARGINAL_KEYS[key]] = (
            [0] + [10 ** (-12 + 18 * i / 159) for i in range(160)]
            if key == "pT"
            else [end * i / 160 for i in range(161)]
        )
    return axes


def index(edges, value):
    if value < edges[0]:
        return -1
    if value > edges[-1]:
        return len(edges) - 1
    return min(len(edges) - 2, bisect.bisect_right(edges, value) - 1)


def joint(theta, pt, x, y):
    counts = [0] * 16384
    ex = {k: 0 for k in frames.EXCLUSIONS}
    for a, p in zip(theta, pt):
        ex["theta_nonpositive"] += int(a <= 0)
        ex["pT_nonpositive"] += int(p <= 0)
        ex["both_nonpositive"] += int(a <= 0 and p <= 0)
        if a <= 0 or p <= 0:
            ex["nonpositive_records"] += 1
            continue
        ix, iy = index(x, a), index(y, p)
        if min(ix, iy) < 0:
            ex["underflow_records"] += 1
        elif max(ix, iy) >= 128:
            ex["overflow_records"] += 1
        else:
            counts[iy * 128 + ix] += 1
    return dict(counts=counts, samples=len(theta), in_range_records=sum(counts), excluded=ex)


def marginal(values, edges):
    counts = [0] * 160
    under = over = 0
    for v in values:
        i = index(edges, v)
        if i < 0:
            under += 1
        elif i >= 160:
            over += 1
        else:
            counts[i] += 1
    statistics = None
    if values:
        mean = math.fsum(values) / len(values)
        density = [c / (b - a) for c, a, b in zip(counts, edges, edges[1:])]
        i = density.index(max(density))
        statistics = dict(
            mean=mean,
            variance=math.fsum((v - mean) ** 2 for v in values) / len(values),
            min=min(values),
            max=max(values),
            mode=(edges[i] + edges[i + 1]) / 2 if sum(counts) else None,
            mode_method=frames.MODE if sum(counts) else "unavailable: no in-range marginal records",
        )
    return dict(
        counts=counts,
        samples=len(values),
        underflow=under,
        overflow=over,
        total_weight=len(values),
        statistics=statistics,
    )


def native_frame(i, values, axes, available=True):
    # Each tuple is an independently specified (theta, pT, signed-charge marker).
    if not available:
        return dict(
            id=i,
            diagonal=i * 2,
            available=False,
            reason="Unsupported test mode",
            nrecords=0,
            joint=None,
            joint_folded=None,
            marginals=None,
            selection=None,
        )
    theta = [x[0] for x in values]
    pt = [x[1] for x in values]
    folded = [min(x, math.pi - x) for x in theta]
    n = len(values)
    return dict(
        id=i,
        diagonal=i * 2,
        available=True,
        reason=None,
        nrecords=n,
        joint=joint(theta, pt, axes["theta_edges_rad"], axes["pT_edges_GeV"]),
        joint_folded=joint(folded, pt, axes["theta_folded_edges_rad"], axes["pT_edges_GeV"]),
        marginals={
            k: marginal(v, axes[frames.MARGINAL_KEYS[k]])
            for k, v in [("theta", theta), ("theta_folded", folded), ("pT", pt)]
        },
        selection=dict(
            process_counts={"0": n} if n else {},
            signed_energy_counts=dict(
                positive_energy=sum(s > 0 for _, _, s in values),
                negative_energy=sum(s < 0 for _, _, s in values),
            ),
            admitted_tracker_count=n,
            authoritative_reserve_count=n,
        ),
    )


def raw_fixture(values=None):
    if values is None:
        values = [[], [(0.2, 1.0, 1), (0.9, 2.0, -1)], [(0.3, 1.5, 1), (1.1, 3.0, -1)]]
    axes = make_axes()
    return dict(
        schema_version=frames.SCHEMA,
        phase=frames.PHASE,
        coordinate_frame="native_pair_tracking",
        weighting=frames.WEIGHTING,
        evaluated_flags=copy.deepcopy(FLAGS),
        axes=axes,
        definitions=dict(
            selection="Authoritative reserve includes completed and escaped records exactly once",
            timing="After the diagonal tracker deactivation",
            normalization="One current stored record, no physical weighting",
            coordinates="Native tracker axes",
            precision="Full native precision, unlike final text serialization",
        ),
        frames=[native_frame(i, v, axes, v is not None) for i, v in enumerate(values)],
    )


def install(root, raw=None, store_pairs=1):
    man = fixture(root)
    raw = raw_fixture() if raw is None else raw
    source = (
        (root / "out.dat").read_text().replace("store_pairs = 1", f"store_pairs = {store_pairs}")
    )
    (root / "out.dat").write_text(source)
    raw["evaluated_flags"]["store_pairs"] = store_pairs
    man["frames"] = []
    for f in raw["frames"]:
        cloud = {}
        for beam in ("beam1", "beam2"):
            name = f'{beam}.{f["id"]}'
            (root / name).write_text("125 .2 -.3 1 .04 .05\n")
            cloud[beam] = finalizer.describe_frame(root, name)
        man["frames"].append(
            dict(id=f["id"], diagonal=f["diagonal"], elapsed_s=f["id"] * 1e-12, **cloud)
        )
    # The synthetic diagnostics are only used to exercise the binding mechanism.
    diag = json.loads((root / "gui_diagnostics.json").read_text())
    diag["frames"] = [
        dict(
            diag["frames"][0],
            id=f["id"],
            diagonal_start=0 if f["id"] == 0 else raw["frames"][f["id"] - 1]["diagonal"] + 1,
            diagonal_end=f["diagonal"],
        )
        for f in raw["frames"]
    ]
    diag["total"]["diagonal_end"] = raw["frames"][-1]["diagonal"]
    h = diag["total"]["luminosity_spectrum"]
    h["total_weight"] *= len(raw["frames"])
    for b in h["bins"]:
        b["weight"] *= len(raw["frames"])
    (root / "gui_diagnostics.json").write_text(json.dumps(diag))
    man["diagnostics"] = ref(root / "gui_diagnostics.json")
    man["secondary_frames_file"] = "gui_secondary_frames.json"
    (root / "gui_secondary_frames.json").write_text(json.dumps(raw, separators=(",", ":")))
    (root / "gui_manifest.json").write_text(json.dumps(man))
    return man, raw


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.man, self.raw = install(self.root)

    def analyze(self):
        return adapter.analyze_run(self.root, manifest=self.man)

    def test_current_maps_change_without_summing_into_final(self):
        data = self.analyze()
        ps = [f["populations"]["pairs"] for f in data["frames"]]
        self.assertTrue(ps[0]["available"])
        self.assertEqual(ps[0]["nrecords"], 0)
        self.assertIsNone(ps[0]["distributions"]["pT"]["statistics"])
        self.assertEqual([p["nrecords"] for p in ps], [0, 2, 2])
        self.assertEqual(
            [ps[i]["distributions"]["pT"]["statistics"]["mean"] for i in (1, 2)], [1.5, 2.25]
        )
        self.assertNotEqual(ps[1]["joint"]["counts"], ps[2]["joint"]["counts"])
        self.assertEqual(data["populations"]["pairs"]["nrecords"], 2)
        self.assertEqual(
            data["populations"]["pairs"]["distributions"]["pT"]["statistics"]["mean"], 1.0
        )
        self.assertFalse(ps[2]["normalization"]["completeness_verified"])

    def test_frames_survive_no_final_file_and_store_disabled(self):
        self.man, self.raw = install(self.root, store_pairs=0)
        (self.root / "pairs.dat").unlink()
        data = self.analyze()
        self.assertTrue(data["frame_availability"])
        self.assertFalse(data["populations"]["pairs"]["available"])
        self.assertEqual(data["frames"][2]["populations"]["pairs"]["nrecords"], 2)

    def test_empty_and_unavailable_are_distinct(self):
        self.man, self.raw = install(self.root, raw_fixture([[], None]))
        data = self.analyze()
        self.assertTrue(data["frames"][0]["available"])
        self.assertFalse(data["frames"][1]["available"])
        self.assertTrue(data["frame_availability"])
        self.assertEqual(
            data["frames"][1]["populations"]["pairs"]["reason"], "Unsupported test mode"
        )

    def test_only_unavailable_frames_still_preserve_source(self):
        self.man, self.raw = install(self.root, raw_fixture([None]))
        data = self.analyze()
        self.assertFalse(data["frame_availability"])
        self.assertIn("frame_snapshots", data["source_artifacts"])

    def test_fixed_geometry_and_all_flow_statistics(self):
        self.man, self.raw = install(self.root, raw_fixture([[(0.0, 0.0, 1)], [(0.3, 2e6, -1)]]))
        data = self.analyze()
        ps = [f["populations"]["pairs"] for f in data["frames"]]
        self.assertEqual(ps[0]["joint"]["excluded"]["both_nonpositive"], 1)
        self.assertEqual(ps[1]["joint"]["excluded"]["overflow_records"], 1)
        stat = ps[1]["distributions"]["pT"]["statistics"]
        self.assertIsNone(stat["mode"])
        self.assertEqual(stat["mean"], 2e6)

    def test_missing_declared_raw_rejected(self):
        (self.root / "gui_secondary_frames.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.analyze()

    def test_undeclared_raw_rejected(self):
        self.man.pop("secondary_frames_file")
        with self.assertRaisesRegex(ValueError, "not declared"):
            self.analyze()

    def test_standalone_obtains_manifest(self):
        self.assertEqual(adapter.analyze_run(self.root)["frames"], self.analyze()["frames"])

    def test_old_dataset_without_frames_remains_valid(self):
        (self.root / "gui_secondary_frames.json").unlink()
        self.man.pop("secondary_frames_file")
        data = self.analyze()
        self.assertFalse(data["frame_availability"])
        self.assertNotIn("frames", data)

    def test_raw_bound_and_symlink(self):
        with patch.object(frames, "MAX_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "byte bound"):
                self.analyze()
        path = self.root / "gui_secondary_frames.json"
        saved = self.root / "saved.json"
        path.rename(saved)
        try:
            path.symlink_to(saved)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows account lacks symbolic-link privilege")
            raise
        with self.assertRaises(ValueError):
            self.analyze()

    def test_flags_precision_and_legacy_missing_bhabha(self):
        raw = copy.deepcopy(self.raw)
        raw["evaluated_flags"]["pair_ecut"] = 0.0005000000237487
        frames.validate_raw(raw, self.man, FLAGS)
        flags = dict(FLAGS, do_bhabhas=None)
        frames.validate_raw(raw, self.man, flags)
        raw["evaluated_flags"]["pair_ratio"] = 0.9
        with self.assertRaises(ValueError):
            frames.validate_raw(raw, self.man, FLAGS)

    def test_native_count_geometry_identity_corruptions(self):
        mutations = [
            lambda r: r["frames"][1]["selection"].__setitem__("admitted_tracker_count", 3),
            lambda r: r["frames"][1]["joint"]["counts"].__setitem__(0, 1),
            lambda r: r["frames"][1]["marginals"]["pT"]["counts"].__setitem__(0, 0.5),
            lambda r: r["axes"]["pT_edges_GeV"].__setitem__(8, 1e-8),
            lambda r: r["frames"][1].__setitem__("diagonal", 3),
            lambda r: r["frames"][1]["selection"]["signed_energy_counts"].__setitem__(
                "positive_energy", 2
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                raw = copy.deepcopy(self.raw)
                mutate(raw)
                with self.assertRaises(ValueError):
                    frames.validate_raw(raw, self.man, FLAGS)

    def test_expanded_timing_weighting_label_and_source_binding(self):
        data = self.analyze()
        mutations = [
            lambda d: d["frames"][1].__setitem__("elapsed_s", 9),
            lambda d: d["frames"][1]["populations"]["pairs"]["normalization"].__setitem__(
                "completeness_verified", True
            ),
            lambda d: d["frames"][1]["populations"]["pairs"].__setitem__(
                "label", "Electron detector spectrum"
            ),
            lambda d: d["frames"][1]["populations"]["pairs"]["distributions"]["pT"][
                "statistics"
            ].__setitem__("mean", 1.6),
            lambda d: d["frame_definition"]["axes"]["pT_edges_GeV"].__setitem__(0, 1e-15),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                corrupted = copy.deepcopy(data)
                mutate(corrupted)
                with self.assertRaises(ValueError):
                    adapter.validate(corrupted, self.man, self.raw)

    def test_compact_writer_and_expanded_readback(self):
        data = self.analyze()
        path = adapter.write_analysis(self.root, data)
        self.assertLess(path.stat().st_size, 600000)
        self.assertEqual(json.loads(path.read_text()), data)


class BindingTests(unittest.TestCase):
    setUp = FrameTests.setUp

    def test_finalizer_frame_authentication_and_native_preservation(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        result = finalizer.verify(self.root, self.man)
        self.assertEqual(result["status"], "PASS_FOR_STATED_TARGET")
        self.assertTrue(all((self.root / n).read_bytes() == v for n, v in before.items()))
        self.assertIn(
            "collider_secondary_frames",
            self.man["diagnostic_analysis_provenance"]["adapter_sha256"],
        )
        refs = {r["path"]: r for r in self.man["diagnostic_analysis_sources"]}
        self.assertEqual(
            refs["gui_secondary_frames.json"], ref(self.root / "gui_secondary_frames.json")
        )

    def test_finalizer_counts_each_physical_source_path_once(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        result = finalizer.verify(self.root, self.man)
        paths = {a["path"]: a for f in self.man["frames"] for a in (f["beam1"], f["beam2"])}
        for k in ("analysis", "diagnostics", "monochromatization", "secondary_kinematics",
                  "event_deflections"):
            paths[self.man[k]["path"]] = self.man[k]
        for r in self.man["diagnostic_analysis_sources"]:
            paths[r["path"]] = r
        self.assertEqual(result["bytes"], sum(r["bytes"] for r in paths.values()))

    def test_finalizer_declared_snapshot_requires_bound_derived(self):
        with self.assertRaisesRegex(ValueError, "must be authenticated"):
            finalizer.verify(self.root, self.man)

    def test_finalizer_coupled_semantic_corruption_rejected(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        path = self.root / self.man["secondary_kinematics"]["path"]
        data = json.loads(path.read_text())
        data["frames"][1]["populations"]["pairs"]["distributions"]["pT"]["statistics"]["mean"] = 1.6
        path.write_text(json.dumps(data))
        self.man["secondary_kinematics"] = ref(path)
        with self.assertRaisesRegex(ValueError, "differs from authenticated"):
            finalizer.verify(self.root, self.man)

    def test_finalizer_missing_raw_and_particle_alias_rejected(self):
        finalizer.bind_diagnostic_analysis(self.root, self.man)
        self.man["analysis_sources"] = [self.man["frames"][0]["beam1"]]
        with self.assertRaisesRegex(ValueError, "aliases a particle"):
            finalizer.verify(self.root, self.man)
        self.man.pop("analysis_sources")
        (self.root / "gui_secondary_frames.json").unlink()
        with self.assertRaises(ValueError):
            finalizer.verify(self.root, self.man)


if __name__ == "__main__":
    unittest.main()
