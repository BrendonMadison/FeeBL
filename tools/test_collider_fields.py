"""Failure and artifact-binding checks for the optional native field sidecar."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import collider_fields as c


def fixture():
    sources = [
        dict(
            id="beam1",
            kind="beam_beam",
            label="Source beam 1",
            quantity="native transverse electric-equivalent field magnitude",
            unit="GV/nm",
            source_beam=1,
            target_beam=2,
        ),
        dict(
            id="beam2",
            kind="beam_beam",
            label="Source beam 2",
            quantity="native transverse electric-equivalent field magnitude",
            unit="GV/nm",
            source_beam=2,
            target_beam=1,
        ),
    ]
    frames = [
        dict(
            id=i,
            diagonal=i,
            elapsed_s=i * 1e-12,
            available=True,
            reason=None,
            encounter=dict(
                beam1_slice=0,
                beam2_slice=i,
                substep_index=1,
                encounter_counter=2 + i,
                nominal_z_nm=10.0 * i,
                phase="post-solve/pre-kick",
            ),
            maps=dict(beam1=[0, 1.0, 2.0, 3.0], beam2=[4.0, 3.0, 2.0, 1.0]),
        )
        for i in range(2)
    ]
    return dict(
        schema_version=c.SCHEMA,
        plane=dict(axes=["x", "y"], coordinate_unit="nm", frame="native simulated frame"),
        geometry=dict(
            nx=2,
            ny=2,
            x_min_nm=-1.0,
            x_max_nm=1.0,
            y_min_nm=-2.0,
            y_max_nm=2.0,
            native_domain_nm=[-1.5, 1.5, -2.5, 2.5],
            storage_order="row-major-y-x",
            sampling="uniform CIC gather points; bounds are sample centres",
        ),
        sources=sources,
        conventions=dict(
            selection="central active slice encounter, final substep, pre-kick",
            scope="single source slice; not whole-bunch or diagonal-averaged field",
            charge_sign=-1,
            scale_step=[1, 1],
            n_z=2,
            n_t=2,
        ),
        frames=frames,
    )


class FieldsTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.frames = [
            dict(id=f["id"], diagonal=f["diagonal"], elapsed_s=f["elapsed_s"])
            for f in self.data["frames"]
        ]

    def reject(self, change):
        d = copy.deepcopy(self.data)
        change(d)
        with self.assertRaises(ValueError):
            c.validate(d, self.frames)

    def test_valid(self):
        self.assertEqual(c.validate(self.data, self.frames)["available_frames"], 2)

    def test_zero_field_valid(self):
        for f in self.data["frames"]:
            f["maps"] = {k: [0] * 4 for k in f["maps"]}
        self.assertEqual(
            c.validate(self.data, self.frames)["source_maxima"], dict(beam1=0, beam2=0)
        )

    def test_unavailable_valid(self):
        self.data["frames"][1].update(
            available=False, reason="Unsupported backend", maps=None, encounter=None
        )
        self.assertEqual(c.validate(self.data, self.frames)["available_frames"], 1)

    def test_all_unavailable_nonstandard_normalization(self):
        self.data["conventions"]["scale_step"] = [2, 3]
        for f in self.data["frames"]:
            f.update(available=False, reason="Unadmitted scale step", maps=None, encounter=None)
        self.assertEqual(c.validate(self.data, self.frames)["available_frames"], 0)

    def test_zero_source_unavailable_contract(self):
        self.data["conventions"]["charge_sign"] = 0
        for f in self.data["frames"]:
            f.update(
                available=False,
                reason="Zero-source control has no admitted field maps",
                maps=None,
                encounter=None,
            )
        self.assertEqual(c.validate(self.data, self.frames)["available_frames"], 0)

    def test_zero_source_cannot_admit_nonzero_field_maps(self):
        self.reject(lambda d: d["conventions"].update(charge_sign=0))

    def test_external_layer_slot(self):
        self.data["sources"].append(
            dict(
                id="external",
                kind="external",
                label="Synthetic contract fixture",
                quantity="electric field magnitude",
                unit="V/m",
                source_beam=None,
                target_beam=None,
            )
        )
        for f in self.data["frames"]:
            f["maps"]["external"] = [0] * 4
        self.assertIn("external", c.validate(self.data, self.frames)["source_maxima"])

    def test_schema(self):
        self.reject(lambda d: d.update(schema_version="other"))

    def test_axes(self):
        self.reject(lambda d: d["plane"].update(axes=["x", "z"]))

    def test_unit(self):
        self.reject(lambda d: d["plane"].update(coordinate_unit="um"))

    def test_frame(self):
        self.reject(lambda d: d["plane"].update(frame="lab"))

    def test_grid_count(self):
        self.reject(lambda d: d["geometry"].update(nx=65))

    def test_grid_bool(self):
        self.reject(lambda d: d["geometry"].update(nx=True))

    def test_grid_nan(self):
        self.reject(lambda d: d["geometry"].update(x_min_nm=float("nan")))

    def test_grid_order(self):
        self.reject(lambda d: d["geometry"].update(y_max_nm=-3))

    def test_missing_domain(self):
        self.reject(lambda d: d["geometry"].pop("native_domain_nm"))

    def test_domain_overrun(self):
        self.reject(lambda d: d["geometry"].update(native_domain_nm=[-0.5, 0.5, -2.5, 2.5]))

    def test_domain_nonfinite(self):
        self.reject(
            lambda d: d["geometry"].update(native_domain_nm=[-1.5, float("inf"), -2.5, 2.5])
        )

    def test_storage(self):
        self.reject(lambda d: d["geometry"].update(storage_order="row-major-x-y"))

    def test_sampling(self):
        self.reject(lambda d: d["geometry"].update(sampling="cell edges"))

    def test_source_duplicate(self):
        self.reject(lambda d: d["sources"].append(copy.deepcopy(d["sources"][0])))

    def test_source_unknown(self):
        self.reject(lambda d: d["sources"][0].update(kind="inferred"))

    def test_source_units(self):
        self.reject(lambda d: d["sources"][0].update(unit="V/m"))

    def test_beam_identity(self):
        self.reject(lambda d: d["sources"][0].update(target_beam=1))

    def test_source_quantity(self):
        self.reject(lambda d: d["sources"][0].update(quantity="density"))

    def test_bad_id(self):
        self.reject(lambda d: d["sources"][0].update(id="../field"))

    def test_frame_count(self):
        self.reject(lambda d: d["frames"].pop())

    def test_frame_id(self):
        self.reject(lambda d: d["frames"][1].update(id=0))

    def test_diagonal(self):
        self.reject(lambda d: d["frames"][1].update(diagonal=2))

    def test_time(self):
        self.reject(lambda d: d["frames"][1].update(elapsed_s=2e-12))

    def test_availability_type(self):
        self.reject(lambda d: d["frames"][1].update(available=1))

    def test_unavailable_stale_map(self):
        self.reject(lambda d: d["frames"][1].update(available=False, reason="Unsupported"))

    def test_unavailable_missing_reason(self):
        self.reject(lambda d: d["frames"][1].update(available=False, maps=None, encounter=None))

    def test_available_reason(self):
        self.reject(lambda d: d["frames"][1].update(reason="Unsupported"))

    def test_slice(self):
        self.reject(lambda d: d["frames"][1]["encounter"].update(beam1_slice=1))

    def test_phase(self):
        self.reject(lambda d: d["frames"][1]["encounter"].update(phase="after move"))

    def test_substep(self):
        self.reject(lambda d: d["frames"][1]["encounter"].update(substep_index=0))

    def test_counter(self):
        self.reject(lambda d: d["frames"][1]["encounter"].update(encounter_counter=-1))

    def test_nominal_z(self):
        self.reject(lambda d: d["frames"][1]["encounter"].update(nominal_z_nm=float("inf")))

    def test_scale(self):
        self.reject(lambda d: d["conventions"].update(scale_step=[1, 2]))

    def test_charge(self):
        self.reject(lambda d: d["conventions"].update(charge_sign=1))

    def test_maps_missing_source(self):
        self.reject(lambda d: d["frames"][0]["maps"].pop("beam2"))

    def test_maps_extra_source(self):
        self.reject(lambda d: d["frames"][0]["maps"].update(fake=[0] * 4))

    def test_maps_truncated(self):
        self.reject(lambda d: d["frames"][0]["maps"]["beam1"].pop())

    def test_maps_negative(self):
        self.reject(lambda d: d["frames"][0]["maps"]["beam1"].__setitem__(0, -1))

    def test_maps_nan(self):
        self.reject(lambda d: d["frames"][0]["maps"]["beam1"].__setitem__(0, float("nan")))

    def test_maps_bool(self):
        self.reject(lambda d: d["frames"][0]["maps"]["beam1"].__setitem__(0, True))


class ArtifactTests(unittest.TestCase):
    def test_field_hash_and_schema_bound(self):
        from finalize_gui_dataset import verify, describe_frame, digest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            d = fixture()
            frames = []
            for f in d["frames"]:
                arts = {}
                for b in (1, 2):
                    name = f"gui_b{b}.{f['id']}"
                    (root / name).write_text("125 0 0 0 0 0\n")
                    arts[f"beam{b}"] = describe_frame(root, name)
                frames.append(
                    dict(id=f["id"], diagonal=f["diagonal"], elapsed_s=f["elapsed_s"], **arts)
                )
            p = root / "gui_fields.json"
            p.write_text(json.dumps(d))
            m = dict(
                schema_version="feebl.collider/2",
                units=dict(energy="GeV", position="um", angle="urad", time="s"),
                coordinates=dict(frame="legacy_collider_dump"),
                frames=frames,
            )
            verify(root, m)  # Legacy datasets remain valid.
            m["fields"] = dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
            verify(root, m)
            p.write_text(p.read_text().replace("post-solve/pre-kick", "post-solve/postkick"))
            with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                verify(root, m)
            m["fields"] = dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
            with self.assertRaisesRegex(ValueError, "encounter phase"):
                verify(root, m)


if __name__ == "__main__":
    unittest.main()
