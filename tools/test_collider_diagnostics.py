import copy
import unittest
from collider_diagnostics import validate, MOMENTS


def hist(weight):
    return dict(
        quantity="W",
        unit="GeV",
        weight_unit="m^-2 per BX",
        weighting="native collision pair luminosity",
        normalization="bin-integrated",
        bins=[dict(low=0, high=1, weight=weight)],
        underflow=0,
        overflow=0,
        total_weight=weight,
    )


def record(weight, pol):
    return dict(
        luminosity_spectrum=hist(weight),
        polarization=dict(
            weighting="native collision pair luminosity", weight_sum=weight, beam1=pol, beam2=0.3
        ),
        beam_moments=None,
        distributions=dict(chi_e=None, chi_gamma=None, photon_energy=None, pair_energy=None),
    )


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.frames = [dict(id=0, diagonal=0), dict(id=1, diagonal=2)]
        self.d = dict(
            schema_version="feebl.collider.diagnostics/1",
            nominal_polarization=dict(
                convention="stored spin / reference direction",
                beam1=[0, 0, 0.8],
                beam2=[0, 0, -0.3],
                reference_helicity=[0.8, 0.3],
            ),
            unavailable_reasons=dict(
                chi_e="not recorded",
                chi_gamma="coherent mode disabled",
                photon_energy="disabled",
                pair_energy="disabled",
            ),
            frames=[
                dict(record(1, 0.8), id=0, diagonal_start=0, diagonal_end=0),
                dict(record(3, 0.4), id=1, diagonal_start=1, diagonal_end=2),
            ],
            total=dict(record(4, 0.5), diagonal_start=0, diagonal_end=2),
        )

    def reject(self, mutate):
        d = copy.deepcopy(self.d)
        mutate(d)
        with self.assertRaises((ValueError, KeyError)):
            validate(d, self.frames)

    def test_valid_weighted_closure(self):
        self.assertEqual(validate(self.d, self.frames)["frame_total_closure"], "PASS")

    def test_nonfinite(self):
        self.reject(
            lambda d: d["total"]["luminosity_spectrum"]["bins"][0].update(weight=float("nan"))
        )

    def test_histogram_sum(self):
        self.reject(lambda d: d["total"]["luminosity_spectrum"].update(total_weight=5))

    def test_interval_gap(self):
        self.reject(lambda d: d["frames"][1].update(diagonal_start=2))

    def test_wrong_snapshot_alignment(self):
        self.reject(lambda d: d["frames"][1].update(diagonal_end=3))

    def test_frame_total_weight(self):
        self.reject(lambda d: d["total"].update(luminosity_spectrum=hist(5)))

    def test_polarization_unweighted_average(self):
        self.reject(lambda d: d["total"]["polarization"].update(beam1=0.6))

    def test_missing_reason(self):
        self.reject(lambda d: d["unavailable_reasons"].pop("chi_gamma"))

    def test_missing_distribution_availability(self):
        self.reject(lambda d: d["frames"][0]["distributions"].pop("chi_e"))

    def test_changed_edges(self):
        self.reject(lambda d: d["frames"][0]["luminosity_spectrum"]["bins"][0].update(high=2))

    def test_negative_bin(self):
        self.reject(lambda d: d["frames"][0]["luminosity_spectrum"]["bins"][0].update(weight=-1))

    def test_polarization_range(self):
        self.reject(lambda d: d["total"]["polarization"].update(beam1=1.1))

    def test_absent_nominal_with_reason(self):
        self.d["nominal_polarization"] = None
        self.d["unavailable_reasons"]["nominal_polarization"] = "spin storage disabled"
        validate(self.d, self.frames)

    def test_oversize_histogram(self):
        self.reject(
            lambda d: d["total"]["luminosity_spectrum"].update(
                bins=[dict(low=i, high=i + 1, weight=0) for i in range(2049)]
            )
        )

    def add_moments(self):
        def m(w, mean, rms):
            return dict(
                weighting="native collision pair luminosity",
                **{
                    b: dict(weight_sum=w, **{k: dict(mean=mean, rms=rms) for k in MOMENTS})
                    for b in ("beam1", "beam2")
                },
            )

        self.d["frames"][0]["beam_moments"] = m(1, 0, 0)
        self.d["frames"][1]["beam_moments"] = m(3, 2, 0)
        self.d["total"]["beam_moments"] = m(4, 1.5, 3**0.5 / 2)

    def test_parallel_moment_combination(self):
        self.add_moments()
        validate(self.d, self.frames)

    def test_wrong_total_moment_mean(self):
        self.add_moments()
        self.reject(lambda d: d["total"]["beam_moments"]["beam1"]["energy_GeV"].update(mean=1))

    def test_averaging_frame_RMS_is_wrong(self):
        self.add_moments()
        self.reject(lambda d: d["total"]["beam_moments"]["beam1"]["energy_GeV"].update(rms=0))

    def test_zero_angular_variance_rejected(self):
        self.add_moments()
        for r in self.d["frames"] + [self.d["total"]]:
            r["beam_moments"]["beam1"]["vx"] = dict(mean=0, rms=1e-6)
        self.reject(lambda d: d["total"]["beam_moments"]["beam1"]["vx"].update(rms=0))

    def test_nonrepresentable_narrow_energy_and_unequal_intervals(self):
        self.add_moments()
        low, high = 45.6 - 1e-8, 45.6 + 1e-8
        mean = low + .75 * (high - low)
        rms = 3**.5 * (high - low) / 4
        for beam in ("beam1", "beam2"):
            self.d["frames"][0]["beam_moments"][beam]["energy_GeV"] = dict(mean=low, rms=0)
            self.d["frames"][1]["beam_moments"][beam]["energy_GeV"] = dict(mean=high, rms=0)
            self.d["total"]["beam_moments"][beam]["energy_GeV"] = dict(mean=mean, rms=rms)
        validate(self.d, self.frames)
        self.reject(lambda d: d["total"]["beam_moments"]["beam1"]["energy_GeV"].update(rms=0))

    def add_count_histogram(self):
        for r, weight in zip(self.d["frames"] + [self.d["total"]], [1, 3, 4]):
            r["distributions"]["chi_e"] = dict(hist(weight), samples=weight)

    def test_count_histogram(self):
        self.add_count_histogram()
        validate(self.d, self.frames)

    def test_wrong_count_rejected(self):
        self.add_count_histogram()
        self.reject(lambda d: d["total"]["distributions"]["chi_e"].update(samples=5))


if __name__ == "__main__":
    unittest.main()
