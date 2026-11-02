"""Verify stored luminosity normalization and scalar EPA checks without FeeBL runs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    """Return a content hash used by the portable evidence manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio_estimate(vectors):
    """Return ratios of means and their complete covariance and jackknife."""
    mean = vectors.mean(axis=0)
    centered = vectors-mean
    covariance = centered.T@centered/(len(vectors)*(len(vectors)-1))
    jacobian = np.c_[np.eye(3)/mean[3], -mean[:3]/mean[3]**2]
    omitted = (vectors.sum(axis=0)-vectors)/(len(vectors)-1)
    ratios = omitted[:, :3]/omitted[:, 3, None]
    differences = ratios-ratios.mean(axis=0)
    return (mean[:3]/mean[3], jacobian@covariance@jacobian.T,
            (len(vectors)-1)/len(vectors)*(differences.T@differences))


def check_ensemble():
    """Recompute both model results from the packaged whole-collision vectors."""
    results = json.loads((HERE/"ALL_PROCESS_LUMINOSITY_COMPARISON.json").read_text())
    for model in (1, 2):
        expected = results["models"][str(model)]
        path = HERE.parent/f"qed_figure_campaign/reduction/ilc_model{model}_joint_yield_luminosity.npz"
        if digest(path) != expected["source_sha256"]:
            raise ValueError("Changed whole-collision input: "+path.name)
        with np.load(path, allow_pickle=False) as stored:
            column = list(stored["columns"]).index("lumi_ee")
            native = stored["vectors"]
            # Native luminosity is m^-2; the ratio uses inverse millibarns.
            vectors = np.c_[native[:, :3], native[:, column]*1e-31]
        measured = ratio_estimate(vectors)
        for values, key in zip(measured, ("mean_mb", "covariance_of_mean_mb2", "jackknife_covariance_mb2")):
            np.testing.assert_allclose(values, expected["common_ee"][key], rtol=1e-11, atol=1e-12)


def check_scalar_integrals():
    """Check an analytic EPA flux integral; this is not a collider calculation."""
    results = json.loads((HERE/"EPA_SCALAR_CHECKS.json").read_text())
    mass_GeV = .51099906e-3
    alpha_ratio = (1/137.035999084/math.pi)/.00232461
    born_ratio = (math.pi*(2.81794092e-15)**2/2)/1.23088e-29
    np.testing.assert_allclose(born_ratio, results["born_new_over_old"], rtol=1e-13)
    for row in results["fixed_fraction_examples"]:
        fraction = row["x"]
        splitting = (1+(1-fraction)**2)/2
        lower = mass_GeV**2*fraction**2/(1-fraction)
        upper = (250*fraction)**2
        span = math.log(upper/lower)
        analytic = 1-(1-fraction)/splitting*(1-lower/upper)/span
        numerical = quad(lambda t: 1-(1-fraction)/splitting*math.exp(-t),
                         0, span, epsabs=1e-12, epsrel=1e-12)[0]/span
        np.testing.assert_allclose([analytic, numerical],
            [row["finite_mass_integrated_one_photon_ratio"], row["quadrature_ratio"]],
            rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose((alpha_ratio*analytic)**2,
                                  row["two_virtual_photon_flux_ratio"], rtol=1e-12)


def main():
    """Check immutable inputs and numerical results, writing no new artifacts."""
    provenance = json.loads((HERE/"provenance.json").read_text())
    for record in provenance["files"]:
        path = ROOT/record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError("Evidence file differs from its manifest: "+record["path"])
    check_ensemble()
    check_scalar_integrals()
    print("PASS: two ensemble ratios, full covariances, jackknives and four scalar EPA integrals.")
    print("No collision, native build or comparison-significance claim is part of this check.")


if __name__ == "__main__":
    main()
