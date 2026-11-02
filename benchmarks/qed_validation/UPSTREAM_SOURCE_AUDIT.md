# Source audit of the inherited pair implementation

This note supports the pair-model table in the FeeBL paper. It records source
inspection and scalar checks performed on 30 October 2026. It does not assign
the same defects to every GuineaPig++ release or infer the numerical bias of
published collider predictions without their source and input configurations.

## Source identity

The inspected source is `guinea-pig-master.zip`, SHA-256
`35b982c220a779ae3d3b3b429d9d392c0e443424f581d5791f38ea42bd8deab8`.
Its ZIP comment identifies upstream revision
`6702c99f9182eca0e94ff8cb893bab73d454e157` of the
[CERN GuineaPig++ repository](https://gitlab.cern.ch/clic-software/guinea-pig).
The local source-import commit is
`9eeb18572e12e84e4c05b0eb34b924481927fdd5`; this is an audit import identifier,
not an upstream commit. All 14 inspected files were compared byte for byte
between that import and the archive. The revision identity is taken from the
archive; a fresh remote checkout was not authenticated in this inspection.

[UPSTREAM_SOURCE_AUDIT.json](UPSTREAM_SOURCE_AUDIT.json) contains the file
hashes, numbered source excerpts, revision-qualified links and numerical
normalization check. Line numbers below refer to this archived source, not the
current FeeBL tree.

## Findings and scope

| Finding | Reproducible source anchors | Consequence and correction scope |
|---|---|---|
| Born normalization | `src/pairsCPP.cc:158–160`, `:251`; `src/physconst.h:16` | The coefficient is the literal `1.23088e-29 m²`, including the mass-scaled muon analogue. It is 1.319334% below `pi*RE²/2` using the source's own electron radius. FeeBL corrects it in every pair model. |
| Forward-angle inversion | `src/mathematicalTools.h:68–92`; `src/physicalTools.cc:7–25`; `src/physicalTools.h:80–90` | `equal_newton` can stop when the absolute bracket width reaches `1e-6`, before resolving the forward tail. The characteristic bisection endpoint is `1−abs(cos(theta*))≈2^-20` in the small-mass regime; this is a numerical artifact, not a physical angular bound. The replacement sampler requires `pair_model>=1`. |
| Landau–Lifshitz dispatch | `src/guineapigCPP.cc:687`, `:719–744`; `src/gridCPP.h:1110–1122`; `src/gridCPP.cc:1680–1754` | Virtual photons can be prepared while the real-photon switch prevents their collision routine from running. With both real-photon generation and loading disabled, this removes the virtual–virtual channel. FeeBL's independent pair/muon dispatch requires `pair_model>=1`; the virtual-photon population must also be enabled by the input configuration. |
| Spectator bookkeeping | `src/pairsCPP.cc:224–233`; `src/pairsCPP.h:245–254`; `src/resultsCPP.h:242–252`; `src/resultsCPP.cc:346–374` | With `beam_pair=1`, scattered beam leptons intentionally enter the stored population and increment the same counters as the two produced daughters. If these counters are interpreted as produced-daughter yields, the BH and LL expectations are larger by `3/2` and `2`, respectively. These factors concern weighted expectations; finite stored samples fluctuate. FeeBL excludes spectators from daughter counters for `pair_model>=1`; their stored tracks still require an explicit population selection. |

The spectator option itself is intentional. The bookkeeping change separates
two populations whose combined counters are ambiguous when used as
produced-pair yields. Source inspection establishes this mixing; it does not
establish that including spectator tracks was an unintended upstream feature.

The normalization calculation is

```python
from math import pi
reference = pi * (2.81794092e-15)**2 / 2
deficit_percent = 100 * (1 - 1.23088e-29 / reference)
# reference = 1.2473365379587158e-29 m²
# deficit_percent = 1.3193342340189318
```

Thus the paper's rounded value, **1.319% below the reference**, is correct.
It describes this coefficient comparison, not the total change in a collider
background after new kinematics, acceptance cuts and tracking.

## Secondary transport and attribution limits

The archived source also contains the float-based secondary rescaling:
`src/particlesCPP.cc:157–168` updates float velocities, energy and cached inverse
energy; `:360–376` reconstructs a final velocity scale using `E²−m²`.
The float state and velocity norm are visible at
`src/abstractParticle.h:30–39` and `src/particlesCPP.h:338–346`, `:451–454`.
Repeated rounded rescaling does not preserve an independent on-shell momentum
state and is particularly fragile near rest.

The same rescaling expression is present in the separate FeeBL 0.6.0-dev source
snapshot, local import commit `17a893cefda84e3128f5e22233d38781ef6e670e`, at
`src/particlesCPP.cc:138–149` and `src/particlesCPP.h:490–493`. That derivative
snapshot is **not pristine upstream**. Its recorded fork point,
`3f79200f4a6b7dcbc08718157c85f378cc688995`, is provenance from `ORIGIN.md`, not a
second upstream revision independently rechecked here.

The later HALHF failure was observed in FeeBL with additional validity checks.
Source ancestry establishes that the float rescaling was inherited; it does
not establish an identical failure in an unmodified upstream HALHF run.
The current tracker instead retains double Cartesian momentum, derives energy
from its mass shell, preserves magnetic energy and checks photon recoil against
available kinetic energy. Its regression tests and collision replay must be
evaluated separately from this source-attribution audit.

The exact-kernel proposal-density boundary and kernel/host mass inconsistencies
were found in the new QED extension and are not attributed to pristine
GuineaPig++. Finite-mass EPA corrections and full tree-level amplitudes are
model improvements, distinct from the implementation defects and
population-bookkeeping issue above.
