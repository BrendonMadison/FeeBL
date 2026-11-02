# Origin and attribution

FeeBL (Fast e+- e+- Beam Luminosity) extends GUINEA-PIG++ 1.2.2.
The inherited source records permission from the original authors to use the
code. FeeBL does not claim original authorship of inherited physics kernels.

The upstream repository is
<https://gitlab.cern.ch/clic-software/guinea-pig.git>. The recorded fork point
is commit `3f79200f4a6b7dcbc08718157c85f378cc688995`, dated 20 April 2026.
The source snapshot used for this distribution is based on FeeBL
commit `f819b996b0dd99313b2decb15f5c9dd84feb989d`. `SOURCE_REVISION` records a content digest of the exported native
build inputs. `SHA256SUMS` identifies the exact distributed files. The ZIP is a
source export and does not contain Git history.

Original authors credited by the inherited project include:

- Daniel Schulte: original beam-beam simulation.
- Guy Le Meur: object-oriented implementation.
- C. Rimbault, B. Dalena, J. Esberg, and J. Snuverink: GUINEA-PIG++ development.

The principal physics reference is D. Schulte, *Study of Electromagnetic and
Hadronic Background in the Interaction Region of the TESLA Collider*,
TESLA-97-08. The manual in `docs/FeeBL_Manual_Paper.pdf` gives primary references for the
spin, spectra, deflection, and monochromatization studies. Existing source
copyright notices and contributed-code attribution are preserved.

FeeBL additions include explicit incoming maps, native diagnostic outputs,
browser visualization, reproducible comparison tools, and experimental
parallel and adaptive algorithms. Their validation scope is recorded in the
manual and the registered tests. Numerical agreement for a particular test does
not establish accuracy outside that test's physical and numerical conditions.

## Distribution terms

This source distribution contains code derived from GUINEA-PIG++ 1.2.2.
The supplied project records permission from the original authors to use
that code, but supplies no standalone software license text. This
distribution preserves that attribution and does not invent or extend a
license grant. See `ORIGIN.md` for the upstream repository and revision.

Generated build-support files retain their own notices. Digitized numerical
reference data identify their primary publications and extraction method in
`benchmarks/reference`; the original publication PDFs are not redistributed.
Inclusion of derived comparison data does not transfer rights to the papers.
