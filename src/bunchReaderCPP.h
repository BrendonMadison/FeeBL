#ifndef BUNCHREADER_SEEN
#define BUNCHREADER_SEEN

#include <string>
#include "canonicalBunchCPP.h"

// FEEBL: reader for the gp++-arbitrary-bunch/1.0 HDF5 schema.
//
// Compiled only when FEEBL_HDF5 is defined, so a build without libhdf5 is
// unaffected. Requesting an HDF5 import from a build without it fails loudly at
// the call site rather than silently doing nothing.
//
// The schema is explicit that a reader "must reject negative or non-finite
// source weights and must not silently infer units". This one additionally
// refuses on:
//
//   * a schema string it does not recognise -- a future 2.0 may reinterpret
//     fields, and guessing would be worse than stopping;
//   * a coordinate_system other than beam_local_canonical, because converting
//     from an undeclared system is exactly the silent inference the schema
//     forbids;
//   * a missing or unrecognised longitudinal_convention, because FEEBL has no
//     per-beam zeta of its own (doc/CONVENTIONS.md C5) and an RF map's
//     sin(k*zeta) flips sign with it.
//
// Weight handling follows the file's own declaration. The example ships
// weight_normalization = "sum(weight)=1; multiply by bunch_population", so the
// weights are a probability distribution, not particle counts.

struct BunchReadResult
{
  bool        ok;
  std::string error;      // empty when ok
  std::size_t n_read;
  std::size_t n_rejected; // non-positive or non-finite weights
  BunchReadResult() : ok(false), n_read(0), n_rejected(0) {}
};

// Reads path into bunch. The caller supplies the longitudinal convention it
// intends to use; the reader checks the file's declaration against it and
// refuses on a mismatch rather than quietly preferring one.
BunchReadResult read_bunch_hdf5(const std::string& path,
                                CANONICAL_BUNCH& bunch,
                                const LongitudinalConvention& intended);

// True when this build can actually read HDF5.
bool feebl_have_hdf5();

#endif
