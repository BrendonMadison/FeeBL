// Stage C reader gate. A reader that only proves it can load a good file is
// half a test: the schema's actual requirement is that it REFUSES bad input
// rather than silently inferring. Most of these cases are rejections.
#include "bunchReaderCPP.h"
#include <cstdio>
#include <cmath>
#include <string>

static int fails = 0;
static void check(bool ok, const char* what)
{
  printf("  %-56s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

int main(int argc, char** argv)
{
  // The unsupported build has a useful, testable contract: reject an import
  // explicitly and do not populate a bunch. It needs no fixture file.
  if (!feebl_have_hdf5())
    {
      CANONICAL_BUNCH b;
      const BunchReadResult r = read_bunch_hdf5(
          "unsupported-reader-fixture.h5", b, LongitudinalConvention(+1, 0.0));
      check(!r.ok && r.n_read == 0 && b.size() == 0,
            "unsupported reader refuses import without creating particles");
      check(r.error.find("no HDF5 support") != std::string::npos,
            "unsupported reader identifies the missing capability");
      return fails ? 1 : 0;
    }

  // The runner generates a fixture in its own temporary output directory.
  if (argc <= 1)
    {
      printf("  usage: test_reader <arbitrary_bunch_example.h5>\n");
      printf("  NOT RUN -- no bunch file given.\n");
      return 2;
    }
  const std::string good = argv[1];

  // The file declares "zeta>0 toward bunch head", i.e. sign +1.
  const LongitudinalConvention head(+1, 0.0);
  const LongitudinalConvention tail(-1, 0.0);
  LongitudinalConvention undeclared;

  // --- refuses without a caller convention (C5) ------------------------
  {
    CANONICAL_BUNCH b;
    BunchReadResult r = read_bunch_hdf5(good, b, undeclared);
    check(!r.ok && r.error.find("longitudinal") != std::string::npos,
          "refuses when the caller declares no convention");
  }

  // --- refuses on a convention mismatch --------------------------------
  {
    CANONICAL_BUNCH b;
    BunchReadResult r = read_bunch_hdf5(good, b, tail);
    check(!r.ok && r.error.find("mismatch") != std::string::npos,
          "refuses when file and caller disagree on the sign");
  }

  // --- refuses a missing file ------------------------------------------
  {
    CANONICAL_BUNCH b;
    BunchReadResult r = read_bunch_hdf5("/nonexistent/bunch.h5", b, head);
    check(!r.ok, "refuses a missing file");
  }

  // --- the happy path ---------------------------------------------------
  {
    CANONICAL_BUNCH b;
    BunchReadResult r = read_bunch_hdf5(good, b, head);
    check(r.ok, "reads the generated HDF5 fixture");
    check(r.n_read == 20000 && b.size() == 20000, "reads all 20000 particles");
    check(std::fabs(b.reference_energy_GeV() - 62.5) < 1e-9,
          "picks up reference_energy_GeV = 62.5 from the file");
    check(b.provenance().schema == "gp++-arbitrary-bunch/1.0",
          "records the schema in provenance");
    check(b.provenance().source == good, "records the source path");

    CANONICAL_BUNCH::Validation v = b.validate();
    check(v.ok(), "imported bunch passes weight and finiteness validation");

    // The file's weights are a probability distribution (sum = 1), so they are
    // emphatically not uniform. If uniform_weight() were true, the weight
    // column had been ignored.
    check(!b.uniform_weight(), "non-uniform weights are detected, not ignored");

    double sw = 0.0;
    for (std::size_t i = 0; i < b.size(); i++) sw += b.weight(i);
    printf("       sum(weight) = %.6f  (file declares sum = 1)\n", sw);
    check(std::fabs(sw - 1.0) < 1e-6, "weights sum to 1 as the file declares");

    // Round-trip a few through FEEBL units and back.
    bool rt = true;
    for (std::size_t i = 0; i < 5 && i < b.size(); i++)
      {
        float X,Y,Z,VX,VY,E;
        if (!b.to_feebl(i, X, Y, Z, VX, VY, E)) { rt = false; break; }
        const double dx = std::fabs((double)X * 1e-9 - b.x(i));
        if (dx > 1e-15 + 1e-6 * std::fabs(b.x(i))) rt = false;
      }
    check(rt, "canonical -> FEEBL units round trip is consistent");
  }

  printf("\n  %s\n", fails ? "FAILURES PRESENT" : "all reader checks passed");
  return fails ? 1 : 0;
}
