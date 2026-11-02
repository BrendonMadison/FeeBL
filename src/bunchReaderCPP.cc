#include "bunchReaderCPP.h"

#ifndef FEEBL_HDF5

bool feebl_have_hdf5() { return false; }

BunchReadResult read_bunch_hdf5(const std::string&, CANONICAL_BUNCH&,
                                const LongitudinalConvention&)
{
  BunchReadResult r;
  r.ok = false;
  r.error = "this build has no HDF5 support (rebuild with FEEBL_HDF5 and libhdf5)";
  return r;
}

#else

#include <hdf5.h>
#include <vector>
#include <cmath>

bool feebl_have_hdf5() { return true; }

namespace {

// Read a string attribute, fixed- or variable-length.
bool read_str_attr(hid_t loc, const char* name, std::string& out)
{
  if (H5Aexists(loc, name) <= 0) return false;
  hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
  if (a < 0) return false;
  hid_t t = H5Aget_type(a);
  bool ok = false;
  if (H5Tget_class(t) == H5T_STRING)
    {
      if (H5Tis_variable_str(t))
        {
          char* s = 0;
          if (H5Aread(a, t, &s) >= 0 && s) { out = s; ok = true; }
        }
      else
        {
          size_t sz = H5Tget_size(t);
          hid_t mt = H5Tcopy(H5T_C_S1);
          H5Tset_size(mt, sz + 1);
          std::vector<char> buf(sz + 1, 0);
          if (H5Aread(a, mt, &buf[0]) >= 0) { out = &buf[0]; ok = true; }
          H5Tclose(mt);
        }
    }
  H5Tclose(t); H5Aclose(a);
  return ok;
}

bool read_dbl_attr(hid_t loc, const char* name, double& out)
{
  if (H5Aexists(loc, name) <= 0) return false;
  hid_t a = H5Aopen(loc, name, H5P_DEFAULT);
  if (a < 0) return false;
  bool ok = (H5Aread(a, H5T_NATIVE_DOUBLE, &out) >= 0);
  H5Aclose(a);
  return ok;
}

// Read a 1-D double dataset. Returns false if absent or the wrong length.
bool read_col(hid_t g, const char* name, std::vector<double>& out, hsize_t expect)
{
  if (H5Lexists(g, name, H5P_DEFAULT) <= 0) return false;
  hid_t d = H5Dopen2(g, name, H5P_DEFAULT);
  if (d < 0) return false;
  hid_t s = H5Dget_space(d);
  hsize_t dims[2] = {0, 0};
  int nd = H5Sget_simple_extent_dims(s, dims, 0);
  bool ok = false;
  if (nd == 1 && (expect == 0 || dims[0] == expect))
    {
      out.resize((size_t)dims[0]);
      ok = (H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &out[0]) >= 0);
    }
  H5Sclose(s); H5Dclose(d);
  return ok;
}

// The schema records the convention as prose. Map only the forms we actually
// understand; anything else must stop the read rather than be guessed at.
bool parse_longitudinal(const std::string& decl, int& sign)
{
  if (decl.find("zeta>0 toward bunch head") != std::string::npos ||
      decl.find("zeta > 0 toward bunch head") != std::string::npos)
    { sign = +1; return true; }
  if (decl.find("zeta>0 toward bunch tail") != std::string::npos ||
      decl.find("zeta > 0 toward bunch tail") != std::string::npos)
    { sign = -1; return true; }
  return false;
}

} // namespace

BunchReadResult read_bunch_hdf5(const std::string& path,
                                CANONICAL_BUNCH& bunch,
                                const LongitudinalConvention& intended)
{
  BunchReadResult r;

  if (!intended.declared)
    { r.error = "caller supplied no longitudinal convention; FEEBL has no "
                "per-beam zeta to fall back on (CONVENTIONS.md C5)"; return r; }

  // HDF5 prints its whole error stack to stderr on a failed open. We report
  // failures through BunchReadResult, so silence it -- an expected rejection
  // should not look like a crash.
  H5E_auto2_t oldfn = 0; void* olddata = 0;
  H5Eget_auto2(H5E_DEFAULT, &oldfn, &olddata);
  H5Eset_auto2(H5E_DEFAULT, 0, 0);
  hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  H5Eset_auto2(H5E_DEFAULT, oldfn, olddata);
  if (f < 0) { r.error = "cannot open " + path; return r; }

  std::string schema, coords, longdecl, species, wnorm;
  double eref = 0.0, pop = 0.0;
  read_str_attr(f, "schema", schema);
  read_str_attr(f, "coordinate_system", coords);
  read_str_attr(f, "longitudinal_convention", longdecl);
  read_str_attr(f, "species", species);
  read_str_attr(f, "weight_normalization", wnorm);
  read_dbl_attr(f, "reference_energy_GeV", eref);
  read_dbl_attr(f, "bunch_population", pop);

  if (schema != "gp++-arbitrary-bunch/1.0")
    { r.error = "unsupported schema \"" + schema + "\""; H5Fclose(f); return r; }
  if (coords != "beam_local_canonical")
    { r.error = "coordinate_system \"" + coords + "\" is not beam_local_canonical; "
                "refusing to infer a conversion"; H5Fclose(f); return r; }
  if (eref <= 0.0)
    { r.error = "missing or non-positive reference_energy_GeV"; H5Fclose(f); return r; }

  int file_sign = 0;
  if (!parse_longitudinal(longdecl, file_sign))
    { r.error = "unrecognised longitudinal_convention \"" + longdecl + "\"";
      H5Fclose(f); return r; }
  if (file_sign != intended.sign)
    { r.error = "longitudinal convention mismatch: file declares \"" + longdecl +
                "\", caller intends the opposite sign"; H5Fclose(f); return r; }

  if (H5Lexists(f, "particles", H5P_DEFAULT) <= 0)
    { r.error = "no /particles group"; H5Fclose(f); return r; }
  hid_t g = H5Gopen2(f, "particles", H5P_DEFAULT);

  std::vector<double> x, px, y, py, ze, de, w;
  bool got = read_col(g, "x", x, 0);
  const hsize_t n = (hsize_t)x.size();
  got = got && read_col(g, "px_over_p0", px, n)
            && read_col(g, "y",          y,  n)
            && read_col(g, "py_over_p0", py, n)
            && read_col(g, "zeta",       ze, n)
            && read_col(g, "delta",      de, n)
            && read_col(g, "weight",     w,  n);
  H5Gclose(g); H5Fclose(f);

  if (!got)
    { r.error = "missing or mis-shaped required dataset in /particles"; return r; }

  // Reject rather than repair: the schema requires it, and a silently dropped
  // particle is a change in the physics.
  for (size_t i = 0; i < w.size(); i++)
    if (!(w[i] > 0.0) || !std::isfinite(w[i])) r.n_rejected++;
  if (r.n_rejected)
    { char b[128];
      snprintf(b, sizeof b, "%zu of %zu weights are non-positive or non-finite",
               r.n_rejected, w.size());
      r.error = b; return r; }

  bunch.resize(x.size());
  bunch.set_reference_energy(eref);
  bunch.set_longitudinal(intended);
  for (size_t i = 0; i < x.size(); i++)
    bunch.set(i, x[i], px[i], y[i], py[i], ze[i], de[i], w[i]);

  BunchProvenance p;
  p.source = path;
  p.schema = schema;
  p.coordinate_system = coords;
  p.reference_energy_GeV = eref;
  bunch.set_provenance(p);

  r.ok = true;
  r.n_read = x.size();
  return r;
}

#endif
