#ifndef CANONICALBUNCH_SEEN
#define CANONICALBUNCH_SEEN

#include <vector>
#include <string>
#include <cstddef>
#include <cmath>

// Canonical six-dimensional storage for import, incoming maps and export.
// Double precision resolves the 1e-11 symplecticity tolerance. Conversion and
// map application occur once per incoming bunch, outside the tracking loop.
// Coordinate conventions are defined in doc/CONVENTIONS.md:
//
//   C2  vx_/vy_ are trajectory ANGLES in radians, not canonical momenta.
//       px = x'(1+delta), so they coincide only on-momentum. Measured cost of
//       conflating them: the relative error on a crab kick is exactly delta,
//       which is ~0.1% of beam divergence at FCC-ee-Z energy spread and ~0.8%
//       for tail particles. Systematic, not random -- it will not average away.
//
//   C5  FEEBL has NO per-beam zeta. zpos_ is one global lab coordinate shared
//       by both beams. zeta must therefore be DEFINED, per beam, and the sign
//       and reference plane have no value in the code today. That is why
//       LongitudinalConvention below has no default: an RF map's sin(k*zeta)
//       flips with it, and getting it wrong tilts the bunch the wrong way,
//       which surfaces as a plausible luminosity reduction rather than an
//       obvious failure.

// How this beam's zeta relates to the shared global z. Must be supplied
// explicitly; there is no defensible default (see C5).
struct LongitudinalConvention
{
  int    sign;            // +1 if this beam travels toward +z, -1 otherwise
  double z_ref_nm;        // reference plane on the shared axis, in nm
  bool   declared;        // false until someone actually sets it

  LongitudinalConvention() : sign(0), z_ref_nm(0.0), declared(false) {}
  LongitudinalConvention(int s, double zref)
    : sign(s), z_ref_nm(zref), declared(true) {}
};

struct BunchProvenance
{
  std::string source;             // file path, or "generated"
  std::string schema;             // e.g. gp++-arbitrary-bunch/1.0
  std::string coordinate_system;  // as declared by the source
  double      reference_energy_GeV;
  bool        recentred;
  bool        rescaled;
  BunchProvenance()
    : source("generated"), schema(""), coordinate_system(""),
      reference_energy_GeV(0.0), recentred(false), rescaled(false) {}
};

class CANONICAL_BUNCH
{
  std::vector<double> x_, px_, y_, py_, zeta_, delta_, w_;
  LongitudinalConvention lc_;
  BunchProvenance        prov_;
  double                 e_ref_GeV_;
  bool                   uniform_weight_;

 public:

  CANONICAL_BUNCH() : e_ref_GeV_(0.0), uniform_weight_(true) {}

  inline std::size_t size() const { return x_.size(); }
  inline bool uniform_weight() const { return uniform_weight_; }
  inline double reference_energy_GeV() const { return e_ref_GeV_; }
  inline const LongitudinalConvention& longitudinal() const { return lc_; }
  inline const BunchProvenance& provenance() const { return prov_; }
  inline void set_provenance(const BunchProvenance& p) { prov_ = p; }
  inline void set_longitudinal(const LongitudinalConvention& c) { lc_ = c; }

  inline void resize(std::size_t n)
    {
      x_.resize(n); px_.resize(n); y_.resize(n);
      py_.resize(n); zeta_.resize(n); delta_.resize(n); w_.assign(n, 1.0);
    }

  inline double x(std::size_t i)     const { return x_[i]; }
  inline double px(std::size_t i)    const { return px_[i]; }
  inline double y(std::size_t i)     const { return y_[i]; }
  inline double py(std::size_t i)    const { return py_[i]; }
  inline double zeta(std::size_t i)  const { return zeta_[i]; }
  inline double delta(std::size_t i) const { return delta_[i]; }
  inline double weight(std::size_t i)const { return w_[i]; }

  inline void set(std::size_t i, double X, double PX, double Y, double PY,
                  double Z, double D, double W)
    {
      x_[i]=X; px_[i]=PX; y_[i]=Y; py_[i]=PY; zeta_[i]=Z; delta_[i]=D; w_[i]=W;
      if (W != 1.0) uniform_weight_ = false;
    }

  inline void set_reference_energy(double e) { e_ref_GeV_ = e; }

  // ---- conversions, per doc/CONVENTIONS.md ----------------------------
  //
  // FEEBL side: positions in nm, angles in rad, energy absolute in GeV.
  // Canonical side: positions in m, momenta normalised to p0, delta relative.

  // Returns false if the longitudinal convention was never declared, rather
  // than silently inventing one (C5).
  inline bool from_feebl(std::size_t i,
                         float xpos_nm, float ypos_nm, float zpos_nm,
                         float vx_rad, float vy_rad, float energy_GeV)
    {
      if (!lc_.declared || e_ref_GeV_ <= 0.0) return false;
      const double d = (double)energy_GeV / e_ref_GeV_ - 1.0;
      x_[i]     = (double)xpos_nm * 1e-9;
      y_[i]     = (double)ypos_nm * 1e-9;
      zeta_[i]  = lc_.sign * ((double)zpos_nm - lc_.z_ref_nm) * 1e-9;
      delta_[i] = d;
      px_[i]    = (double)vx_rad * (1.0 + d);   // C2: NOT simply vx_
      py_[i]    = (double)vy_rad * (1.0 + d);
      return true;
    }

  inline bool to_feebl(std::size_t i,
                       float& xpos_nm, float& ypos_nm, float& zpos_nm,
                       float& vx_rad, float& vy_rad, float& energy_GeV) const
    {
      if (!lc_.declared || e_ref_GeV_ <= 0.0) return false;
      const double d = delta_[i];
      xpos_nm   = (float)(x_[i] * 1e9);
      ypos_nm   = (float)(y_[i] * 1e9);
      zpos_nm   = (float)(lc_.z_ref_nm + lc_.sign * zeta_[i] * 1e9);
      vx_rad    = (float)(px_[i] / (1.0 + d));
      vy_rad    = (float)(py_[i] / (1.0 + d));
      energy_GeV= (float)(e_ref_GeV_ * (1.0 + d));
      return true;
    }

  // Validate the coordinate and weight contract.
  // "A reader must reject negative or non-finite source weights and must not
  //  silently infer units."
  struct Validation { std::size_t bad_weight, non_finite; bool ok() const
                      { return bad_weight == 0 && non_finite == 0; } };

  inline Validation validate() const
    {
      Validation v; v.bad_weight = 0; v.non_finite = 0;
      for (std::size_t i = 0; i < x_.size(); i++)
        {
          if (!(w_[i] > 0.0)) v.bad_weight++;
          if (!std::isfinite(x_[i])  || !std::isfinite(px_[i]) ||
              !std::isfinite(y_[i])  || !std::isfinite(py_[i]) ||
              !std::isfinite(zeta_[i])||!std::isfinite(delta_[i]) ||
              !std::isfinite(w_[i])) v.non_finite++;
        }
      return v;
    }
};

#endif
