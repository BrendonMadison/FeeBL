#ifndef FEEBL_GPU_CIC_SEEN
#define FEEBL_GPU_CIC_SEEN

#include <cstddef>
#include <string>

namespace feebl {
namespace gpu {

// The names intentionally describe algorithms, rather than CUDA launch
// details.  They are stable integration points for SWITCHES/readData and for
// a future resident GPU tracking backend.
enum class CicVariant
{
  Atomic,
  Shared,
  Sorted
};

const char* cic_variant_name(CicVariant variant);
bool parse_cic_variant(const std::string& text, CicVariant& variant);

struct CicStats
{
  std::size_t particles_in;
  std::size_t particles_out;
  double setup_ms;
  double h2d_ms;
  double kernel_ms;
  double d2h_ms;
  double total_ms;

  CicStats();
};

// Owns reusable device allocations and a stream.  Reusing one instance avoids
// cudaMalloc/cudaFree in FeeBL's frequently called slice loop.  The class has
// no CUDA types in its public header, so CPU-only translation units and builds
// remain independent of CUDA headers.
class CicDepositor
{
 public:
  CicDepositor();
  ~CicDepositor();

  CicDepositor(const CicDepositor&) = delete;
  CicDepositor& operator=(const CicDepositor&) = delete;

  bool initialize(int device, std::string& error);
  bool is_available() const;
  std::string device_description() const;

  // Deposit a uniform macroparticle charge on an nx-by-ny host grid.  xy is
  // interleaved [x0,y0,x1,y1,...].  rho is overwritten, not accumulated.
  // This exactly matches SLICE_ON_GRID::razRho followed by
  // assignChargeToCIC for an ordinary beam slice.
  bool deposit(const float* xy,
               std::size_t particle_count,
               double charge,
               int nx,
               int ny,
               float delta_x_inverse,
               float delta_y_inverse,
               float offset_x,
               float offset_y,
               CicVariant variant,
               double* rho,
               CicStats& stats,
               std::string& error);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace gpu
}  // namespace feebl

#endif
