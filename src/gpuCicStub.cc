#include "gpuCic.h"

namespace feebl {
namespace gpu {

CicStats::CicStats()
    : particles_in(0),
      particles_out(0),
      setup_ms(0.0),
      h2d_ms(0.0),
      kernel_ms(0.0),
      d2h_ms(0.0),
      total_ms(0.0)
{
}

const char* cic_variant_name(CicVariant variant)
{
  switch (variant)
    {
    case CicVariant::Atomic: return "atomic";
    case CicVariant::Shared: return "shared";
    case CicVariant::Sorted: return "sorted";
    }
  return "unknown";
}

bool parse_cic_variant(const std::string& text, CicVariant& variant)
{
  if (text == "atomic")
    {
      variant = CicVariant::Atomic;
      return true;
    }
  if (text == "shared" || text == "tiled")
    {
      variant = CicVariant::Shared;
      return true;
    }
  if (text == "sorted" || text == "binning")
    {
      variant = CicVariant::Sorted;
      return true;
    }
  return false;
}

struct CicDepositor::Impl
{
};

CicDepositor::CicDepositor() : impl_(0)
{
}

CicDepositor::~CicDepositor()
{
}

bool CicDepositor::initialize(int, std::string& error)
{
  error = "FeeBL was built without CUDA support; rebuild with "
          "-DFEEBL_ENABLE_CUDA=ON";
  return false;
}

bool CicDepositor::is_available() const
{
  return false;
}

std::string CicDepositor::device_description() const
{
  return "CUDA unavailable in this build";
}

bool CicDepositor::deposit(const float*,
                           std::size_t,
                           double,
                           int,
                           int,
                           float,
                           float,
                           float,
                           float,
                           CicVariant,
                           double*,
                           CicStats& stats,
                           std::string& error)
{
  stats = CicStats();
  error = "CUDA CIC deposition was requested from a CPU-only FeeBL build";
  return false;
}

}  // namespace gpu
}  // namespace feebl
