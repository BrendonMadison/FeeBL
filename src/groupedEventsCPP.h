#ifndef FEEBL_GROUPED_EVENTS_CPP_H
#define FEEBL_GROUPED_EVENTS_CPP_H

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

// Qualification API: no live GRID field adapter is admitted yet. Units are
// GeV, mm and ns. The public format never uses energy sign as charge identity.
namespace feebl { namespace events {
using Vec3 = std::array<double, 3>;
struct FourMomentum { double px=0, py=0, pz=0, energy=0; };
struct Vertex { Vec3 position_mm{{0,0,0}}; double time_ns=0; std::string method; };
struct Particle {
  std::string id;
  std::size_t original_row=0;
  int pdg=0, status=1;
  std::array<int,2> mothers{{0,0}}, colors{{0,0}};
  FourMomentum input, output;
  double mass_GeV=0, lifetime_mm=0, spin=9;
};
struct Event {
  std::string id;
  std::size_t file_event_index=0;
  int process_id=0;
  double weight=1, scale=0, alpha_qed=0, alpha_qcd=0;
  std::vector<Particle> particles;
  std::string extensions;
  std::string opening_tag="<event>"; // preserve LHEF event attributes verbatim
};
struct LheDocument {
  std::string prefix, suffix;
  std::vector<Event> events;
};

// Declared line-oriented LHEF 3.x profile: exact 6/13 field rows; extension
// markup and event attributes are preserved verbatim. DTD/entity declarations
// and nested event tags are rejected.
// Caller supplies a source namespace (normally SHA-256 of the original file).
// IDs are independent of later sorting/chunking; they are not re-derived.
LheDocument read_lhe(std::istream&, const std::string& source_namespace);
void write_lhe(std::ostream&, const LheDocument&, bool transported=false);
void validate_event(const Event&);

struct Species { double charge_e, mass_GeV; };
Species species(int pdg); // e+/e-, mu+/mu-, gamma, neutrinos only; else throws
struct Field {
  Vec3 electric_GV_per_mm{{0,0,0}};
  Vec3 cB_GV_per_mm{{0,0,0}};
};
class FieldProvider {
public:
  virtual ~FieldProvider() = default;
  virtual Field at(const Vec3& position_mm, double time_ns) const = 0;
  virtual std::string version() const = 0;
};
struct TransportPolicy {
  double end_time_ns=0, max_step_ns=0;
  std::size_t max_steps=100000;
  std::string radiation="none", decay="none";
};
struct Track {
  std::string event_id, particle_id, field_version;
  std::size_t original_row=0, steps=0, file_event_index=0;
  int pdg=0, outgoing_pz_sign=0, process_id=0;
  double original_event_weight=1;
  Vertex input_vertex, output_vertex;
  FourMomentum input, output;
  double charge_e=0, mass_GeV=0, electric_work_GeV=0;
  double delta_theta=0, delta_phi=0, delta_theta_out=0;
  std::string surface="fixed_lab_time", stop_reason="end_time";
};
// Fixed-step RK4 qualification integrator, not an adaptive or detector-plane
// propagator. Each stage requests the provider at its actual x,t. The event
// is committed only after every supported final particle succeeds.
std::vector<Track> transport(Event&, const Vertex&, const FieldProvider&,
                             const TransportPolicy&);
// Caller must compute this digest from the exact output LHE bytes. The API
// checks syntax only; it cannot authenticate a caller-supplied file digest.
void write_sidecar(std::ostream&, const std::string& event_file_sha256,
                   const std::vector<Track>&);
} }
#endif
