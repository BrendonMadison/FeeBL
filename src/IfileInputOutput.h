#ifndef IFILEINPUTOUTPUT_SEEN
#define IFILEINPUTOUTPUT_SEEN

#include <string>
#include <vector>

#include "abstractParticle.h"

class IFILE_IN_OUT
{

 public:

  IFILE_IN_OUT() {;}
  virtual ~IFILE_IN_OUT() {;}

  virtual  void open_file(std::string name, const char* mode) =0;
  virtual  void set_header(std::string head) =0;
  virtual  void close() = 0;

  virtual void set_jet_header(float ptmin, float sqrt_s) = 0;
  virtual void  save_jet(float energy1, float energy2, int process) = 0;
  virtual void  save_jet(float energy1, float energy2, float pz1, float pz2, float pt, int process) = 0;

  virtual void  save_hadron(float energy1, float energy2, float z) = 0;
  //  virtual void  save_photon(const ABSTRACT_PARTICLE& part, int no_beam) = 0;
  virtual void save_compton_photon(float y, float px, float py) = 0;

  //  virtual  void  save_particle(const ABSTRACT_PARTICLE& part)  = 0;
  virtual bool  read_particle(PARTICLE_INTERFACE& part)  = 0;

  virtual bool read_line(std::string& line)  = 0;
  virtual void write_line(std::string& line)  = 0;


  //  virtual void save_pair_particle(const ABSTRACT_PAIR_PARTICLE& pair_part)  = 0;
  virtual void save_bhabhasamples(const ABSTRACT_BHABHASAMPLES* const bhabhas)  =0;
  virtual bool read_bhabhasamples(ABSTRACT_BHABHASAMPLES* const bhabhas) =0;

  virtual void save_bhabhaPhotonSamples(const ABSTRACT_BHABHA_PHOTON_SAMPLES*  bhabhaPhot)  = 0;


  virtual bool read_bhabhaPhotonsamples(ABSTRACT_BHABHA_PHOTON_SAMPLES* const bhabhasPhoton) = 0;
  virtual int read_pythia_file(int& logx, int& logy, std::vector<double>& x, std::vector<double>& y) = 0;
 
  virtual bool read_cross(ABSTRACT_CROSS_DATA* const crossIni) = 0;
  virtual void save_lumi_heap(const ABSTRACT_LUMI_HEAP* const lumi_heap)  =0;
  //  virtual void save_lumi_heap_full(const ABSTRACT_LUMI_HEAP* const lumi_heap)  =0;
  // virtual void dump_photons(const ABSTRACT_BEAM& beam,int istep, int timestep, float step, float max_z) = 0;

  //  virtual void dump_beam(const ABSTRACT_BEAM& beam, int istep, int every_particle, int timestep, float step, float max_z) = 0;

  /*   virtual void save_beam_parameters(const ABSTRACT_BEAM_PARAMETERS* const beam_par1, const ABSTRACT_BEAM_PARAMETERS* const beam_par2) = 0; */

  virtual void save_object_on_output_listing(const ABSTRACT_IO_CLASS* const obj) = 0;

  virtual void save_object_on_persistent_file(const ABSTRACT_IO_CLASS* const obj) = 0;
  virtual void save_object_on_persistent_file(const int evtIdx, const ABSTRACT_IO_CLASS* const obj) = 0;


};



#endif
