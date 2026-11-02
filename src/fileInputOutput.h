#ifndef FILEINPUTOUTPUT_SEEN
#define FILEINPUTOUTPUT_SEEN
#include "abstractParticle.h"
//#include "IfileInputOutput.h"
#include "fileInputOutputAscii.h"
#include <string>
#include <vector>
//#include <cstdio>

class FILE_IN_OUT
{

  IFILE_IN_OUT* out_to_file_;

  /// assignment and copy constructor not implemented nor used
  FILE_IN_OUT& operator=(const FILE_IN_OUT&);
  FILE_IN_OUT(FILE_IN_OUT&);
 
 public: 


  FILE_IN_OUT() : out_to_file_(NULL) 
  { 
    out_to_file_ = new FILE_IN_OUT_ASCII();
  }

  virtual ~FILE_IN_OUT() 
    {
      delete out_to_file_;
    }

  virtual inline void open_file(std::string name, const char* mode)
    { 
//	  std::cout << "Opening file " << name << std::endl;
      out_to_file_->open_file(name, mode);
    }


  virtual inline void set_header(std::string head)
    { 
      out_to_file_->set_header(head);
    }
  
  virtual inline void close() { out_to_file_->close();}
  
  virtual inline void set_jet_header(float ptmin, float sqrt_s)
    {
      out_to_file_->set_jet_header(ptmin, sqrt_s);
    }
  
  virtual inline void  save_jet(float energy1, float energy2, int process) 
    {   
      out_to_file_->save_jet(energy1, energy2, process); 
    }
  
  virtual inline void  save_jet(float energy1, float energy2, float pz1, float pz2, float pt, int process) 
    {
      out_to_file_-> save_jet(energy1, energy2, pz1, pz2, pt, process);
    }

  virtual inline void  save_hadron(float energy1, float energy2, float z) 
    {   
      out_to_file_->save_hadron(energy1, energy2, z);
    }

  virtual inline void save_compton_photon(float y, float px, float py)
    {
      out_to_file_->save_compton_photon(y, px, py);
    }
  
  virtual inline  bool  read_particle(PARTICLE_INTERFACE& part) 
    {
      return  out_to_file_->read_particle(part);
    }
  
  virtual inline  bool  read_line(std::string& line) 
    {
      return  out_to_file_->read_line(line);
    }
  
  virtual inline  void  write_line(std::string& line) 
    {
      return  out_to_file_->write_line(line);
    }
  
  virtual inline void save_bhabhasamples(const ABSTRACT_BHABHASAMPLES* const bhabhas) 
    {
      out_to_file_->save_bhabhasamples(bhabhas);
    }
  
  virtual inline bool read_bhabhasamples(ABSTRACT_BHABHASAMPLES* const bhabhas) 
    {
      return  out_to_file_->read_bhabhasamples(bhabhas);
    }
  
  virtual inline void save_bhabhaPhotonSamples(const ABSTRACT_BHABHA_PHOTON_SAMPLES*  bhabhaPhot)  
    {
      return  out_to_file_->save_bhabhaPhotonSamples(bhabhaPhot);
    }

  virtual inline bool read_bhabhaPhotonsamples(ABSTRACT_BHABHA_PHOTON_SAMPLES* const bhabhasPhoton) 
    {
      return  out_to_file_->read_bhabhaPhotonsamples( bhabhasPhoton);
    }
  
  virtual int read_pythia_file(int& logx, int& logy, std::vector<double>& x, std::vector<double>& y)
    {
      return  out_to_file_->read_pythia_file(logx, logy, x, y);
    }
  
  virtual bool read_cross(ABSTRACT_CROSS_DATA* const crossIni)
    {
      return  out_to_file_->read_cross(crossIni);
    }
  
  virtual inline void save_lumi_heap(const ABSTRACT_LUMI_HEAP* const lumi_heap) 
    {
      out_to_file_->save_lumi_heap(lumi_heap);
    }
  
  virtual inline void save_object_on_output_listing(const ABSTRACT_IO_CLASS* const obj)
    {
      if (obj == NULL) return;
      out_to_file_->save_object_on_output_listing(obj);
    }
  
  virtual inline void save_object_on_persistent_file(const ABSTRACT_IO_CLASS* const obj)
    {
      if (obj == NULL) return;
      out_to_file_->save_object_on_persistent_file(obj);
    }

  virtual inline void save_object_on_persistent_file(const int evtIndex, const ABSTRACT_IO_CLASS* const obj)
    {
      if (obj == NULL) return;
      out_to_file_->save_object_on_persistent_file(evtIndex, obj);
    }

};

#endif
