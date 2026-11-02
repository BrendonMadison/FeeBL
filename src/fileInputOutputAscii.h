#ifndef FILEOUTPUTASCII_SEEN
#define FILEOUTPUTASCII_SEEN
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>


#include "IfileInputOutput.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class FILE_IN_OUT_ASCII : public IFILE_IN_OUT
{
  std::ifstream infile_;
  std::ofstream outfile_;

 public :

  FILE_IN_OUT_ASCII()  {;}
  virtual ~FILE_IN_OUT_ASCII() {;}
      
  virtual void open_file(std::string name, const char* mode)
  {  
    switch (mode[0])
      {
	// read 
      case 'r':
	infile_.open(name.c_str(), std::ios::in);
	if (!infile_) 
	  {
	    std::cerr << " error opening input stream " << name << std::endl;
	    exit(1);
	  }
	break;
	// write
      case 'w':
	      
	outfile_.open(name.c_str(), std::ios::out);
	if (!outfile_) 
	  {
	    std::cerr << " error opening output stream " << name << std::endl;
	    exit(1);
	  }
	break;
      default:
	std::cerr << " FILE_IN_OUT_ASCII::open_file: unknown open mode : " << mode[0] << std::endl;
	exit(1);
      }
  }

  virtual void set_header(std::string head)
  {  
    outfile_ << head << std::endl;
  }
      
  virtual void close()
  {
    infile_.close();
    outfile_.close();
  }
 
  virtual bool read_line(std::string& line) 
  {
    getline(infile_, line);
    return infile_.good();
  }
      
  virtual void write_line(std::string& line) 
  {
    outfile_ << line << std::endl;
  }
  virtual void set_jet_header(float ptmin, float sqrt_s)
  {
    outfile_ << ptmin << " " << sqrt_s << std::endl;
  }
      
  virtual void  save_jet(float energy1, float energy2, int process) 
  {   
    outfile_ << energy1 << " "  << energy2 << " "  << process << std::endl;
  }
  virtual void  save_jet(float energy1, float energy2, float pz1, float pz2, float pt, int process) 
  {   
    outfile_ << energy1 << " "  << energy2 << " "  << pz1 << " " << pz2 << " " << pt << " " << process << std::endl;
  }
      
  virtual void  save_hadron(float energy1, float energy2, float z) 
  {   
    outfile_ << energy1 << " "  << energy2 << " "  << z << std::endl;
  }

  /*  virtual void  save_photon(const ABSTRACT_PARTICLE& part, int no_beam)  */
  /*   {    */
  /*     float  energy, xpos, ypos, zpos, vx,vy, dummy; */
  /*     part.get_parameters_for_output(energy, xpos, ypos, zpos, vx,vy, dummy, dummy, dummy); */
  /*     if (no_beam != 1) energy = -energy; */
  /*     outfile_ << energy << " "  << vx << " "  << vy << std::endl; */
  /*   } */

  virtual void save_compton_photon(float y, float px, float py)
  {
    outfile_ << y << " " << px << " " << py << std::endl;
  }
      
  /* virtual void  save_particle(const ABSTRACT_PARTICLE& part)  */
  /*   {    */
  /*     float  energy, xpos, ypos, zpos, vx,vy, spinx, spiny, spinz ; */
      
  /*     part.get_parameters_for_output(energy, xpos, ypos, zpos, vx,vy,spinx, spiny, spinz ); */
  /*     outfile_ << energy << " "  << xpos << " "  << ypos <<" "  <<  zpos << " "  << vx << " "  << vy << " "  << spinx << " "  << spiny << " "  << spinz << std::endl; */
  /*   } */
      
  virtual bool  read_particle(PARTICLE_INTERFACE& part) 
  {   
    int k;
    // float  energy, xpos, ypos, zpos, vx,vy ;
    float readValue[9];
    for ( k =0; k < 9; k++) readValue[k] = 0.0;
    bool test = false;
    std::string slu;
    int number = 0;
    if ( getline(infile_, slu) )
      {
	bool goodline = false;
	std::istringstream ss(slu);
	float temp;
	while ( ss >> temp && number < 9) 
	  {
	    goodline = true;
	    readValue[number] = temp;
	    number++;
	  }
	part.init_from_input_file(readValue[0],readValue[1],readValue[2],readValue[3], readValue[4], readValue[5], readValue[6], readValue[7], readValue[8] );
	if ( !goodline) std::cerr << " read_particle : reading failure " << std::endl;
	test = true;
      }
    return test;
  }
      
  virtual bool  read_photon(PARTICLE_INTERFACE& part) 
  {   
    float  energy, xpos, ypos, zpos, vx,vy, hel ;
    float dumy = 0.0;
    float dumz = 0.0;
    bool test;
    if ( infile_ >> energy >> xpos >> ypos >> zpos >> vx >> vy >> hel) 
      {
	part.init_from_input_file(energy, xpos, ypos, zpos, vx, vy, hel, dumy,dumz);
	test= true;
      }
    else test =  false;
    return test;
  }
      
  /* virtual void save_pair_particle(const ABSTRACT_PAIR_PARTICLE& pair_part)  */
  /*   { */
  /*     float x,y,z,vx,vy,vz, energy; */
  /*     pair_part.get_parametersX(x,y,z,vx,vy,vz,energy); */
  /*     //   outfile_ << energy << " "  << vx << " "  << vy << " "  << vz << " "  << x << " "  << y << " "  << z << " "  << pair_part.get_process() <<  " "  << pair_part.get_label() << std::endl; */
  /*        outfile_ << energy << " "  << vx << " "  << vy << " "  << vz <<  " "  << pair_part.get_process() <<  " "  << pair_part.get_label() << std::endl; */
  /*   } */
 
  virtual void save_bhabhasamples(const ABSTRACT_BHABHASAMPLES* const bhabhas)  
  {
	  
    unsigned int k, evtIndex;
    float eCM, mother1_en, e1, vx1, vy1, vz1, mother2_en, e2, vx2, vy2, vz2;
    int nbphot;
    for (k=0; k < bhabhas->nb_samples(); k++)
      {
	bhabhas->get_parameters_for_output(k, evtIndex, eCM, mother1_en, e1, vx1, vy1, vz1, mother2_en, e2, vx2, vy2, vz2, nbphot);
	outfile_ << std::setw(8) << evtIndex << " " << std::setw(10) << std::fixed << std::setprecision(4) << eCM << " ";
	outfile_ << std::setw(10) << std::setprecision(4) << mother1_en << " " << std::setw(10) << e1 << " "  << std::setprecision(6) << std::setw(10) << vx1 << " "  << std::setw(10) << vy1 << " "  << std::setw(10) << vz1 << " ";
	outfile_ << std::setw(10) << std::setprecision(4) << mother2_en << " " << std::setw(10) << e2 << " "  << std::setprecision(6) << std::setw(10) << vx2 << " "  << std::setw(10) << vy2 << " "  << std::setw(10) << vz2 << " ";
	outfile_ << std::setw(4) << nbphot << std::endl;
      }
  }
      
      
  virtual bool read_bhabhasamples(ABSTRACT_BHABHASAMPLES* const bhabhas)
  {
    struct Row { unsigned int id; float px1,py1,pz1,e1,px2,py2,pz2,e2; int photons; };
    std::vector<Row> rows;
    std::set<unsigned int> ids;
    std::string line;
    // Validate the complete legacy file before exposing any records. A
    // partial/truncated row used to terminate the read as apparent success.
    while (std::getline(infile_,line)) {
      if (line.find_first_not_of(" \t\r")==std::string::npos) continue;
      std::istringstream data(line);
      Row row; long long id,photons; std::string extra;
      if (!(data>>id>>row.px1>>row.py1>>row.pz1>>row.e1
                 >>row.px2>>row.py2>>row.pz2>>row.e2>>photons) || (data>>extra))
        throw std::runtime_error("malformed/truncated legacy bhabha.ini row");
      if (id<0 || id>std::numeric_limits<unsigned int>::max() || photons<0 ||
          photons>std::numeric_limits<int>::max())
        throw std::runtime_error("invalid legacy Bhabha ID/photon count");
      row.id=static_cast<unsigned int>(id);row.photons=static_cast<int>(photons);
      for (float value:{row.px1,row.py1,row.pz1,row.e1,row.px2,row.py2,row.pz2,row.e2})
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite legacy Bhabha row");
      if (row.e1==0 || row.e2==0 || !ids.insert(row.id).second)
        throw std::runtime_error("zero energy or duplicate legacy Bhabha ID");
      rows.push_back(row);
    }
    if (infile_.bad()) throw std::runtime_error("legacy Bhabha read failed");
    for (const auto& row:rows)
      bhabhas->add_bhabha(row.id,row.px1,row.py1,row.pz1,row.e1,
                       row.px2,row.py2,row.pz2,row.e2,row.photons);
    return true;
  }

  virtual void save_bhabhaPhotonSamples(const ABSTRACT_BHABHA_PHOTON_SAMPLES*  bhabhaPhot)  
  {
	  
    unsigned int k;
    float en, vx, vy, vz;
    int evtIndex;
    for (k=0; k < bhabhaPhot->nb_samples(); k++)
      {
	bhabhaPhot->get_parameters_for_output(k,evtIndex, en, vx, vy, vz);
	//outfile_ << std::setw(8) << k+1 << " " << std::setw(8) << evtIndex << " " << std::setw(10) << std::fixed << std::setprecision(4) << en << " "  << std::setw(10) << vx << " "  << std::setw(10) << vy << " "  << std::setw(10) << vz << std::endl;
	outfile_ << std::setw(8) << k+1 << " " << std::setw(8) << evtIndex << " " << std::setw(10) << std::fixed << std::setprecision(12) << en << " "  << std::setw(10) << vx << " "  << std::setw(10) << vy << " "  << std::setw(10) << vz << std::endl;
      }
  }
      
  virtual bool read_bhabhaPhotonsamples(ABSTRACT_BHABHA_PHOTON_SAMPLES* const bhabhasPhoton)
  {
    int evtIdx;
    float px, py, pz, en;
    //int nbphot;
    while (infile_ >> evtIdx >> px >> py >> pz >> en)
      {
	bhabhasPhoton->add_bhabha_photon(evtIdx, px, py, pz, en);
      }
    return true;
  }
      
      
  virtual int read_pythia_file(int& logx, int& logy, std::vector<double>& x, std::vector<double>& y)
  {
    int k;
    int nentries;
    x.clear();
    y.clear();
    if (infile_ >> nentries >> logx >> logy )
      {
	x.resize(nentries);
	y.resize(nentries);     
	for (k = 0; k < nentries;k++)
	  {
	    if ( !(infile_ >> x[k] >> y[k]) )
	      {
		std::cerr << " FILE_IN_OUT_ASCII::read_pythia_file : the list of pythia values is interrupted " << std::endl;
		exit(1);
	      }
	  }
      }
    else
      {
	std::cerr << " FILE_IN_OUT_ASCII::read_pythia_file error reading pythia file " << std::endl;
	exit(1);
      }
    return nentries; 
  }
      
  virtual bool read_cross(ABSTRACT_CROSS_DATA* const crossIni)
  {
    int k;
    int n, nval, dummy1, dummy2;
    float energy;
    float* values = NULL;
    bool test = false;
    if (infile_ >> n >> nval >> dummy1 >> dummy2)
      {
	crossIni->resize(n, nval);
	values = new float[nval];
	while (  infile_ >> energy )
	  {
	    for (k=0; k< nval; k++)
	      {
		if (!(infile_ >> values[k]))
		  {
		    std::cerr << " FILE_IN_OUT_ASCII::read_cross : the list of cross values is interrupted " << std::endl;
		    exit(1);
		  }
	      }
	    crossIni->add_data(energy, values);
	  }
	if (values != NULL) delete [] values;
	test = true;
      }
    else
      {
	std::cerr << " FILE_IN_OUT_ASCII::read_cross error reading cross values from file " << std::endl;
	exit(1);
      }
    return test; 
  }


  virtual void save_lumi_heap(const ABSTRACT_LUMI_HEAP* const lumi_heap) 
  {
    outfile_ << lumi_heap->persistent_flow() << std::endl;
  }
      
  virtual void save_object_on_output_listing(const ABSTRACT_IO_CLASS* const obj)
  {
    outfile_ << obj->output_flow();
  }
      
  virtual void save_object_on_persistent_file(const ABSTRACT_IO_CLASS* const obj)
  {
    outfile_ << obj->persistent_flow() << std::endl;
  }

  virtual void save_object_on_persistent_file(const int eventIndex, const ABSTRACT_IO_CLASS* const obj)
  {
    outfile_ << eventIndex << " " << obj->persistent_flow() << std::endl;
  }

};



#endif
