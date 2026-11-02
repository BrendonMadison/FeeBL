#include "groupedEventsCPP.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv)try {
  if(argc!=4)throw std::runtime_error("usage: grouped_events_roundtrip INPUT OUTPUT ORIGINAL_SOURCE_NAMESPACE");
  std::ifstream input(argv[1]);if(!input)throw std::runtime_error("cannot open input");
  auto doc=feebl::events::read_lhe(input,argv[3]);
  // Parsing completes before opening the destination, preventing partial
  // success/truncation of an output when the input is unsupported.
  std::ofstream output(argv[2]);if(!output)throw std::runtime_error("cannot open output");
  feebl::events::write_lhe(output,doc,false);
  std::cout<<"PASS LHE profile roundtrip events="<<doc.events.size()<<" transport_performed=false\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"REFUSED "<<e.what()<<'\n';return 1;}
