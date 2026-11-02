
#include "option_args.h"
#include <getopt.h>
#include <cstdlib>
#include <iostream>

std::string electron_input_file; // = "electron.ini";
std::string positron_input_file; // = "positron.ini";
std::string acc_file;            // = "acc.dat";
bool make_gui_files_ = false;


std::string get_elfilename()
{
  return electron_input_file;
}
std::string get_posfilename()
{
  return positron_input_file;
}


std::string get_acc_filename()
{
  return acc_file;
}

bool get_make_gui_files()
{
  return make_gui_files_;
}
const struct option long_options[] = {
    {"help",           no_argument, 0, 'h'},
    {"el_file",  required_argument, 0, 'e'},
    {"pos_file", required_argument, 0, 'p'},
    {"acc_file", required_argument, 0, 'a'},
    {"make_gui_files",   no_argument, 0, 'g'},
    {0, 0, 0, 0}
};

void set_defaults() {
  electron_input_file=std::string("electron.ini");
  positron_input_file=std::string("positron.ini");
  acc_file=std::string("acc.dat");
  make_gui_files_=false;
}

void print_usage(char *program_name) {
                std::cout << "USAGE:\t" << program_name << " [OPTIONS] accelerator parameter_set output_file\n\n"
                    "DESCRIPTION:\n"
                    "  Calculates luminosity from a distribution...\n\n"
                    "OPTIONS [defaults]:\n"
                    "  --help             Display this help\n  --version, -V      Print version and provenance\n"
                    "  --el_file          electron distribution [electron.ini]\n"
                    "  --pos_file         positron distribution [positron.ini]\n"
                    "  --acc_file         parameters            [acc.dat]\n"
                    "  --make_gui_files   dump per-step beams and write gui_manifest.json\n\n";
}
  
std::vector< std::string > read_args(int argc, char* argv[]) {
  std::vector<std::string> required_args(3);
  set_defaults();


  // First we retrieve the optional arguments..
  int index = 0;
  int iarg = 0;
  while(iarg!=-1) {
    iarg = getopt_long_only( argc, argv,"", long_options, &index);
    switch (iarg) {
      case '?':
        print_usage(argv[0]);
        exit(1);
      case 'h':
        print_usage(argv[0]);
        exit(0);
      case 'e':
        electron_input_file=optarg;
        break;
      case 'p':
        positron_input_file=optarg;
        break;
      case 'a':
        acc_file=optarg;
        break;
      case 'g':
        make_gui_files_=true;
        break;
      default:
        break;
    }
  }

  // Now we retrieve the required arguments..
  int num_required_args = 0;
  for ( int i=optind; i<argc; i++ ) {
    if (num_required_args < 3) required_args[num_required_args]=argv[i];
    num_required_args++;
  }
  // Check that we have correct number of required arguments
  if ( num_required_args != 3 ) {
    std::cerr << "ERROR: Wrong number of arguments" << std::endl;
    print_usage(argv[0]);
    exit(2);
  }

  return required_args;
};

