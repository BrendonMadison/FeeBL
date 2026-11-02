#include "feeblCPP.h"
#include "adaptiveRootReplaySignalCPP.h"
#include <exception>
#include <iostream>
#include "particleBeamCPP.h"
#include "option_args.h"

// FEEBL: report identity and lineage without needing a run.
static int feebl_print_version()
{
  std::cout << "FEEBL " << PACKAGE_VERSION << "  (Fast e+e- Beam Luminosity)\n"
            << "derived from GUINEA-PIG++ 1.2.2 by D. Schulte et al.,"
            << " used with permission\n"
            << "upstream: https://gitlab.cern.ch/clic-software/guinea-pig.git\n"
            << "fork point: 3f79200f4a6b7dcbc08718157c85f378cc688995\n"
            << "physics reference: D. Schulte, TESLA-97-08 (DESY)\n"
            << "see ORIGIN.md for the full lineage\n";
  return 0;
}

int mainFeebl(std::vector<std::string> input_arguments)
{
  // There is probably a nicer way to do this...
   std::string arg1 = input_arguments[0];
   std::string arg2 = input_arguments[1];
   std::string arg3 = input_arguments[2];

   FEEBL feebl_run((char*)(arg1.c_str()));
   feebl_run.run((char*)(arg2.c_str()),(char*)(arg3.c_str()));
   return 0;
}

           
  
int main (int argc,char *argv[])
{
  // FEEBL: handled before read_args(), which rejects flags it does not know.
  for (int i = 1; i < argc; i++)
    {
      std::string a(argv[i]);
      if (a == "--version" || a == "-V") return feebl_print_version();
    }
  std::vector<std::string> input_arguments = read_args(argc,argv);
  try
    {
      return mainFeebl(input_arguments);
    }
  catch (const feebl::adaptive_particles::RootReplayRequired& error)
    {
      std::cerr << "FEEBL: ADAPTIVE ROOT REPLAY REQUIRED: "
                << error.what() << std::endl;
      return feebl::adaptive_particles::kRootReplayRequiredExitCode;
    }
  catch (const std::exception& error)
    {
      std::cerr << "FEEBL: ERROR: " << error.what() << std::endl;
      return 2;
    }
  catch (...)
    {
      std::cerr << "FEEBL: ERROR: unrecognized fatal exception" << std::endl;
      return 2;
    }
}
