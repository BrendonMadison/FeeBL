# This file sets all the config variables needed for config.h 

#include(CheckLibraryExists)
include(CheckFunctionExists)
include(CheckIncludeFiles)

macro(check_header name)
    string(TOLOWER ${name}.h _FNAME)
    string(REGEX REPLACE "_" "/" _FNAME "${_FNAME}")
    check_include_files(${_FNAME} HAVE_${name}_H)
endmacro(check_header)


# config.h:
# Single-sourced from project(feebl VERSION ...) in the top-level
# CMakeLists.txt. It must expand to a C STRING literal; the value
# inherited here was '1.2.1' in single quotes, i.e. a multi-character
# constant, and was two versions stale.
set(PACKAGE_VERSION  ${FEEBL_VERSION_FULL})
set(PACKAGE_NAME     ${PROJECT_NAME})
set(PACKAGE_TARNAME  ${PROJECT_NAME})
set(PACKAGE_STRING   "${PROJECT_NAME} ${FEEBL_VERSION_FULL}")
set(PACKAGE_BUGREPORT brendon_madison@ku.edu)

check_header("ALLOCA")
check_header("ARPA_INET")
check_header("FCNTL")
check_header("INTTYPES")
check_header("MEMORY")
check_header("NETDB")
check_header("STDBOOL")
check_header("STDINT")
check_header("STDLIB")
check_header("STRINGS")
check_header("STRING")
check_header("SYS_SELECT")
check_header("SYS_SOCKET")
check_header("SYS_STAT")
check_header("SYS_TIME")
check_header("SYS_TYPES")
check_header("UNISTD")
check_header("VFORK")
check_header("WCHAR")
check_header("WCTYPE")
check_header("FFTW")
if(NOT HAVE_FFTW_H)
   check_header("DFFTW")
   if(NOT HAVE_DFFTW_H)
      check_header("SFFTW")
   endif()
endif()
check_header("FFTW3")
check_function_exists(stpcpy  HAVE_STPCPY)
