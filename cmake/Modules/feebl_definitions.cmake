
# Adds preprocessor definitions...

if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
   add_definitions(-DCOMPUTER64b)
endif()

find_package(FFTW)
if(FFTW_FOUND AND NOT FFTW2)
   option(FFTW2 "FFTW2 libraries for field computation" OFF)
   option(FFTW3 "FFTW3 libraries for field computation" ON)
else()
   option(FFTW3 "FFTW3 libraries for field computation" OFF)
   find_package(FFTW2)
   if(FFTW2_FOUND AND NOT FFTW3)
      option(FFTW2 "FFTW2 libraries for field computation" ON)
   else()
      option(FFTW2 "FFTW2 libraries for field computation" OFF)
   endif()
endif()

if(FFTW2 AND FFTW3)
   message(FATAL_ERROR "Cannot set both FFTW2 and FFTW3 on, please select one")
endif()

if(FFTW2)
   add_definitions(-DUSE_FFTW2)
elseif(FFTW3)
   add_definitions(-DUSE_FFTW3)
else()
   add_definitions(-DUSE_FFT_LOCAL)
endif()


