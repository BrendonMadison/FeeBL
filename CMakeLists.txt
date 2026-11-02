cmake_minimum_required(VERSION 3.16)

project(feebl VERSION 0.6.0)
# project(VERSION) accepts only numeric components, but the release identity carries
# a suffix. Single-source it here so config.h, the binary banner and configure.ac
# remain consistent; examples/version_identity.sh enforces it.
set(FEEBL_VERSION_SUFFIX "-dev")
set(FEEBL_VERSION_FULL "${PROJECT_VERSION}${FEEBL_VERSION_SUFFIX}")
include(cmake/SourceIdentity.cmake)

get_filename_component(FEEBL_SOURCE_ROOT_REAL "${CMAKE_SOURCE_DIR}" REALPATH)
execute_process(
  COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --show-toplevel
  OUTPUT_VARIABLE FEEBL_GIT_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE FEEBL_GIT_ROOT_RESULT)
if(FEEBL_GIT_ROOT_RESULT EQUAL 0 AND FEEBL_GIT_ROOT)
  get_filename_component(FEEBL_GIT_ROOT_REAL "${FEEBL_GIT_ROOT}" REALPATH)
endif()

# An exported tree may be unpacked below an unrelated repository. Git parent
# discovery must not attribute that outer repository's revision to FeeBL.
if(FEEBL_GIT_ROOT_RESULT EQUAL 0 AND
   FEEBL_GIT_ROOT_REAL STREQUAL FEEBL_SOURCE_ROOT_REAL)
  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --verify HEAD
    OUTPUT_VARIABLE FEEBL_SOURCE_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(NOT FEEBL_SOURCE_REVISION)
  if(EXISTS "${CMAKE_SOURCE_DIR}/SOURCE_REVISION")
    file(STRINGS "${CMAKE_SOURCE_DIR}/SOURCE_REVISION"
         FEEBL_SOURCE_REVISION LIMIT_COUNT 1)
    feebl_validate_source_identity("${FEEBL_SOURCE_REVISION}" FEEBL_SOURCE_REVISION)
  else()
    set(FEEBL_SOURCE_REVISION "unknown")
  endif()
else()
  execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" status --porcelain
            --untracked-files=normal --ignore-submodules
    OUTPUT_VARIABLE FEEBL_SOURCE_DIRTY
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(FEEBL_SOURCE_DIRTY)
    string(APPEND FEEBL_SOURCE_REVISION "-dirty")
  endif()
endif()

# Use the same language standard for serial and accelerated builds.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(FEEBL_ENABLE_OPENMP
       "Build the experimental OpenMP CPU kernel candidate" OFF)

# Resolve the imported target in the top-level directory so both the
# production src/ tree and sibling examples/ tree can link the same runtime.
# Finding it only inside src/ makes OpenMP::OpenMP_CXX directory-scoped and
# breaks combined CUDA+OpenMP test generation.
if(FEEBL_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Optional, production-object compiler evidence for the bounded electron-lumi
# block kernel. Empty is the normal build; an explicit absolute path asks GCC
# to write its vectorization decisions while compiling the object linked into
# feebl (not a throwaway benchmark-only copy).
set(FEEBL_ELECTRON_LUMI_VECTOR_REPORT "" CACHE FILEPATH
    "GCC -fopt-info-vec-all report for legacyElectronLumiBlockedCPP.cc")
if(FEEBL_ELECTRON_LUMI_VECTOR_REPORT AND
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "FEEBL_ELECTRON_LUMI_VECTOR_REPORT currently requires GCC")
endif()

# CUDA is deliberately opt-in: a normal FeeBL build must neither require a
# CUDA toolkit nor change its serial executable.  CUDA_ARCHITECTURES was added
# in CMake 3.18; keeping the project's historical minimum is therefore safe
# for CPU-only builds while accelerator builds receive an actionable error.
option(FEEBL_ENABLE_CUDA "Build the optional CUDA CIC deposition backend" OFF)
set(FEEBL_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures for FeeBL (for example: 86 or 80;86)")
option(FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER
       "Pass --allow-unsupported-compiler to nvcc (use only after testing)" OFF)
option(FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS
       "Compile deterministic resident-CUDA failure injection for qualification tests" OFF)

if(FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS AND NOT FEEBL_ENABLE_CUDA)
  message(FATAL_ERROR
    "FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS requires FEEBL_ENABLE_CUDA=ON")
endif()

if(FEEBL_ENABLE_CUDA)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    message(FATAL_ERROR
      "FEEBL_ENABLE_CUDA requires CMake 3.18 or newer; the CPU-only build still supports older CMake")
  endif()
  include(CheckLanguage)
  # The compiler check itself invokes nvcc, so this compatibility escape hatch
  # must be present before check_language/enable_language, not only on targets.
  if(FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER)
    string(FIND "${CMAKE_CUDA_FLAGS}" "--allow-unsupported-compiler"
           _feebl_cuda_allow_index)
    if(_feebl_cuda_allow_index EQUAL -1)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    endif()
  endif()
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "FEEBL_ENABLE_CUDA=ON was requested, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  # Imported CUDA runtime and cuFFT targets are required by the resident
  # pipeline.  Finding them here fails at configure time with a useful error,
  # while CPU-only builds remain entirely CUDA-free.
  find_package(CUDAToolkit REQUIRED)
  if(FEEBL_CUDA_ARCHITECTURES STREQUAL "native" AND
     CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR
      "CUDA architecture 'native' requires CMake 3.24 or newer; set FEEBL_CUDA_ARCHITECTURES explicitly")
  endif()
endif()

# add source dir as a place for CMake modules (e.g. FindGSL.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(feebl_definitions)

include(compilers)

# With this true we automatically include current src/binary directory
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

include(CTest)
add_subdirectory(src)

# BUILD_TESTING=OFF provides a small executable-only installation.
if(BUILD_TESTING)
  add_subdirectory(examples)
endif()
