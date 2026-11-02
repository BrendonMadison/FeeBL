
file(GLOB MODULES  besselCPP.cc beamParametersCPP.cc beamCPP.cc backgroundCPP.cc 
pairsCPP.cc minijetsCPP.cc mathematicalTools.cc lumiCPP.cc 
jetParameterCPP.cc feeblCPP.cc grvCPP.cc gridCPP.cc
fourierCPP.cc fieldCPP.cc bhabhaSamplesCPP.cc timerCPP.cc 
switchesCPP.cc splineCPP.cc rndmCPP.cc resultsCPP.cc 
principal.cc physicalTools.cc particlesCPP.cc particleBeamCPP.cc 
readData.cc memory.cc tridentCPP.cc option_args.cc parallelBackendCPP.cc
incomingMapsCPP.cc gpgeomMaps.cc legacyPhotonLumiFastCPP.cc)
list(APPEND MODULES legacyElectronLumiBlockedCPP.cc
                    electronLumiSamplerCPP.cc
                    electronLumiBernoulliSkipCPP.cc
                    encounterDAGCPP.cc
                    legacyWavefrontSchedulerCPP.cc
                    wavefrontFieldWorkspaceCPP.cc
                    wavefrontFieldValidationCPP.cc
                    uniformWindowPlanCPP.cc
                    adaptiveResolutionControllerCPP.cc
                    adaptiveTerminalPatchCPP.cc
    adaptiveMacroparticleCPP.cc
    adaptiveMacroparticleControllerCPP.cc
                    adaptiveGridScheduleCPP.cc
                    sparseDirectFieldCPP.cc)
if(FEEBL_ELECTRON_LUMI_VECTOR_REPORT)
  set_source_files_properties(legacyElectronLumiBlockedCPP.cc PROPERTIES
    COMPILE_OPTIONS
      "-fopt-info-vec-all=${FEEBL_ELECTRON_LUMI_VECTOR_REPORT}")
endif()

# The factorized luminosity kernel relies on finite-value validation and an
# audited rounded threshold predicate.  It must not inherit the historical
# project-wide fast-math contract.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(legacyPhotonLumiFastCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math")
  set_source_files_properties(electronLumiSamplerCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(electronLumiBernoulliSkipCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(sparseDirectFieldCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(uniformWindowPlanCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(adaptiveResolutionControllerCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(adaptiveTerminalPatchCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(adaptiveMacroparticleCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(wavefrontFieldValidationCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(adaptiveGridScheduleCPP.cc PROPERTIES
                              COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()

# Pure CPU reference for the versioned physical electron-luminosity collision
# raster.  The production executable links it, but runtime selection remains
# strictly opt-in and Mode 0 never calls across this boundary.
add_library(feebl_electron_lumi_spatial STATIC electronLumiSpatialCPP.cc)
target_include_directories(feebl_electron_lumi_spatial PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(feebl_electron_lumi_spatial PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_electron_lumi_spatial PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()

# Standalone, strict-FP ratio-two composite finite-volume Poisson operator.
# It remains outside the production executable until its interface,
# source-ownership, reciprocal-gather, and uniform-fine gates all pass.
add_library(feebl_adaptive_composite_poisson STATIC
            adaptiveCompositePoissonCPP.cc)
target_include_directories(feebl_adaptive_composite_poisson PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(feebl_adaptive_composite_poisson PUBLIC cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_composite_poisson PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()

# Source-once physical tent deposition and its exactly reciprocal composite
# gather.  This remains a strict standalone qualification seam: runtime Mode 3
# may consume it only after its interface and uniform-fine gates pass.
add_library(feebl_adaptive_composite_coupling STATIC
            adaptiveCompositeCouplingCPP.cc)
target_include_directories(feebl_adaptive_composite_coupling PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(feebl_adaptive_composite_coupling PUBLIC cxx_std_17)
target_link_libraries(feebl_adaptive_composite_coupling PUBLIC
                      feebl_adaptive_composite_poisson)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_composite_coupling PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()

# Set up header macro generations
include(feebl_config)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

if(FEEBL_ENABLE_CUDA)
  # Shared dependency surface for the narrow mode-1 CIC library and the
  # resident mode-2 library.  The latter links this interface when present.
  add_library(feebl_gpu2_dependencies INTERFACE)
  target_link_libraries(feebl_gpu2_dependencies INTERFACE
    CUDA::cudart CUDA::cufft)
  add_library(feebl_gpu_cic STATIC gpuCic.cu)
  set_target_properties(feebl_gpu_cic PROPERTIES
    CUDA_ARCHITECTURES "${FEEBL_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(feebl_gpu_cic PUBLIC FEEBL_HAVE_CUDA=1)
  target_link_libraries(feebl_gpu_cic PUBLIC feebl_gpu2_dependencies)
  target_compile_options(feebl_gpu_cic PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--fmad=false>)
  if(FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(feebl_gpu_cic PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>)
  endif()

  add_library(feebl_gpu_resident STATIC
    gpuResident.cu
    gpuPrimaryResident.cu
    gpuWaveBatch.cu)
  set_target_properties(feebl_gpu_resident PROPERTIES
    CUDA_ARCHITECTURES "${FEEBL_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(feebl_gpu_resident PUBLIC
    FEEBL_HAVE_GPU2=1
    FEEBL_HAVE_GPU_PRIMARY_RESIDENT=1)
  # An explicitly qualified test build can exercise deterministic failure
  # after valid stream work has been queued.  Default production builds do not
  # compile the hook, and its declaration remains hidden from normal callers.
  if(FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS)
    target_compile_definitions(feebl_gpu_resident PRIVATE
      FEEBL_GPU_RESIDENT_TEST_HOOKS=1)
  endif()
  target_link_libraries(feebl_gpu_resident PUBLIC feebl_gpu2_dependencies)
  target_compile_options(feebl_gpu_resident PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--fmad=false;-Xcompiler=-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>)
  if(FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(feebl_gpu_resident PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>)
  endif()

  # gpu_mode=5 is a separate adaptive topology implementation.  Its complete
  # class bank, cache identities and result binding are intentionally not
  # folded into the fixed-topology mode-4 library.
  add_library(feebl_gpu_adaptive STATIC gpuAdaptiveWaveBatch.cu)
  set_target_properties(feebl_gpu_adaptive PROPERTIES
    CUDA_ARCHITECTURES "${FEEBL_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(feebl_gpu_adaptive PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(feebl_gpu_adaptive PUBLIC FEEBL_HAVE_GPU5=1)
  target_link_libraries(feebl_gpu_adaptive PUBLIC feebl_gpu2_dependencies)
  target_compile_options(feebl_gpu_adaptive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off>)
  if(FEEBL_CUDA_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(feebl_gpu_adaptive PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>)
  endif()
else()
  add_library(feebl_gpu_cic STATIC gpuCicStub.cc)
  add_library(feebl_gpu_resident STATIC
    gpuResidentStub.cc
    gpuPrimaryResidentStub.cc
    gpuWaveBatchStub.cc)
endif()
target_include_directories(feebl_gpu_cic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(feebl_gpu_resident PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Focused host oracle for the compact resident-field gather.  Compiling the
# production grid translation unit with function/data sections and discarding
# unused sections keeps this test limited to GENERAL_GRID construction,
# coordinatesInGridRange, and the real ElectricFieldCIC implementation instead
# of dragging the monolithic FeeBL executable and its event-generation stack
# into the test binary.
if(BUILD_TESTING AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_grid_field_contract_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_grid_electric_field_cic.cc"
    gridCPP.cc)
  target_include_directories(feebl_grid_field_contract_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(feebl_grid_field_contract_test PRIVATE cxx_std_17)
  target_compile_options(feebl_grid_field_contract_test PRIVATE
    -ffunction-sections -fdata-sections
    # Whole-translation-unit UBSan metadata keeps RTTI references from
    # otherwise discarded monolithic grid sections, defeating this target's
    # intentionally narrow link.  ASan remains active in sanitizer builds;
    # the standalone resident target retains both ASan and UBSan coverage.
    -fno-sanitize=undefined)
  set_property(TARGET feebl_grid_field_contract_test APPEND_STRING PROPERTY
    LINK_FLAGS " -Wl,--gc-sections")
  target_link_libraries(feebl_grid_field_contract_test PRIVATE
    feebl_gpu_cic feebl_gpu_resident)
endif()

# Replacement-core foundation.  Most replacement physics remains test-only;
# the deliberately narrow gpu_mode=3 target below links BeamSoA plus the
# audited import/write-back boundary into production without selecting any of
# the otherwise unfinished modern collision engines.
add_library(feebl_modern_core STATIC
  beamSoACPP.cc
  cellCSRCPP.cc
  encounterDAGCPP.cc
  legacyWavefrontSchedulerCPP.cc
  legacyHostWaveConcurrencyCPP.cc
  modernCollisionCPP.cc
  modernCpuExecutorCPP.cc
  modernLegacyAdapterCPP.cc
  modernLegacyWritebackCPP.cc
  modernPhysicsPlanCPP.cc
  modernPhotonSoACPP.cc
  modernPhotonLumiCPP.cc)
target_include_directories(feebl_modern_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(feebl_modern_core PUBLIC cxx_std_17)
# The repository's legacy global flags include -ffast-math.  The scalar modern
# reference must retain IEEE finite/NaN checks and defined comparison behavior.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_core PRIVATE -fno-fast-math)
endif()

# Audited production boundary for gpu_mode=3.  The resident CUDA component is
# usable in CPU/stub builds too, where its capability check fails closed.
# Keeping this boundary separate from the replacement core prevents an
# incomplete modern engine from being selected accidentally.
add_library(feebl_gpu_mode3_adapter STATIC
  gpuPrimaryModernAdapter.cc
  gpuMode3ProductionCPP.cc)
target_include_directories(feebl_gpu_mode3_adapter PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(feebl_gpu_mode3_adapter PUBLIC cxx_std_17)
target_link_libraries(feebl_gpu_mode3_adapter PUBLIC
  feebl_modern_core feebl_gpu_resident)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_gpu_mode3_adapter PRIVATE -fno-fast-math)
endif()

add_executable(feebl ${MODULES})
target_compile_definitions(feebl PRIVATE
  FEEBL_BUILD_SOURCE_HASH="${FEEBL_SOURCE_REVISION}")
# gpgeom/maps.hpp includes gpgeom/phase_space.hpp by that path
target_include_directories(feebl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(feebl feebl_gpu_cic feebl_gpu_resident
                            feebl_gpu_mode3_adapter
                            feebl_electron_lumi_spatial)

if(FEEBL_ENABLE_CUDA)
  target_link_libraries(feebl feebl_gpu_adaptive)
  target_compile_definitions(feebl PRIVATE
    FEEBL_HAVE_CUDA=1
    FEEBL_HAVE_GPU2=1
    FEEBL_HAVE_GPU_PRIMARY_RESIDENT=1
    FEEBL_HAVE_GPU5=1
    FEEBL_CUDA_TOOLKIT_VERSION="${CMAKE_CUDA_COMPILER_VERSION}")
endif()

if(FEEBL_ENABLE_CUDA AND BUILD_TESTING)
  add_executable(feebl_gpu_cic_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_gpu_cic.cc")
  target_link_libraries(feebl_gpu_cic_test PRIVATE feebl_gpu_cic)
  target_compile_features(feebl_gpu_cic_test PRIVATE cxx_std_11)
  add_executable(feebl_gpu_resident_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_gpu_resident_fft.cc")
  if(FEEBL_ENABLE_GPU_RESIDENT_TEST_HOOKS)
    target_compile_definitions(feebl_gpu_resident_test PRIVATE
      FEEBL_GPU_RESIDENT_TEST_HOOKS=1)
  endif()
  target_link_libraries(feebl_gpu_resident_test PRIVATE feebl_gpu_resident)
  target_compile_features(feebl_gpu_resident_test PRIVATE cxx_std_11)
  add_executable(feebl_gpu_wave_batch_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_gpu_wave_batch.cc")
  target_link_libraries(feebl_gpu_wave_batch_test PRIVATE feebl_gpu_resident)
  target_compile_features(feebl_gpu_wave_batch_test PRIVATE cxx_std_11)
  add_executable(feebl_gpu_primary_resident_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_gpu_primary_resident.cc"
    particlesCPP.cc
    tridentCPP.cc
    physicalTools.cc
    rndmCPP.cc
    besselCPP.cc
    splineCPP.cc
    mathematicalTools.cc)
  target_link_libraries(feebl_gpu_primary_resident_test PRIVATE
    feebl_gpu_resident)
  target_compile_features(feebl_gpu_primary_resident_test PRIVATE cxx_std_14)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Capability guards and the direct PARTICLE oracle both require IEEE
    # finite-value semantics; the repository-wide legacy fast-math flag would
    # otherwise optimize those checks away.
    target_compile_options(feebl_gpu_primary_resident_test PRIVATE
                           -fno-fast-math -ffp-contract=off)
  endif()
  add_executable(feebl_gpu_primary_modern_adapter_test
    "${CMAKE_SOURCE_DIR}/examples/gpu/test_gpu_primary_modern_adapter.cc")
  target_link_libraries(feebl_gpu_primary_modern_adapter_test PRIVATE
    feebl_gpu_mode3_adapter)
  target_compile_features(feebl_gpu_primary_modern_adapter_test PRIVATE
    cxx_std_17)
  add_executable(feebl_gpu_adaptive_wave_batch_test
    "${CMAKE_SOURCE_DIR}/examples/adaptive_grid/test_gpu_adaptive_wave_batch.cu"
    uniformWindowPlanCPP.cc)
  set_target_properties(feebl_gpu_adaptive_wave_batch_test PROPERTIES
    CUDA_ARCHITECTURES "${FEEBL_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES)
  target_link_libraries(feebl_gpu_adaptive_wave_batch_test PRIVATE
    feebl_gpu_adaptive)
  target_compile_features(feebl_gpu_adaptive_wave_batch_test PRIVATE
    cxx_std_17)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(feebl_gpu_adaptive_wave_batch_test PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-Xcompiler=-fno-fast-math;-Xcompiler=-ffp-contract=off>)
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(feebl Threads::Threads)

if(FEEBL_ENABLE_OPENMP)
  target_link_libraries(feebl OpenMP::OpenMP_CXX)
  target_link_libraries(feebl_modern_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(feebl PRIVATE FEEBL_HAVE_OPENMP=1)
endif()

# Native text output uses store_native=1 (src/nativeOutputCPP.h). No EDM4HEP
# dependency or writer is included in this build.

	

if(FFTW2)
   if(FFTW2_FOUND)
      target_link_libraries(feebl ${FFTW2_LIBRARIES})
      include_directories(${FFTW2_INCLUDES})
   else()
      target_link_libraries(feebl fftw m)
   endif()
elseif(FFTW3)
   if(FFTW_FOUND)
      target_link_libraries(feebl ${FFTW_LIBRARIES})
      include_directories(${FFTW_INCLUDES})
   else()
      target_link_libraries(feebl fftw3 m)
   endif()
endif()

# Installing targets and headers:
install(TARGETS feebl
  ARCHIVE
  DESTINATION lib
  COMPONENT Libraries
  RUNTIME
  DESTINATION bin
  COMPONENT Runtime)

file(GLOB headers *.h)
install(FILES ${headers}
  DESTINATION include/feebl
  COMPONENT Headers)

# Grouped-event qualification API is available to the collider executable.
# No unqualified live-field/runtime selector is enabled by this linkage.
target_sources(feebl PRIVATE groupedEventsCPP.cc)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(groupedEventsCPP.cc PROPERTIES
      COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  set_source_files_properties(bhabhaSamplesCPP.cc PROPERTIES
      COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()
