file(COPY acc.dat electron.ini.gz result.ref result.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# The fine design-axis observable is accumulated by private collision lanes.
# Check geometry, reduction, reset, and fail-closed publication with both the
# production arithmetic policy and a strict reference policy.
foreach(design_axis_suite IN ITEMS spectrum parallel)
  foreach(design_axis_policy IN ITEMS production reference)
    set(design_axis_target feebl_design_axis_${design_axis_suite}_${design_axis_policy})
    add_executable(${design_axis_target}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/test_design_axis_${design_axis_suite}.cc)
    target_include_directories(${design_axis_target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
    target_compile_features(${design_axis_target} PRIVATE cxx_std_17)
    if(design_axis_policy STREQUAL "reference" AND
       CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${design_axis_target} PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME feebl-design-axis-${design_axis_suite}-${design_axis_policy}
      COMMAND ${design_axis_target})
    set_tests_properties(feebl-design-axis-${design_axis_suite}-${design_axis_policy}
      PROPERTIES LABELS "luminosity;design-axis;unit;${design_axis_policy}" TIMEOUT 60)
  endforeach()
endforeach()

# Checkpoint and card admission use production flags. A strict-float-only test
# would not catch a build policy that removes std::isfinite from the simulator.
add_executable(feebl_rng_checkpoint_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/rng/test_checkpoint.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc)
target_include_directories(feebl_rng_checkpoint_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_rng_checkpoint_unit PRIVATE cxx_std_17)
add_test(NAME feebl-source-identity
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../examples/source_identity.cmake)
set_tests_properties(feebl-source-identity PROPERTIES LABELS "provenance;unit" TIMEOUT 10)
add_test(NAME feebl-rng-checkpoint-unit COMMAND feebl_rng_checkpoint_unit)
set_tests_properties(feebl-rng-checkpoint-unit PROPERTIES
    LABELS "rng;checkpoint;production-flags;transaction" TIMEOUT 60)

add_executable(feebl_collider_moment_contract
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/collider_moment_contract.cc)
target_include_directories(feebl_collider_moment_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_collider_moment_contract PRIVATE cxx_std_17)
add_test(NAME feebl-collider-moment-contract COMMAND feebl_collider_moment_contract)
set_tests_properties(feebl-collider-moment-contract PROPERTIES
    LABELS "gui;moments;stability;production-flags" TIMEOUT 30)

find_program(GUNZIP NAMES gunzip)
execute_process(COMMAND ${GUNZIP} -f electron.ini.gz WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Resolve the executable after generation, including multi-configuration builds.
set(binaryname $<TARGET_FILE:feebl>)

# The worker-pool unit is independent of the monolithic simulation and checks
# repeated dispatch, exactly-once task execution, exception propagation, and
# recovery. It intentionally does not need OpenMP.
find_package(Threads REQUIRED)
add_executable(feebl_cpu_backend_unit
               cpu_backend_unit.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/parallelBackendCPP.cc)
target_include_directories(feebl_cpu_backend_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_link_libraries(feebl_cpu_backend_unit Threads::Threads)
add_test(NAME feebl-cpu-backend-unit COMMAND feebl_cpu_backend_unit)

# Selector/provenance contract for the adaptive CUDA engine.  This unit does
# not claim device availability; it enables the compile-time capability only
# to exercise fail-closed mode-5 routing and its persistent host pool.
add_executable(feebl_gpu5_backend_unit
               gpu5_backend_unit.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/parallelBackendCPP.cc)
target_include_directories(feebl_gpu5_backend_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_definitions(feebl_gpu5_backend_unit PRIVATE
                           FEEBL_HAVE_GPU5=1)
target_compile_features(feebl_gpu5_backend_unit PRIVATE cxx_std_17)
target_link_libraries(feebl_gpu5_backend_unit Threads::Threads)
add_test(NAME feebl-gpu5-backend-unit COMMAND feebl_gpu5_backend_unit)
set_tests_properties(feebl-gpu5-backend-unit PROPERTIES
    LABELS "adaptive-grid;gpu;mode5;backend;failure;no-fallback;unit"
    TIMEOUT 120)

# Immutable canonical-window planner.  This target is independent of the
# monolithic simulation and covers exact deposit/gather node enumeration,
# nextafter boundaries, deterministic placement/hash, and fail-closed caps.
add_executable(feebl_uniform_window_plan_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_uniform_window_plan.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/uniformWindowPlanCPP.cc)
target_include_directories(feebl_uniform_window_plan_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_uniform_window_plan_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_uniform_window_plan_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-uniform-window-plan-unit
         COMMAND feebl_uniform_window_plan_unit)
set_tests_properties(feebl-uniform-window-plan-unit PROPERTIES
    LABELS "adaptive-grid;mode1;planner;unit" TIMEOUT 120)

# Grid-independent, fixed-physical-CIC electron collision quadrature.  This is
# a pure Stage-A target: no GRID/MESH, legacy RNG, event generation, field
# solver, or runtime-dispatch dependency is permitted across this boundary.
add_executable(feebl_electron_lumi_spatial_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_spatial/test_electron_lumi_spatial.cc)
target_link_libraries(feebl_electron_lumi_spatial_unit PRIVATE
                      feebl_electron_lumi_spatial)
target_compile_features(feebl_electron_lumi_spatial_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_electron_lumi_spatial_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-electron-lumi-spatial-unit
         COMMAND feebl_electron_lumi_spatial_unit)
set_tests_properties(feebl-electron-lumi-spatial-unit PROPERTIES
    LABELS "electron-luminosity;spatial-kernel;cic;collision-plan;closure;determinism;unit"
    TIMEOUT 120)

# Process-level Stage-B boundary: Mode-0 inertness, parser/capability refusals,
# runtime admission/resource failures, and byte-identical aggregate/spectrum
# output across nominal 64/128/256 force grids for frozen physical roots.
add_test(NAME feebl-electron-lumi-spatial-process-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_spatial/run_process_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/electron-lumi-spatial-process)
set_tests_properties(feebl-electron-lumi-spatial-process-validation PROPERTIES
    LABELS "electron-luminosity;spatial-kernel;parser;capability;integration;closure;grid-invariance;determinism;failure;no-fallback"
    TIMEOUT 180 RUN_SERIAL TRUE
    DEPENDS "feebl-electron-lumi-spatial-unit")

# Two-level ratio-2 leaf-space finite-volume operator.  This Stage-A target
# verifies matrix symmetry/SPD, unique interface fluxes, Gauss balance,
# deterministic PCG, manufactured recovery, continuum contraction and
# fail-closed publication before any Mode-3 runtime wiring is permitted.
add_executable(feebl_adaptive_composite_poisson_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_composite_poisson.cc)
target_link_libraries(feebl_adaptive_composite_poisson_unit PRIVATE
                      feebl_adaptive_composite_poisson)
target_compile_features(feebl_adaptive_composite_poisson_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_composite_poisson_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-adaptive-composite-poisson-unit
         COMMAND feebl_adaptive_composite_poisson_unit)
set_tests_properties(feebl-adaptive-composite-poisson-unit PROPERTIES
    LABELS "adaptive-grid;mode3;composite;poisson;flux;manufactured;determinism;unit"
    TIMEOUT 120)

add_executable(feebl_adaptive_composite_coupling_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_composite_coupling.cc)
target_link_libraries(feebl_adaptive_composite_coupling_unit PRIVATE
                      feebl_adaptive_composite_coupling)
target_compile_features(feebl_adaptive_composite_coupling_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_composite_coupling_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-adaptive-composite-coupling-unit
         COMMAND feebl_adaptive_composite_coupling_unit)
set_tests_properties(feebl-adaptive-composite-coupling-unit PROPERTIES
    LABELS "adaptive-grid;mode3;composite;deposit;gather;adjoint;self-force;determinism;unit"
    TIMEOUT 120)

# Independent scientific diagnostics for the composite prototype.  The
# executable exits successfully after emitting fixed Stage-4 gates so the
# complete regression suite retains the measured failure evidence.  The
# strict qualification test invokes --require-stage4, which exits 2 while any
# numerical physics gate remains unresolved.
add_executable(feebl_adaptive_composite_physics_validation
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_composite_physics_validation.cc)
target_link_libraries(feebl_adaptive_composite_physics_validation PRIVATE
                      feebl_adaptive_composite_coupling
                      feebl_adaptive_composite_poisson)
target_compile_features(feebl_adaptive_composite_physics_validation PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_composite_physics_validation PRIVATE
                         -Wall -Wextra -Wpedantic -Werror -Wconversion
                         -Wsign-conversion -Wshadow
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-adaptive-composite-physics-validation
         COMMAND feebl_adaptive_composite_physics_validation)
set_tests_properties(feebl-adaptive-composite-physics-validation PROPERTIES
    LABELS "adaptive-grid;mode3;composite;physics-validation;interface;gaussian;diagnostic"
    TIMEOUT 180 RUN_SERIAL TRUE
    DEPENDS "feebl-adaptive-composite-poisson-unit;feebl-adaptive-composite-coupling-unit")

# Strict scientific qualification of the experimental composite solver.
# Failed numerical gates remain test failures; WILL_FAIL would conceal them.
add_test(NAME feebl-adaptive-v3-stage4-strict-physics-qualification
         COMMAND feebl_adaptive_composite_physics_validation --require-stage4)
set_tests_properties(
    feebl-adaptive-v3-stage4-strict-physics-qualification PROPERTIES
    LABELS "adaptive-grid;mode4;v3;stage4;physics-qualification;strict;release"
    TIMEOUT 180 RUN_SERIAL TRUE
    DEPENDS "feebl-adaptive-composite-physics-validation")

add_executable(feebl_adaptive_resolution_controller_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_resolution_controller.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveResolutionControllerCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/uniformWindowPlanCPP.cc)
target_include_directories(feebl_adaptive_resolution_controller_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_adaptive_resolution_controller_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_resolution_controller_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
if(FFTW3)
  if(FFTW_FOUND)
    target_link_libraries(feebl_adaptive_resolution_controller_unit PRIVATE
                          ${FFTW_LIBRARIES})
  else()
    target_link_libraries(feebl_adaptive_resolution_controller_unit PRIVATE
                          fftw3 m)
  endif()
endif()
add_test(NAME feebl-adaptive-resolution-controller-unit
         COMMAND feebl_adaptive_resolution_controller_unit)
set_tests_properties(feebl-adaptive-resolution-controller-unit PROPERTIES
    LABELS "adaptive-grid;mode2;mode3;controller;richardson;amr;unit"
    TIMEOUT 120)

# Pure fail-closed certificate for the opt-in terminal two-patch Mode-3 path.
# Manufactured cases cover localized smooth convergence, nonlocal support,
# stochastic/noncontracting sequences, boundary/tail failures and exact
# deterministic replay of the scalar decision.
add_executable(feebl_adaptive_terminal_patch_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_terminal_patch.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveTerminalPatchCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveResolutionControllerCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/uniformWindowPlanCPP.cc)
target_include_directories(feebl_adaptive_terminal_patch_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_adaptive_terminal_patch_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_terminal_patch_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
if(FFTW3)
  if(FFTW_FOUND)
    target_link_libraries(feebl_adaptive_terminal_patch_unit PRIVATE
                          ${FFTW_LIBRARIES})
  else()
    target_link_libraries(feebl_adaptive_terminal_patch_unit PRIVATE
                          fftw3 m)
  endif()
endif()
add_test(NAME feebl-adaptive-terminal-patch-unit
         COMMAND feebl_adaptive_terminal_patch_unit)
set_tests_properties(feebl-adaptive-terminal-patch-unit PROPERTIES
    LABELS "adaptive-grid;mode3;terminal-patch;manufactured;failure;determinism;unit"
    TIMEOUT 120)

# Weighted, lineage-aware projected field-source core.  This is deliberately
# independent of the monolithic simulation so conservation, exact CIC
# projection, many-root ancestry and clone-support invariance are exercised
# under the strict floating-point contract.
add_executable(feebl_adaptive_macroparticle_core_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/test_adaptive_macroparticle_core.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveMacroparticleCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveResolutionControllerCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/uniformWindowPlanCPP.cc)
target_include_directories(feebl_adaptive_macroparticle_core_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_adaptive_macroparticle_core_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_macroparticle_core_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
if(FFTW3)
  if(FFTW_FOUND)
    target_link_libraries(feebl_adaptive_macroparticle_core_unit PRIVATE
                          ${FFTW_LIBRARIES})
  else()
    target_link_libraries(feebl_adaptive_macroparticle_core_unit PRIVATE
                          fftw3 m)
  endif()
endif()
add_test(NAME feebl-adaptive-macroparticle-core-unit
         COMMAND feebl_adaptive_macroparticle_core_unit)
set_tests_properties(feebl-adaptive-macroparticle-core-unit PROPERTIES
    LABELS "adaptive-particles;field-source;weights;ancestry;unit"
    TIMEOUT 120)

add_executable(feebl_adaptive_macroparticle_controller_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/test_adaptive_macroparticle_controller.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveMacroparticleControllerCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveMacroparticleCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveResolutionControllerCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/uniformWindowPlanCPP.cc)
target_include_directories(feebl_adaptive_macroparticle_controller_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_adaptive_macroparticle_controller_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_macroparticle_controller_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
if(FFTW3)
  if(FFTW_FOUND)
    target_link_libraries(feebl_adaptive_macroparticle_controller_unit PRIVATE
                          ${FFTW_LIBRARIES})
  else()
    target_link_libraries(feebl_adaptive_macroparticle_controller_unit PRIVATE
                          fftw3 m)
  endif()
endif()
add_test(NAME feebl-adaptive-macroparticle-controller-unit
         COMMAND feebl_adaptive_macroparticle_controller_unit)
set_tests_properties(feebl-adaptive-macroparticle-controller-unit PROPERTIES
    LABELS "adaptive-particles;controller;error-budget;unit"
    TIMEOUT 120)

# The external replay planner and standalone algorithm comparator are Python
# verification artifacts.  Register them when an interpreter is available so
# their deterministic/fail-closed contracts are part of the normal inventory,
# while keeping Python optional for building the FeeBL executable itself.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME feebl-production-input-admission
      COMMAND ${Python3_EXECUTABLE} -B
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/input/test_production_cards.py
      --feebl ${binaryname}
      --output ${CMAKE_CURRENT_BINARY_DIR}/production-input-admission)
  set_tests_properties(feebl-production-input-admission PROPERTIES
      LABELS "parser;nonfinite;production-flags;rng" TIMEOUT 240 RUN_SERIAL TRUE)
endif()
if(Python3_Interpreter_FOUND)
  # Freeze the direct production link closure as configured.  The G0 audit
  # combines it with an ELF defined-symbol scan, so standalone composite
  # libraries cannot be mistaken for production integration.
  get_target_property(FEEBL_V3_G0_DIRECT_LINK_LIBRARIES feebl LINK_LIBRARIES)
  string(REPLACE ";" "\n" FEEBL_V3_G0_DIRECT_LINK_TEXT
         "${FEEBL_V3_G0_DIRECT_LINK_LIBRARIES}")
  set(FEEBL_V3_G0_DIRECT_LINK_RECEIPT
      "${CMAKE_CURRENT_BINARY_DIR}/feebl_v3_g0_direct_link_libraries.txt")
  file(WRITE "${FEEBL_V3_G0_DIRECT_LINK_RECEIPT}"
       "${FEEBL_V3_G0_DIRECT_LINK_TEXT}\n")

  add_test(NAME feebl-adaptive-v3-g0-baseline-blocker-audit
           COMMAND ${Python3_EXECUTABLE} -B
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_v3_g0_baseline_blockers.py
                   --validator $<TARGET_FILE:feebl_adaptive_composite_physics_validation>
                   --feebl $<TARGET_FILE:feebl>
                   --source-root ${CMAKE_SOURCE_DIR}
                   --link-libraries-receipt ${FEEBL_V3_G0_DIRECT_LINK_RECEIPT})
  set_tests_properties(feebl-adaptive-v3-g0-baseline-blocker-audit PROPERTIES
      LABELS "adaptive-grid;v3;g0;baseline;audit;failure;no-fallback;reproducibility"
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      TIMEOUT 600 RUN_SERIAL TRUE
      DEPENDS "feebl-adaptive-composite-physics-validation")

  add_test(NAME feebl-adaptive-root-replay-unit
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/test_adaptive_root_replay.py)
  set_tests_properties(feebl-adaptive-root-replay-unit PROPERTIES
      LABELS "adaptive-particles;roots;replay;statistics;unit;failure;reproducibility"
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      TIMEOUT 120)

  add_test(NAME feebl-coupled-grid-root-campaign-unit
           COMMAND ${Python3_EXECUTABLE} -B
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/test_coupled_grid_root_campaign.py)
  set_tests_properties(feebl-coupled-grid-root-campaign-unit PROPERTIES
      LABELS "adaptive-particles;grid;roots;campaign;statistics;unit;failure;reproducibility"
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      TIMEOUT 180 RUN_SERIAL TRUE)

  add_test(NAME feebl-adaptive-resampling-prototypes-unit
           COMMAND ${Python3_EXECUTABLE}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/test_compare_algorithms.py)
  set_tests_properties(feebl-adaptive-resampling-prototypes-unit PROPERTIES
      LABELS "adaptive-particles;prototype;resampling;conservation;unit;failure;reproducibility"
      TIMEOUT 120)
endif()

add_test(NAME feebl-adaptive-macroparticle-card-contract-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_particles/run_card_contract_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-macroparticle-card-validation)
set_tests_properties(feebl-adaptive-macroparticle-card-contract-validation
    PROPERTIES
    LABELS "adaptive-particles;parser;capability;integration;failure;no-fallback;reproducibility"
    TIMEOUT 300 RUN_SERIAL TRUE
    DEPENDS "feebl-adaptive-macroparticle-core-unit;feebl-adaptive-macroparticle-controller-unit")

add_executable(feebl_adaptive_runtime_safety_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_runtime_safety.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/wavefrontFieldValidationCPP.cc)
target_include_directories(feebl_adaptive_runtime_safety_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_adaptive_runtime_safety_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_runtime_safety_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-adaptive-runtime-safety-unit
         COMMAND feebl_adaptive_runtime_safety_unit)
set_tests_properties(feebl-adaptive-runtime-safety-unit PROPERTIES
    LABELS "adaptive-grid;mode2;mode3;runtime-safety;unit" TIMEOUT 120)

# Versioned deterministic schedule serialization, checksums, atomic save,
# strict replay, malformed-input rejection, and canonical encounter ordering.
add_executable(feebl_adaptive_grid_schedule_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/test_adaptive_grid_schedule.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/adaptiveGridScheduleCPP.cc)
target_include_directories(feebl_adaptive_grid_schedule_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_adaptive_grid_schedule_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_adaptive_grid_schedule_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-adaptive-grid-schedule-unit
         COMMAND feebl_adaptive_grid_schedule_unit)
set_tests_properties(feebl-adaptive-grid-schedule-unit PROPERTIES
    LABELS "adaptive-grid;mode1;schedule;unit" TIMEOUT 120)

# Independent direct-Green/production-FFT oracle over randomized translations,
# aspect ratios, boundaries, CIC closure, symmetry and binding failures.  The
# wrapper deliberately compiles the production translation units with the
# frozen no-reassociation contract rather than reimplementing their kernels.
add_test(NAME feebl-mode1-field-equivalence
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/build_and_run_cpu_mode1_oracle.sh
                 ${CMAKE_BINARY_DIR}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-grid-field-oracle)
set_tests_properties(feebl-mode1-field-equivalence PROPERTIES
    LABELS "adaptive-grid;mode1;cpu;oracle;property;manufactured;binding"
    TIMEOUT 300 RUN_SERIAL TRUE
    DEPENDS "feebl-uniform-window-plan-unit")

# Sequential process-level contract: parser omission, full/cropped identity,
# save/replay, corrupt/incompatible schedules, caps, unsupported Mode 3 and
# no-artifact fail-closed behavior.
add_test(NAME feebl-mode1-process-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/run_mode1_process_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-grid-process-validation)
set_tests_properties(feebl-mode1-process-validation PROPERTIES
    LABELS "adaptive-grid;mode0;mode1;parser;schedule;integration;failure;no-fallback;reproducibility"
    TIMEOUT 300 RUN_SERIAL TRUE
    DEPENDS "feebl-adaptive-grid-schedule-unit;feebl-mode1-field-equivalence")

# Mode-2/3 executable card contract and invalid-input fail-closed checks.
add_test(NAME feebl-mode23-card-contract-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/run_modes23_card_contract_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-grid-mode23-card-validation)
set_tests_properties(feebl-mode23-card-contract-validation PROPERTIES
    LABELS "adaptive-grid;mode2;mode3;parser;configuration;integration;failure;no-fallback"
    # The reference CPU card suite takes up to 230 s on the documented host. Keep
    # measured margin so this remains an assertion test rather than a timing
    # race; performance acceptance is recorded by separate benchmarks.
    TIMEOUT 600 RUN_SERIAL TRUE)

# The non-aborting Mode-2/3 path is deliberately isolated behind an explicit
# unqualified-pilot opt-in.  Exercise both terminal uniform publications and
# prove that every incomplete or contradictory opt-in remains fail-closed.
add_test(NAME feebl-mode23-pilot-diagnostics-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/run_modes23_pilot_diagnostics_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-grid-mode23-pilot-diagnostics-validation)
set_tests_properties(feebl-mode23-pilot-diagnostics-validation PROPERTIES
    LABELS "adaptive-grid;mode2;mode3;pilot;diagnostics;unqualified;failure;no-fallback"
    TIMEOUT 120 RUN_SERIAL TRUE)

# Reproducible Mode-2/3 mechanism/operator validation.  This checks routing,
# changed physical spacing, controller receipts, patch/promotion invariants,
# and fail-closed behavior.  It is deliberately not a 1--3% collider-physics
# qualification; that requires the documented analytic and statistical
# convergence programme.
add_test(NAME feebl-mode23-physics-validation
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/adaptive_grid/run_modes23_physics_validation.sh
                 ${binaryname}
                 ${CMAKE_CURRENT_BINARY_DIR}/adaptive-grid-mode23-physics-validation)
set_tests_properties(feebl-mode23-physics-validation PROPERTIES
    LABELS "adaptive-grid;mode2;mode3;mechanism;operator;integration;reference"
    TIMEOUT 180 RUN_SERIAL TRUE
    DEPENDS "feebl-mode23-card-contract-validation")

if(TARGET feebl_gpu_adaptive_wave_batch_test)
  add_test(NAME feebl-gpu-adaptive-wave-batch-unit
           COMMAND feebl_gpu_adaptive_wave_batch_test)
  set_tests_properties(feebl-gpu-adaptive-wave-batch-unit PROPERTIES
      LABELS "adaptive-grid;gpu;mode5;cuda;unit;binding;no-fallback"
      SKIP_RETURN_CODE 77 TIMEOUT 240 RUN_SERIAL TRUE)
endif()

# The target is available in both CPU/stub and CUDA configurations.  It links
# only the retained production GENERAL_GRID field/domain sections and checks
# the exact host arithmetic contract used by the resident-GPU gather oracle.
if(TARGET feebl_grid_field_contract_test)
  add_test(NAME feebl-grid-field-contract-unit
           COMMAND feebl_grid_field_contract_test)
  set_tests_properties(feebl-grid-field-contract-unit PROPERTIES
      LABELS "gpu2;grid;field;unit;production-oracle" TIMEOUT 120)
endif()

# Dense-state/CSR/encounter-DAG foundation for the replacement engine.  It is
# a focused unit target, not a production-path integration test.
add_executable(feebl_modern_core_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_core/test_modern_core.cc)
target_link_libraries(feebl_modern_core_unit PRIVATE feebl_modern_core)
target_compile_features(feebl_modern_core_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_core_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-modern-core-unit COMMAND feebl_modern_core_unit)
set_tests_properties(feebl-modern-core-unit PROPERTIES
    LABELS "modern-core;unit" TIMEOUT 120)

add_executable(feebl_modern_physics_plan_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_core/test_physics_plan.cc)
target_link_libraries(feebl_modern_physics_plan_unit PRIVATE feebl_modern_core)
target_compile_features(feebl_modern_physics_plan_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_physics_plan_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-modern-physics-plan-unit
         COMMAND feebl_modern_physics_plan_unit)
set_tests_properties(feebl-modern-physics-plan-unit PROPERTIES
    LABELS "modern-core;physics-plan;unit" TIMEOUT 120)

add_executable(feebl_modern_photon_lumi_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_photon_lumi/test_modern_photon_lumi.cc)
target_link_libraries(feebl_modern_photon_lumi_unit PRIVATE feebl_modern_core)
target_compile_features(feebl_modern_photon_lumi_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_photon_lumi_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-modern-photon-lumi-unit
         COMMAND feebl_modern_photon_lumi_unit)
set_tests_properties(feebl-modern-photon-lumi-unit PROPERTIES
    LABELS "modern-core;photon-luminosity;unit" TIMEOUT 120)

# Transitional production fast path: direct legacy Cartesian oracle versus
# the strictly gated analytic factorization used by GRID::photon_lumi.
add_executable(feebl_legacy_photon_lumi_fast_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_photon_lumi_fast/test_legacy_photon_lumi_fast.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/legacyPhotonLumiFastCPP.cc)
target_include_directories(feebl_legacy_photon_lumi_fast_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_legacy_photon_lumi_fast_unit PRIVATE cxx_std_11)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_legacy_photon_lumi_fast_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-legacy-photon-lumi-fast-unit
         COMMAND feebl_legacy_photon_lumi_fast_unit)
set_tests_properties(feebl-legacy-photon-lumi-fast-unit PROPERTIES
    LABELS "legacy;photon-luminosity;unit" TIMEOUT 120)
add_executable(feebl_legacy_photon_threshold_production_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_photon_lumi_fast/test_production_threshold.cc)
target_include_directories(feebl_legacy_photon_threshold_production_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_legacy_photon_threshold_production_unit PRIVATE
                        cxx_std_11)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_legacy_photon_threshold_production_unit PRIVATE
                         -ffast-math)
endif()
add_test(NAME feebl-legacy-photon-threshold-production-unit
         COMMAND feebl_legacy_photon_threshold_production_unit)
set_tests_properties(feebl-legacy-photon-threshold-production-unit PROPERTIES
    LABELS "legacy;photon-luminosity;unit;production-flags" TIMEOUT 120)
add_test(NAME feebl-legacy-photon-lumi-fast-e2e
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_photon_lumi_fast/run_e2e.sh
                 ${binaryname})
set_tests_properties(feebl-legacy-photon-lumi-fast-e2e PROPERTIES
    LABELS "legacy;photon-luminosity;integration;reproducibility"
    TIMEOUT 180 RUN_SERIAL TRUE)

add_test(NAME feebl-legacy-electron-lumi-precompute-e2e
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_electron_lumi_production/run_e2e.sh
                 ${binaryname})
set_tests_properties(feebl-legacy-electron-lumi-precompute-e2e PROPERTIES
    LABELS "legacy;electron-luminosity;integration;reproducibility"
    TIMEOUT 180 RUN_SERIAL TRUE)

add_executable(feebl_electron_lumi_sampler_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_sampler/test_electron_lumi_sampler.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/electronLumiSamplerCPP.cc)
target_include_directories(feebl_electron_lumi_sampler_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_electron_lumi_sampler_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_electron_lumi_sampler_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-electron-lumi-sampler-unit
         COMMAND feebl_electron_lumi_sampler_unit)
set_tests_properties(feebl-electron-lumi-sampler-unit PROPERTIES
    LABELS "electron-luminosity;counter-rng;weighted-reservoir;unit"
    TIMEOUT 120)

add_executable(feebl_electron_lumi_sampling2_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_sampling2/test_fast_bernoulli.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/electronLumiSamplerCPP.cc)
target_include_directories(feebl_electron_lumi_sampling2_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_electron_lumi_sampling2_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_electron_lumi_sampling2_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-electron-lumi-sampling2-unit
         COMMAND feebl_electron_lumi_sampling2_unit)
set_tests_properties(feebl-electron-lumi-sampling2-unit PROPERTIES
    LABELS "electron-luminosity;counter-rng;bernoulli;unit"
    TIMEOUT 120)

add_executable(feebl_electron_lumi_bernoulli_skip_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_bernoulli_skip/test_electron_lumi_bernoulli_skip.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/electronLumiBernoulliSkipCPP.cc)
target_include_directories(feebl_electron_lumi_bernoulli_skip_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_electron_lumi_bernoulli_skip_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_electron_lumi_bernoulli_skip_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-electron-lumi-bernoulli-skip-unit
         COMMAND feebl_electron_lumi_bernoulli_skip_unit)
set_tests_properties(feebl-electron-lumi-bernoulli-skip-unit PROPERTIES
    LABELS "electron-luminosity;counter-rng;geometric-skip;statistics;unit"
    TIMEOUT 120)

# The hot luminosity output path streams records directly rather than creating
# an ostringstream per event.  Compare it byte-for-byte with the retained
# string-returning representation for base/e-e records, legacy/deterministic
# heaps, native output, and the historical final blank line.
add_executable(feebl_lumi_direct_stream_unit
               lumi_direct_stream_unit.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/lumiCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/electronLumiSamplerCPP.cc)
target_include_directories(feebl_lumi_direct_stream_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_lumi_direct_stream_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_lumi_direct_stream_unit PRIVATE
                         -fno-fast-math -ffp-contract=off
                         -ffunction-sections -fdata-sections)
  target_link_options(feebl_lumi_direct_stream_unit PRIVATE
                      -Wl,--gc-sections)
endif()
add_test(NAME feebl-lumi-direct-stream-unit
         COMMAND feebl_lumi_direct_stream_unit
                 ${CMAKE_CURRENT_BINARY_DIR}/lumi-direct-stream-regression)
set_tests_properties(feebl-lumi-direct-stream-unit PROPERTIES
    LABELS "electron-luminosity;serialization;native-output;reproducibility;unit"
    TIMEOUT 120 RUN_SERIAL TRUE)

if(FEEBL_ENABLE_OPENMP)
  add_test(NAME feebl-electron-lumi-mode3-e2e
           COMMAND bash
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/electron_lumi_mode3/run_e2e.sh
                   ${binaryname})
  set_tests_properties(feebl-electron-lumi-mode3-e2e PROPERTIES
      LABELS "electron-luminosity;blocked;openmp;counter-rng;integration"
      TIMEOUT 240 RUN_SERIAL TRUE)
else()
  message(STATUS
      "Skipping feebl-electron-lumi-mode3-e2e: it requires FEEBL_ENABLE_OPENMP")
endif()

# Production PHOTON_BEAM append/accounting contract.  The test drives the real
# per-slice vectors under randomized dynamic OpenMP schedules and requires the
# aggregate bits, stable IDs, and dump-selection signatures to match at
# 1/2/4/8 threads.  The same target remains a serial contract test in builds
# configured without OpenMP.
add_executable(feebl_photon_accounting_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/photon_accounting/test_photon_accounting.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particleBeamCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc)
target_include_directories(feebl_photon_accounting_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_photon_accounting_unit PRIVATE cxx_std_17)
target_link_libraries(feebl_photon_accounting_unit PRIVATE Threads::Threads)
if(FEEBL_ENABLE_OPENMP)
  target_link_libraries(feebl_photon_accounting_unit PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_photon_accounting_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-photon-accounting-unit
         COMMAND feebl_photon_accounting_unit)
set_tests_properties(feebl-photon-accounting-unit PROPERTIES
    LABELS "legacy;photon;openmp;determinism;unit" TIMEOUT 120)

# The independent source-level oracle compiles strict optimized and ASan/UBSan
# executables in a temporary directory. This exercises its randomized and edge
# cases without adding sanitizer flags to FeeBL itself. The runner uses
# GCC/Clang sanitizer options, so register it only for those toolchains.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_test(NAME feebl-legacy-electron-lumi-precompute-oracle
           COMMAND ${CMAKE_COMMAND} -E env
                   "CXX=${CMAKE_CXX_COMPILER}"
                   bash
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_electron_lumi_precompute/run.sh)
  set_tests_properties(feebl-legacy-electron-lumi-precompute-oracle PROPERTIES
      LABELS "legacy;electron-luminosity;unit;oracle;sanitizer"
      TIMEOUT 240 RUN_SERIAL TRUE)
endif()

# Independent scalar oracle for electron_lumi_precompute=2.  The runner builds
# both strict optimized and ASan/UBSan variants in a temporary directory; the
# production executable retains its ordinary Release flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_test(NAME feebl-legacy-electron-lumi-blocked-oracle
           COMMAND ${CMAKE_COMMAND} -E env
                   "CXX=${CMAKE_CXX_COMPILER}"
                   bash
                   ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_electron_lumi_blocked/run.sh)
  set_tests_properties(feebl-legacy-electron-lumi-blocked-oracle PROPERTIES
      LABELS "legacy;electron-luminosity;blocked;simd;oracle;sanitizer"
      TIMEOUT 300 RUN_SERIAL TRUE)
endif()

# Audited, read-only boundary from legacy PARTICLE_BEAM slices into dense
# modern BeamSoA storage.  Legacy implementation units below construct real
# generated particles; modernLegacyAdapterCPP itself lives in modern_core.
add_executable(feebl_modern_legacy_adapter_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_adapter/test_modern_legacy_adapter.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particleBeamCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc)
target_link_libraries(feebl_modern_legacy_adapter_unit PRIVATE
                      feebl_modern_core)
target_include_directories(feebl_modern_legacy_adapter_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_modern_legacy_adapter_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_legacy_adapter_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-modern-legacy-adapter-unit
         COMMAND feebl_modern_legacy_adapter_unit)
set_tests_properties(feebl-modern-legacy-adapter-unit PROPERTIES
    LABELS "modern-core;legacy-adapter;unit" TIMEOUT 120)

add_executable(feebl_modern_legacy_writeback_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_adapter/test_modern_legacy_writeback.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particleBeamCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc)
target_link_libraries(feebl_modern_legacy_writeback_unit PRIVATE
                      feebl_modern_core)
target_include_directories(feebl_modern_legacy_writeback_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_modern_legacy_writeback_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_legacy_writeback_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-modern-legacy-writeback-unit
         COMMAND feebl_modern_legacy_writeback_unit)
set_tests_properties(feebl-modern-legacy-writeback-unit PROPERTIES
    LABELS "modern-core;legacy-adapter;transaction;unit" TIMEOUT 120)

add_executable(feebl_modern_collision_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_collision/test_modern_collision.cc)
target_link_libraries(feebl_modern_collision_unit PRIVATE feebl_modern_core)
target_compile_features(feebl_modern_collision_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_collision_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-modern-collision-unit
         COMMAND feebl_modern_collision_unit)
set_tests_properties(feebl-modern-collision-unit PROPERTIES
    LABELS "modern-core;collision;luminosity;unit" TIMEOUT 120)

add_executable(feebl_modern_cpu_executor_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_cpu_executor/test_modern_cpu_executor.cc)
target_link_libraries(feebl_modern_cpu_executor_unit PRIVATE
                      feebl_modern_core Threads::Threads)
target_compile_features(feebl_modern_cpu_executor_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_cpu_executor_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-modern-cpu-executor-unit
         COMMAND feebl_modern_cpu_executor_unit)
set_tests_properties(feebl-modern-cpu-executor-unit PROPERTIES
    LABELS "modern-core;cpu;executor;unit" TIMEOUT 120)

# Fail-closed bridge for future same-wave host-physics concurrency.  The unit
# exercises the only admitted first tier (orbit/BMT plus fine luminosity),
# exhaustive shared-state rejection, typed lanes, deterministic merge,
# worker-count identity, telemetry, and transactional failures.  There is no
# production dispatch until the documented legacy GRID/RNG/output gates exist.
add_executable(feebl_legacy_host_wave_concurrency_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_host_wave_concurrency/test_legacy_host_wave_concurrency.cc)
target_link_libraries(feebl_legacy_host_wave_concurrency_unit PRIVATE
                      feebl_modern_core Threads::Threads)
target_compile_features(feebl_legacy_host_wave_concurrency_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_legacy_host_wave_concurrency_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-legacy-host-wave-concurrency-unit
         COMMAND feebl_legacy_host_wave_concurrency_unit)
set_tests_properties(feebl-legacy-host-wave-concurrency-unit PROPERTIES
    LABELS "legacy;wavefront;host-physics;transaction;unit" TIMEOUT 120)

# Production-facing ordered traversal of EncounterDAG.  This target proves
# the FCC-Z 160x160 geometry, canonical strict visitation, bounded launch
# descriptors, and the explicit mutable-state gate on same-wave concurrency.
add_executable(feebl_legacy_wavefront_scheduler_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_wavefront_scheduler/test_legacy_wavefront_scheduler.cc)
target_link_libraries(feebl_legacy_wavefront_scheduler_unit PRIVATE
                      feebl_modern_core)
target_compile_features(feebl_legacy_wavefront_scheduler_unit PRIVATE
                        cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_legacy_wavefront_scheduler_unit PRIVATE
                         -fno-fast-math)
endif()
add_test(NAME feebl-legacy-wavefront-scheduler-unit
         COMMAND feebl_legacy_wavefront_scheduler_unit)
set_tests_properties(feebl-legacy-wavefront-scheduler-unit PROPERTIES
    LABELS "legacy;wavefront;encounter-dag;unit" TIMEOUT 120)
add_test(NAME feebl-legacy-wavefront-scheduler-e2e
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/legacy_wavefront_scheduler/run_e2e.sh
                 ${binaryname})
set_tests_properties(feebl-legacy-wavefront-scheduler-e2e PROPERTIES
    LABELS "legacy;wavefront;encounter-dag;integration;reproducibility"
    TIMEOUT 300 RUN_SERIAL TRUE)

add_test(NAME feebl-sparse-cells-unit
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/sparse_cells/run.sh)
set_tests_properties(feebl-sparse-cells-unit PROPERTIES
    LABELS "legacy;sparse-cells;unit;sanitizer" TIMEOUT 300 RUN_SERIAL TRUE)

add_executable(feebl_sparse_direct_field_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/sparse_direct_field/test_sparse_direct_field.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/sparseDirectFieldCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/fieldCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/fourierCPP.cc)
target_include_directories(feebl_sparse_direct_field_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_sparse_direct_field_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_sparse_direct_field_unit PRIVATE
                         -fno-fast-math -ffp-contract=off)
endif()
if(FFTW2)
  if(FFTW2_FOUND)
    target_link_libraries(feebl_sparse_direct_field_unit PRIVATE
                          ${FFTW2_LIBRARIES})
  else()
    target_link_libraries(feebl_sparse_direct_field_unit PRIVATE fftw m)
  endif()
elseif(FFTW3)
  if(FFTW_FOUND)
    target_link_libraries(feebl_sparse_direct_field_unit PRIVATE
                          ${FFTW_LIBRARIES})
  else()
    target_link_libraries(feebl_sparse_direct_field_unit PRIVATE fftw3 m)
  endif()
endif()
add_test(NAME feebl-sparse-direct-field-unit
         COMMAND feebl_sparse_direct_field_unit)
set_tests_properties(feebl-sparse-direct-field-unit PROPERTIES
    LABELS "legacy;wavefront;sparse-direct;field;unit" TIMEOUT 120)
add_test(NAME feebl-sparse-direct-field-e2e
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/sparse_direct_field/run_e2e.sh
                 ${binaryname})
set_tests_properties(feebl-sparse-direct-field-e2e PROPERTIES
    LABELS "legacy;wavefront;sparse-direct;field;integration;reproducibility"
    TIMEOUT 300 RUN_SERIAL TRUE)

add_executable(feebl_modern_photon_soa_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern_photon_soa/test_modern_photon_soa.cc)
target_link_libraries(feebl_modern_photon_soa_unit PRIVATE feebl_modern_core)
target_compile_features(feebl_modern_photon_soa_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_modern_photon_soa_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-modern-photon-soa-unit
         COMMAND feebl_modern_photon_soa_unit)
set_tests_properties(feebl-modern-photon-soa-unit PROPERTIES
    LABELS "modern-core;photon;soa;unit" TIMEOUT 120)

# Schedule-independent physical random-identity schema.  This remains a
# storage-free unit so CPU CI exercises the same header used by CUDA kernels.
add_executable(feebl_counter_rng_v2_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/rng/test_counter_rng_v2.cc)
target_include_directories(feebl_counter_rng_v2_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_counter_rng_v2_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_counter_rng_v2_unit PRIVATE -fno-fast-math)
endif()
add_test(NAME feebl-counter-rng-v2-unit COMMAND feebl_counter_rng_v2_unit)
set_tests_properties(feebl-counter-rng-v2-unit PROPERTIES
    LABELS "modern-core;unit;rng" TIMEOUT 120)

# Worker-lane sufficient-statistic reduction and the adapter that presents a
# physical-key CounterRngV2 stream through the legacy RNDM interface.  Link the
# production implementations, not copies in the test.  resultsCPP.cc is a
# monolithic legacy translation unit, so unused sections are discarded to keep
# this target's dependency boundary focused.
add_executable(feebl_reduction_rng_unit
               ${CMAKE_CURRENT_SOURCE_DIR}/../examples/reduction_rng/test_reduction_rng.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc)
target_include_directories(feebl_reduction_rng_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_features(feebl_reduction_rng_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_reduction_rng_unit PRIVATE
                         -fno-fast-math -ffp-contract=off
                         -ffunction-sections -fdata-sections)
  target_link_options(feebl_reduction_rng_unit PRIVATE -Wl,--gc-sections)
endif()
add_test(NAME feebl-reduction-rng-unit COMMAND feebl_reduction_rng_unit)
set_tests_properties(feebl-reduction-rng-unit PROPERTIES
    LABELS "collision-reduction;pair-spectrum;counter-rng;legacy-rng;unit"
    TIMEOUT 120)

# Backend-state unit for GPU2.  Compile-time availability is simulated here so
# the configuration/provenance state machine is testable on CPU-only CI; no
# CUDA API is called and this is not a device-correctness test.
add_executable(feebl_gpu2_backend_unit
               gpu2_backend_unit.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/../src/parallelBackendCPP.cc)
target_include_directories(feebl_gpu2_backend_unit PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           ${CMAKE_BINARY_DIR}/src)
target_compile_definitions(feebl_gpu2_backend_unit PRIVATE
                           FEEBL_HAVE_CUDA=1 FEEBL_HAVE_GPU2=1)
target_link_libraries(feebl_gpu2_backend_unit Threads::Threads)
if(FEEBL_ENABLE_OPENMP)
  target_compile_definitions(feebl_gpu2_backend_unit PRIVATE
                             FEEBL_HAVE_OPENMP=1)
  target_link_libraries(feebl_gpu2_backend_unit OpenMP::OpenMP_CXX)
endif()
add_test(NAME feebl-gpu2-backend-unit COMMAND feebl_gpu2_backend_unit)

add_test(NAME feebl-cpu-backends-end-to-end
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../examples/cpu_backends.sh
                      ${binaryname})
set_tests_properties(feebl-cpu-backends-end-to-end PROPERTIES TIMEOUT 600)

add_test(NAME feebl-cpu-primary-move-e2e
         COMMAND bash
                 ${CMAKE_CURRENT_SOURCE_DIR}/../examples/cpu_primary_move/run_e2e.sh
                 ${binaryname})
set_tests_properties(feebl-cpu-primary-move-e2e PROPERTIES
    LABELS "legacy;cpu;primary-movement;integration;reproducibility"
    TIMEOUT 300 RUN_SERIAL TRUE)

add_test(NAME feebl-backend-interface
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../examples/backend_interface.sh
                      ${binaryname})
set_tests_properties(feebl-backend-interface PROPERTIES TIMEOUT 600)

set(TESTID ${CMAKE_CXX_COMPILER_ID}_${CMAKE_BUILD_TYPE})
set(FEEBL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# CLIC-like regression against the frozen source snapshot. The bundled
# comparator checks every line; a run-only smoke check is insufficient.
add_test(NAME CLIC-default_${TESTID}
         COMMAND ${binaryname} --pos_file=electron.ini default default result.out
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME CLIC-default-compare_${TESTID}
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/compare_numeric.sh
                 ${CMAKE_CURRENT_BINARY_DIR}/result.out
                 ${CMAKE_CURRENT_BINARY_DIR}/result.ref
                 1e-6
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(CLIC-default-compare_${TESTID} PROPERTIES
    DEPENDS CLIC-default_${TESTID})

if(Python3_Interpreter_FOUND)
  set(FEEBL_COMPARISON_PYTHON "${Python3_EXECUTABLE}")
else()
  set(FEEBL_COMPARISON_PYTHON python3)
endif()
add_test(NAME feebl-numeric-comparison-unit
         COMMAND "${FEEBL_COMPARISON_PYTHON}" -B
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_compare_numeric.py)
set_tests_properties(feebl-numeric-comparison-unit PROPERTIES
    LABELS "regression;comparator;unit" TIMEOUT 30)

# Shell integration suites locate the source tree from their own paths.
# The meta-test checks that all wrappers propagate simulation failures.
# Use its full discovery mode, including the refinement wrappers.
add_test(NAME feebl-meta-dead_binary
         COMMAND bash ${FEEBL_ROOT}/examples/meta/dead_binary.sh
         WORKING_DIRECTORY ${FEEBL_ROOT})

set(FEEBL_SUITES
    version_identity.sh
    negative_checks.sh
    native_output.sh
    native_roundtrip.sh
    conventions/run.sh
    canonical/run.sh
    maps/run.sh
    geometry/run.sh
    rng/run.sh
    pairs/run.sh
    spin/run.sh)

foreach(suite ${FEEBL_SUITES})
    string(REPLACE "/" "-" suitename ${suite})
    string(REPLACE ".sh" "" suitename ${suitename})
    add_test(NAME feebl-${suitename}
             COMMAND bash ${FEEBL_ROOT}/examples/${suite}
             WORKING_DIRECTORY ${FEEBL_ROOT})
    set_tests_properties(feebl-${suitename} PROPERTIES
        ENVIRONMENT
          "FEEBL_TEST_BINARY=${binaryname};FEEBL_TEST_BUILD_DIR=${CMAKE_BINARY_DIR}")
endforeach()

# dead_binary copies the tree and runs every other suite against a stub, so it
# must not race them.
set_tests_properties(feebl-meta-dead_binary PROPERTIES RUN_SERIAL TRUE TIMEOUT 3600)
set_tests_properties(feebl-geometry-run PROPERTIES TIMEOUT 3600)
set_tests_properties(feebl-maps-run PROPERTIES TIMEOUT 3600)
set_tests_properties(feebl-spin-run PROPERTIES TIMEOUT 3600)

if(TARGET feebl_gpu_cic_test)
  add_test(NAME feebl-gpu-cic-correctness
           COMMAND feebl_gpu_cic_test --correctness)
  set_tests_properties(feebl-gpu-cic-correctness PROPERTIES
      LABELS "gpu;cuda" TIMEOUT 300)
endif()

if(TARGET feebl_gpu_resident_test)
  add_test(NAME feebl-gpu2-resident-fft-correctness
           COMMAND feebl_gpu_resident_test)
  set_tests_properties(feebl-gpu2-resident-fft-correctness PROPERTIES
      LABELS "gpu;cuda;gpu2" TIMEOUT 300)

  add_test(NAME feebl-gpu2-fields-end-to-end
           COMMAND bash ${FEEBL_ROOT}/examples/gpu/run_gpu2_fields_e2e.sh
                        ${binaryname})
  set_tests_properties(feebl-gpu2-fields-end-to-end PROPERTIES
      LABELS "gpu;cuda;gpu2" TIMEOUT 600 RUN_SERIAL TRUE)

  add_test(NAME feebl-gpu2-primary-end-to-end
           COMMAND bash ${FEEBL_ROOT}/examples/gpu/run_gpu2_primary_e2e.sh
                        ${binaryname})
  set_tests_properties(feebl-gpu2-primary-end-to-end PROPERTIES
      LABELS "gpu;cuda;gpu2;primary;integration;physics;reproducibility"
      TIMEOUT 900 RUN_SERIAL TRUE
      DEPENDS "feebl-gpu2-resident-fft-correctness;feebl-gpu2-fields-end-to-end")
endif()

if(TARGET feebl_gpu_wave_batch_test)
  add_test(NAME feebl-gpu-wave-batch-correctness
           COMMAND feebl_gpu_wave_batch_test)
  set_tests_properties(feebl-gpu-wave-batch-correctness PROPERTIES
      LABELS "gpu;cuda;gpu-mode4;wavefront;fields;physics;reproducibility"
      TIMEOUT 300 RUN_SERIAL TRUE)

  add_test(NAME feebl-gpu-mode4-wave-batch-end-to-end
           COMMAND bash ${FEEBL_ROOT}/examples/gpu/run_gpu_mode4_e2e.sh
                        ${binaryname})
  set_tests_properties(feebl-gpu-mode4-wave-batch-end-to-end PROPERTIES
      LABELS "gpu;cuda;gpu-mode4;wavefront;integration;physics;reproducibility"
      TIMEOUT 600 RUN_SERIAL TRUE
      DEPENDS "feebl-gpu-wave-batch-correctness")
endif()

if(TARGET feebl_gpu_primary_resident_test)
  add_test(NAME feebl-gpu-primary-resident-correctness
           COMMAND feebl_gpu_primary_resident_test)
  set_tests_properties(feebl-gpu-primary-resident-correctness PROPERTIES
      LABELS "gpu;cuda;gpu2;primary;resident;wavefront;physics"
      TIMEOUT 300 RUN_SERIAL TRUE)
endif()

if(TARGET feebl_gpu_primary_modern_adapter_test)
  add_test(NAME feebl-gpu-primary-modern-adapter-correctness
           COMMAND feebl_gpu_primary_modern_adapter_test 0)
  set_tests_properties(feebl-gpu-primary-modern-adapter-correctness PROPERTIES
      LABELS "gpu;cuda;gpu-mode3;primary;resident;physics;reproducibility"
      TIMEOUT 300 RUN_SERIAL TRUE)

  add_test(NAME feebl-gpu-mode3-production-end-to-end
           COMMAND bash ${FEEBL_ROOT}/examples/gpu/run_gpu_mode3_e2e.sh
                        ${binaryname})
  set_tests_properties(feebl-gpu-mode3-production-end-to-end PROPERTIES
      LABELS "gpu;cuda;gpu-mode3;primary;integration;physics;reproducibility"
      TIMEOUT 900 RUN_SERIAL TRUE
      DEPENDS "feebl-gpu-primary-resident-correctness;feebl-gpu-primary-modern-adapter-correctness")
endif()

add_executable(feebl_grouped_events_unit
  ${CMAKE_CURRENT_SOURCE_DIR}/../examples/grouped_events/grouped_events_unit.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/groupedEventsCPP.cc)
target_include_directories(feebl_grouped_events_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_grouped_events_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_grouped_events_unit PRIVATE -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-grouped-events-unit COMMAND feebl_grouped_events_unit)
set_tests_properties(feebl-grouped-events-unit PROPERTIES
  LABELS "event-transport;lhe;qualification;manufactured-fields" TIMEOUT 30)

add_executable(feebl_lhe_roundtrip
  ${CMAKE_CURRENT_SOURCE_DIR}/../examples/grouped_events/grouped_events_roundtrip.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/groupedEventsCPP.cc)
target_include_directories(feebl_lhe_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_lhe_roundtrip PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_lhe_roundtrip PRIVATE -fno-fast-math -ffp-contract=off)
endif()
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_legacy_event_adapter_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/grouped_events/legacy_adapter_unit.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/bhabhaSamplesCPP.cc)
  target_include_directories(feebl_legacy_event_adapter_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_options(feebl_legacy_event_adapter_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  set_target_properties(feebl_legacy_event_adapter_unit PROPERTIES LINK_FLAGS "-Wl,--gc-sections")
  add_test(NAME feebl-legacy-event-adapter-unit
    COMMAND feebl_legacy_event_adapter_unit ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(feebl-legacy-event-adapter-unit PROPERTIES
    LABELS "event-transport;legacy;adapter;regression" TIMEOUT 30)
endif()

# Actual native GUI field sampling, checked against independent analytic
# potentials. No solver, beam generator, RNG, or simulation is invoked.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_collider_fields_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_collider_fields.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gridCPP.cc)
  target_include_directories(feebl_collider_fields_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_collider_fields_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_collider_fields_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections
    # Match the existing narrow production-grid oracle: discarded monolithic
    # sections otherwise retain unrelated RTTI via whole-TU UBSan metadata.
    -fno-sanitize=undefined)
  set_property(TARGET feebl_collider_fields_unit APPEND_STRING PROPERTY
    LINK_FLAGS " -Wl,--gc-sections")
  target_link_libraries(feebl_collider_fields_unit PRIVATE
    feebl_gpu_cic feebl_gpu_resident)
  add_test(NAME feebl-collider-fields-unit COMMAND feebl_collider_fields_unit)
  set_tests_properties(feebl-collider-fields-unit PROPERTIES
    LABELS "gui;field;manufactured;production-sampler;unit" TIMEOUT 60)
endif()

# Geometry-only GUI luminosity partition and native histogram conservation.
add_executable(feebl_gui_spectrum_binning_unit
  ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_spectrum/test_gui_spectrum_binning.cc)
target_include_directories(feebl_gui_spectrum_binning_unit PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(feebl_gui_spectrum_binning_unit PRIVATE cxx_std_17)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(feebl_gui_spectrum_binning_unit PRIVATE
    -fno-fast-math -ffp-contract=off)
endif()
add_test(NAME feebl-gui-spectrum-binning-unit COMMAND feebl_gui_spectrum_binning_unit)
set_tests_properties(feebl-gui-spectrum-binning-unit PROPERTIES
  LABELS "gui;spectrum;partition;unit" TIMEOUT 30)

# Species-separated primary chi uses the real RESULTS merge/reset operations.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_primary_chi_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_diagnostics/test_primary_chi.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc)
  target_include_directories(feebl_primary_chi_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_primary_chi_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_primary_chi_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_primary_chi_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-primary-chi-unit COMMAND feebl_primary_chi_unit)
  set_tests_properties(feebl-primary-chi-unit PROPERTIES
    LABELS "gui;diagnostics;primary-species;unit" TIMEOUT 30)
endif()


# Full-bunch GUI snapshots use private native FFTs and copied primary states.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_snapshot_fields_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_snapshot_fields.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gridCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fieldCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fourierCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  target_include_directories(feebl_snapshot_fields_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_snapshot_fields_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_snapshot_fields_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections
    -fno-sanitize=undefined)
  target_link_options(feebl_snapshot_fields_unit PRIVATE -Wl,--gc-sections)
  target_link_libraries(feebl_snapshot_fields_unit PRIVATE feebl_gpu_cic feebl_gpu_resident)
  if(FFTW2)
    if(FFTW2_FOUND)
      target_link_libraries(feebl_snapshot_fields_unit PRIVATE ${FFTW2_LIBRARIES})
    else()
      target_link_libraries(feebl_snapshot_fields_unit PRIVATE fftw m)
    endif()
  elseif(FFTW3)
    if(FFTW_FOUND)
      target_link_libraries(feebl_snapshot_fields_unit PRIVATE ${FFTW_LIBRARIES})
    else()
      target_link_libraries(feebl_snapshot_fields_unit PRIVATE fftw3 m)
    endif()
  endif()
  add_test(NAME feebl-snapshot-fields-unit COMMAND feebl_snapshot_fields_unit)
  set_tests_properties(feebl-snapshot-fields-unit PROPERTIES
    LABELS "gui;field;snapshot;native-fft;noninterference;unit" TIMEOUT 60)
endif()

# Read-only coverage audit binds real deposited source counts and charge.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_mesh_coverage_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_mesh_coverage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/coverage_production_cic.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/gridCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fieldCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fourierCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  target_include_directories(feebl_mesh_coverage_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_mesh_coverage_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_mesh_coverage_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections
    -fno-sanitize=undefined)
  target_link_options(feebl_mesh_coverage_unit PRIVATE -Wl,--gc-sections)
  target_link_libraries(feebl_mesh_coverage_unit PRIVATE feebl_gpu_cic feebl_gpu_resident)
  if(FFTW2)
    if(FFTW2_FOUND)
      target_link_libraries(feebl_mesh_coverage_unit PRIVATE ${FFTW2_LIBRARIES})
    else()
      target_link_libraries(feebl_mesh_coverage_unit PRIVATE fftw m)
    endif()
  elseif(FFTW3)
    if(FFTW_FOUND)
      target_link_libraries(feebl_mesh_coverage_unit PRIVATE ${FFTW_LIBRARIES})
    else()
      target_link_libraries(feebl_mesh_coverage_unit PRIVATE fftw3 m)
    endif()
  endif()
  add_test(NAME feebl-mesh-coverage-unit COMMAND feebl_mesh_coverage_unit)
  set_tests_properties(feebl-mesh-coverage-unit PROPERTIES
    LABELS "gui;field;mesh-coverage;native-deposition;noninterference;unit" TIMEOUT 60)
endif()

# Match the production CIC translation unit arithmetic in the charge audit oracle.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/coverage_production_cic.cc
  PROPERTIES COMPILE_OPTIONS "-ffast-math;-ffp-contract=fast")


# GUI secondary snapshots read authoritative tracker state after deactivation.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_secondary_frames_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_secondary_frames.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pairsCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  target_include_directories(feebl_secondary_frames_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_secondary_frames_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_secondary_frames_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_secondary_frames_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-secondary-frames-unit COMMAND feebl_secondary_frames_unit)
  set_tests_properties(feebl-secondary-frames-unit PROPERTIES
    LABELS "gui;secondary;snapshot;noninterference;unit" TIMEOUT 30)
endif()

# Exact conditional luminosity observables and the real RESULTS interval/reduction lifecycle.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_peak_diagnostics_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_peak_diagnostics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc)
  target_include_directories(feebl_peak_diagnostics_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_peak_diagnostics_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_peak_diagnostics_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_peak_diagnostics_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-peak-diagnostics-unit COMMAND feebl_peak_diagnostics_unit)
  set_tests_properties(feebl-peak-diagnostics-unit PROPERTIES
    LABELS "gui;diagnostics;peak;noninterference;unit" TIMEOUT 30)
endif()

# Imported event identities and delayed-input consumption are observational.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_imported_trajectories_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/gui_fields/test_imported_trajectories.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pairsCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  target_include_directories(feebl_imported_trajectories_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_imported_trajectories_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_imported_trajectories_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_imported_trajectories_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-imported-trajectories-unit COMMAND feebl_imported_trajectories_unit)
  set_tests_properties(feebl-imported-trajectories-unit PROPERTIES
    LABELS "gui;imported;identity;noninterference;unit" TIMEOUT 30)
endif()

# Correlated spin observables and production RESULTS reduction lifecycle.
if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_spin_correlation_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_spin_correlation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/resultsCPP.cc)
  target_include_directories(feebl_spin_correlation_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_spin_correlation_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_spin_correlation_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_spin_correlation_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-spin-correlation-unit COMMAND feebl_spin_correlation_unit)
  set_tests_properties(feebl-spin-correlation-unit PROPERTIES
    LABELS "spin;diagnostics;correlation;unit" TIMEOUT 30)
endif()

if(UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_executable(feebl_spin_physics_unit
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_spin_physics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  target_include_directories(feebl_spin_physics_unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_spin_physics_unit PRIVATE cxx_std_17)
  target_compile_options(feebl_spin_physics_unit PRIVATE
    -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
  target_link_options(feebl_spin_physics_unit PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-spin-physics-unit COMMAND feebl_spin_physics_unit)
  set_tests_properties(feebl-spin-physics-unit PROPERTIES
    LABELS "spin;physics;unit" TIMEOUT 90)
  get_target_property(spin_physics_sources feebl_spin_physics_unit SOURCES)
  add_executable(feebl_spin_physics_production ${spin_physics_sources})
  target_include_directories(feebl_spin_physics_production PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
  target_compile_features(feebl_spin_physics_production PRIVATE cxx_std_17)
  target_compile_options(feebl_spin_physics_production PRIVATE
    -ffunction-sections -fdata-sections)
  target_link_options(feebl_spin_physics_production PRIVATE -Wl,--gc-sections)
  add_test(NAME feebl-spin-physics-production COMMAND feebl_spin_physics_production)
  set_tests_properties(feebl-spin-physics-production PROPERTIES
    LABELS "spin;physics;production;unit" TIMEOUT 90)

  # Test the actual imported-track field-step methods in both production and
  # strict floating-point configurations; the analytic oracle is independent.
  foreach(spin_variant IN ITEMS production strict)
    add_executable(feebl_compton_kinematics_${spin_variant}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_compton_kinematics.cc)
    target_include_directories(feebl_compton_kinematics_${spin_variant} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_compile_features(feebl_compton_kinematics_${spin_variant} PRIVATE cxx_std_17)
    if(spin_variant STREQUAL "strict")
      target_compile_options(feebl_compton_kinematics_${spin_variant} PRIVATE
        -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME feebl-compton-kinematics-${spin_variant}
      COMMAND feebl_compton_kinematics_${spin_variant})
    set_tests_properties(feebl-compton-kinematics-${spin_variant} PROPERTIES
      LABELS "compton;physics;unit" TIMEOUT 30)
    add_executable(feebl_pair_cross_section_${spin_variant}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_pair_cross_section.cc)
    target_include_directories(feebl_pair_cross_section_${spin_variant} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_compile_features(feebl_pair_cross_section_${spin_variant} PRIVATE cxx_std_17)
    if(spin_variant STREQUAL "strict")
      target_compile_options(feebl_pair_cross_section_${spin_variant} PRIVATE
        -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME feebl-pair-cross-section-${spin_variant}
      COMMAND feebl_pair_cross_section_${spin_variant})
    set_tests_properties(feebl-pair-cross-section-${spin_variant} PROPERTIES
      LABELS "pairs;physics;unit" TIMEOUT 30)
    add_executable(feebl_imported_spin_${spin_variant}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_imported_spin.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
    target_include_directories(feebl_imported_spin_${spin_variant} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
    target_compile_features(feebl_imported_spin_${spin_variant} PRIVATE cxx_std_17)
    target_compile_options(feebl_imported_spin_${spin_variant} PRIVATE
      -ffunction-sections -fdata-sections)
    if(spin_variant STREQUAL "strict")
      target_compile_options(feebl_imported_spin_${spin_variant} PRIVATE
        -fno-fast-math -ffp-contract=off)
    endif()
    target_link_options(feebl_imported_spin_${spin_variant} PRIVATE -Wl,--gc-sections)
    add_test(NAME feebl-imported-spin-${spin_variant}
      COMMAND feebl_imported_spin_${spin_variant})
    set_tests_properties(feebl-imported-spin-${spin_variant} PROPERTIES
      LABELS "spin;physics;imported;unit" TIMEOUT 60)
  endforeach()
endif()

# QED integration contracts exercise the shipped headers under both numerical
# policies. The production variants explicitly retain fast-math, so validation
# cannot pass merely because a test has stronger finite-value semantics.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(qed_variant IN ITEMS reference production)
    foreach(qed_suite IN ITEMS integration photon-domain)
      if(qed_suite STREQUAL "integration")
        set(qed_test_source test_qed_integration.cc)
      else()
        set(qed_test_source test_photon_domain.cc)
      endif()
      string(REPLACE "-" "_" qed_target_suite "${qed_suite}")
      set(qed_target feebl_qed_${qed_target_suite}_${qed_variant})
      add_executable(${qed_target}
        ${CMAKE_CURRENT_SOURCE_DIR}/../examples/qed/${qed_test_source})
      target_include_directories(${qed_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
      target_compile_features(${qed_target} PRIVATE cxx_std_17)
      if(qed_variant STREQUAL "reference")
        target_compile_options(${qed_target} PRIVATE -fno-fast-math -ffp-contract=off)
      else()
        target_compile_options(${qed_target} PRIVATE -ffast-math)
      endif()
      add_test(NAME feebl-qed-${qed_suite}-${qed_variant} COMMAND ${qed_target})
      set_tests_properties(feebl-qed-${qed_suite}-${qed_variant} PROPERTIES
        LABELS "qed;pairs;physics;unit;${qed_variant}" TIMEOUT 60)
    endforeach()
  endforeach()
endif()

# These independent amplitude identities use GCC binary128 reference
# arithmetic. They do not invoke Mathematica or the long collision ensembles.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND UNIX
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  foreach(qed_oracle IN ITEMS tree bh-bw)
    if(qed_oracle STREQUAL "tree")
      set(qed_oracle_source test_tree_engine.cc)
    else()
      set(qed_oracle_source test_bh_bw_engine.cc)
    endif()
    string(REPLACE "-" "_" qed_oracle_target "${qed_oracle}")
    set(qed_target feebl_qed_${qed_oracle_target}_oracle)
    add_executable(${qed_target}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/qed/${qed_oracle_source})
    target_include_directories(${qed_target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
    target_compile_features(${qed_target} PRIVATE cxx_std_17)
    target_compile_options(${qed_target} PRIVATE -fno-fast-math -ffp-contract=off)
    target_link_libraries(${qed_target} PRIVATE quadmath)
    add_test(NAME feebl-qed-${qed_oracle}-oracle COMMAND ${qed_target})
    set_tests_properties(feebl-qed-${qed_oracle}-oracle PROPERTIES
      LABELS "qed;pairs;physics;oracle;binary128" TIMEOUT 180)
  endforeach()
endif()

if(Python3_Interpreter_FOUND)
  add_test(NAME feebl-qed-input-contracts
    COMMAND ${Python3_EXECUTABLE} -B
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/qed/test_qed_input_contracts.py
      --feebl $<TARGET_FILE:feebl>
      --card ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/cases/smoke/acc.dat
      --output ${CMAKE_CURRENT_BINARY_DIR}/qed-input-contracts)
  set_tests_properties(feebl-qed-input-contracts PROPERTIES
    LABELS "qed;pairs;input;production" TIMEOUT 180)
endif()

# Exercise canonical momentum and the actual PAIR_PARTICLE stepping methods.
# Link the same production dependencies as the imported-spin regression; dead
# unrelated sections are discarded rather than replaced with transport stubs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND UNIX)
  set(secondary_transport_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_secondary_transport.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/particlesCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rndmCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tridentCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/physicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/mathematicalTools.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/besselCPP.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/splineCPP.cc)
  foreach(secondary_variant IN ITEMS reference production)
    add_executable(feebl_secondary_transport_${secondary_variant} ${secondary_transport_sources})
    add_executable(feebl_secondary_momentum_${secondary_variant}
      ${CMAKE_CURRENT_SOURCE_DIR}/../examples/spin/test_secondary_momentum.cc)
    foreach(secondary_kind IN ITEMS transport momentum)
      set(secondary_target feebl_secondary_${secondary_kind}_${secondary_variant})
      target_include_directories(${secondary_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_BINARY_DIR}/src)
      target_compile_features(${secondary_target} PRIVATE cxx_std_17)
      target_compile_options(${secondary_target} PRIVATE -ffunction-sections -fdata-sections)
      target_link_options(${secondary_target} PRIVATE -Wl,--gc-sections)
      if(secondary_variant STREQUAL "reference")
        target_compile_options(${secondary_target} PRIVATE -fno-fast-math -ffp-contract=off)
      else()
        target_compile_options(${secondary_target} PRIVATE -ffast-math)
      endif()
      add_test(NAME feebl-secondary-${secondary_kind}-${secondary_variant}
        COMMAND ${secondary_target})
      set_tests_properties(feebl-secondary-${secondary_kind}-${secondary_variant} PROPERTIES
        LABELS "pairs;transport;physics;unit;${secondary_variant}" TIMEOUT 90)
    endforeach()
  endforeach()
endif()
