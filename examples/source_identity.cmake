include("${CMAKE_CURRENT_LIST_DIR}/../cmake/SourceIdentity.cmake")

string(REPEAT "a" 40 commit)
string(REPEAT "B" 64 digest)
foreach(identity IN ITEMS "${commit}" "${commit}-dirty" "sha256:${digest}")
  feebl_validate_source_identity("${identity}" result)
  if(NOT result STREQUAL identity)
    message(FATAL_ERROR "Valid source identity was rejected: ${identity}")
  endif()
endforeach()

foreach(identity IN ITEMS "" "unknown" "${commit}x" "${commit}-dirty-dirty"
    "sha256:${commit}" "sha256:${digest}0" "sha256:${digest}-dirty"
    "${commit}\ninjected" "${commit}\"" "sha256:bad")
  feebl_validate_source_identity("${identity}" result)
  if(NOT result STREQUAL "unknown")
    message(FATAL_ERROR "Invalid source identity was admitted: ${identity}")
  endif()
endforeach()
message(STATUS "Git and SHA256 export source identities validated")
