#!/bin/bash
# Every place that declares a version must agree, and the binary must report it.
#
# configure.ac is the source of truth. CMake's project(VERSION) accepts only
# numeric components, so it carries the numeric part and appends
# FEEBL_VERSION_SUFFIX; this checks the composition, not just the literal.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
fails=0
ok () { printf "  %-52s %s\n" "$2" "$([ "$1" = 1 ] && echo PASS || echo FAIL)"
        [ "$1" = 1 ] || fails=$((fails+1)); }

TRUE=$(sed -n 's/^AC_INIT(\[feebl\],\[\([^]]*\)\].*/\1/p' "$ROOT/configure.ac" | head -1)
[ -n "$TRUE" ] || { echo "  cannot read the version from configure.ac"; exit 1; }
echo "  source of truth (configure.ac): $TRUE"
echo

GOT=$(sed -n "s/^PACKAGE_VERSION='\(.*\)'/\1/p" "$ROOT/configure" | head -1)
[ "$GOT" = "$TRUE" ] && ok 1 "generated configure agrees" \
                     || ok 0 "generated configure agrees (says '$GOT')"

NUM=$(sed -n 's/^project(feebl VERSION \([0-9.]*\)).*/\1/p' "$ROOT/CMakeLists.txt" | head -1)
SUF=$(sed -n 's/^set(FEEBL_VERSION_SUFFIX "\(.*\)").*/\1/p' "$ROOT/CMakeLists.txt" | head -1)
[ "${NUM}${SUF}" = "$TRUE" ] && ok 1 "CMake composes to the same version" \
                             || ok 0 "CMake composes to the same version (got '${NUM}${SUF}')"

grep -q "Version ${TRUE}\." "$ROOT/README.md" && ok 1 "README agrees" || ok 0 "README agrees"

if [ -f "$ROOT/src/config.h" ]; then
  GOT=$(sed -n 's/^#define PACKAGE_VERSION "\(.*\)"/\1/p' "$ROOT/src/config.h" | head -1)
  [ "$GOT" = "$TRUE" ] && ok 1 "src/config.h agrees" \
                       || ok 0 "src/config.h agrees (says '$GOT')"
fi

B="${FEEBL_TEST_BINARY:-$ROOT/src/feebl}"
if [ -x "$B" ]; then
  GOT=$("$B" --version 2>/dev/null | head -1 | awk '{print $2}')
  [ "$GOT" = "$TRUE" ] && ok 1 "the built binary reports it" \
                       || ok 0 "the built binary reports it (says '$GOT')"
else
  ok 0 "the built binary reports it (no binary present)"
fi

# Every DECLARATION of a version in the tree must equal the truth. A blanket
# scan for stale strings would be wrong: "1.2.1" legitimately appears in
# provenance notes about the GUINEA-PIG++ original, and superseded numbers
# belong in CHANGELOG history. Only literal declaration sites are checked, so
# comments and shell-variable templates are excluded.
echo
echo "  scanning every version DECLARATION in the tree:"
PAT="PACKAGE_VERSION='[^']*'|project\(feebl VERSION [0-9.]*\)|AC_INIT\(\[feebl\]"
BAD=$(grep -rns --include='*.ac' --include='configure' --include='*.cmake' \
        --include='CMakeLists.txt' --include='config.h' -E "$PAT" "$ROOT" 2>/dev/null \
      | grep -v '\$PACKAGE_VERSION' \
      | grep -vE ':[0-9]+:[[:space:]]*#' \
      | grep -v "$TRUE" \
      | grep -vE "project\(feebl VERSION ${NUM}\)" || true)
if [ -z "$BAD" ]; then
  ok 1 "every declaration site matches"
else
  ok 0 "every declaration site matches"
  echo "$BAD" | sed 's/^/      /' | head -6
fi

echo
if [ "$fails" -eq 0 ]; then echo "  version identity is consistent"; else
  echo "  $fails FAILURE(S)"; fi
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
