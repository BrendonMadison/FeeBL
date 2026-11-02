#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.71])
AC_INIT([feebl],[0.6.0-dev],[brendon_madison@ku.edu])

EXTERNAL_CFLAGS="$CFLAGS"
EXTERNAL_CXXFLAGS="$CXXFLAGS"

DEFAULT_CFLAGS="-O3"
DEFAULT_CXXFLAGS="-std=gnu++17 -pthread"

AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE

AC_CONFIG_SRCDIR([src/feeblCPP.h])
AC_CONFIG_HEADERS([src/config.h])

FEEBL_SOURCE_ROOT=`cd "$srcdir" 2>/dev/null && pwd -P`
FEEBL_GIT_ROOT=`git -C "$srcdir" rev-parse --show-toplevel 2>/dev/null || true`
if test -n "$FEEBL_GIT_ROOT" ; then
  FEEBL_GIT_ROOT=`cd "$FEEBL_GIT_ROOT" 2>/dev/null && pwd -P`
fi
if test -n "$FEEBL_SOURCE_ROOT" && test "x$FEEBL_GIT_ROOT" = "x$FEEBL_SOURCE_ROOT" ; then
  FEEBL_SOURCE_REVISION=`git -C "$srcdir" rev-parse --verify HEAD 2>/dev/null || true`
else
  FEEBL_SOURCE_REVISION=
fi
if test -n "$FEEBL_SOURCE_REVISION" ; then
  FEEBL_SOURCE_DIRTY=`git -C "$srcdir" status --porcelain --untracked-files=normal --ignore-submodules 2>/dev/null || true`
  if test -n "$FEEBL_SOURCE_DIRTY" ; then
    FEEBL_SOURCE_REVISION="$FEEBL_SOURCE_REVISION-dirty"
  fi
elif test -r "$srcdir/SOURCE_REVISION" ; then
  FEEBL_SOURCE_REVISION=`sed -n '1p' "$srcdir/SOURCE_REVISION"`
  if ! printf '%s\n' "$FEEBL_SOURCE_REVISION" | LC_ALL=C grep -Eq '^([[0-9a-fA-F]]{40}(-dirty)?|sha256:[[0-9a-fA-F]]{64})$' ; then
    FEEBL_SOURCE_REVISION=unknown
  fi
else
  FEEBL_SOURCE_REVISION=unknown
fi
AC_DEFINE_UNQUOTED([FEEBL_BUILD_SOURCE_HASH],["$FEEBL_SOURCE_REVISION"],
                   [Git revision or exported content identity recorded in run provenance])

if test "$prefix" = "NONE" ; then
  prefix="/usr/local"
fi

make_all=feebl

USE_FFTW2=false
USE_FFTW3=false
USE_ROOT=false
USE_OPENMP=false

AC_ARG_ENABLE([fftw2], [AS_HELP_STRING([--enable-fftw2],[enable the fftw2 libraries for field computation (default is no)])], [USE_FFTW2=true], [USE_FFTW2=false])
AC_ARG_ENABLE([fftw3], [AS_HELP_STRING([--enable-fftw3],[enable the fftw3 libraries for field computation (default is no)])], [USE_FFTW3=true], [USE_FFTW3=false])
AC_ARG_ENABLE([root], [AS_HELP_STRING([--enable-root],[enable the ROOT libraries for outputs (default is no)])], [USE_ROOT=true], [USE_ROOT=false])
AC_ARG_ENABLE([openmp], [AS_HELP_STRING([--enable-openmp],[build the experimental OpenMP CPU kernel candidate (default is no)])], [USE_OPENMP=true], [USE_OPENMP=false])


AC_ARG_WITH([fftwdir], [AS_HELP_STRING([--with-fftwdir=DIR],[FFTW home directory (such that `$DIR/lib' and  `$DIR/include' exist)])], [FFTW_HOME="$with_fftwdir"])
AC_ARG_WITH([rootdir], [AS_HELP_STRING([--with-rootdir=DIR],[ROOT home directory (such that `$DIR/lib' and  `$DIR/include' exist)])], [ROOTSYS="$with_rootdir"])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

OPENMP_CXXFLAGS=""
if test "$USE_OPENMP" = "true" ; then
  AC_LANG_PUSH([C++])
  feebl_save_CXXFLAGS="$CXXFLAGS"
  CXXFLAGS="$CXXFLAGS -fopenmp"
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[#include <omp.h>]], [[return omp_get_max_threads() < 1;]])],
    [OPENMP_CXXFLAGS="-fopenmp -DFEEBL_HAVE_OPENMP=1"
     AC_DEFINE([FEEBL_HAVE_OPENMP],[1],[Define when the OpenMP CPU candidate is built])],
    [AC_MSG_ERROR([--enable-openmp was requested, but the C++ compiler cannot link OpenMP])])
  CXXFLAGS="$feebl_save_CXXFLAGS"
  AC_LANG_POP([C++])
fi

CFLAGS="$DEFAULT_CFLAGS"
CXXFLAGS="$DEFAULT_CXXFLAGS"

#
# setting up hardware dependent options
#

CPPFLAGS="$CPPFLAGS -DUSE_NON_CONST"
CFLAGS="$CFLAGS -ffast-math -fno-keep-inline-functions -Wall -Wundef"

case "$target" in
i[[3456]]86-*-linux-*)
  LDFLAGS="$LDFLAGS"
  CFLAGS="$CFLAGS -fPIC"
  CPPFLAGS="$CPPFLAGS -malign-double"
;;
x86_64-*-linux-*)
  LDFLAGS="$LDFLAGS"
  CFLAGS="$CFLAGS -fPIC"
;;
*-apple-darwin*)
  CFLAGS="$CFLAGS"
  LDFLAGS="$LDFLAGS -L/sw/lib"
;;
esac

# Check for 64 bit
AC_CHECK_SIZEOF([void*])

if SIZEOF_VOID_P=8; then
   CFLAGS="$CFLAGS -m64 -DCOMPUTER64b"
fi

# Checks for libraries.
#
# checks for FFTW
#
if test "$USE_FFTW2" = "true" ; then

   test -z $FFTW_HOME  && AC_CHECK_PROG(FFTW_HOME, $FFTW_HOME/include/fftw.h)
     
     if test -z $FFTW_HOME; then
	AC_MSG_ERROR([\`$FFTW_HOME/include/fftw.h' not found. FFTW2 is required if to compile and run feebl. If you know that FFTW is properly installed, you can set the environment variable \`FFTW_HOME' to point to your installation directory.])
     fi

     FFTW_CFLAGS="-DUSE_FFTW2 -I$FFTW_HOME/include"
     FFTW_LDFLAGS="-L$FFTW_HOME/lib -lfftw -lm"
fi   

if test "$USE_FFTW3" = "true" ; then

   test -z $FFTW_HOME  && AC_CHECK_PROG(FFTW_HOME, $FFTW_HOME/include/fftw3.h)
     
     if test -z $FFTW_HOME; then
	AC_MSG_ERROR([\`$FFTW_HOME/include/fftw3.h' not found. FFTW3 is required if to compile and run feebl. If you know that FFTW is properly installed, you can set the environment variable \`FFTW_HOME' to point to your installation directory.])
     fi

     FFTW_CFLAGS="-DUSE_FFTW3 -I$FFTW_HOME/include"
     FFTW_LDFLAGS="-L$FFTW_HOME/lib -lfftw3 -lm"
fi   
#
# checks for ROOT
#

if test "$USE_ROOT" = "true" ; then

   test -z $ROOTSYS && AC_CHECK_PROG(ROOTSYS)
   
   if test -z $ROOTSYS; then
      AC_MSG_ERROR([\`ROOT' not found. ROOT is required to compile and run feeblroot. If you know that ROOT is properly installed, you can set the environment variable \`ROOTSYS' to point to your installation directory.])
   fi

   make_all="$make_all feeblroot"
   ROOT_CFLAGS="-I$(ROOTSYS)/include"
   ROOT_LDFLAGS="-L$(ROOTSYS)/lib"

fi

# Checks for header files.
m4_warn([obsolete],
[The preprocessor macro `STDC_HEADERS' is obsolete.
  Except in unusual embedded environments, you can safely include all
  ISO C90 headers unconditionally.])dnl
# Autoupdate added the next two lines to ensure that your configure
# script's behavior did not change.  They are probably safe to remove.
AC_CHECK_INCLUDES_DEFAULT
AC_PROG_EGREP

AC_CHECK_HEADERS([stdlib.h])


# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor pow sqrt stpcpy strstr strtol])

#
# last operations before substitutions
#

if test -n "$EXTERNAL_CFLAGS" ; then
  CFLAGS="$CFLAGS $EXTERNAL_CFLAGS"
fi

if test -n "$EXTERNAL_CXXFLAGS" ; then
  CXXFLAGS="$CXXFLAGS $EXTERNAL_CXXFLAGS"
fi

if test "$USE_FFTW2" = "false" && test "$USE_FFTW3" = "false"; then
   FFTW_CFLAGS="-DUSE_FFT_LOCAL"   
fi

if test "$USE_ROOT" = "false" ; then
   ROOT_CFLAGS=""   
fi


AC_SUBST(make_all)
AC_SUBST(FFTW_CFLAGS)
AC_SUBST(ROOT_CFLAGS)
AC_SUBST(FFTW_LDFLAGS)
AC_SUBST(ROOT_LDFLAGS)
AC_SUBST(OPENMP_CXXFLAGS)

AC_CONFIG_FILES([Makefile src/Makefile])


AC_OUTPUT
