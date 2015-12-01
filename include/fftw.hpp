#pragma once

#include <complex>

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)) \
    && defined(__CUDACC__)

/**
 * If this happens then this file is being compiled by nvcc pretending that it
 * is gcc.
 *
 * This triggers a bug in certain versions of fftw which enables quad precision
 * support when it detects gcc >= 4.4.6.  Unfortuately nvcc does not support
 * quad precision and chokes on the __float128 type.
 *
 * As a work around we pretend to be a different version of GCC, just while we
 * include the fftw header, to convince fftw to not define prototypes for the
 * quad precision functions.
 *
 * This bug is discussed here:
 *
 * https://github.com/FFTW/fftw3/issues/18
 *
 * and fixed in fftw3 mainline here:
 *
 * https://github.com/FFTW/fftw3/commit/07ef78dc1b273a40fb4f7db1797d12d3423b1f40
 *
 * Unfortunately we can't rely on having access to a bleeding-edge version of
 * fftw so we use this workaround.
 *
 * This bug is definitely present in the following versions of fftw:
 *
 * 3.3.2
 * 3.3.3
 * 3.3.4
 *
 */

#define GNUC_VERSION_REAL __GNUC__
#undef __GNUC__
#define __GNUC__ 3
#include <fftw3.h>
#undef __GNUC__
#define __GNUC__ GNUC_VERSION_REAL

#else

#include <fftw3.h>

#endif

namespace acdc {

template<typename Dtype>
struct fftw;

}

#define FFTW_TRAITS_EXEC_IMPL

#define fftw_(NAME) fftw_ ## NAME
#define DTYPE double
#include "fftw_impl.hpp"
#undef fftw_
#undef DTYPE

#define fftw_(NAME) fftwf_ ## NAME
#define DTYPE float
#include "fftw_impl.hpp"
#undef fftw_
#undef DTYPE

#undef FFTW_TRAITS_EXEC_IMPL


