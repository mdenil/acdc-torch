#pragma once

#include <complex>
#include <cufft.h>

#include "complex_util.hpp"

namespace acdc {

template<typename Dtype>
struct cufft;

}

#define CUFFT_EXEC_IMPL

#define cufft_r2c_(NAME) NAME ## D2Z
#define cufft_c2r_(NAME) NAME ## Z2D
#define cufft_c2c_(NAME) NAME ## Z2Z
#define DTYPE double
#include "cufft_impl.hpp"
#undef cufft_r2c_
#undef cufft_c2r_
#undef cufft_c2c_
#undef DTYPE

#define cufft_r2c_(NAME) NAME ## R2C
#define cufft_c2r_(NAME) NAME ## C2R
#define cufft_c2c_(NAME) NAME ## C2C
#define DTYPE float
#include "cufft_impl.hpp"
#undef cufft_r2c_
#undef cufft_c2r_
#undef cufft_c2c_
#undef DTYPE

#undef CUFFT_EXEC_IMPL

