#ifndef CAFFE_COMPLEX_UTIL_
#define CAFFE_COMPLEX_UTIL_

#include <complex>

namespace acdc {

/**
 * Blobs don't support complex numbers, so we fake it using float/double
 * Blobs and then reinterpret the data as complex numbers. These functions
 * make doing this less notationally cumbersome.
 */
template<typename Dtype>
inline std::complex<Dtype>* complex_ptr(Dtype* ptr) {
    return reinterpret_cast<std::complex<Dtype>*>(ptr);
}

template<typename Dtype>
inline std::complex<Dtype> const* complex_ptr(Dtype const* ptr) {
    return reinterpret_cast<std::complex<Dtype> const*>(ptr);
}

inline int complex_count(int real_count) {
    return real_count / 2;
}

inline int real_count(int complex_count) {
    return 2 * complex_count;
}

#ifndef CPU_ONLY
/**
 * Helpers for complex cuda types.  Whenever possible you should write wrapper
 * functions that work with Dtype or std::complex<Dtype> and cast to the cuda
 * type in your wrapper if needed.
 */
template<typename Dtype>
struct complex_cuda_types;

template<>
struct complex_cuda_types<float> {
    typedef cuFloatComplex gpu_complex;
};

template<>
struct complex_cuda_types<double> {
    typedef cuDoubleComplex gpu_complex;
};

template<typename Dtype>
struct cuda_types : public complex_cuda_types<Dtype> {
    typedef Dtype cpu_real;
    typedef Dtype gpu_real;
    typedef std::complex<Dtype> cpu_complex;
};

#endif


}

#endif

