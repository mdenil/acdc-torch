#ifndef CUFFT_EXEC_IMPL
#error "Do not include this header directly.  Include cufft.hpp instead."
#endif

#define UNUSED(x) (void)(x);

// hack
#include <iostream>
#define LOG(x) std::cerr
#define FATAL

namespace acdc {

template<>
struct cufft<DTYPE>
{
    typedef cufftHandle plan_type;

    typedef typename cuda_types<DTYPE>::gpu_real cuda_real;
    typedef typename cuda_types<DTYPE>::gpu_complex cuda_complex;

    static cufftHandle plan_many_dft_r2c_1d(int size, int how_many)
    {
        cufftHandle plan;

        // http://docs.nvidia.com/cuda/cufft/index.html#function-cufftplanmany
        cufftResult result = cufftPlanMany(
            &plan,
            1, // 1d
            &size, // size of each transform
            NULL, // input is not embedded in a larger matrix
            1, // stride between elements of each input
            size, // stride between different inputs
            NULL, // output is not embedded in a larger matrix
            1, // stride between elements of each output
            size, // stride between different outputs
            cufft_r2c_(CUFFT_),
            how_many);

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT planning failed (err: " << result << ")";
        }

        return plan;
    }

    static cufftHandle plan_many_dft_c2r_1d(int size, int how_many)
    {
        cufftHandle plan;

        cufftResult result = cufftPlanMany(
            &plan,
            1,
            &size,
            NULL,
            1,
            size,
            NULL,
            1,
            size,
            cufft_c2r_(CUFFT_),
            how_many);

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT planning failed (err: " << result << ")";
        }

        return plan;
    }

    static cufftHandle plan_many_dft_c2c_1d(int size, int how_many)
    {
        cufftHandle plan;

        cufftResult result = cufftPlanMany(
            &plan,
            1,
            &size,
            NULL,
            1,
            size,
            NULL,
            1,
            size,
            cufft_c2c_(CUFFT_),
            how_many);

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT planning failed (err: " << result << ")";
        }

        return plan;
    }

    static void execute_r2c(cufftHandle& plan, DTYPE* idata, std::complex<DTYPE>* odata)
    {
        cufftResult result = cufft_r2c_(cufftExec)(
            plan,
            reinterpret_cast<cuda_real*>(idata),
            reinterpret_cast<cuda_complex*>(odata));

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT failed to execute (err: " << result << ")";
        }
    }

    static void execute_c2r(cufftHandle& plan, std::complex<DTYPE>* idata, DTYPE* odata)
    {
        cufftResult result = cufft_c2r_(cufftExec)(
            plan,
            reinterpret_cast<cuda_complex*>(idata),
            reinterpret_cast<cuda_real*>(odata));

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT failed to execute (err: " << result << ")";
        }
    }

    static void execute_c2c(cufftHandle& plan, std::complex<DTYPE>* idata, std::complex<DTYPE>* odata, int direction)
    {
        cufftResult result = cufft_c2c_(cufftExec)(
            plan,
            reinterpret_cast<cuda_complex*>(idata),
            reinterpret_cast<cuda_complex*>(odata),
            direction);

        if (result != CUFFT_SUCCESS) {
            LOG(FATAL) << "CUFFT failed to execute (err: " << result << ")";
        }
    }

    static void destroy_plan(cufftHandle& plan)
    {
        cufftDestroy(plan);
    }

    static DTYPE* malloc_real(size_t n)
    {
        cufftReal* data;
        cudaMalloc((void**)&data, sizeof(cuda_real)*n);
        return reinterpret_cast<DTYPE*>(data);
    }

    static std::complex<DTYPE>* malloc_complex(size_t n)
    {
        cufftComplex* data;
        cudaMalloc((void**)&data, sizeof(cuda_complex)*n);
        return reinterpret_cast<std::complex<DTYPE>*>(data);
    }

    static void free(void* data)
    {
        cudaFree(data);
    }
};

}

