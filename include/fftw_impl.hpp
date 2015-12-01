#ifndef FFTW_TRAITS_EXEC_IMPL
#error "Do not include this header directly.  Include fftw.hpp instead."
#endif

/**
 * This is a preprocessor template to generate generic type-aware wrappers around
 * the fftw C api.
 *
 * This file relies on the following preprocessor macros:
 *
 * DTYPE : expands to the appropriate width floating point type
 *
 * fftw_(name) : expands to the name of the appropriate fftw function
 * compatable with DTYPE.
 *
 * e.g. fftw_(free) becomes fftw_free when DTYPE=double or fftwf_free when
 * DTYPE=float.
 *
 * The functions in here also translate between fftw and std complex
 * types.  These types are memory layout compatible, but the standard library
 * types are nicer to work with.  e.g. fftw_complex is a double[2],
 * which has the same memory layout as std::complex<double> but the fftw type
 * doesn't have arithmetic or stream operators defined.
 *
 * ( see: http://www.fftw.org/fftw3_doc/Complex-numbers.html#Complex-numbers )
 * 
 */

namespace acdc {

template<>
struct fftw<DTYPE>
{
    typedef fftw_(plan) plan_type;

    static fftw_(plan) plan_dft_1d(
        int size,
        std::complex<DTYPE>* in,
        std::complex<DTYPE>* out,
        int sign,
        unsigned int flags)
    {
        return fftw_(plan_dft_1d)(
            size,
            reinterpret_cast<fftw_(complex)*>(in),
            reinterpret_cast<fftw_(complex)*>(out),
            sign,
            flags);
    }

    static fftw_(plan) plan_many_dft_1d(
        int size,
        int how_many,
        std::complex<DTYPE>* in,
        std::complex<DTYPE>* out,
        int sign,
        unsigned int flags)
    {
        // http://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html#Advanced-Complex-DFTs
        return fftw_(plan_many_dft)(
            1, // 1d
            &size, // size of each transform
            how_many, // number of transforms
            reinterpret_cast<fftw_(complex)*>(in),
            NULL, // input is not embedded in a larger matrix
            1, // stride between elements of each input
            size, // stride between different inputs
            reinterpret_cast<fftw_(complex)*>(out),
            NULL, // output not embedded in a larger matrix
            1, // stride between output elements
            size, // stride btween outputs
            sign,
            flags);
    }

    static fftw_(plan) plan_many_dft_r2c_1d(
        int size,
        int how_many,
        DTYPE* in,
        std::complex<DTYPE>* out,
        unsigned int flags)
    {
        // Arguments:
        // http://www.fftw.org/fftw3_doc/Advanced-Real_002ddata-DFTs.html#Advanced-Real_002ddata-DFTs
        //
        // Data format:
        // http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format

        return fftw_(plan_many_dft_r2c)(
            1, // 1d
            &size, // size of each transform
            how_many, // number of transforms
            reinterpret_cast<DTYPE*>(in),
            NULL, // input is not embedded in a larger matrix
            1, // stride between lements of each input
            size, // stride between different inputs
            reinterpret_cast<fftw_(complex)*>(out),
            NULL, // output is not embedded in a larger matrix
            1, // stride between output elements
            size / 2 + 1, // stride between different inputs (see data format link)
            flags);
    }

    static fftw_(plan) plan_many_dft_c2r_1d(
        int size,
        int how_many,
        std::complex<DTYPE>* in,
        DTYPE* out,
        unsigned int flags)
    {
        // Arguments:
        // http://www.fftw.org/fftw3_doc/Advanced-Real_002ddata-DFTs.html#Advanced-Real_002ddata-DFTs
        //
        // Data format:
        // http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format

        return fftw_(plan_many_dft_c2r)(
            1, // 1d
            &size, // size of each transform
            how_many, // number of transforms
            reinterpret_cast<fftw_(complex)*>(in),
            NULL, // input is not embedded in a larger matrix
            1, // stride between lements of each input
            size / 2 + 1, // stride between different inputs (see data format link)
            reinterpret_cast<DTYPE*>(out),
            NULL, // output is not embedded in a larger matrix
            1, // stride between output elements
            size, // stride between different inputs
            flags);
    }

    static fftw_(plan) plan_many_dft_r2r_1d(
        int size,
        int how_many,
        DTYPE* in,
        DTYPE* out,
        unsigned int flags)
    {
        // http://www.fftw.org/doc/Real_002dto_002dReal-Transforms.html
        // http://www.fftw.org/doc/Advanced-Real_002dto_002dreal-Transforms.html#Advanced-Real_002dto_002dreal-Transforms
        
        static fftw_(r2r_kind) kind = FFTW_REDFT10; // http://www.fftw.org/doc/Real_002dto_002dReal-Transform-Kinds.html#Real_002dto_002dReal-Transform-Kinds

        return fftw_(plan_many_r2r)(
            1,
            &size,
            how_many,
            in,
            NULL,
            1,
            size,
            out,
            NULL,
            1,
            size,
            &kind,
            flags);
    }

    static fftw_(plan) plan_many_dift_r2r_1d(
        int size,
        int how_many,
        DTYPE* in,
        DTYPE* out,
        unsigned int flags)
    {
        // http://www.fftw.org/doc/Real_002dto_002dReal-Transforms.html
        // http://www.fftw.org/doc/Advanced-Real_002dto_002dreal-Transforms.html#Advanced-Real_002dto_002dreal-Transforms

        static fftw_(r2r_kind) kind = FFTW_REDFT01; // http://www.fftw.org/doc/Real_002dto_002dReal-Transform-Kinds.html#Real_002dto_002dReal-Transform-Kinds

        return fftw_(plan_many_r2r)(
            1,
            &size,
            how_many,
            in,
            NULL,
            1,
            size,
            out,
            NULL,
            1,
            size,
            &kind,
            flags);
    }

    static void destroy_plan(fftw_(plan)& plan)
    {
        fftw_(destroy_plan)(plan);
    }

    static DTYPE* malloc_real(size_t n)
    {
        return reinterpret_cast<DTYPE*>(fftw_(malloc)(n * sizeof(DTYPE)));
    }

    static std::complex<DTYPE>* malloc_complex(size_t n)
    {
        return reinterpret_cast<std::complex<DTYPE>*>(
            fftw_(malloc)(n * sizeof(std::complex<DTYPE>)));
    }

    static void free(void* v)
    {
        fftw_(free)(v);
    }


    static void execute(fftw_(plan)& plan)
    {
        fftw_(execute)(plan);
    }
};

}

