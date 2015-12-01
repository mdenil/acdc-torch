/* Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/


/* NOTES:

1) These functions expect a real input. Weights are length "length", input/outputs length "length * batchSize".
2) The tmp variable must have space 2 * length * batchSize.
3) There is still some performance work to be done.

For questions/comments, please contact Jeremy Appleyard (jappleyard@nvidia.com)

*/

// TODO:
// Fusion for performance
// Shared memory setup/final for smaller problems to avoid strided access
// Get rid of static cuFFT plans.


#include <lua.hpp>
#include <luaT.h>

#include "cutorch_state.h"
#include "THC.h"


#include <stdio.h>
#include <cufft.h>
#include <cufftXt.h>

#include <stdio.h>
#include <cufft.h>
#include <cufftXt.h>

// Useful to have
#define PI 3.14159265359f
#define ROOT2 1.4142135623730951f

template<typename Dtype, int tblockSize, bool forward>
__global__ void DCT_setup(int length,
                          int batchSize,
                          int groupSize,
                          const Dtype * __restrict__ A,
                          const Dtype * __restrict__ Ab,
                          const Dtype * __restrict__ in,
                          Dtype * __restrict__ out) {
   int batchID = blockIdx.x;
   int groupID = blockIdx.y;

   if (forward) in += length * batchID;
   else         in += length * (batchID * groupSize + groupID);
   
   out += 2 * length * (batchID * groupSize + groupID);

   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;
      cufftComplex val;
      
      int index;
      if (element < length / 2) {
         index = element * 2;
      }
      else {
         index = length - 2 * (element - length / 2) - 1;
      }

      // For forward we're relying on cache hits for perf here.
      if (element < length / 2) {
         val.x = ((float*)(in))[element * 2];
      }
      else {
         val.x = ((float*)(in))[length - 2 * (element - length / 2) - 1];
      }
      val.y = 0.f;
      
      if (A != NULL) {           
         val.x *= A[groupID * length + index];
         if (Ab != NULL) {
           val.x += Ab[groupID * length + index];
         }
      }      

      ((cufftComplex*)(out))[element] = val;            
   }
}


template<typename Dtype, int tblockSize>
__global__ void DCT_final(int length,
                          int batchSize,
                          int groupSize,
                          const Dtype * __restrict__ A,
                          const Dtype * __restrict__ Ab,
                          const Dtype * __restrict__ in,
                          Dtype * __restrict__ out) {
   int batchID = blockIdx.x;
   int groupID = blockIdx.y;

   in += 2 * length * (batchID * groupSize + groupID);
   out += length * (batchID * groupSize + groupID);

   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;

      cufftComplex val = ((cufftComplex*)(in))[element];

      cufftComplex val2;
      cufftComplex ret;

      __sincosf(element * PI / (2.f * (length)), &(val2.y), &(val2.x));

      val2.y = -val2.y;

      ret.x = val.x * val2.x - val.y * val2.y;

      // Normalisation
      if (element == 0) {
         ret.x *= rsqrt((float)length);
      }
      else {
         ret.x *= ROOT2 * rsqrt((float)length);
      }

      if (A != NULL) {
         ret.x *= A[groupID * length + element];
         if (Ab != NULL) {
           ret.x += Ab[groupID * length + element];
         }
      }

      ((float*)(out))[element] = ret.x;
   }
}
/* 
template<typename Dtype, int tblockSize>
__global__ void IDCT_setup(int length,
                           int batchSize,
                           int groupSize,
                           const Dtype * __restrict__ D,
                           const Dtype * __restrict__ in,
                           Dtype * __restrict__ out) {
   int batchID = blockIdx.x;

   in += batchID * length;
   out += 2 * batchID * length;


   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;
      cufftComplex val;

      float re_in = ((float*)(in))[element];

      if (D != NULL) {
         re_in *= D[element];
      }

      // Un-normalisation
      if (element == 0) {
         re_in *= rsqrtf((float)length);
      }
      else {
         re_in *= ROOT2 * rsqrtf((float)length);
      }

      float2 val2;
      __sincosf(element * PI / (2.f * length), &(val2.y), &(val2.x));

      val.x = re_in * val2.x;
      val.y = -re_in * val2.y;

      ((cufftComplex*)(out))[element] = val;
   }
} */



template<typename Dtype, int tblockSize, bool accumulate>
__global__ void IDCT_final(int length,
                          int batchSize,
                          int groupSize,
                          const Dtype * __restrict__ A,
                          const Dtype * __restrict__ Ab,
                          const Dtype * __restrict__ in,
                          Dtype * __restrict__ out) {
   int batchID = blockIdx.x;
   int groupID = blockIdx.y;

   in += 2 * length * (batchID * groupSize + groupID);
   out += length * (batchID * groupSize + groupID);

   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;

      int index;
      if (element < length / 2) {
         index = element * 2;
      }
      else {
         index = length - 2 * (element - length / 2) - 1;
      }      
      
      cufftComplex val = ((cufftComplex*)(in))[element];
      
      // "A" for backward pass
      if (A != NULL) {
         val.x *= A[groupID * length + index];
         if (Ab != NULL) {
           val.x += Ab[groupID * length + index];
         }
      }

      if (accumulate) {
         ((float*)(out))[index] += val.x;
      }
      else {
         ((float*)(out))[index] = val.x;
      }
      
   }
}

template<typename Dtype, int tblockSize, bool accumulate>
__global__ void DCT_final_IDCT_setup(
                      int length,
                      int batchSize,
                      int groupSize,
                      const Dtype * __restrict__ D,
                      const Dtype * __restrict__ Db,
                      const Dtype * __restrict__ in,
                      Dtype * __restrict__ out,
                      Dtype * __restrict__ deltaMid) {

   int batchID = blockIdx.x;
   int groupID = blockIdx.y;

   in += 2 * length * (batchID * groupSize + groupID);
   out += 2 * length * (batchID * groupSize + groupID);
   if (deltaMid) deltaMid += length * (batchID * groupSize + groupID);

   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;

      cufftComplex val = ((cufftComplex*)(in))[element];

      cufftComplex val2;
      cufftComplex ret;

      __sincosf(element * PI / (2.f * (length)), &(val2.y), &(val2.x));

      val2.y = -val2.y;

      ret.x = val.x * val2.x - val.y * val2.y;

      // Normalisation
      if (element == 0) {
         ret.x *= rsqrt((float)length);
      }
      else {
         ret.x *= ROOT2 * rsqrt((float)length);
      }

      float re_in = ret.x;

      if (D != NULL) {
        re_in *= D[groupID * length + element];
        if (Db != NULL) {
          re_in += Db[groupID * length + element];
        }
      }

      if (deltaMid) {
         if (accumulate) deltaMid[element] += re_in;
         else            deltaMid[element] = re_in;
      }

      // Un-normalisation
      if (element == 0) {
         re_in *= rsqrtf((float)length);
      }
      else {
         re_in *= ROOT2 * rsqrtf((float)length);
      }

      __sincosf(element * PI / (2.f * length), &(val2.y), &(val2.x));

      val.x = re_in * val2.x;
      val.y = -re_in * val2.y;

      ((cufftComplex*)(out))[element] = val;
      
   }
}

template<typename Dtype, int tblockSize>
__global__ void updateWeights(int length, 
                              int batchSize,
                              int groupSize,
                              const Dtype * __restrict__ D,
                              const Dtype * __restrict__ input,
                              const Dtype * __restrict__ gradOutput,
                              Dtype * __restrict__ delta_D,
                              Dtype * __restrict__ delta_Db) {
   int batchID = blockIdx.x;
   int groupID = blockIdx.y;

   input      += length * batchID;
   gradOutput += length * (batchID * groupSize + groupID);

   D += length * groupID;
   delta_D += length * groupID;
   delta_Db += length * groupID;
   
   for (int i = 0; i < (length + tblockSize - 1) / tblockSize; i++) {
      int element = i * tblockSize + threadIdx.x;
      if (element >= length) return;

      float val = gradOutput[element] / D[element];

      atomicAdd((float*)(&(delta_D[element])), (float)(val * input[element]));
      atomicAdd((float*)(&(delta_Db[element])), val);

   }
}


template<typename Dtype>
void acdc_fp_fast2(
                  cudaStream_t stream,
                  int length, int batchSize, int groupSize,
                  const Dtype * __restrict__ in,
                  const Dtype * __restrict__ A,
                  const Dtype * __restrict__ Ab,
                  const Dtype * __restrict__ D,
                  const Dtype * __restrict__ Db,
                  Dtype * __restrict__ out,
                  Dtype * __restrict__ tmp1,
                  Dtype * __restrict__ tmp2) {


   // This is awful. TODO: Store the plans more sensibly.
   static cufftHandle plan;
   static int planLength = -1;
   static int planBatchSize = -1;

   if (planLength != length || planBatchSize != batchSize * groupSize) {
      if (planLength != -1 && planBatchSize != -1) {
         cufftDestroy(plan);
      }
      cufftPlan1d(&plan, length, CUFFT_C2C, batchSize * groupSize);
      planLength = length;
      planBatchSize = batchSize  * groupSize;
   }

   cufftSetStream(plan, stream);
   
   const int blockSize = 128;
   dim3 blockDim;
   dim3 gridDim;
   
   blockDim.x = blockSize;
   gridDim.x = batchSize;
   gridDim.y = groupSize;

   // Two DCTs required. Inverse is handled in the custom setup.
   DCT_setup<Dtype, blockSize, true> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, Ab, in, tmp1);
     
   cufftExecC2C(plan, (cufftComplex*)tmp1, (cufftComplex*)tmp2, CUFFT_FORWARD);

   DCT_final_IDCT_setup<Dtype, blockSize, false> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, D, Db, tmp2, tmp1, NULL);

   cufftExecC2C(plan, (cufftComplex*)tmp1, (cufftComplex*)tmp2, CUFFT_FORWARD);
   
   IDCT_final<Dtype, blockSize, false> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, NULL, NULL, tmp2, out);
}



// NOTE: For the backward pass "in" is bottom, "out" is top, so we write to in.
template<typename Dtype>
void acdc_bp_fast2(
                  cudaStream_t stream,
                  int length, 
                  int batchSize,
                  int groupSize,
                  Dtype * __restrict__ delta_in,
                  const Dtype * __restrict__ A,
                  const Dtype * __restrict__ Ab,
                  const Dtype * __restrict__ D,
                  const Dtype * __restrict__ Db,
                  const Dtype * __restrict__ delta_out,
                  Dtype * __restrict__ delta_mid,
                  Dtype * __restrict__ tmp1,
                  Dtype * __restrict__ tmp2) {

   // This is awful. Don't do this. TODO: Store the plans more sensibly.
   static cufftHandle plan;
   static int planLength = -1;
   static int planBatchSize = -1;

   if (planLength != length || planBatchSize != batchSize * groupSize) {
      if (planLength != -1 && planBatchSize != -1) {
         cufftDestroy(plan);
      }
      cufftPlan1d(&plan, length, CUFFT_C2C, batchSize * groupSize);
      planLength = length;
      planBatchSize = batchSize * groupSize;
   }

   cufftSetStream(plan, stream);

   const int blockSize = 128;
   dim3 blockDim;
   dim3 gridDim;
   
   blockDim.x = blockSize;
   gridDim.x = batchSize;
   gridDim.y = groupSize;   
   
   // Backward through CD
   DCT_setup<Dtype, 128, false> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, NULL, NULL, delta_out, tmp1);
   cufftExecC2C(plan, (cufftComplex*)tmp1, (cufftComplex*)tmp2, CUFFT_FORWARD);

   DCT_final_IDCT_setup<Dtype, 128, false> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, D, NULL, tmp2, tmp1, delta_mid);

   // Backward through CA
   cufftExecC2C(plan, (cufftComplex*)tmp1, (cufftComplex*)tmp2, CUFFT_FORWARD);
   IDCT_final<Dtype, 128, false> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, NULL, tmp2, delta_in);
}

template<typename Dtype>
void acdc_bp_acc_fast2(
                  cudaStream_t stream,
                  int length, 
                  int batchSize,
                  int groupSize,
                  Dtype * __restrict__ delta_in,
                  Dtype * __restrict__ delta_mid,
                  const Dtype * __restrict__ A,
                  const Dtype * __restrict__ Ab,
                  const Dtype * __restrict__ D,
                  //const Dtype * __restrict__ Db,
                  const Dtype * __restrict__ inputA,
                  Dtype * __restrict__ inputD,
                  Dtype * __restrict__ delta_A,
                  Dtype * __restrict__ delta_Ab,
                  Dtype * __restrict__ delta_D,
                  Dtype * __restrict__ delta_Db,
                  Dtype * __restrict__ tmp1,
                  Dtype * __restrict__ tmp2) {

   // This is awful. Don't do this. TODO: Store the plans more sensibly.
   static cufftHandle plan;
   static int planLength = -1;
   static int planBatchSize = -1;

   if (planLength != length || planBatchSize != batchSize * groupSize) {
      if (planLength != -1 && planBatchSize != -1) {
         cufftDestroy(plan);
      }
      cufftPlan1d(&plan, length, CUFFT_C2C, batchSize * groupSize);
      planLength = length;
      planBatchSize = batchSize * groupSize;
   }

   cufftSetStream(plan, stream);

   const int blockSize = 128;
   dim3 blockDim;
   dim3 gridDim;
   
   blockDim.x = blockSize;
   gridDim.x = batchSize;
   gridDim.y = groupSize;   
 
   updateWeights<Dtype, 128> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, inputA, delta_in, delta_A, delta_Ab);

   // Forward thorugh AC to calculate input going into D
   DCT_setup<Dtype, 128, true> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, Ab, inputA, tmp1);
   cufftExecC2C(plan, (cufftComplex*)tmp1, (cufftComplex*)tmp2, CUFFT_FORWARD);
   DCT_final<Dtype, 128> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, NULL, NULL, tmp2, inputD);

   updateWeights<Dtype, 128> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, D, inputD, delta_mid, delta_D, delta_Db);
 }



#define Tensor THCudaTensor
#define TensorTypename "torch.CudaTensor"
#define Tensor_(fn) THCudaTensor_ ## fn

int Tensor_(Fast_ACDC_updateOutput)(lua_State* L)
{
    THCState *state = getCutorchState(L);
    Tensor* input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    Tensor* A = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "A", TensorTypename));
    Tensor* Ab = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Ab", TensorTypename));
    Tensor* D = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "D", TensorTypename));
    Tensor* Db = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Db", TensorTypename));
    Tensor* tmp1 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp1", TensorTypename));
    Tensor* tmp2 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp2", TensorTypename));
    Tensor* output = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "output", TensorTypename));

    int batch_size;
    int input_size;
    int group_size;
    if (Tensor_(nDimension)(state, input) == 1) {
        batch_size = 1;
        group_size = 1;
        input_size = Tensor_(size)(state, input, 0);
    }
    else if (Tensor_(nDimension)(state, input) == 2) {
        batch_size = Tensor_(size)(state, input, 0);
        group_size = 1;
        input_size = Tensor_(size)(state, input, 1);
    }
    else if (Tensor_(nDimension)(state, input) == 3) {
        batch_size = Tensor_(size)(state, input, 0);
        group_size = Tensor_(size)(state, output, 1);
        input_size = Tensor_(size)(state, input, 2);
    }    
    else {
        luaL_error(L, "input must have 1 or 2 or 3 dimensions");
    }

    cudaStream_t stream = THCState_getCurrentStream(state);

    acdc_fp_fast2<float>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  Tensor_(data)(state, input),
                  Tensor_(data)(state, A),
                  Tensor_(data)(state, Ab),
                  Tensor_(data)(state, D),
                  Tensor_(data)(state, Db),
                  Tensor_(data)(state, output),
                  Tensor_(data)(state, tmp1),
                  Tensor_(data)(state, tmp2));

    cudaDeviceSynchronize();
    return 1;
}


int Tensor_(Fast_ACDC_updateGradInput)(lua_State* L)
{
    THCState *state = getCutorchState(L);
    Tensor* input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    Tensor* gradOutput = static_cast<Tensor*>(
        luaT_checkudata(L, 3, TensorTypename));
    Tensor* A = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "A", TensorTypename));
    Tensor* Ab = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Ab", TensorTypename));
    Tensor* D = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "D", TensorTypename));
    Tensor* Db = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Db", TensorTypename));
    Tensor* tmp1 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp1", TensorTypename));
    Tensor* tmp2 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp2", TensorTypename));
    Tensor* delta_mid = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "delta_mid", TensorTypename));
    Tensor* gradInput = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradInput", TensorTypename));

    int batch_size;
    int input_size;
    int group_size;
    if (Tensor_(nDimension)(state, gradOutput) == 1) {
        batch_size = 1;
        group_size = 1;
        input_size = Tensor_(size)(state, gradOutput, 0);
    }
    else if (Tensor_(nDimension)(state, gradOutput) == 2) {
        batch_size = Tensor_(size)(state, gradOutput, 0);
        group_size = 1;
        input_size = Tensor_(size)(state, gradOutput, 1);
    }
    else if (Tensor_(nDimension)(state, gradOutput) == 3) {
        batch_size = Tensor_(size)(state, gradOutput, 0);
        group_size = Tensor_(size)(state, gradOutput, 1);
        input_size = Tensor_(size)(state, gradOutput, 2);
    }    
    else {
        luaL_error(L, "input must have 1 or 2 or 3 dimensions");
    }

    cudaStream_t stream = THCState_getCurrentStream(state);

    acdc_bp_fast2<float>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  Tensor_(data)(state, gradInput),
                  Tensor_(data)(state, A),
                  Tensor_(data)(state, Ab),
                  Tensor_(data)(state, D),
                  Tensor_(data)(state, Db),
                  Tensor_(data)(state, gradOutput),
                  Tensor_(data)(state, delta_mid),
                  Tensor_(data)(state, tmp1),
                  Tensor_(data)(state, tmp2));

    return 1;
}


int Tensor_(Fast_ACDC_accGradParams)(lua_State* L)
{
    THCState *state = getCutorchState(L);
    Tensor* input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    Tensor* gradOutput = static_cast<Tensor*>(
        luaT_checkudata(L, 3, TensorTypename));
    Tensor* A = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "A", TensorTypename));
    Tensor* Ab = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Ab", TensorTypename));
    Tensor* D = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "D", TensorTypename));
    Tensor* Db = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "Db", TensorTypename));
    Tensor* tmp1 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp1", TensorTypename));
    Tensor* tmp2 = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "tmp2", TensorTypename));
    Tensor* inputD = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "activationsD", TensorTypename));
    Tensor* gradA = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradA", TensorTypename));
    Tensor* gradD = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradD", TensorTypename));
    Tensor* gradAb = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradAb", TensorTypename));
    Tensor* gradDb = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradDb", TensorTypename));
    Tensor* delta_mid = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "delta_mid", TensorTypename));
    Tensor* gradInput = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradInput", TensorTypename));
    int outputIdx = lua_gettop(L);

    int batch_size;
    int input_size;
    int group_size;
    if (Tensor_(nDimension)(state, gradInput) == 1) {
        batch_size = 1;
        group_size = 1;
        input_size = Tensor_(size)(state, gradInput, 0);
    }
    else if (Tensor_(nDimension)(state, gradInput) == 2) {
        batch_size = Tensor_(size)(state, gradInput, 0);
        group_size = 1;
        input_size = Tensor_(size)(state, gradInput, 1);
    }
    else if (Tensor_(nDimension)(state, gradInput) == 3) {
        batch_size = Tensor_(size)(state, gradInput, 0);
        group_size = Tensor_(size)(state, gradInput, 1);
        input_size = Tensor_(size)(state, gradInput, 2);
    }    
    else {
        luaL_error(L, "input must have 1 or 2 or 3 dimensions");
    }

    cudaStream_t stream = THCState_getCurrentStream(state);

    acdc_bp_acc_fast2<float>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  Tensor_(data)(state, gradInput),
                  Tensor_(data)(state, delta_mid),
                  Tensor_(data)(state, A),
                  Tensor_(data)(state, Ab),
                  Tensor_(data)(state, D),
                  //Tensor_(data)(state, Db),
                  Tensor_(data)(state, input), // inputA
                  Tensor_(data)(state, inputD),
                  Tensor_(data)(state, gradA),
                  Tensor_(data)(state, gradAb),
                  Tensor_(data)(state, gradD),
                  Tensor_(data)(state, gradDb),
                  Tensor_(data)(state, tmp1),
                  Tensor_(data)(state, tmp2));

    lua_pushvalue(L, outputIdx);

    return 1;
}



static const struct luaL_Reg Tensor_(Fast_ACDC_functions_)[] = {
    {"Fast_ACDC_updateOutput", Tensor_(Fast_ACDC_updateOutput)},
    {"Fast_ACDC_updateGradInput", Tensor_(Fast_ACDC_updateGradInput)},
    {"Fast_ACDC_accGradParams", Tensor_(Fast_ACDC_accGradParams)},
    {NULL, NULL}
};

namespace acdc {

void Tensor_(initFastACDC)(lua_State* L) {
    luaT_pushmetatable(L, TensorTypename);
    luaT_registeratname(L, Tensor_(Fast_ACDC_functions_), "nn");
    lua_pop(L, 1);
}

}
