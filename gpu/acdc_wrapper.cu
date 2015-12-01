#include <lua.hpp>
#include <luaT.h>

#include "cutorch_state.h"
#include "THC.h"

#include "acdc.cu"


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
    
    
    float *cinput = Tensor_(data)(state, input);
    float *cA = Tensor_(data)(state, A);
    float *cAb = Tensor_(data)(state, Ab);
    float *cD = Tensor_(data)(state, D);
    float *cDb = Tensor_(data)(state, Db);
    float *coutput = Tensor_(data)(state, output);
    
    // It should be up to the calling framework to provide plans.
    // This is a hack to make it work, though will be very slow if the 
    // length/batchSize/groupSize change frequently.
    static cufftHandle planR2C;
    static cufftHandle planC2C;
    static int planLength = -1;
    static int planBatchSize = -1;
   
    if (planLength != input_size || planBatchSize != batch_size * group_size) {
       if (planLength != -1 && planBatchSize != -1) {
          cufftDestroy(planR2C);
          cufftDestroy(planC2C);
       }
       cufftPlan1d(&planR2C, input_size, CUFFT_R2C, batch_size * group_size);
       cufftPlan1d(&planC2C, input_size, CUFFT_C2C, batch_size * group_size);
       planLength = input_size;
       planBatchSize = batch_size * group_size;
    }
    
    acdc_fp<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,                  
                  Tensor_(data)(state, input),
                  Tensor_(data)(state, A),
                  Tensor_(data)(state, Ab),
                  Tensor_(data)(state, D),
                  Tensor_(data)(state, Db),
                  Tensor_(data)(state, output),
                  Tensor_(data)(state, tmp1),
                  Tensor_(data)(state, tmp2));

    
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

    // It should be up to the calling framework to provide plans.
    // This is a hack to make it work, though will be very slow if the 
    // length/batchSize/groupSize change frequently.
    static cufftHandle planR2C;
    static cufftHandle planC2C;
    static int planLength = -1;
    static int planBatchSize = -1;
   
    if (planLength != input_size || planBatchSize != batch_size * group_size) {
       if (planLength != -1 && planBatchSize != -1) {
          cufftDestroy(planR2C);
          cufftDestroy(planC2C);
       }
       cufftPlan1d(&planR2C, input_size, CUFFT_R2C, batch_size * group_size);
       cufftPlan1d(&planC2C, input_size, CUFFT_C2C, batch_size * group_size);
       planLength = input_size;
       planBatchSize = batch_size * group_size;
    }
        
    acdc_bp<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,
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

    // It should be up to the calling framework to provide plans.
    // This is a hack to make it work, though will be very slow if the 
    // length/batchSize/groupSize change frequently.
    static cufftHandle planR2C;
    static cufftHandle planC2C;
    static int planLength = -1;
    static int planBatchSize = -1;

    if (planLength != input_size || planBatchSize != batch_size * group_size) {
       if (planLength != -1 && planBatchSize != -1) {
          cufftDestroy(planR2C);
          cufftDestroy(planC2C);
       }
       cufftPlan1d(&planR2C, input_size, CUFFT_R2C, batch_size * group_size);
       cufftPlan1d(&planC2C, input_size, CUFFT_C2C, batch_size * group_size);
       planLength = input_size;
       planBatchSize = batch_size * group_size;
    }
        
    acdc_bp_acc<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,
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
