#include <lua.hpp>
#include <luaT.h>

#include "cutorch_state.h"
#include "THC.h"

#include <cuda_util.hpp>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace acdc {

#define Tensor THCudaTensor
#define TensorTypename "torch.CudaTensor"
#define Tensor_(fn) THCudaTensor_ ## fn
#define DTYPE float
#include "Permutation_impl.cu"
#undef Tensor
#undef TensorTypename
#undef Tensor_
#undef DTYPE

} // namespace acdc

