#include <algorithm>

#include <lua.hpp>
#include <luaT.h>

#include "TH.h"

namespace acdc {

#define Tensor THFloatTensor
#define TensorTypename "torch.FloatTensor"
#define Tensor_(fn) THFloatTensor_ ## fn
#define DTYPE float
#include "Permutation_impl.cpp"
#undef Tensor
#undef TensorTypename
#undef Tensor_
#undef DTYPE


#define Tensor THDoubleTensor
#define TensorTypename "torch.DoubleTensor"
#define Tensor_(fn) THDoubleTensor_ ## fn
#define DTYPE double
#include "Permutation_impl.cpp"
#undef Tensor
#undef TensorTypename
#undef Tensor_
#undef DTYPE

}

