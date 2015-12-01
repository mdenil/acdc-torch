#include "luaT.h"

namespace acdc {

void THCudaTensor_initPermutation(lua_State* L);
void THCudaTensor_initFastACDC(lua_State* L);

}

using namespace acdc;

extern "C" {

LUA_EXTERNC DLL_EXPORT int luaopen_libacdc_gpu(lua_State* L);

int luaopen_libacdc_gpu(lua_State* L)
{
    THCudaTensor_initPermutation(L);
    THCudaTensor_initFastACDC(L);

    return 0;
}

}
