#include "luaT.h"

#define DECLARE_INIT(name) \
void THFloatTensor_ ## name(lua_State* L); \
void THDoubleTensor_ ## name(lua_State* L);

#define INVOKE_INIT(name, state) \
THFloatTensor_ ## name(L); \
THDoubleTensor_ ## name(L);

namespace acdc {

DECLARE_INIT(initDCT);
DECLARE_INIT(initPermutation);

}

using namespace acdc;

extern "C" {

LUA_EXTERNC DLL_EXPORT int luaopen_libacdc_cpu(lua_State* L);

int luaopen_libacdc_cpu(lua_State* L)
{
    INVOKE_INIT(initDCT, L);
    INVOKE_INIT(initPermutation, L);

    return 0;
}

}

