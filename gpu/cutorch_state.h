#pragma once

extern "C"
{
#include <lua.h>
}
#include <lua.hpp>

#include "THC.h"

THCState* getCutorchState(lua_State* L);
