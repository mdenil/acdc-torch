namespace {

/*************************************
 * Helpers
 ************************************/

void Tensor_(accumulate_permutation)(
    DTYPE const* input,
    DTYPE* output,
    DTYPE const* perm,
    int const input_size,
    int const batch_size,
    int const output_size)
{
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            output[i*output_size + static_cast<int>(perm[j])] += input[i*input_size + j];
        }
    }
}

void Tensor_(accumulate_reverse_permutation)(
    DTYPE const* input,
    DTYPE* output,
    DTYPE const* perm,
    int const input_size,
    int const batch_size,
    int const output_size)
{
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            output[i*output_size + j] += input[i*input_size + static_cast<int>(perm[j])];
        }
    }
}

/*************************************
 * Permutation layer
 ************************************/

int Tensor_(Permutation_updateOutput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto perm = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "perm", TensorTypename));
    int output_size = luaT_getfieldcheckint(L, 1, "output_size");
    auto output = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "output", TensorTypename));
    int outputIdx = lua_gettop(L);

    int batch_size;
    int input_size;
    long output_shape[2];
    if (Tensor_(nDimension)(input) == 1) {
        batch_size = 1;
        input_size = Tensor_(size)(input, 0);
        output_shape[0] = output_size;
        output_shape[1] = -1;
    }
    else if (Tensor_(nDimension)(input) == 2) {
        batch_size = Tensor_(size)(input, 0);
        input_size = Tensor_(size)(input, 1);
        output_shape[0] = batch_size;
        output_shape[1] = output_size;
    }
    else {
        luaL_error(L, "input must have 1 or 2 dimensions");
    }

    input = Tensor_(newContiguous)(input);
    Tensor_(resize2d)(output, output_shape[0], output_shape[1]);

    std::fill_n(Tensor_(data)(output), batch_size*output_size, static_cast<DTYPE>(0));

    Tensor_(accumulate_permutation)(
        Tensor_(data)(input),
        Tensor_(data)(output),
        Tensor_(data)(perm),
        input_size,
        batch_size,
        output_size);

    lua_pushvalue(L, outputIdx);
    Tensor_(free)(input);

    return 1;
}

int Tensor_(Permutation_updateGradInput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto perm = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "perm", TensorTypename));
    int output_size = luaT_getfieldcheckint(L, 1, "output_size");
    auto gradOutput = static_cast<Tensor*>(
        luaT_checkudata(L, 3, TensorTypename));
    auto gradInput = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradInput", TensorTypename));
    int gradInputIdx = lua_gettop(L);

    int batch_size;
    int input_size;
    long input_shape[2];
    if (Tensor_(nDimension)(input) == 1) {
        batch_size = 1;
        input_size = Tensor_(size)(input, 0);
        input_shape[0] = input_size;
        input_shape[1] = -1;
    }
    else if (Tensor_(nDimension)(input) == 2) {
        batch_size = Tensor_(size)(input, 0);
        input_size = Tensor_(size)(input, 1);
        input_shape[0] = batch_size;
        input_shape[1] = input_size;
    }
    else {
        luaL_error(L, "input must have 1 or 2 dimensions");
    }

    gradOutput = Tensor_(newContiguous)(gradOutput);
    Tensor_(resize2d)(gradInput, input_shape[0], input_shape[1]);

    std::fill_n(Tensor_(data)(gradInput), batch_size*input_size, static_cast<DTYPE>(0));

    Tensor_(accumulate_reverse_permutation)(
        Tensor_(data)(gradOutput),
        Tensor_(data)(gradInput),
        Tensor_(data)(perm),
        output_size,
        batch_size,
        input_size);

    Tensor_(free)(gradOutput);

    lua_pushvalue(L, gradInputIdx);

    return 1;
}

static const struct luaL_Reg Tensor_(Permutation_functions_)[] = {
    {"Permutation_updateOutput", Tensor_(Permutation_updateOutput)},
    {"Permutation_updateGradInput", Tensor_(Permutation_updateGradInput)},
    {nullptr, nullptr}
};

} // anon namespace

void Tensor_(initPermutation)(lua_State* L) {
    luaT_pushmetatable(L, TensorTypename);
    luaT_registeratname(L, Tensor_(Permutation_functions_), "nn");
    lua_pop(L, 1);
}

