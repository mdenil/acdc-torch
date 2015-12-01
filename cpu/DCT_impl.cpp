namespace {

/***********************************************
 * DCT/IDCT Helpers
 **********************************************/

bool Tensor_(run_dct)(Tensor* input, Tensor* output)
{
    int batch_size;
    int input_size;
    if (Tensor_(nDimension)(input) == 1) {
        batch_size = 1;
        input_size = Tensor_(size)(input, 0);
    }
    else if (Tensor_(nDimension)(input) == 2) {
        batch_size = Tensor_(size)(input, 0);
        input_size = Tensor_(size)(input, 1);
    }
    else {
        return false;
    }

    typename fftw<DTYPE>::plan_type fft_plan = fftw<DTYPE>::plan_many_dft_r2r_1d(
        input_size,
        batch_size,
        Tensor_(data)(input),
        Tensor_(data)(output),
        FFTW_ESTIMATE);

    fftw<DTYPE>::execute(fft_plan);

    fftw<DTYPE>::destroy_plan(fft_plan);

    // normalize
    for (int i = 0; i < batch_size; ++i) {
        DTYPE* example = Tensor_(data)(output) + i * input_size;

        example[0] *= std::sqrt(1.0 / (4.0 * input_size));

        for (int j = 1; j < input_size; ++j) {
            example[j] *= std::sqrt(1.0 / (2.0 * input_size));
        }
    }

    return true;
}

bool Tensor_(run_idct)(Tensor* input, Tensor* output)
{
    int batch_size;
    int input_size;
    if (Tensor_(nDimension)(input) == 1) {
        batch_size = 1;
        input_size = Tensor_(size)(input, 0);
    }
    else if (Tensor_(nDimension)(input) == 2) {
        batch_size = Tensor_(size)(input, 0);
        input_size = Tensor_(size)(input, 1);
    }
    else {
        return false;
    }

    // normalize
    for (int i = 0; i < batch_size; ++i) {
        DTYPE* example = Tensor_(data)(input) + i * input_size;
        example[0] *= std::sqrt(1.0 / input_size);

        for (int j = 1; j < input_size; ++j) {
            example[j] *= std::sqrt(1.0 / (2.0 * input_size));
        }
    }

    typename fftw<DTYPE>::plan_type fft_plan = fftw<DTYPE>::plan_many_dift_r2r_1d(
        input_size,
        batch_size,
        Tensor_(data)(input),
        Tensor_(data)(output),
        FFTW_ESTIMATE);

    fftw<DTYPE>::execute(fft_plan);

    fftw<DTYPE>::destroy_plan(fft_plan);

    return true;
}


/***********************************************
 * DCT Layer
 **********************************************/

int Tensor_(DCT_updateOutput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto output = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "output", TensorTypename));
    int outputIdx = lua_gettop(L);

    input = Tensor_(newContiguous)(input);
    output = Tensor_(newContiguous)(output);

    Tensor_(resizeAs)(output, input);

    if (!Tensor_(run_dct)(input, output)) {
        luaL_error(L, "DCT updateOutput failed");
    }

    lua_pushvalue(L, outputIdx);
    Tensor_(free)(input);
    Tensor_(free)(output);

    return 1;
}

int Tensor_(DCT_updateGradInput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto gradOutput = static_cast<Tensor*>(
        luaT_checkudata(L, 3, TensorTypename));
    auto gradInput = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradInput", TensorTypename));
    int gradInputIdx = lua_gettop(L);

    gradOutput = Tensor_(newClone)(gradOutput);
    gradInput = Tensor_(newContiguous)(gradInput);

    Tensor_(resizeAs)(gradInput, gradOutput);

    if (!Tensor_(run_idct)(gradOutput, gradInput)) {
        luaL_error(L, "DCT updateGradInput failed");
    }

    Tensor_(free)(gradOutput);
    Tensor_(free)(gradInput);

    lua_pushvalue(L, gradInputIdx);

    return 1;
}

/***********************************************
 * IDCT Layer
 **********************************************/

int Tensor_(IDCT_updateOutput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto output = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "output", TensorTypename));
    int outputIdx = lua_gettop(L);

    input = Tensor_(newClone)(input);
    output = Tensor_(newContiguous)(output);

    Tensor_(resizeAs)(output, input);

    if (!Tensor_(run_idct)(input, output)) {
        luaL_error(L, "IDCT updateOutput failed");
    }

    Tensor_(free)(input);
    Tensor_(free)(output);

    lua_pushvalue(L, outputIdx);
    return 1;
}

int Tensor_(IDCT_updateGradInput)(lua_State* L)
{
    auto input = static_cast<Tensor*>(
        luaT_checkudata(L, 2, TensorTypename));
    auto gradOutput = static_cast<Tensor*>(
        luaT_checkudata(L, 3, TensorTypename));
    auto gradInput = static_cast<Tensor*>(
        luaT_getfieldcheckudata(L, 1, "gradInput", TensorTypename));
    int gradInputIdx = lua_gettop(L);

    gradOutput = Tensor_(newContiguous)(gradOutput);
    gradInput = Tensor_(newContiguous)(gradInput);

    Tensor_(resizeAs)(gradInput, gradOutput);

    if (!Tensor_(run_dct)(gradOutput, gradInput)) {
        luaL_error(L, "IDCT updateGradInput failed");
    }

    Tensor_(free)(gradOutput);
    Tensor_(free)(gradInput);

    lua_pushvalue(L, gradInputIdx);
    return 1;
}


static const struct luaL_Reg Tensor_(DCT_functions_)[] = {
    {"DCT_updateOutput", Tensor_(DCT_updateOutput)},
    {"DCT_updateGradInput", Tensor_(DCT_updateGradInput)},
    {"IDCT_updateOutput", Tensor_(IDCT_updateOutput)},
    {"IDCT_updateGradInput", Tensor_(IDCT_updateGradInput)},
    {nullptr, nullptr}
};

} // anon namespace

void Tensor_(initDCT)(lua_State* L) {
    luaT_pushmetatable(L, TensorTypename);
    luaT_registeratname(L, Tensor_(DCT_functions_), "nn");
    lua_pop(L, 1);
}

