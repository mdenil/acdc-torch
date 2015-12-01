
function acdc.ACDC(dim)
    local module = nn.Sequential()

    module:add(acdc.DiagonalProduct(dim))
    module:add(acdc.DCT())
    module:add(acdc.DiagonalProduct(dim))
    module:add(acdc.IDCT())

    return module
end


function acdc.ScaledPermutation(input_size, output_size)
    local module = nn.Sequential()

    if input_size > output_size then
        module:add(acdc.Permutation(input_size, output_size))
        module:add(acdc.DiagonalProduct(output_size))
    else
        module:add(acdc.DiagonalProduct(input_size))
        module:add(acdc.Permutation(input_size, output_size))
    end

    return module
end

