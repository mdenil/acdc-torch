local IDCT, parent = torch.class('acdc.IDCT', 'nn.Module')

function IDCT:updateOutput(input)
    return input.nn.IDCT_updateOutput(self, input)
end

function IDCT:updateGradInput(input, gradOutput)
    return input.nn.IDCT_updateGradInput(self, input, gradOutput)
end

