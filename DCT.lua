local DCT, parent = torch.class('acdc.DCT', 'nn.Module')

function DCT:updateOutput(input)
    return input.nn.DCT_updateOutput(self, input)
end

function DCT:updateGradInput(input, gradOutput)
    return input.nn.DCT_updateGradInput(self, input, gradOutput)
end

