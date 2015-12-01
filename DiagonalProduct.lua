local DiagonalProduct, parent = torch.class('acdc.DiagonalProduct', 'nn.Module')

function DiagonalProduct:__init(dim)
    parent.__init(self)

    self.dim = dim
    self.weight = torch.Tensor(dim)
    self.bias = torch.Tensor(dim)
    self.gradWeight = torch.Tensor(dim)
    self.gradBias = torch.Tensor(dim)

    self:reset()
end

function DiagonalProduct:reset()
    self.weight:fill(1.0)
    self.bias:fill(0.0)
    self.gradWeight:zero()
    self.gradBias:zero()
end

function DiagonalProduct:updateOutput(input)
    self.output:resizeAs(input)
    self.output:copy(input)

    if input:dim() == 1 then
        self.output:cmul(self.weight)
        self.output:add(self.bias)

    elseif input:dim() == 2 then
        self.output:cmul(self.weight:view(1, self.dim):expandAs(self.output))
        self.output:add(self.bias:view(1, self.dim):expandAs(self.output))
    
    else
        error('input must be a vector or a matrix')
    end

    return self.output
end

function DiagonalProduct:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(gradOutput)

    if input:dim() == 1 then
        self.gradInput:cmul(self.weight)

    elseif input:dim() == 2 then
        self.gradInput:cmul(self.weight:view(1, self.dim):expandAs(self.gradInput))

    else
        error('input must be a vector or a matrix')

    end

    return self.gradInput
end

function DiagonalProduct:accGradParameters(input, gradOutput, scale)
    scale = scale or 1

    local this_gradWeight = input:clone():cmul(gradOutput):mul(scale)
    local this_gradBias = gradOutput:clone():mul(scale)

    if input:dim() == 1 then
        self.gradWeight:add(this_gradWeight)
        self.gradBias:add(this_gradBias)

    elseif input:dim() == 2 then
        self.gradWeight:add(this_gradWeight:sum(1))
        self.gradBias:add(this_gradBias:sum(1))

    else
        error('input must be a vector or a matrix')

    end
end


DiagonalProduct.sharedAccUpdateGradParameters = DiagonalProduct.accUpdateGradParameters

