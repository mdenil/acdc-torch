local Permutation, parent = torch.class('acdc.Permutation', 'nn.Module')

function Permutation:__init(size)
    parent.__init(self)

    local _
    self.perm = torch.randperm(size):long()
    _, self.inv_perm = self.perm:sort()
end

function Permutation:updateOutput(input)
    local dim = input:nDimension()
    self.output = input:index(dim, self.perm):contiguous()
    return self.output
end

function Permutation:updateGradInput(input, gradOutput)
    local dim = input:nDimension()
    self.gradInput = gradOutput:index(dim, self.inv_perm):contiguous()
    return self.gradInput
end

function Permutation:type(t)
    if t == 'torch.CudaTensor' then
        self.perm = self.perm:cuda()
        self.inv_perm = self.inv_perm:cuda()
    else
        self.perm = self.perm:long()
        self.inv_perm = self.inv_perm:long()
    end
    return self
end
