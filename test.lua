require 'torch'
require 'acdc'
require 'nn'

local mytester = torch.Tester()

local precision = 1e-5

local acdc_test = {}

function acdc_test.DiagonalProduct()
    local function run_test_batch(module, input)
        local err = nn.Jacobian.testJacobian(module, input)
        mytester:assertlt(err, precision, "error on state")

        local err = nn.Jacobian.testJacobianParameters(module, input, module.bias, module.gradBias)
        mytester:assertlt(err, precision, "error on bias")

        local err = nn.Jacobian.testJacobianParameters(module, input, module.weight, module.gradWeight)
        mytester:assertlt(err, precision, "error on weight")

        local err = nn.Jacobian.testJacobianUpdateParameters(module, input, module.bias)
        mytester:assertlt(err, precision, "error on bias [direct update]")

        local err = nn.Jacobian.testJacobianUpdateParameters(module, input, module.weight)
        mytester:assertlt(err, precision, "error on weight [direct update]")

        for t, err in pairs(nn.Jacobian.testAllUpdate(module, input, 'bias', 'gradBias')) do
            mytester:assertlt(err, precision, string.format("error on bias [%s]", t))
        end

        for t, err in pairs(nn.Jacobian.testAllUpdate(module, input, 'weight', 'gradWeight')) do
            mytester:assertlt(err, precision, string.format("error on weight [%s]", t))
        end

        local ferr, berr = nn.Jacobian.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward error')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward error')
    end

    local batch_size = 5
    local input_size = 7

    local module = acdc.DiagonalProduct(input_size)
    local input

    module.weight:uniform(-1, 1)
    module.bias:uniform(-1, 1)

    -- 1d
    input = torch.randn(input_size)
    run_test_batch(module, input)

    -- 2d
    input = torch.randn(batch_size, input_size)
    run_test_batch(module, input)
end

function acdc_test.Permutation()
    local function run_test_batch(module, input)
        local err = nn.Jacobian.testJacobian(module, input)
        mytester:assertlt(err, precision, "error on state")

        local ferr, berr = nn.Jacobian.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward error')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward error')
    end

    local batch_size = 5
    local input_size = 30

    for _, output_size in pairs({input_size - 2, input_size + 2, input_size * 5, input_size / 5}) do
        local module = acdc.Permutation(input_size, output_size)
        local input
        
        -- 1d
        input = torch.randn(input_size)
        run_test_batch(module, input)

        -- 2d
        input = torch.randn(batch_size, input_size)
        run_test_batch(module, input)
    end
end

function acdc_test.Permutation_gpu()
    local function run_test_batch(cpu_module, gpu_module, input)
        local out_cpu = cpu_module:forward(input:clone():float())
        local out_gpu = gpu_module:forward(input:clone():cuda())
        local err = (out_cpu - out_gpu:float()):abs():max()
        mytester:assertlt(err, precision, "error in forward")

        local gradOutput = input:clone():randn(out_cpu:size())
        local back_cpu = cpu_module:backward(input:clone():float(), gradOutput:clone():float())
        local back_gpu = gpu_module:backward(input:clone():cuda(), gradOutput:clone():cuda())
        local err = (back_cpu - back_gpu:float()):abs():max()
        mytester:assertlt(err, precision, "error in backward")

        --local ferr, berr = nn.Jacobian.testIO(gpu_module, input:clone():cuda())
        --mytester:asserteq(ferr, 0, torch.typename(gpu_module) .. ' - i/o forward error')
        --mytester:asserteq(berr, 0, torch.typename(gpu_module) .. ' - i/o backward error')
    end

    local batch_size = 5
    local input_size = 30

    for _, output_size in pairs({input_size - 2, input_size + 2, input_size * 5, input_size / 5}) do
        local cpu_module = acdc.Permutation(input_size, output_size):float()
        local gpu_module = cpu_module:clone():cuda()
        local input
        
        -- 1d
        input = torch.randn(input_size):float()
        run_test_batch(cpu_module, gpu_module, input)

        -- 2d
        input = torch.randn(batch_size, input_size):float()
        run_test_batch(cpu_module, gpu_module, input)
    end
end

function acdc_test.DCT()
    local function run_test_batch(module, input)
        local err = nn.Jacobian.testJacobian(module, input)
        mytester:assertlt(err, precision, "error on state")

        local ferr, berr = nn.Jacobian.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward error')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward error')
    end

    local batch_size = 5
    local input_size = 7

    local module = acdc.DCT()
    local input
    
    -- 1d
    input = torch.randn(input_size)
    run_test_batch(module, input)

    -- 2d
    input = torch.randn(batch_size, input_size)
    run_test_batch(module, input)

    local output = module:forward(input)
    local out_norm = output:clone():cmul(output):sum(2)
    local in_norm = input:clone():cmul(input):sum(2)
    for i = 1, out_norm:size(1) do
        mytester:assertlt((out_norm[i] - in_norm[i]):abs():squeeze(), precision, "normalization error")
    end

end

function acdc_test.IDCT()
    local function run_test_batch(module, input)
        local err = nn.Jacobian.testJacobian(module, input)
        mytester:assertlt(err, precision, "error on state")

        local ferr, berr = nn.Jacobian.testIO(module, input)
        mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward error')
        mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward error')
    end

    local batch_size = 5
    local input_size = 7

    local module = acdc.IDCT()
    local input
    
    -- 1d
    input = torch.randn(input_size)
    run_test_batch(module, input)

    -- 2d
    input = torch.randn(batch_size, input_size)
    run_test_batch(module, input)

    local output = module:forward(input)
    local out_norm = output:clone():cmul(output):sum(2)
    local in_norm = input:clone():cmul(input):sum(2)
    for i = 1, out_norm:size(1) do
        mytester:assertlt((out_norm[i] - in_norm[i]):abs():squeeze(), precision, "normalization error")
    end
end

mytester:add(acdc_test)

mytester:run()

