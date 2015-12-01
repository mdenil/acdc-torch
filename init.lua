require 'nn'
require 'libacdc_cpu'

require 'cunn'
require 'libacdc_gpu'

acdc = {}

include('ACDC.lua')
include('DCT.lua')
include('IDCT.lua')
include('DiagonalProduct.lua')
include('Permutation.lua')

include('FastACDC.lua')
include('FastACDC_grouped.lua')

return acdc
