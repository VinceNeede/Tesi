using Distributed
using MKL

addprocs(20)
@everywhere maxdim = 8192
chain_length = 100
include("main.jl")
set_mpo()
const θs = 0.1:0.1:0.9
const γs = [0.08, 0.16]

# @everywhere MKL.set_num_threads(3)
# for γ in γs, θ in θs
# 	exe(θ, γ, 40, 1000)
# end
rmprocs(7:nprocs()...)
@everywhere MKL.set_num_threads(5)
for θ in θs
	exe(θ, 0.04, 40, 1000)
end
