using Distributed

addprocs(8)
include("main.jl")

const chain_length = 100
const maxdim = 5096
# const final_time = 15
for γ in [0.2, 0.4, 0.8]
    final_time = ceil(Int, 2.5/γ)
    execute(
		γ, 
        "Z", 
        chain_length, 
        maxdim, 
        final_time,
        [chain_length ÷ 2], 
        500;
        flush_every=final_time+1,
        nthreads=2,
        )
end
