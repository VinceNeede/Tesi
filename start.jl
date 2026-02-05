using Distributed

addprocs(14)
include("main.jl")

const chain_length = 100
const maxdim = 5096
# const final_time = 15
for γ in [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4]
    final_time = ceil(Int, 2.5/γ)
    execute(
		γ, 
        "Z", 
        chain_length, 
        maxdim, 
        final_time,
        [chain_length ÷ 2], 
        5000;
        flush_every=final_time+1,
        nthreads=1,
        )
end
