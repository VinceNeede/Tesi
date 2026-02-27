using Distributed

include("main.jl")

function @main(_)
    addprocs(2)
    θ = 0.5
    γ = 0.2
    final_time = 12#ceil(Int, 2.5/γ)
    for χ in [256, 512, 1024]
        execute(
            θ,
            γ,
            "X",
            50,
            χ,
            final_time,
            3:25,
            10;
            flush_every=final_time+1,
            nthreads=2,
        )
    end
end