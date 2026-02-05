using Distributed

include("main.jl")

function @main(_)
    addprocs(10)
    θs = 0.001:0.001:0.01
    γ = 0.16
    final_time = ceil(Int, 2.5/γ)
    for th in θs
        execute(
            th,
            γ,
            "Z",
            100,
            2048,
            final_time,
            [50],
            5_000;
            flush_every=final_time + 1,
            nthreads=1,
        )
    end
end