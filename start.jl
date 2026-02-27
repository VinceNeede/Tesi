using Distributed

include("main.jl")

function @main(_)
    addprocs(3)
	tf = 12#ceil(Int, 2.5/γ)
	for y in [0.5, 0.4]
		execute(
			0.5,
			y,
			"X",
			50,
			4096,
			tf,
			3:25,
			2_000;
			flush_every=tf + 1,
			nthreads=5,
		)
	end
end
