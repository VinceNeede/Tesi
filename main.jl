using MonitoredSystems
import ITensors: IndexSet
import ITensorMPS: MPS, siteinds
import LoggingExtras: FileLogger, with_logger
using Glob

"""
    set_folder(density::Float64, per_site_prob::Float64, chain_length::Int, maxdim::Int, final_time::Int)::String
Create a folder name based on the simulation parameters. If the folder does not
exist, it is created. The folder name has the format:
`data_<density>_<per_site_prob>_<measure_op>_<chain_length>_<maxdim>_<final_time>`
- `density::Float64`: Density of quasiparticles.
- `per_site_prob::Float64`: Probability of measurement per site.
- `measure_op::String`: Measurement operator ("X" or "Z").
- `chain_length::Int`: Length of the spin chain.
- `maxdim::Int`: Maximum bond dimension for MPS evolution.
- `final_time::Int`: Total number of time steps for the evolution.
Returns the folder name as a `String`.
"""
function set_folder(
    density::Float64,
    per_site_prob::Float64,
    measure_op::String,
    chain_length::Int,
    maxdim::Int,
    final_time::Int,
)
    # if there are only floats and integers, Julia may cast integers to floats when
    # constructing the folder name, so we explicitly set the vector type as
    # Union{Float64, Int}
    # In this case measure_op is a String, so no risk of casting
    folder_name =
        "data_" * join(
            [
                round(density; sigdigits = 4);
                round(per_site_prob; sigdigits = 4);
                measure_op;
                chain_length;
                maxdim;
                final_time
            ],
            "_",
        )

    isdir(folder_name) || mkdir(folder_name)

    return folder_name
end

"""
    archive_results(folder_name::String)
Archive all `.dat` files in the specified folder into a tar file named
`<folder_name>.tar` and remove the original `.dat` files.
"""
function archive_results(folder_name::String)
    cd(folder_name) do
		archive_path = "../$(folder_name * ".tar.gz")"
		if isfile(archive_path)
			run(pipeline(`find . -maxdepth 1 -name "*.dat" -print0`, `tar -rzf $archive_path --null -T -`))
		else
			run(pipeline(`find . -maxdepth 1 -name "*.dat" -print0`, `tar -czf $archive_path --null -T -`))
		end
        foreach(rm, glob("*.dat"))
    end
    nothing
end

"""
    workers_scope(
        starting_mps::MPS,
        ops::Operators,
        params::MCMCParameters;
        folder_name::String=".";
        flush_every::Int=1,
        nthreads::Int=1,
    )
Define the `main` function on all workers for parallel execution and the 
global logger. The MPS is copied at each call since it gets modified during evolution.
Operators and parameters are serialized from the outer scope.
Each worker logs to a separate file named `logfile_worker_<worker_id>.log` in the specified folder.
- flush_every::Int: Number of time steps between flushing results to disk.
- nthreads::Int: Number of threads to use in each worker.
"""
function workers_scope(
    starting_mps::MPS,
    ops::Operators,
    params::MCMCParameters;
    folder_name::String = ".",
    flush_every::Int = 1,
    nthreads::Int = 1,
)
    # can't use @everywhere for `using MonitoredSystems` since it would try to
    # load the package on the master process, but it would result in a 
    # toplevel expression not at top level error
    remotecall_eval(Main, procs(), :(using MKL, MonitoredSystems, LoggingExtras))

    @everywhere begin
        MKL.set_num_threads($nthreads)

        logger = timestamp_logger(
            FileLogger(
                joinpath($folder_name, "logfile_worker_$(myid()).log");
                append = true,
            ),
        )

        global_logger(logger)

        main(_) = evolve_trajectory(
            MPSQtMCMC(copy($starting_mps); root_folder = $folder_name),
            $ops,
            $params;
            flush_every=$flush_every
        )

    end
end

function get_projectors(measure_op::String)
    if measure_op == "X"
        projs = [[1 1; 1 1] / 2, [1 -1; -1 1] / 2]
    elseif measure_op == "Z"
        projs = [[1 0; 0 0], [0 0; 0 1]]
    else
        error("Unknown measurement operator: $measure_op")
    end
    return projs
end

"""
    execute(
        density::Float64,
        per_site_prob_measure::Float64,
        measure_op::String,
        chain_length::Int,
        maxdim::Int,
        final_time::Int,
        subsystems::AbstractVector{Int},
        num_trajectories::Int;
        flush_every::Int=1,
        nthreads::Int=1,
    )
Execute quantum trajectory simulations with the specified parameters.
- `density::Float64`: Density of quasiparticles.
- `per_site_prob_measure::Float64`: Probability of measurement per site.
- `measure_op::String`: Measurement operator ("X" or "Z").
- `chain_length::Int`: Length of the spin chain.
- `maxdim::Int`: Maximum bond dimension for MPS evolution.
- `final_time::Int`: Total number of time steps for the evolution.
- `subsystems::AbstractVector{Int}`: Subsystems to monitor.
- `num_trajectories::Int`: Number of quantum trajectories to simulate.
- `flush_every::Int`: Number of time steps between flushing results to disk.
- `nthreads::Int`: Number of threads to use in each worker.
"""
function execute(
    density::Float64,
    per_site_prob_measure::Float64,
    measure_op::String,
    chain_length::Int,
    maxdim::Int,
    final_time::Int,
    subsystems::AbstractVector{Int},
    num_trajectories::Int;
    flush_every::Int=1,
    nthreads::Int=1,
)
    sites = siteinds("S=1/2", chain_length)
    projs = get_projectors(measure_op)
    ops = Operators(sites, projs)
    params =
        MCMCParameters(maxdim, per_site_prob_measure * chain_length, final_time, subsystems; cutoff=1.e-12)
    folder_name =
        set_folder(density, per_site_prob_measure, measure_op, chain_length, maxdim, final_time)

    starting_mps = BiasedNeelState(sites, density)

    workers_scope(starting_mps, ops, params; folder_name = folder_name, flush_every=flush_every, nthreads=nthreads)

    pmap(
        main,
        1:num_trajectories;
        on_error = (e -> @error "Error" exception = (e, catch_backtrace())),
    )
    @info "All tasks finished" density per_site_prob_measure chain_length maxdim final_time num_trajectories "number of trajectories in folder" =
        length(glob("*.dat", folder_name))

    archive_results(folder_name)
end
