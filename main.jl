using MonitoredSystems
import ITensors: IndexSet
import ITensorMPS: MPS, siteinds
import LoggingExtras: FileLogger, with_logger
using Glob

"""
    set_folder(density::Float64, per_site_prob::Float64, chain_length::Int, maxdim::Int, final_time::Int)::String
Create a folder name based on the simulation parameters. If the folder does not
exist, it is created. The folder name has the format:
`data_<density>_<per_site_prob>_<chain_length>_<maxdim>_<final_time>.dat`
- `density::Float64`: Density of quasiparticles.
- `per_site_prob::Float64`: Probability of measurement per site.
- `chain_length::Int`: Length of the spin chain.
- `maxdim::Int`: Maximum bond dimension for MPS evolution.
- `final_time::Int`: Total number of time steps for the evolution.
Returns the folder name as a `String`.
"""
function set_folder(
    density::Float64,
    per_site_prob::Float64,
    chain_length::Int,
    maxdim::Int,
    final_time::Int,
)
    folder_name =
        "data_" * join(
            Union{Float64,Int}[   # prevent int casted to float
                round(density; sigdigits = 2);
                round(per_site_prob; sigdigits = 2);
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
    run(pipeline(`tar -cvf $(folder_name * ".tar") $(glob("*.dat", folder_name))`, devnull))
    run(pipeline(`rm $(glob("*.dat", folder_name))`, devnull))
    nothing
end

"""
    workers_scope(
        starting_mps::MPS,
        ops::Operators,
        params::MCMCParameters;
        folder_name::String="."
    )
Define the `main` function on all workers for parallel execution and the 
global logger. The MPS is copied at each call since it gets modified during evolution.
Operators and parameters are serialized from the outer scope.
Each worker logs to a separate file named `logfile_worker_<worker_id>.log` in the specified folder.
"""
function workers_scope(
    starting_mps::MPS,
    ops::Operators,
    params::MCMCParameters;
    folder_name::String = ".",
)
    # can't use @everywhere for `using MonitoredSystems` since it would try to
    # load the package on the master process, but it would result in a 
    # toplevel expression not at top level error
    remotecall_eval(Main, procs(), :(using MonitoredSystems, LoggingExtras))


    @everywhere begin
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
            $params,
        )

    end
end

"""
    execute(
        density::Float64,
        per_site_prob_measure::Float64,
        chain_length::Int,
        maxdim::Int,
        final_time::Int,
        subsystems::AbstractVector{Int},
        num_trajectories::Int,
    )
"""
function execute(
    density::Float64,
    per_site_prob_measure::Float64,
    chain_length::Int,
    maxdim::Int,
    final_time::Int,
    subsystems::AbstractVector{Int},
    num_trajectories::Int,
)
    sites = siteinds("S=1/2", chain_length)
    projs = [[1 0; 0 0], [0 0; 0 1]]
    ops = Operators(sites, projs)
    params =
        MCMCParameters(maxdim, per_site_prob_measure * chain_length, final_time, subsystems)
    folder_name =
        set_folder(density, per_site_prob_measure, chain_length, maxdim, final_time)

    starting_mps = BiasedNeelState(sites, density)

    workers_scope(starting_mps, ops, params; folder_name = folder_name)

    pmap(
        main,
        1:num_trajectories;
        on_error = (e -> @error "Error" exception = (e, catch_backtrace())),
    )
    @info "All tasks finished" density per_site_prob_measure chain_length maxdim final_time num_trajectories "number of trajectories in folder" =
        length(glob("*.dat", folder_name))

    archive_results(folder_name)
end
