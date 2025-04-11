using MonitoredSystems, MCMC
import HDF5
import ITensorMPS: siteinds, MPS, linkdims, apply

const parsed_args = parse_args()
const θ = parsed_args["theta"]
const chain_length = parsed_args["chain_length"]
const maxdim = parsed_args["maxdim"]
const measure_rate = parsed_args["measure_rate"]
const final_time = parsed_args["final_time"]
const num_trajectories = parsed_args["num_trajectories"]
const subsystems = 3:2:(chain_length÷2)

const folder_name = "data_$(round(measure_rate; sigdigits=2))_$(chain_length)"

isdir(folder_name) || mkdir(folder_name)

const sites = siteinds("S=1/2", chain_length)
const mpo_odd = r54_odd(sites)
const mpo_even = r54_even(sites)

function MonitoredSystems.evolve!(mcmc::MPSQtMCMC)
    starting_χ = maxlinkdim(state(mcmc))
    starting_χ ≥ maxdim && return false

    state(mcmc)[:] = apply(mpo_odd, state(mcmc); mcmc.evol_keys...)
    state(mcmc)[:] = apply(mpo_even, state(mcmc); mcmc.evol_keys...)

    finishing_χ = maxlinkdim(state(mcmc))
    if finishing_χ ≥ maxdim
        @warn "id $(id(mcmc)) reached maximum dimension $(maxdim)"
        return false
    end
    return true
end

function MCMC.observables(::MPSQtMCMC)
    return [x -> Renyi_entropy(state(x), p, 1) for p in subsystems]
end

MCMC.should_save(::MPSQtMCMC, ::Int) = true

import Base: Semaphore, acquire, release

const semaphore = Semaphore(Threads.nthreads())

function main(file::HDF5.File)
    acquire(semaphore)
    mcmc = MPSQtMCMC(
        MPS(BertiniState(sites, θ), 0, 2),
        measure_rate * chain_length,
        [[1 0; 0 0], [0 0; 0 1]];
        checkpoint_file=file,
        cutoff=1.e-15,
        maxdim=maxdim,
    )

    write(save_file(mcmc), join(subsystems, ", "), "\n")

    Base.with_logger(
        timestamp_logger(
            mcmc_logger(id(mcmc), 1)
        )
    ) do
        try
            run!(mcmc, final_time)
        finally
            close(save_file(mcmc))
        end
    end

    release(semaphore)
end

cd(folder_name)
HDF5.h5open("checkpoint.h5", "cw") do file
    tasks = [Threads.@spawn main($file) for _ in 1:num_trajectories]
    wait.(tasks)
end
cd("../")