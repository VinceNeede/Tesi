"""
    Operators(
        sites::ITensors.Indices,
        projectors::Vector{Matrix{T}},
    )
Create the MPOs and measurement operators needed for the MPSQtMCMC sampler.
- `sites::ITensors.Indices`: The site indices of the spin chain.
- `projectors::Vector{Matrix{T}}`: The projectors to be used during measurements.
It returns an `Operators` struct containing the odd and even site MPOs,
the projectors, and the density operators for quasiparticle measurements.
"""
struct Operators
    odd_sites_mpo::MPO
    even_sites_mpo::MPO
    projectors::Vector{Matrix{<:Number}}
    density_ops::Vector{Array{Float64}}
    function Operators(
        sites::ITensors.Indices,
        projectors::Vector{Matrix{T}},
    ) where {T<:Number}
        odd_sites_mpo = r54_odd(sites)
        even_sites_mpo = r54_even(sites)
        density_ops = qp_tensors(sites)
        new(odd_sites_mpo, even_sites_mpo, projectors, density_ops)
    end
end

"""
    MCMCParameters(
        maxdim::Int,
        cutoff::Float64,
        distribution::DiscreteUnivariateDistribution,
        final_time::Int,
        subsystems::AbstractVector{Int} 
    )
Wrapper struct to hold the parameters for the MPSQtMCMC sampler.
- `maxdim::Int`: Maximum bond dimension for the MPS evolution.
- `cutoff::Float64`: Cutoff for singular value decomposition during MPS evolution.
- `distribution::DiscreteUnivariateDistribution`: Distribution for sampling measurement positions.
- `final_time::Int`: Total number of time steps for the evolution.
- `subsystems::AbstractVector{Int}`: List of subsystem sizes for Renyi entropy
measurements.
"""
struct MCMCParameters
    maxdim::Int
    cutoff::Float64
    distribution::DiscreteUnivariateDistribution
    final_time::Int
    subsystems::AbstractVector{Int}
end

"""
    MCMCParameters(
        maxdim::Int,
        measure_rate::Float64,
        final_time::Int,
        subsystems::AbstractVector{Int};
        cutoff::Float64=eps(),
    )
Helper function to create an `MCMCParameters` struct.
- `maxdim::Int`: Maximum bond dimension for the MPS evolution.
- `measure_rate::Float64`: Probability that a measurement occurs at any site in a time step.
- `final_time::Int`: Total number of time steps for the evolution.
- `subsystems::AbstractVector{Int}`: List of subsystem sizes for Renyi entropy measurements.
The distribution is set to a Poisson distribution with mean `measure_rate`.
"""
function MCMCParameters(
    maxdim::Int,
    measure_rate::Float64,
    final_time::Int,
    subsystems::AbstractVector{Int};
    cutoff::Float64 = eps(),
)
    return MCMCParameters(maxdim, cutoff, Poisson(measure_rate), final_time, subsystems)
end


"""
    MCMCStatus
Wrapper of Bool to represent the status of the MPSQtMCMC sampler.
If `errored` is true, the sampler has encountered an error during evolution.
It is used to keep MPSQtMCMC an immutable struct.
"""
mutable struct MCMCStatus
    errored::Bool
end

"""
    MPSQtMCMC(state::MPS; root_folder::String=".")
Create an MPS quantum trajectory MCMC. Each instance has a unique UUID,
a Xoshiro random number generator, the MPS state, status, and file paths for saving
entropy and density measurements. The UUID and the rng are initialized automatically,
and the entropy and density files are created in the specified `root_folder` with
unique names based on the UUID.

- `state::MPS`: The initial MPS state for the sampler.
- `root_folder::String`: The root folder where the result files will be saved.
"""
struct MPSQtMCMC
    id::UUID
    rng::Xoshiro
    state::MPS
    status::MCMCStatus
    entropy_file::String
    density_file::String

    function MPSQtMCMC(state::MPS; root_folder::String=".")
        id = uuid4()
        rng = Xoshiro([rand(UInt64) for _ = 1:5]...)
        status = MCMCStatus(false)
        entropy_file = joinpath(root_folder, "entropy_$(id).csv")
        density_file = joinpath(root_folder, "density_$(id).csv")
        if isfile(entropy_file) || isfile(density_file)
            @error "Chain with id $(id) already exists.\\
            Chain will not be execudted to avoid overwriting data."
            status.errored = true
        end

        new(id, rng, state, status, entropy_file, density_file)
    end
end

"""
    check_running(mcmc::MPSQtMCMC)::Bool
Check if the MPSQtMCMC sampler is still running (i.e., has not errored).
"""
check_running(mcmc::MPSQtMCMC) = !mcmc.status.errored

"""
    set_error!(mcmc::MPSQtMCMC)
Set the MPSQtMCMC sampler status to errored.
"""
set_error!(mcmc::MPSQtMCMC) = (mcmc.status.errored = true)

"""
    evolve!(mcmc::MPSQtMCMC, ops::Operators, params::MCMCParameters)::Bool
Evolve the MPSQtMCMC sampler `mcmc` by applying the odd and even site MPOs
from `ops` using the parameters in `params`. If the evolution is successful,
it returns true; if an error occurs or the maximum bond dimension is reached,
it sets the sampler status to errored and returns false.
"""
function evolve!(mcmc::MPSQtMCMC, ops::Operators, params::MCMCParameters)
    check_running(mcmc) || return false
    cutoff = params.cutoff
    maxdim = params.maxdim
    for mpo in (ops.odd_sites_mpo, ops.even_sites_mpo)
        try
            mcmc.state[:] = apply(mpo, mcmc.state; maxdim = maxdim, cutoff = cutoff)
        catch e
            set_error!(mcmc)
            @error "id $(mcmc.id) errored during evolution: $e"
            return false
        end
        finishing_χ = maxlinkdim(mcmc.state)
        if finishing_χ ≥ maxdim
            @warn "id $(mcmc.id) reached maximum dimension $(maxdim)"
            set_error!(mcmc)
            return false
        end
    end
    return true
end

"""
    sample_measurement_sites(mcmc::MPSQtMCMC, params::MCMCParameters)::Vector{Int}
Sample measurement positions according to the distribtuion of the MPSQtMCMC
sampler.
"""
function sample_measurement_sites(mcmc::MPSQtMCMC, params::MCMCParameters)
    num_sites = length(siteinds(mcmc.state))
    num_measurements = rand(mcmc.rng, params.distribution)
    return rand(mcmc.rng, 1:num_sites, num_measurements)
end

"""
    project_on_site!(mcmc::MPSQtMCMC, isite::Int, projectors::Vector{Matrix{<:Number}}, params::MCMCParameters)
Perform a projective measurement on site `isite` of the MPSQtMCMC sampler `mcmc`
using the provided `projectors`. The measurement outcome is sampled according
to the probabilities computed from the current MPS state.
"""
function project_on_site!(
    mcmc::MPSQtMCMC,
    isite::Int,
    projectors::Vector{Matrix{<:Number}},
    params::MCMCParameters,
)
    probs = expect(mcmc.state, projectors; sites = isite)
    proj_index = rand(mcmc.rng, Categorical(probs))
    proj = projectors[proj_index]
    prob = probs[proj_index]
    mcmc.state = apply(proj, mcmc.state; cutoff = params.cutoff) / sqrt(prob) # ensures normalizations
    norm(mcmc.state) ≈ 1.0 ||
        @warn "MPS norm deviated from 1.0 after measurement at site $isite"
end

"""
    compute_save_measurements(
        mcmc::MPSQtMCMC,
        ops::Operators,
        params::MCMCParameters,
        time::Int,
        entropy_io::IO,
        density_io::IO
    )
Compute and save the measurements (Renyi entropies and quasiparticle densities)
at the current time step `time` for the MPSQtMCMC sampler `mcmc` using the
operators in `ops` and parameters in `params`. The results are written to the
provided IO streams `entropy_io` and `density_io`.
"""
function compute_save_measurements(
    mcmc::MPSQtMCMC,
    ops::Operators,
    params::MCMCParameters,
    time::Int,
    entropy_io::IO,
    density_io::IO,
)
    entropies = Renyi_entropy(mcmc.state, params.subsystems, 1)
    densities = measure_qp(mcmc.state, ops.density_ops)

    println(entropy_io, join([time; entropies], ","))
    println(density_io, join([time; densities], ","))
end

"""
    evolve_trajectory(mcmc::MPSQtMCMC, ops::Operators, params::MCMCParameters)
Evolve the MPSQtMCMC sampler `mcmc` for the total time specified in `params`,
applying projective measurements at sampled positions after each evolution step.
"""
function evolve_trajectory(mcmc::MPSQtMCMC, ops::Operators, params::MCMCParameters; flush_every::Int=1)
    try
        entropy_io = open(mcmc.entropy_file, "w")
        density_io = open(mcmc.density_file, "w")
        println(entropy_io, join(["time"; params.subsystems], ","))
        println(density_io, join(["time"; 1:length(ops.density_ops)], ","))

        compute_save_measurements(mcmc, ops, params, 0, entropy_io, density_io)
        for time = 1:params.final_time
            evolve!(mcmc, ops, params) || break # if an error occurs, stop evolution
            norm(mcmc.state) ≈ 1.0 ||
                @warn "MPS norm deviated from 1.0 after evolution at time $time"
            samples = sample_measurement_sites(mcmc, params)
            for isite in samples
                project_on_site!(mcmc, isite, ops.projectors, params)
            end
            compute_save_measurements(mcmc, ops, params, time, entropy_io, density_io)
            if time % flush_every == 0
                flush(entropy_io)
                flush(density_io)
            end
        end
    finally
        close(entropy_io)
        close(density_io)
    end
end
