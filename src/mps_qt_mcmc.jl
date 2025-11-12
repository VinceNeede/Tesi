abstract type AbstractMPSQtMCMC <: AbstractBaseMCMC{MPS} end

"""
    MPSQtMCMC(
        state::MPS,
        λ::Real,
        projectors::Vector{Matrix{T}};
        id::UUID=uuid4(),
        rng::Xoshiro=Xoshiro([rand(UInt64) for _ in 1:5]...),
        checkpoint_file::HDF5.File=default_checkpoint_file(id),
        save_file_::IO=default_save_file(id),
        kwargs...,
    ) where T<:Number
Create an MPS quantum trajectory MCMC sampler with Poisson-distributed
measurement times with rate `λ` and local projective measurements defined
by the set of `projectors`. Each element of `projectors` is a matrix defining a
projector on a single site in the local basis. An internal variable `status` is used
to check if the sampler erroed during evolution.

# Arguments
- `state::MPS`: Initial MPS state.
- `λ::Real`: Measurement rate for the Poisson distribution of the whole state.
- `projectors::Vector{Matrix{T}}`: Set of local projectors defining the measurements.
# Keyword Arguments
- `id::UUID=uuid4()`: Unique identifier for the MCMC sampler.
- `rng::Xoshiro=Xoshiro([rand(UInt64) for _ in 1:5]...)`: Random number generator.
- `checkpoint_file::HDF5.File=default_checkpoint_file(id)`: HDF5 file for checkpointing.
- `save_file_::IO=default_save_file(id)`: IO stream for saving the results of the observable.
- `kwargs...`: Additional keyword arguments for the MCMC sampler e.g. cutoff during evolution.
"""
@BaseMCMC_def MPS mutable struct MPSQtMCMC <: AbstractMPSQtMCMC
    distr::DiscreteUnivariateDistribution
    projectors::Vector{Matrix{<:Number}}
    status::Symbol
    evol_keys::Base.Pairs
end

function MPSQtMCMC(
    state::MPS,
    λ::Real,
    projectors::Vector{Matrix{T}};
    id::UUID = uuid4(),
    rng::Xoshiro = Xoshiro([rand(UInt64) for _ = 1:5]...),
    checkpoint_file::HDF5.File = default_checkpoint_file(id),
    save_file_::IO = default_save_file(id),
    kwargs...,
) where {T<:Number}
    return MPSQtMCMC(
        id,
        rng,
        state,
        checkpoint_file,
        save_file_,
        Poisson(λ),
        projectors,
        :ok,
        kwargs,
    )
end

distr(mcmc::MPSQtMCMC) = mcmc.distr
projectors(mcmc::MPSQtMCMC) = mcmc.projectors
evolve!(mcmc::AbstractMPSQtMCMC) = throw(MethodError(evolve!, typeof(mcmc)))

"""
    MCMC.sample(mcmc::MPSQtMCMC)::Vector{Int}
Sample measurement positions according to the distribtuion of the MPSQtMCMC
sampler.
"""
function MCMC.sample(mcmc::MPSQtMCMC)
    rng_ = rng(mcmc)
    n_measures = rand(rng_, distr(mcmc))
    chain_length = length(mcmc.state)
    return rand(rng_, 1:chain_length, n_measures)
end

"""
    _compute_probs_on_site(T::ITensor, projs::Vector{ITensor})::Vector{Float64}
Compute the probabilities of obtaining each projector outcome on a given site
for the local tensor `T` which is the orthogonal center.
"""
function _compute_probs_on_site(T::ITensor, projs::Vector{ITensor})
    n = length(projs)
    probs = zeros(n)
    s = 0
    for i = 1:n
        probs[i] = real(scalar(conj(dag(T)) * replaceprime(projs[i] * T, 1 => 0)))
        s += probs[i]
    end
    s ≈ 1.0 || @warn "Probabilities on site do not sum to 1.0, got $s"
    return probs
end

"""
    MCMC.update!(mcmc::MPSQtMCMC, samples::Vector{Int})::Int
Perform the MPSQtMCMC update by evolving the state and applying
projective measurements at the sampled `samples` positions.
"""
function MCMC.update!(mcmc::MPSQtMCMC, samples::Vector{Int})
    mcmc.status == :ok || return 0 # check for errors

    state = MCMC.state # just a shorthand

    evolve!(mcmc) || return 0   # when getting false, return so that measures are not performed

    cutoff = get(mcmc.evol_keys, :cutoff, 1.e-15)

    for isite in samples
        orthogonalize!(state(mcmc), isite)
        site = siteind(only, state(mcmc), isite)
        proj, prob = let
            d = dim(site)
            projs = [itensor(projectors(mcmc)[i], site', dag(site)) for i = 1:d] # projectors as ITensors
            probs = _compute_probs_on_site(state(mcmc)[isite], projs)
            proj_index = rand(rng(mcmc), Categorical(probs)) # sample the index according to probs
            projs[proj_index], probs[proj_index]
        end

        state(mcmc) = apply(proj, state(mcmc); cutoff = cutoff) / sqrt(prob) # apply projector and normalize
    end
    return length(samples)
end
