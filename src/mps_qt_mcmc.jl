abstract type AbstractMPSQtMCMC <: AbstractBaseMCMC{MPS} end

@BaseMCMC_def MPS mutable struct MPSQtMCMC <: AbstractMPSQtMCMC
    distr::Poisson
    projectors::Vector{Matrix{<:Number}}
    evol_keys::Base.Pairs
end

function MPSQtMCMC(state::MPS, λ::Real, projectors::Vector{Matrix{T}};
    id::UUID=uuid4(),
    rng::Xoshiro=Xoshiro([rand(UInt64) for _ in 1:5]...),
    checkpoint_file::HDF5.File=default_checkpoint_file(id),
    save_file_::IO=default_save_file(id),
    kwargs...,
) where T<:Number
    return MPSQtMCMC(
        id,
        rng,
        state,
        checkpoint_file,
        save_file_,
        Poisson(λ),
        projectors,
        kwargs
    )
end

distr(mcmc::MPSQtMCMC) = mcmc.distr
projectors(mcmc::MPSQtMCMC) = mcmc.projectors
evolve!(mcmc::AbstractMPSQtMCMC) = throw(MethodError(evolve!, typeof(mcmc)))

function MCMC.sample(mcmc::MPSQtMCMC)
    rng_ = rng(mcmc)
    n_measures = rand(rng_, distr(mcmc))
    chain_length = length(mcmc.state)
    return rand(rng_, 1:chain_length, n_measures)
end


function _compute_probs_on_site(T::ITensor, projs::Vector{ITensor})
    n = length(projs)
    probs = zeros(n)
    s = 0
    for i in 1:n-1
        probs[i] = real(scalar(conj(dag(T)) * replaceprime(projs[i] * T, 1 => 0)))
        s += probs[i]
    end
    probs[n] = 1 - s
    return probs
end

function MCMC.update!(mcmc::MPSQtMCMC, samples::Vector{Int})
    state = MCMC.state
    chain_length = length(state(mcmc))
    evolve!(mcmc) || return 0   # when getting false, return so that measures are not performed
    cutoff = get(mcmc.evol_keys, :cutoff, 1.e-15)
    for isite in samples
        orthogonalize!(state(mcmc), isite)
        site = siteind(only, state(mcmc), isite)
        proj = let
            d = dim(site)
            projs = [itensor(projectors(mcmc)[i], site', dag(site)) for i in 1:d]
            probs = _compute_probs_on_site(state(mcmc)[isite], projs)
            r = rand(rng(mcmc))
            proj_index = findfirst(p -> p ≥ r, cumsum(probs))
            projs[proj_index]
        end

        state(mcmc)[isite] = apply(proj, state(mcmc)[isite]) 
        isite < chain_length &&
            setindex!(state(mcmc), state(mcmc)[isite] * state(mcmc)[isite+1], isite:isite+1; orthocenter=isite, cutoff=cutoff)
        isite > 1 &&
            setindex!(state(mcmc), state(mcmc)[isite-1] * state(mcmc)[isite], isite-1:isite; orthocenter=isite, cutoff=cutoff)
        state(mcmc)[isite] ./= norm(state(mcmc)[isite])
    end
    return length(samples)
end


