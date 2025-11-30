"""
Utility function to parse command-line arguments for the simulation.

Arguments can be accessed from the returned dictionary.
"""
function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--theta"
        help = "Parameter θ"
        arg_type = Float64
        required = true

        "--chain_length"
        help = "Length of the chain"
        arg_type = Int
        required = true

        "--maxdim"
        help = "Maximum dimension"
        arg_type = Int
        required = true

        "--measure_rate"
        help = "Measurement rate"
        arg_type = Float64
        required = true

        "--final_time"
        help = "Final time"
        arg_type = Int
        required = true

        "--num_trajectories"
        help = "Number of trajectories"
        arg_type = Int
        required = true

    end

    return ArgParse.parse_args(s)
end

"""
    timesetamp_logger(logger::AbstractLogger)

Wrap a logger to prepend timestamps to each log message in the format
`[yyyy-mm-dd HH:MM:SS] message`.
"""
function timestamp_logger(logger::AbstractLogger)
    date_format = "yyyy-mm-dd HH:MM:SS"
    TransformerLogger(logger) do log
        merge(log, (; message = "[$(format(now(), date_format))] $(log.message)"))
    end
end


"""
    Itensors.state(::StateName"ψᶿ", ::SiteType"Qubit"; θ::Float64)

Define a custom single-qubit state `ψᶿ` parameterised by `θ` as :
``|ψᶿ⟩ = sqrt(1 - θ) |0⟩ + sqrt(θ) |1⟩``
"""
ITensors.state(::StateName"ψᶿ", ::SiteType"Qubit"; θ::Float64) = [sqrt(1 - θ), sqrt(θ)]

"""
    BiasedNeelState(sites::IndexSet, θ::Float64)::MPS

Construct the initial state of a chain with alternating qubits in state
`|ψᶿ⟩` and `|0⟩`, i.e.,

`|ψ₀⟩ = |ψᶿ⟩ |0⟩ |ψᶿ⟩ |0⟩ ...`
"""
function BiasedNeelState(sites::IndexSet, θ::Float64)::MPS
    return LimitedBiasedNeelState(sites, θ, 1:length(sites))
end

"""
    LimitedBiasedNeelState(sites::IndexSet, θ::Float64, range::UnitRange{Int})::MPS
Construct the initial state of a chain with qubits in state
`|ψᶿ⟩` at odd sites within `range` and `|0⟩` elsewhere, i.e.,
`|ψ₀⟩ = |0⟩ ... |0⟩ |ψᶿ⟩ |0⟩ |ψᶿ⟩ ... |0⟩`
"""
function LimitedBiasedNeelState(sites::IndexSet, θ::Float64, range::UnitRange{Int})::MPS
    return MPS([
        n in range && isodd(n) ? ITensorMPS.state(site, "ψᶿ"; θ = θ) : ITensorMPS.state(site, "Up") for
        (n, site) in enumerate(sites)
    ])
end

"""
    CentralQuasiParticle(sites::IndexSet)::MPS
Construct an initial MPS state with a quasi-particle localized at the center
of the chain, in a superposition of moving left and right.
"""
function CentralQuasiParticle(sites::IndexSet)::MPS
    center = length(sites) ÷ 2
    qp_moving_right = MPS(
        sites,
        [n == center - 2 || n == center - 1 ? "Dn" : "Up" for n in 1:length(sites)]
    )
    qp_moving_left = MPS(
        sites,
        [n == center + 1 || n == center + 2 ? "Dn" : "Up" for n in 1:length(sites)]
    )
    psi = normalize(qp_moving_right + qp_moving_left)
    return orthogonalize(psi, 1)
end

"""
    measure_singular_eigvals[!](psi::MPS, position::Int)
Measure singular values (Schmidt coefficients) at a given MPS cut.
The function orthogonalizes the `MPS` around `position` in place, extracts the
tensor at that site, performs an SVD across the bond to the right, and
returns the vector of singular values. A small warning is emitted if the
singular values squared do not sum approximately to 1.
"""
function measure_singular_eigvals!(psi::MPS, position::Int)
    chain_length = length(psi)
    (1 ≤ position < chain_length) ||
        error("Position $position is out of bounds for MPS of length $chain_length")
    psi = orthogonalize!(psi, position)
    T = psi[position]
    Linds = uniqueinds(T, psi[position+1])
    _, S, _ = svd(T, Linds...)
    segs = diag(matrix(S))
    sum(segs .^ 2) ≈ 1.0 ||
        @warn "The singular values do not sum to 1.0, state may not be normalized"
    return segs
end

measure_singular_eigvals(psi::MPS, position::Int) =
    measure_singular_eigvals!(copy(psi), position)

"""
    measure_singular_eigvals[!](psi::MPS, pos::AbstractVector{Int})
Measure singular values (Schmidt coefficients) at multiple MPS cuts.
It is faster then `[measure_singular_eigvals!(psi, p) for p in pos]` 
in case pos is unsorted.
"""
function measure_singular_eigvals!(psi::MPS, pos::AbstractVector{Int})
    p = sortperm(pos)
    res = [measure_singular_eigvals!(psi, pos[i]) for i in p]
    return res[invperm(p)]
end

measure_singular_eigvals(psi::MPS, pos::AbstractVector{Int}) =
    measure_singular_eigvals!(copy(psi), pos)

"""
    Renyi_entropy(singualar_evals::Vector{Float64}, n::Int; e=eps()/2)
Compute the Rényi entropy of order `n` from the singular eigenvalues.
`e` is a small cutoff to prevent ``0log(0)`` issues.
"""
function Renyi_entropy(singualar_evals::Vector{Float64}, n::Int; e = eps()/2)
    prob = singualar_evals .^ 2
    prob = filter(>(e), prob)
    if n == 1
        return -sum(prob .* log.(prob))
    end
    return log(sum(prob .^ n)) / (1.0 - n)
end

"""
    Renyi_entropy(mps::MPS, pos::Int, n::Int; e=eps()/2)
    Renyi_entropy(mps::MPS, pos::AbstractVector{Int}, n::Int; e=eps()/2)
Compute the Rényi entropy of order `n` at a given cut position `pos` in
the MPS `mps`.
"""
function Renyi_entropy(mps::MPS, pos::AbstractVector{Int}, n::Int; e = eps()/2)
    all_sevals = measure_singular_eigvals(mps, pos)
    return [Renyi_entropy(sevals, n; e = e) for sevals in all_sevals]
end

function Renyi_entropy(mps::MPS, pos::Int, n::Int; e = eps()/2)
    all_sevals = measure_singular_eigvals(mps, pos)
    return [Renyi_entropy(sevals, n; e = e) for sevals in all_sevals]
end
