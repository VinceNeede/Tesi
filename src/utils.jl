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

function timestamp_logger(logger::AbstractLogger)
    date_format = "yyyy-mm-dd HH:MM:SS"
    TransformerLogger(logger) do log
        merge(log, (; message="[$(format(now(), date_format))] $(log.message)"))
    end
end

ITensors.state(::StateName"ψᶿ", ::SiteType"Qubit"; θ::Float64) = [sqrt(1 - θ), sqrt(θ)]

function BertiniState(sites, θ::Float64)::Vector{ITensor}
    return [isodd(n) ? ITensorMPS.state(sites[n], "ψᶿ"; θ=θ) : ITensorMPS.state(sites[n], "Up") for n in 1:length(sites)]
end

function Renyi_entropy(singualar_evals::Vector{Float64}, n::Int)
    prob = singualar_evals .^ 2
    prob = filter(x -> x > 1e-15, prob)
    if n == 1
        return -sum(prob .* log.(prob))
    end
    return log(sum(prob .^ n)) / (1.0 - n)
end

function Renyi_entropy(mps::MPS, pos::Int, n::Int)
	prob = measure_singular_eigvals(mps, pos) .^ 2
	if n == 1
		return -sum(prob .* log.(prob))
	end
	return log(sum(prob .^ n)) / (1.0 - n)
end

Renyi_entropy(mps::MPS, pos::AbstractVector{Int}, n::Int) = [Renyi_entropy(mps, p, n) for p in pos]

function measure_singular_eigvals(psi::MPS, position::Int)
    orthogonalize!(psi, position)
    T = psi[position]
    Linds = uniqueinds(T, psi[position + 1])
    U, S, V = svd(T, Linds...)
    segs = diag(matrix(S))
    sum(segs .^ 2) ≈ 1.0 || @warn "The singular values do not sum to 1.0, state may not be normalized"
    return segs
end
measure_singular_eigvals(psi::MPS, pos::AbstractVector{Int}) = [measure_singular_eigvals(psi, p) for p in pos]