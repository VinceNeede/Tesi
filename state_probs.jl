using DataStructures
using StatsBase

struct SumOrderedDict{K, V} <: AbstractDict{K, V}
    # Buckets by total sum; index at position (sum + 1) to include sum==0
    buckets::Vector{Dict{K, V}}
end

"""
    SumOrderedDict{K,V}(max_sum::Int)

Create an empty SumOrderedDict able to hold states with total sum in 0:max_sum.
K is expected to be an NTuple type (fixed length states).
"""
function SumOrderedDict{K, V}(max_sum::Int) where {K, V}
    # Allocate max_sum + 1 buckets to include the 0-sum bucket at index 1
    return SumOrderedDict{K, V}([Dict{K, V}() for _ in 1:(max_sum + 1)])
end

"""
    SumOrderedDict(probs::AbstractDict{K,V}, max_sum::Int)

Build a SumOrderedDict from an existing dictionary of probabilities.
"""
function SumOrderedDict(probs::AbstractDict{K,V}, max_sum::Int) where {K,V}
    sod = SumOrderedDict{K,V}(max_sum)
    for (k, v) in probs
        sod[k] = v
    end
    return sod
end

max_sum(sod::SumOrderedDict) = length(sod.buckets) - 1

# --- AbstractDict interface ---

Base.length(sod::SumOrderedDict) = sum(length, sod.buckets)

Base.haskey(sod::SumOrderedDict{K,V}, key::K) where {K,V} = haskey(sod.buckets[sum(key) + 1], key)

Base.getindex(sod::SumOrderedDict{K,V}, key::K) where {K,V} = sod.buckets[sum(key) + 1][key]

function Base.setindex!(sod::SumOrderedDict{K,V}, value::V, key::K) where {K,V} 
    while length(sod.buckets) <= sum(key) + 1
        push!(sod.buckets, Dict{K,V}())
    end
    sod.buckets[sum(key) + 1][key] = value
end

function Base.iterate(sod::SumOrderedDict{K,V}, state=(1, nothing)) where {K,V}
    bucket_index, inner_state = state
    while bucket_index <= length(sod.buckets)
        bucket = sod.buckets[bucket_index]
        if isnothing(inner_state)
            inner_iter = iterate(bucket)
        else
            inner_iter = iterate(bucket, inner_state)
        end

        if !isnothing(inner_iter)
            (key, value), new_inner_state = inner_iter
            return (key, value), (bucket_index, new_inner_state)
        else
            bucket_index += 1
            inner_state = nothing
        end
    end
    return nothing
end

const IntTuple = NTuple{N, Int} where {N}

tuple_append(t::Tuple, v) = (t..., v)
add_at(t::Tuple, i::Int, v) = ntuple(j -> j == i ? t[j] + v : t[j], length(t))

"""
    next_states_and_weights(current_state::IntTuple, factor::Float64)
Creates an interator over the next possibile states and their associated weights
from the given `current_state`. The `factor` parameter scales the weights.
"""
function next_states_and_weights(current_state::IntTuple, factor::Float64, max_N::Int)
    state_sum = sum(current_state)
    state_sum == max_N && return [] # if we reached the max number of particles, no next states

    N = length(current_state)
    weights = Vector{Float64}(undef, N + 1)
    weights[2:end] .= current_state[1:N] .* (factor / state_sum)
    weights[1] = 1 - sum(@view weights[2:end])

    new_states = Vector{IntTuple}(undef, N + 1)
    for operator in 1:N
        new_states[operator] = add_at(current_state, operator, 1)
    end
    new_states[N+1] = tuple_append(current_state, 1)
    return zip(new_states, weights)
end

"""
    get_states!(states_queue::Deque{IntTuple}, probs::SumOrderedDict{IntTuple, Float64}, factor::Float64, max_N::Int)
Process the states in `states_queue`, updating the `probs` dictionary with new
states and their probabilities, using the given `factor` to weight transitions,
up to a maximum total sum of `max_N`.
!!! Note
If the starting states in `states_queue` have trailing zeros, states that have `1`
at last position and `0` before could be generated, but they will have null probability.
"""
function get_states!(states_queue::Deque{IntTuple}, probs::SumOrderedDict{IntTuple, Float64}, factor::Float64, max_N::Int)
    while !isempty(states_queue)
        state = popfirst!(states_queue)
        current_prob = probs[state]
        for (next_state, weight) in next_states_and_weights(state, factor, max_N)
            if haskey(probs, next_state)
                probs[next_state] += current_prob * weight
            else
                probs[next_state] = current_prob * weight
                push!(states_queue, next_state)
            end
        end
    end
    
    return probs
end

"""
    get_states(starting_state::IntTuple, factor::Float64, max_N::Int)
Generate all possible states starting from `starting_state`, using the given
`factor` to weight transitions, up to a maximum total sum of `max_N`.
"""
function get_states(starting_state::IntTuple, factor::Float64, max_N::Int)
    probs = SumOrderedDict{IntTuple, Float64}(max_N)
    probs[starting_state] = 1.0

    states_queue = Deque{IntTuple}()
    push!(states_queue, starting_state)
    return get_states!(states_queue, probs, factor, max_N)
end

"""
    get_states!(probs::SumOrderedDict{IntTuple, Float64}, factor::Float64, max_N::Int)
Process the states in the last bucket of `probs`, updating the `probs` dictionary with new
states and their probabilities, using the given `factor` to weight transitions,
up to a maximum total sum of `max_N`.
"""
function get_states!(probs::SumOrderedDict{IntTuple, Float64}, factor::Float64, max_N::Int)
    states_queue = Deque{IntTuple}()
    last_bucket = nothing
    for bucket in reverse(probs.buckets)
        if !isempty(bucket)
            last_bucket = bucket
            break
        end
    end

    for (state, _) in last_bucket
        push!(states_queue, state)
    end
    return get_states!(states_queue, probs, factor, max_N)
end

"""
    entanglement(state::IntTuple)
Compute the entanglement entropy for a given state represented as an NTuple.
"""
function entanglement(state::IntTuple)
    s = 0.0
    for (i, state) in enumerate(state)
        s += state/2^(i-1)
    end

    return log(2) * s
end

using Distributions

"""
    entanglement_distribution(probs::SumOrderedDict{IntTuple, Float64})
Return a SumOrderedDict mapping each state to a tuple containing its probability
and its entanglement entropy.
"""
function entanglement_distribution(probs::SumOrderedDict{IntTuple, Float64})
    entanglements = SumOrderedDict{IntTuple, NTuple{2, Float64}}(max_sum(probs))
    for (state, prob) in probs
        entanglements[state] = (prob, entanglement(state))
    end
end


function entanglement_evolution_bernoulli(chain_length::Int, final_time::Int, factor::Float64; speed::Float64=2.0)
    # factor = 50 / chain_length

    entanglements = zeros(final_time)
    probs = get_states((1,), factor, final_time)
 
    for time in 1:final_time # for bernoulli measurements, at time t we have t particles
        @info "Time step $time: computing entanglement contributions"
        entanglements[time] += sum(
                prob * entanglement(state)
                for (state, prob) in probs.buckets[time+1]
            )

        entanglements[time] *= speed * time / chain_length # rescale by light cone width
    end
    return entanglements
end

function entanglement_evolution(per_site_prob_measure::Float64, chain_length::Int, final_time::Int; speed::Float64=2.0)
    λ = per_site_prob_measure * chain_length
    factor = 50 / chain_length

    entanglements = zeros(final_time)

    estimated_max_N = ceil(Int, λ * final_time + 3 * sqrt(λ * final_time)) # rough estimate of max number of particles as sum of Poisson distribution + 3 sigma

    probs = get_states((1,), factor, estimated_max_N)
    mean_entanglements = Float64[]
    sizehint!(mean_entanglements, estimated_max_N)

    for time in 1:final_time
        distr = Poisson(λ * time)
        N = 1 # already the initial state has 1 particle
        @info "Time step $time: computing entanglement contributions"
        while cdf(distr, N) < 0.999 # we stop when the probability of N particles is less than 2%
            if N > estimated_max_N
                @warn "Estimated max N was too small, computing further probabilities."
                estimated_max_N += ceil(Int, sqrt(λ * final_time)/2)
                get_states!(probs, factor, estimated_max_N)
            end

            if length(mean_entanglements) < N
                push!(mean_entanglements, sum(
                    prob * entanglement(state)
                    for (state, prob) in probs.buckets[N + 1]
                ))
            end
            entanglements[time] += pdf(distr, N) * mean_entanglements[N]

            N += 1
        end
        entanglements[time] *= speed * time / chain_length # rescale by light cone width
    end
    return entanglements
end