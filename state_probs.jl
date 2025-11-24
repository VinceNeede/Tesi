using DataStructures
using StatsBase

"""
    effective_length(state::NTuple{N, Int}) where {N}
Returns the effective length of the state, defined as the position of the first
zero in the state tuple. If there are no zeros, it returns the length of the tuple.
"""
function effective_length(state::NTuple{N, Int}) where {N}
    m = findfirst(==(0), state)
    return isnothing(m) ? N : m # in case there are no zeros, findfirst returns nothing
end

"""
    next_states_and_weights(current_state::NTuple{N, Int}, factor::Float64) where {N}
Creates an interator over the next possibile states and their associated weights
from the given `current_state`. The `factor` parameter scales the weights.
"""
function next_states_and_weights(current_state::NTuple{N, Int}, factor::Float64) where {N}
    m = effective_length(current_state)
    state_sum = sum(current_state)
    state_sum == N && return [] # if we reached the max number of particles, no next states

    weights = Vector{Float64}(undef, m)
    weights[2:end] .= current_state[1:(m-1)] .* (factor / state_sum)
    weights[1] = 1 - sum(@view weights[2:end])

    new_states = NTuple{N, Int}[
        ntuple(j -> j==operator ? current_state[j] + 1 : current_state[j], N)
        for operator in 1:m
    ]

    return zip(new_states, weights)
end

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

Base.setindex!(sod::SumOrderedDict{K,V}, value::V, key::K) where {K,V} = (sod.buckets[sum(key) + 1][key] = value)

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

function get_states(starting_state::NTuple{N, Int}, max_N::Int, factor::Float64) where {N}
    padded_state = ntuple(i -> i<=length(starting_state) ? starting_state[i] : 0, max_N)
    probs = SumOrderedDict{NTuple{max_N, Int}, Float64}(max_N)
    probs[padded_state] = 1.0

    states_queue = Deque{NTuple{max_N, Int}}()
    push!(states_queue, padded_state)
    while !isempty(states_queue)
        state = popfirst!(states_queue)
        current_prob = probs[state]
        for (next_state, weight) in next_states_and_weights(state, factor)
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
    entanglement(state::NTuple{N, Int}) where {N}
Compute the entanglement entropy for a given state represented as an NTuple.
"""
function entanglement(state::NTuple{N, Int}) where {N}
    s = 0.0
    for (i, state) in enumerate(state)
        s += state/2^(i-1)
    end

    return log(2) * s
end

"""
    entanglement_distribution(probs::SumOrderedDict{NTuple{M, Int}, Float64}) where {M}
Return a SumOrderedDict mapping each state to a tuple containing its probability
and its entanglement entropy.
"""
function entanglement_distribution(probs::SumOrderedDict{NTuple{M, Int}, Float64}) where {M}
    entanglements = SumOrderedDict{NTuple{M, Int}, NTuple{2, Float64}}(max_sum(probs))
    for (state, prob) in probs
        entanglements[state] = (prob, entanglement(state))
    end
end