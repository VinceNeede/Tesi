using CSV, DataFrames, Statistics
using ProgressBars
using LinearAlgebra
using Plots
using LsqFit

BLAS.set_num_threads(1)
function get_res(; path::String=".")
    dir = readdir(path)
    dir = joinpath.(path, dir)
    files = filter(x -> (isfile(x) && endswith(x, ".csv")), dir)
    return files
end

function mean_std(dfs::DataFrame...)
    @assert all(all(names(dfs[1]) .== names(df)) for df in dfs[2:end])

    # Precompute matrices for each key
    keys = names(dfs[1])
    matrices = Dict(key => reduce(hcat, [df[!, key] for df in dfs]) for key in keys)

    # Compute mean and standard deviation
    m = Dict(key => mean(matrices[key], dims=2)[:] for key in keys)
    s = Dict(key => std(matrices[key], dims=2)[:] for key in keys)

    return DataFrame(m), DataFrame(s)
end

chain_lengths = [100]
θs = [0.1, 0.5]
γs = [0.04]

function collect_data()
    μ = Dict{Tuple{Int,Float64,Float64},DataFrame}()
    σ = Dict{Tuple{Int,Float64,Float64},DataFrame}()
    for chain_length in chain_lengths, γ in γs, θ in θs
        files = get_res(path="data_$(round(θ, sigdigits=2))_$(round(γ, sigdigits=2))_$chain_length")
        v = DataFrame[]
        sizehint!(v, length(files))
        for (i, file) in ProgressBar(enumerate(files))
            if isempty(file)
                @error "file is empty"
                continue
            end
            tmp = CSV.read(file, DataFrame)
            isempty(tmp) && continue
            for col in eachcol(tmp)
                replace!(col, NaN => 0.0)
            end
            push!(v, tmp)
        end
        μ[chain_length, γ, θ], σ[chain_length, γ, θ] = mean_std(v...)
        μ[chain_length, γ, θ] = μ[chain_length, γ, θ][!, sort(names(μ[chain_length, γ, θ]), by=x -> x == "N" ? 0 : parse(Int, x))]
        σ[chain_length, γ, θ] = σ[chain_length, γ, θ][!, sort(names(σ[chain_length, γ, θ]), by=x -> x == "N" ? 0 : parse(Int, x))] ./ sqrt(length(v))
    end
    @info "collected data"
    return μ, σ
end

s(θ::Float64) = -θ * log(θ) - (1 - θ) * log(1 - θ)

function main(α=1., β=1.)
    plot_S = plot(title="α = $(round(α, sigdigits=4)), β = $(round(β, sigdigits = 4))", xlabel="γᵝt", ylabel="γᵅS/s(θ)", legend=:topright, dpi=300)
    plot_N = plot(xlablel="t", ylabel="N/L", legend=:topright, dpi=300)
    colors = palette(:viridis, length(chain_lengths) * length(γs)*length(θs))
    color_idx = 1
    for chain_length in chain_lengths, γ in γs, θ in θs
        # Use the first column for plot_N
        y = μ[chain_length, γ, θ]
        yerr = σ[chain_length, γ, θ]
        x = 1:size(y, 1)          # x-axis values (row indices)
        plot!(plot_N, 1:size(y, 1), y[!, 1] ./ chain_length, yerror=yerr[!, 1] ./ chain_length,
            label="L = $chain_length, γ = $(γ), θ = $(θ)", color=colors[color_idx], msc=colors[color_idx])

        plot!(plot_S, γ^β .* x, γ^α .* y[!, end] ./ s(θ), yerror=yerr[!, end] .* γ^α ./ s(θ),
            label="L = $chain_length, γ = $(γ), θ = $(θ)", color=colors[color_idx], msc=colors[color_idx])
        color_idx += 1
    end
    savefig(plot_N, "comparison_N.png")
    savefig(plot_S, "comparison.png")
    return plot_S
end

function model(x, p)
    return p[1] .* x .* exp.(-p[2] .* x)
end
function fit(x, y)
    return curve_fit(model, x, y, [1., 1,])
end

# function fit(p, β; x0=1/3)
# 	α = β*x0^β
# 	x = LinRange(0. , 3, 100)
# 	y = log(2) .* x .^α  .* exp.(-x .^ β)
# 	plot!(p, x, y, label="β = $(round(β, sigdigits=4)), x0 = $(round(x0, sigdigits=4))", msc=:auto)
# end