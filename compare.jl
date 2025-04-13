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

chain_length = 50
γs = [0.02, 0.03, 0.04, 0.08]
γs = round.(γs; sigdigits=2)

function collect_data()
    μ = Dict{Float64,DataFrame}()
    σ = Dict{Float64,DataFrame}()
    for γ in γs
        files = get_res(path="data_$(γ)_$chain_length")
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
        μ[γ], σ[γ] = mean_std(v...)
        μ[γ] = μ[γ][!, sort(names(μ[γ]), by=x -> parse(Int, x))]
        σ[γ] = σ[γ][!, sort(names(σ[γ]), by=x -> parse(Int, x))] ./ length(v)
    end
    @info "collected data"
    return μ, σ
end
function main(α, β)
    p = plot(title="l = 25, L = $chain_length", xlabel="γᵝt", ylabel="γᵅS", legend=:topright, dpi=300)
    for γ in γs
        x = 0:size(μ[γ], 1)          # x-axis values (row indices)
        plot!(p, γ^β .* x, γ^α .* [0, μ[γ][!, end]...], yerror=[0, σ[γ][!, end]...] .* γ^α, 
		label="γ = $(γ), α = $(round(α, sigdigits=4)), β = $(round(β, sigdigits = 4))", msc=:auto)
    end
    savefig(p, "comparison.png")
	return p
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