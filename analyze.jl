using CSV, DataFrames, Statistics

function get_res(; path::String=".")
    dir = readdir(path)
    dir = joinpath.(path, dir)
    files = filter(x -> (isfile(x) && endswith(x, ".csv")), dir)
    return files
end

γ = 0.08
chain_length = 50

γ = round(γ; sigdigits=2)
data = let
    files = get_res(path="data_$(γ)_$chain_length")
    data = Vector{DataFrame}(undef, length(files))
    for (i, file) in enumerate(files)
        data[i] = CSV.read(file, DataFrame)
        for col in eachcol(data[i])
            replace!(col, NaN => 0.0)
        end
    end
    data
end

function Statistics.mean(dfs::DataFrame...)
    @assert all(all(names(dfs[1]) .== names(df)) for df in dfs[2:end])
    res = Dict()
    for (key, cols) in zip(names(dfs[1]), zip(eachcol.(dfs)...))
        res[key] = mean(reduce(hcat, cols), dims=2)[:]
    end
    return DataFrame(res)
end
function Statistics.std(dfs::DataFrame...)
    @assert all(all(names(dfs[1]) .== names(df)) for df in dfs[2:end])
    res = Dict()
    for (key, cols) in zip(names(dfs[1]), zip(eachcol.(dfs)...))
        res[key] = std(reduce(hcat, cols), dims=2)[:]
    end
    return DataFrame(res)
end

μ = mean(data...)
σ = std(data...) ./ sqrt(length(data))

μ = μ[!, sort(names(μ), by=x -> parse(Int, x))]
σ = σ[!, sort(names(σ), by=x -> parse(Int, x))]

using Plots

# Extract column names
columns = names(μ)

# Initialize the plot
p = plot(title="γ = $γ, L = $chain_length", xlabel="t", ylabel="S", legend=:topright, dpi=300)

# Loop through each column and add it to the plot
x = 0:size(μ, 1)          # x-axis values (row indices)
for col in columns
    icol = parse(Int, col)
    y = [0, μ[!, col]...]             # Mean values for the current column
    errors = [0, σ[!, col]...]        # Standard errors for the current column
    plot!(p, x, y, yerror=errors, label="l="*col, msc=:auto)  # Add to the plot
end

# Display the plot
# display(p)
savefig("figure_$(γ)_$chain_length.png")

# using LsqFit

# function model(x, p)
# 	a, b = p
# 	return a.*x.*exp.(-b .* x)
# end

# curve_fit(model, 0:size(μ, 1), [0, μ[!, " 25"]...], [1., γ*chain_length])