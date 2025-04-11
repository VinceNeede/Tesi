using CSV, DataFrames, Statistics

function get_res(; path::String=".")
    dir = readdir(path)
    dir = joinpath.(path, dir)
    files = filter(x -> (isfile(x) && endswith(x, ".csv")), dir)
    return files
end

θ = 4.
chain_length = 50

θ = round(θ; sigdigits=2)
data = [CSV.read(file, DataFrame) for file in get_res(path="data_$(θ)_$chain_length")]

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
p = plot(title="Mean with Error Bars", xlabel="Index", ylabel="Value", legend=:topright)

# Loop through each column and add it to the plot
for col in columns
    x = 1:size(μ, 1)          # x-axis values (row indices)
    y = μ[!, col]             # Mean values for the current column
    errors = σ[!, col]        # Standard errors for the current column
    plot!(p, x, y, yerror=errors, label=col, msc=:auto)  # Add to the plot
end

# Display the plot
display(p)