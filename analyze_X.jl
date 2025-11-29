using CSV, DataFrames, Statistics, Plots, StatsPlots


"""
    retrieve_results(data_folder::String; extension::String=".dat")::Tuple{Vector{DataFrame}, Vector{DataFrame}}
Retrieve the entropy and density results from `.dat` files in the specified
`data_folder`. The function looks for files ending with the given `extension`
(default is `.dat`) and separates them into entropy and density files based on
their filenames. It returns two vectors of DataFrames: one for entropies and one
for densities.
"""
function retrieve_results(data_folder::String; extension::String=".dat")
    isdir(data_folder) || error("The specified data folder does not exist.")
    dir_content = readdir(data_folder; join=true)
    files = filter(x -> (isfile(x) && endswith(x, extension)), dir_content)
    entropies_files = filter(x -> occursin("entropy", x), files)
    densities_files = filter(x -> occursin("density", x), files)
    entropies_df = DataFrame[CSV.read(file, DataFrame) for file in entropies_files]
    densities_df = DataFrame[CSV.read(file, DataFrame) for file in densities_files]

    return entropies_df, densities_df
end

# plot()
# for (p, op) in zip([0.8, 0.16], ["X", "Z"])
#     entropies_df, densities_df = retrieve_results("data_0.1_$(p)_$(op)_100_2048_30")
#     entropies_matrix = cat(Matrix.(entropies_df)...; dims=3)
#     errorline!(entropies_matrix[:, 1, 1], entropies_matrix[:, 2, :], errortype=:sem)
# end

s(θ) = -θ*log(θ)-(1-θ)*log(1-θ)
v(θ) = 2 / (1 + 2θ)

line_type = Dict(
    0.1 => :solid,
    0.5 => :dash,
)

color_type = Dict(
    0.8 => :red,
    0.7 => :blue,
    0.6 => :green,
    0.5 => :orange,
)

function execute_plot()
    L = 24
    p = plot(xlabel="γt", ylabel = "γ²S")
    for (θ, χ) in zip([0.1, 0.5], [2048, 3072]), γ in [0.8, 0.7, 0.6, 0.5]
            entropies_df, densities_df = retrieve_results("data_$(join(round.([θ, γ], sigdigits=2), "_"))_X_$(L)_$(χ)_15")
            entropies_matrix = cat(Matrix.(entropies_df)...; dims=3)
            errorline!(p,
                entropies_matrix[:, 1, 1], 
                entropies_matrix[:, 2, :], 
                errortype=:sem, 
                label="θ = $(θ), γ=$(γ)",
                linestyle=line_type[θ],
                groupcolor=color_type[γ],
                )
    end
    return p
end