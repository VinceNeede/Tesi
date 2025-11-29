using DataFrames, CSV, Plots, StatsPlots

traj_path = "qp_bernoulli_0.0_0.0_X_100_2048_10/density_0a50bc59-6a35-4f4f-a281-c7edbf3a9d9f.dat"
# traj_path = "qp_bernoulli_0.0_0.0_X_100_2048_10/density_0ac2a3f1-18c7-4105-a2f0-1ba0f8304c5d.dat"
# traj_path = "qp_0.0_0.01_X_100_2048_10/density_0a4fb248-fcbc-4499-b573-42644ab62a82.dat"
traj_path = "qp_bernoulli_0.0_0.0_X_100_2048_10/density_1b3856e7-22db-4d36-a965-271ed87adb81.dat"
traj_path = "created_trajectory/density.dat"

df = CSV.read(traj_path, DataFrame, header=false)

times = 1:size(df, 1)
densities = Matrix(df)
heatmap(1:size(densities, 2), times, densities, xlabel="Site", ylabel="Time", title="Density Heatmap", colorbar_title="Density", yflip=true)