using Glob

function find_single_line_csvs(dir::String)
	csv_files = glob("*.csv", dir)
	single_line_files = []

	for file in csv_files
		line_count = countlines(file)
		if line_count < 40
			@show line_count
			push!(single_line_files, file)
		end
	end

	return single_line_files
end

# Example usage
dir_path = "data_0.3_0.04_100"
single_line_csvs = find_single_line_csvs(dir_path)
println("CSV files with only one line: ", single_line_csvs)