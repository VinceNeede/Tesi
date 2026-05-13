using Dates, Statistics, Printf

# Function to parse log files and extract trajectory timings
function extract_trajectory_timings(log_dir::String)
    timings = Dict{String, Dict}()
    
    # Get all log files
    logfiles = filter(f -> startswith(f, "logfile_worker_") && endswith(f, ".log"), 
                     readdir(log_dir))
    
    for logfile in sort(logfiles)
        filepath = joinpath(log_dir, logfile)
        println("Processing $logfile...")
        
        open(filepath, "r") do file
            current_trajectory = nothing
            start_time = nothing
            
            for line in eachline(file)
                # Look for starting trajectory lines
                if contains(line, "Starting trajectory")
                    # Extract UUID and timestamp
                    match_start = match(r"\[([^\]]+)\] Starting trajectory ([a-f0-9\-]+)", line)
                    if match_start !== nothing
                        start_time = match_start.captures[1]
                        current_trajectory = match_start.captures[2]
                    end
                # Look for finished trajectory lines
                elseif contains(line, "finished") && current_trajectory !== nothing
                    match_end = match(r"\[([^\]]+)\] Trajectory ([a-f0-9\-]+) finished", line)
                    if match_end !== nothing
                        end_time = match_end.captures[1]
                        trajectory_id = match_end.captures[2]
                        
                        if trajectory_id == current_trajectory && start_time !== nothing
                            # Parse times and calculate duration
                            try
                                start_dt = DateTime(start_time, "yyyy-mm-dd HH:MM:SS")
                                end_dt = DateTime(end_time, "yyyy-mm-dd HH:MM:SS")
                                duration = Dates.value(end_dt - start_dt) / 1000  # Convert to seconds
                                
                                timings[trajectory_id] = Dict(
                                    :worker => logfile,
                                    :start_time => start_time,
                                    :end_time => end_time,
                                    :duration_seconds => duration
                                )
                            catch e
                                println("Error parsing times: $start_time to $end_time")
                            end
                        end
                        current_trajectory = nothing
                        start_time = nothing
                    end
                end
            end
        end
    end
    
    return timings
end

# Main execution
log_dir = "/home/bisogno/Tesi/data_0.1_0.3_X_40_1024_54"
timings = extract_trajectory_timings(log_dir)

# Extract durations for statistics
durations = [v[:duration_seconds] for v in values(timings)]

if !isempty(durations)
    println("\n" * repeat("=", 60))
    println("TRAJECTORY TIMING SUMMARY STATISTICS")
    println(repeat("=", 60))
    println("\nTotal trajectories: $(length(durations))")
    println("\nDuration Statistics (in seconds):")
    println(repeat("-", 60))
    println(@sprintf "  Mean:        %.2f s (%.2f hours)" mean(durations) mean(durations)/3600)
    println(@sprintf "  Median:      %.2f s (%.2f hours)" median(durations) median(durations)/3600)
    println(@sprintf "  Std Dev:     %.2f s" std(durations))
    println(@sprintf "  Min:         %.2f s (%.2f hours)" minimum(durations) minimum(durations)/3600)
    println(@sprintf "  Max:         %.2f s (%.2f hours)" maximum(durations) maximum(durations)/3600)
    println(@sprintf "  Total:       %.2f s (%.2f hours)" sum(durations) sum(durations)/3600)
    println(repeat("-", 60))
    
    # Percentiles
    sorted_durations = sort(durations)
    println("\nPercentiles:")
    for p in [25, 50, 75, 90, 95]
        idx = ceil(Int, p/100 * length(sorted_durations))
        println(@sprintf "  %2d%%:          %.2f s (%.2f hours)" p sorted_durations[idx] sorted_durations[idx]/3600)
    end
    
    println("\n" * repeat("=", 60))
    
    # Convert to hours for easier reading
    durations_hours = durations ./ 3600
    
    println("\nTop 5 Longest Trajectories:")
    println(repeat("-", 60))
    sorted_timings = sort(collect(timings), by = x -> x[2][:duration_seconds], rev=true)
    for (i, (trajectory_id, info)) in enumerate(sorted_timings[1:min(5, length(sorted_timings))])
        println(@sprintf "%d. %s: %.2f hours (%s)" i trajectory_id[1:8] info[:duration_seconds]/3600 info[:worker])
    end
    
    println("\nTop 5 Shortest Trajectories:")
    println(repeat("-", 60))
    for (i, (trajectory_id, info)) in enumerate(sorted_timings[max(1, length(sorted_timings)-4):end])
        println(@sprintf "%d. %s: %.2f hours (%s)" i trajectory_id[1:8] info[:duration_seconds]/3600 info[:worker])
    end
    
    println("\n" * repeat("=", 60))
else
    println("No trajectory timings found!")
end
