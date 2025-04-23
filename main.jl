@everywhere using MonitoredSystems, MCMC
@everywhere import HDF5
@everywhere import ITensorMPS: siteinds, MPS, MPO, linkdims, apply, maxlinkdim
@everywhere import UUIDs: UUID
@everywhere import LinearAlgebra: BLAS, LAPACKException
@everywhere BLAS.set_num_threads(1)

const parsed_args = parse_args()
const θ = parsed_args["theta"]
const chain_length = parsed_args["chain_length"]
@everywhere const maxdim = $parsed_args["maxdim"]
const measure_rate = parsed_args["measure_rate"]
const final_time = parsed_args["final_time"]
const num_trajectories = parsed_args["num_trajectories"]
@everywhere const subsystems = reverse(($chain_length÷2):-2:3)

@everywhere function _try_evolve(mcmc::MPSQtMCMC, mpo::MPO; kwargs...)
	try
		state(mcmc)[:] = apply(mpo, state(mcmc); kwargs...)
	catch e
		if e isa LAPACKException && e.info > 0
			@error "LAPACKException for id: $(id(mcmc)), saving configuration for reproduction. the info field is saved as `iteration` attribute"
			save!(mcmc, e.info)
			mcmc.status = :error
		else
			rethrow(e)
		end
	end
end

@everywhere function MonitoredSystems.evolve!(mcmc::MPSQtMCMC)
    starting_χ = maxlinkdim(state(mcmc))
    starting_χ ≥ maxdim && return false

	_try_evolve(mcmc, mpo_odd; mcmc.evolve_keys...)
	_try_evolve(mcmc, mpo_even; mcmc.evolve_keys...)

	finishing_χ = maxlinkdim(state(mcmc))
    if finishing_χ ≥ maxdim
        @warn "id $(id(mcmc)) reached maximum dimension $(maxdim)"
        mcmc.status = :error
        return false
    end
    return true
end

@everywhere function MCMC.observables(mcmc::MPSQtMCMC)
    mcmc.status === :error && return []
    return [
        x -> measure_qp(state(x), qps),
        (x -> Renyi_entropy(state(x), p, 1) for p in subsystems)...
    ]
end

@everywhere const sites = siteinds("S=1/2", $chain_length)
@everywhere const mpo_odd = r54_odd(sites)
@everywhere const mpo_even = r54_even(sites)
@everywhere const qps = qp_tensors(sites)

@everywhere function main(chain_length::Int, maxdim::Int, θ::Float64, measure_rate::Float64, final_time::Int, subsystems::AbstractVector{Int}, file::HDF5.File)


    mcmc = MPSQtMCMC(
        MPS(BertiniState(sites, θ), 0, 2),
        measure_rate * chain_length,
        [[1 0; 0 0], [0 0; 0 1]];
        checkpoint_file=file,
        cutoff=1.e-13,
        maxdim=maxdim,
    )

    write(save_file(mcmc), join(["N", subsystems...], ", "), "\n")
    Base.with_logger(
        timestamp_logger(
            mcmc_logger(id(mcmc), 1)
        )
    ) do
        try
            run!(mcmc, final_time)
        finally
            close(save_file(mcmc))
        end
    end

    if mcmc.status === :error
        @error "id $(id(mcmc)) erroed, removing files"
        rm("$(id(mcmc)).log")
        rm("$(id(mcmc)).csv")
    end
end

folder_name = "data_$(round(θ; sigdigits=2))_$(round(measure_rate; sigdigits=2))_$chain_length"
isdir(folder_name) || mkdir(folder_name)
@everywhere cd($folder_name)

HDF5.h5open("checkpoint.h5", "cw") do file  # This may be the wrong place to open the file
    pmap(_ -> main(chain_length, maxdim, θ, measure_rate, final_time, subsystems, file), 1:num_trajectories)
end
@info "All tasks finished"

cd("../")
