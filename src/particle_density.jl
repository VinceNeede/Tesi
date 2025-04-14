function qp_tensors(sites::ITensors.Indices)	
	 L = let L = zeros(2, 2, 2, 2)# s1', s2', s1, s2
		L[2, 1, 2, 1] = 2
		L[2, 2, 2, 2] = 1
		L
	end

	 C = let C = zeros(2, 2, 2, 2, 2, 2)# s1' s2' s3' s1 s2 s3
		C[1, 2, 1, 1, 2, 1] = 2
		C[1, 2, 2, 1, 2, 2] = 1
		C[2, 2, 2, 2, 2, 2] = 1
		C
	end
	
	 R = let R = zeros(2, 2, 2, 2)# s1' s2' s1 s2
		R[1, 2, 1, 2] = 2
		R
	end

	 N = length(sites)

	res = Vector{ITensor}(undef, N)
	res[1] = itensor(L, sites[1]', sites[2]', dag(sites[1]), dag(sites[2]))
	for i in 2:N-1
		res[i] = itensor(C, sites[i-1]', sites[i]', sites[i+1]', dag(sites[i-1]), dag(sites[i]), dag(sites[i+1]))
	end
	res[N] = itensor(R, sites[N-1]', sites[N]', dag(sites[N-1]), dag(sites[N]))
	return res
end

function measure_qp(ψ::MPS, tensors::Vector{ITensor})
	s = 0
	for (i, tensor) in enumerate(tensors)
		ψ = orthogonalize(ψ, i)
		T = let
			if i == 1
				ψ[i] * ψ[i+1]
			elseif i == length(ψ)
				ψ[i-1] * ψ[i]
			else
				ψ[i-1] * ψ[i] * ψ[i+1]
			end
		end
		try
			s += scalar(prime(T, tags="Site") * (tensor * T))
		catch e
			@error "Error in measure_qp: " i T tensor
			rethrow(e)
		end
	end
	return real(s)
end
