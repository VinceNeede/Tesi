"""
Construct the tensors needed to measure the quasiparticle density.
The local number of quasiparticles is measured by using 3-site tensors.
On the edges, 2-site tensors are used, they are obtained by
imagining that there are virtual sites outside of the chain in the |0⟩ state.
"""


"""
	qp_tensors(sites::ITensors.Indices)::Vector{ITensor}
Construct the tensors needed to measure the quasiparticle density.
The convention for the indices order is that output site indices come first.
"""
function qp_tensors(sites::ITensors.Indices)
    leftmost = let L = zeros(2, 2, 2, 2)# s1', s2', s1, s2
        L[2, 1, 2, 1] = 2
        L[2, 2, 2, 2] = 1
        L
    end

    central = let C = zeros(2, 2, 2, 2, 2, 2)# s1' s2' s3' s1 s2 s3
        C[1, 2, 1, 1, 2, 1] = 2
        C[1, 2, 2, 1, 2, 2] = 1
        C[2, 2, 2, 2, 2, 2] = 1
        C
    end

    rightmost = let R = zeros(2, 2, 2, 2)# s1' s2' s1 s2
        R[1, 2, 1, 2] = 2
        R
    end

    N = length(sites)

    res = Vector{ITensor}(undef, N)
    res[1] = itensor(leftmost, sites[1]', sites[2]', dag(sites[1]), dag(sites[2]))
    for i = 2:(N-1)
        res[i] = itensor(
            central,
            sites[i-1]',
            sites[i]',
            sites[i+1]',
            dag(sites[i-1]),
            dag(sites[i]),
            dag(sites[i+1]),
        )
    end
    res[N] = itensor(rightmost, sites[N-1]', sites[N]', dag(sites[N-1]), dag(sites[N]))
    return res
end

"""
	measure_qp(ψ::MPS, tensors::Vector{ITensor})::Vector{Float64}
Measure the quasiparticle density per site of the MPS `ψ` using the provided
`tensors` obtained from `qp_tensors`.
"""
function measure_qp(ψ::MPS, tensors::Vector{ITensor})
    res = Vector{Float64}(undef, length(tensors))
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
        res[i] = real(scalar(T * apply(tensor, T)))
    end

    return res
end
