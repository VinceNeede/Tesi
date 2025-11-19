"""
Code for constructing the Rule54 MPO.
The construction is splitted in the part applied to odd sites and the part
applied to even sites.
The boundary conditions are obtained considering 2 additional qubits at
the edges that are always in the |0⟩ state and not included in the MPO.
This leads to the first tensor in the odd MPO, and the last tensor in the even
MPO, being a Controlled-X gate, where the control is given by the adiacent site
via the link index (this is given by the fact that the R54 tensors are alternated
by delta tensors).
"""

"2x2 Pauli-X Matrix"
const X = reshape([0.0, 1.0, 1.0, 0.0], (2, 2))

"2x2 Identity Matrix"
const Id = reshape([1.0, 0.0, 0.0, 1.0], (2, 2))

"Controlled-X Gate where the control is the last index"
const CX = let
    arr = zeros(Float64, 2, 2, 2)
    arr[:, :, 1] = Id
    arr[:, :, 2] = X
    arr
end

"""
This is the base R54 tensor. The first two inidices are the physical ones.
The third one is the incoming link, the fourth one the outgoing link.
"""
const base_R54 = let
    arr = zeros(Float64, 2, 2, 2, 2)
    arr[:, :, 2, 2] = X
    arr[:, :, 1, 2] = X
    arr[:, :, 2, 1] = X
    arr[:, :, 1, 1] = Id
    arr
end


"""
    r54_odd(sites::Vector{Index{Int64}})::MPO
Construct the Rule54 MPO tensor that acts on the odd indices.
"""
function r54_odd(sites::Vector{Index{Int64}})
    chain_length = length(sites)
    mpo = MPO(sites)
    links = [Index(2, tags = "Link, k=$i") for i = 1:(chain_length-1)]

    mpo[1] = ITensor(CX, sites[1]', sites[1], links[1])
    for i = 2:(chain_length-1)
        if iseven(i)
            mpo[i] = delta(sites[i]', sites[i], links[i-1], links[i])
        else
            mpo[i] = ITensor(base_R54, sites[i]', sites[i], links[i-1], links[i])
        end
    end
    mpo[chain_length] =
        delta(sites[chain_length]', sites[chain_length], links[chain_length-1])

    return mpo
end

"""
    r54_even(sites::Vector{Index{Int64}})::MPO
Construct the Rule54 MPO tensor that acts on the even indices.
"""
function r54_even(sites::Vector{Index{Int64}})
    chain_length = length(sites)
    mpo = MPO(sites)
    links = [Index(2, tags = "Link, k=$i") for i = 1:(chain_length-1)]

    mpo[1] = delta(sites[1]', sites[1], links[1])
    for i = 2:(chain_length-1)
        if iseven(i)
            mpo[i] = ITensor(base_R54, sites[i]', sites[i], links[i-1], links[i])
        else
            mpo[i] = delta(sites[i]', sites[i], links[i-1], links[i])
        end
    end
    mpo[chain_length] =
        ITensor(CX, sites[chain_length]', sites[chain_length], links[chain_length-1])

    return mpo
end
