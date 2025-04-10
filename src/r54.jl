const X = reshape([0., 1., 1., 0.], (2, 2))
const Id = reshape([1., 0., 0., 1.], (2, 2))
const CX = let
    arr = zeros(Float64, 2, 2, 2)       # s', s, l
    arr[:, :, 1] = Id
    arr[:, :, 2] = X
    arr
end

const base_R54 = let
    arr = zeros(Float64, 2, 2, 2, 2)
    arr[:, :, 2, 2] = X
    arr[:, :, 1, 2] = X
    arr[:, :, 2, 1] = X
    arr[:, :, 1, 1] = Id
    arr
end

function r54_odd(sites::Vector{Index{Int64}})
    chain_length = length(sites)
    mpo = MPO(sites)
    links = [Index(2, tags="Link, k=$i") for i in 1:(chain_length-1)]
    for i in 1:chain_length
        if i % 2 == 0
            if i == chain_length
                mpo[i] = delta(sites[i]', sites[i], links[i-1])
            else
                mpo[i] = delta(sites[i]', sites[i], links[i-1], links[i])
            end
        else
            if i == 1
                mpo[i] = ITensor(CX, sites[i]', sites[i], links[i])
            else
                mpo[i] = ITensor(base_R54, sites[i]', sites[i], links[i-1], links[i])
            end
        end
    end
    return mpo
end

function r54_even(sites::Vector{Index{Int64}})
    chain_length = length(sites)
    mpo = MPO(sites)
    links = [Index(2, tags="Link, k=$i") for i in 1:(chain_length-1)]
    for i in 1:chain_length
        if i % 2 == 0
            if i == chain_length
                mpo[i] = ITensor(CX, sites[i]', sites[i], links[i-1])
            else
                mpo[i] = ITensor(base_R54, sites[i]', sites[i], links[i-1], links[i])
            end
        else
            if i == 1
                mpo[i] = delta(sites[i]', sites[i], links[i])
            else
                mpo[i] = delta(sites[i]', sites[i], links[i-1], links[i])
            end
        end
    end
    return mpo
end
