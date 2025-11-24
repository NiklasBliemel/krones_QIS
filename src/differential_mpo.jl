function custom_mpo(::Type{ElT}, sites::Vector{<:Index}, linkdim::Int) where {ElT<:Number}
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 0
      return MPO()
    elseif N == 1
      v[1] = ITensor(ElT, dag(sites[1]), sites[1]')
      return MPO(v)
    end
    space_ii = linkdim
    l = [Index(space_ii, "Link,l=$ii") for ii in 1:(N - 1)]
    for ii in eachindex(sites)
      s = sites[ii]
      if ii == 1
        v[ii] = ITensor(ElT, dag(s), s', l[ii])
      elseif ii == N
        v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s')
      else
        v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s', l[ii])
      end
    end
  return MPO(v)
end

function Diff_fill(mpo, J, Jp, alpha_, beta, gamma)
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = Jp
            mat[:,:,3] = J
        elseif i == R
            mat[1,:,:] = (alpha_*Id + beta * J + gamma * Jp)
            mat[2,:,:] = gamma * J
            mat[3,:,:] = beta * Jp
        else
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
        end
        mpo[i] = ITensor(mat, inds(v))
    end
    return mpo
end

function Diff_1_2_x(h, sites)
    out = custom_mpo(Float64, sites, 3)
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    Jp = fill(0.0, (d,d))
    J[2, 1] = 1
    J[4, 3] = 1
    Jp[1,2] = 1
    Jp[3,4] = 1
    out = Diff_fill(out, J, Jp, alpha_, beta, gamma)
    return out
end

function Diff_1_2_y(h, sites)
    out = custom_mpo(Float64, sites, 3)
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    Jp = fill(0.0, (d,d))
    J[3, 1] = 1
    J[4, 2] = 1
    Jp[1,3] = 1
    Jp[2,4] = 1
    out = Diff_fill(out, J, Jp, alpha_, beta, gamma)
    return out
end

function Diff_2_2_x(h, sites)
    out = custom_mpo(Float64, sites, 3)
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    Jp = fill(0.0, (d,d))
    J[2, 1] = 1
    J[4, 3] = 1
    Jp[1,2] = 1
    Jp[3,4] = 1
    out = Diff_fill(out, J, Jp, alpha_, beta, gamma)
    return out
end

function Diff_2_2_y(h, sites)
    out = custom_mpo(Float64, sites, 3)
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    Jp = fill(0.0, (d,d))
    J[3, 1] = 1
    J[4, 2] = 1
    Jp[1,3] = 1
    Jp[2,4] = 1
    out = Diff_fill(out, J, Jp, alpha_, beta, gamma)
    return out
end