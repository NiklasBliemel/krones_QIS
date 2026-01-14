function custom_mpo(::Type{ElT}, sites::Vector{<:Index}, linkdim::Vector{Int}) where {ElT<:Number}
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 0
      return MPO()
    elseif N == 1
      v[1] = ITensor(ElT, dag(sites[1]), sites[1]')
      return MPO(v)
    end
    l = [Index(linkdim[ii], "Link,l=$ii") for ii in 1:(N - 1)]
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

function Diff2_fill(mpo, J, alpha_, beta, gamma)
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
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
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_1_2_y(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_2_2_x(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_2_2_y(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff8_1_fill(mpo, J, a::Vector{Float64})
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
        elseif i == R - 1
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
            mat[2,:,:,4] = Id
            mat[3,:,:,5] = Id
        elseif i == R
            mat[1,:,:] = a[1]*Id + a[2]*Jp + a[3]*J
            mat[2,:,:] = a[2]*J  + a[4]*Id + a[5]*Jp
            mat[3,:,:] = a[3]*Jp - a[4]*Id - a[5]*J
            mat[4,:,:] = a[5]*J  + a[6]*Id
            mat[5,:,:] =-a[5]*Jp - a[6]*Id
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

function Diff8_2_fill(mpo, J, a::Vector{Float64})
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
        elseif i == R - 1
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
            mat[2,:,:,4] = Id
            mat[3,:,:,5] = Id
        elseif i == R
            mat[1,:,:] = a[1]*Id + a[2]*Jp + a[3]*J
            mat[2,:,:] = a[2]*J  + a[4]*Id + a[5]*Jp
            mat[3,:,:] = a[3]*Jp + a[4]*Id + a[5]*J
            mat[4,:,:] = a[5]*J  + a[6]*Id
            mat[5,:,:] = a[5]*Jp + a[6]*Id
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

function Diff_1_8_x(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [0, -4/5, 4/5, 1/5, -4/105, 1/280] / h
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff8_1_fill(out, J, a)
    return out
end

function Diff_1_8_y(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [0, -4/5, 4/5, 1/5, -4/105, 1/280] / h
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff8_1_fill(out, J, a)
    return out
end

function Diff_2_8_x(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [-205/72, 8/5, 8/5, -1/5, 8/315, -1/560] / (h^2)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff8_2_fill(out, J, a)
    return out
end
function Diff_2_8_y(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [-205/72, 8/5, 8/5, -1/5, 8/315, -1/560] / (h^2)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff8_2_fill(out, J, a)
    return out
end

function plot_mps(mps, R, N; min=nothing, max=nothing) # -> 2^N x 2^N Grid
    # convert to TCI Format for evaluation
    mps_tt = TensorCrossInterpolation.TensorTrain(mps)
    n = R - N

    # evaluate function on defined grid
    xvec = 1:2^n:2^R
    yvec = 1:2^n:2^R
    xvals = collect(range(0, 1, 2^N))
    yvals = collect(range(0, 1, 2^N))

    # establish grid
    grid = DiscretizedGrid{2}(R, (0,0), (1,1); includeendpoint = true)

    mps_vals = fill(0.0, (2^N, 2^N))

    for x in 1:2^N
        for y in 1:2^N
            mps_vals[y,x] = mps_tt(grididx_to_quantics(grid, (xvec[x], yvec[y])))
        end
    end

    p1 = contour(xvals, yvals, mps_vals, fill=true)
    xlabel!("x")
    ylabel!("y")
    title!("Plot without clim")

    if !(isnothing(min) && isnothing(max))
        # Plot 2: Mit clim
        p2 = contour(xvals, yvals, mps_vals, fill=true, clim=(min, max))
        xlabel!("x")
        ylabel!("y")
        title!("Plot with clim")
        plot(p1, p2, layout=(1,2), size=(1000,400))
    else
        plot(p1)
    end
end

function contract_except_center(mps_left, mps_right, center)
    result_l = ITensor(1.0)
    result_r = ITensor(1.0)

    mps_p = prime(linkinds, mps_right)

    for i in 1:center-1
        result_l *= dag(mps_left[i]) * mps_p[i]
    end

    for i in length(s):-1:center+1
        result_r *= dag(mps_left[i]) * mps_p[i]
    end

    return mps_p[center] * result_l * result_r
end

function make_beta(v1, v2, a1, a2, b1, b2, center, delta_t, nu, d1x, d1y, d2x, d2y, del, max_bond, cutoff)
    b_dotx = MPO(*(del, b1'')[:])
    b_doty = MPO(*(del, b2'')[:])
    d1x_b1 = apply(d1x, b1; alg="naive", maxdim=max_bond, cutoff=cutoff)
    d1x_b2 = apply(d1x, b2; alg="naive", maxdim=max_bond, cutoff=cutoff)
    d1y_b1 = apply(d1y, b1; alg="naive", maxdim=max_bond, cutoff=cutoff)
    d1y_b2 = apply(d1y, b2; alg="naive", maxdim=max_bond, cutoff=cutoff)
    b_dotx_b1 = apply(b_dotx, b1; alg="naive", maxdim=max_bond, cutoff=cutoff)
    b_dotx_b2 = apply(b_dotx, b2; alg="naive", maxdim=max_bond, cutoff=cutoff)
    b_doty_b1 = apply(b_doty, b1; alg="naive", maxdim=max_bond, cutoff=cutoff)
    b_doty_b2 = apply(b_doty, b2; alg="naive", maxdim=max_bond, cutoff=cutoff)

    B_1 = 0.5 * (apply(b_dotx, d1x_b1; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(b_doty, d1y_b1; alg="naive", maxdim=max_bond, cutoff=cutoff))
    B_1 += 0.5 * (apply(d1x, b_dotx_b1; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(d1y, b_doty_b1; alg="naive", maxdim=max_bond, cutoff=cutoff))
    B_1 += -nu * (apply(d2x, b1; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(d2y, b1; alg="naive", maxdim=max_bond, cutoff=cutoff))

    B_2 = 0.5 * (apply(b_dotx, d1x_b2; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(b_doty, d1y_b2; alg="naive", maxdim=max_bond, cutoff=cutoff))
    B_2 += 0.5 * (apply(d1x, b_dotx_b2; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(d1y, b_doty_b2; alg="naive", maxdim=max_bond, cutoff=cutoff))
    B_2 += -nu * (apply(d2x, b2; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(d2y, b2; alg="naive", maxdim=max_bond, cutoff=cutoff))

    out_1 = contract_except_center(v1, a1 - delta_t * B_1, center)
    out_2 = contract_except_center(v2, a2 - delta_t * B_2, center)

    out = vcat(vec(array(out_1)), vec(array(out_2))) # combine into 1d-vector
    out[abs.(out) .< cutoff] .= 0
    return out
end

function get_c_vec(v1, v2, center)
    return vcat(vec(array(v1[center])), vec(array(v2[center])))
end

function insert_c_vec(c_vec, v1, v2, center)
    length_1 = prod(size(v1[center]))
    length_2 = prod(size(v2[center]))
    out1 = deepcopy(v1)
    out2 = deepcopy(v2)
    out1[center] = ITensor(c_vec[1:length_1], inds(v1[center]))
    out2[center] = ITensor(c_vec[length_1+1:length_1+length_2], inds(v2[center]))
    return out1, out2
end

function apply_H(c_vec, v1, v2, dx_dx, dy_dy, dx_dy, center, max_bond, cutoff)
    v1_p, v2_p = insert_c_vec(c_vec, v1, v2, center)

    out_1 = apply(dx_dx, v1_p; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(dx_dy, v2_p; alg="naive", maxdim=max_bond, cutoff=cutoff)
    out_2 = apply(dx_dy, v1_p; alg="naive", maxdim=max_bond, cutoff=cutoff) + apply(dy_dy, v2_p; alg="naive", maxdim=max_bond, cutoff=cutoff)

    out_1 = contract_except_center(v1, out_1, center)
    out_2 = contract_except_center(v2, out_2, center)

    out = vcat(vec(array(out_1)), vec(array(out_2))) # combine into 1d-vector
    out[abs.(out) .< cutoff] .= 0
    return out
end

function find_max_on_2D_grid(func, R, x_min=0., x_max=1., y_min=0., y_max=1.)
    num_points_per_dim = 2^R
    x_range = range(x_min, x_max, num_points_per_dim)
    y_range = range(y_min, y_max, num_points_per_dim)
    max_val = -Inf
    for x in x_range
        for y in y_range
            max_val = max(max_val, func(x,y))
        end
    end
    return max_val
end