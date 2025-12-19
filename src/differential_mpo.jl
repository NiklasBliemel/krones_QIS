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

    mps_vals = fill(0.0, (2^N, 2^N))

    for x in 1:2^N
        for y in 1:2^N
            mps_vals[y,x] = mps_tt(grididx_to_quantics(qgrid, (xvec[x], yvec[y])))
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

function apply_U_p_k_T(mps, v_k, center)
    result_l = ITensor(1.0)
    result_r = ITensor(1.0)

    mps_p = prime(linkinds, mps)

    for i in 1:center-1
        result_l *= dag(v_k[i]) * mps_p[i]
    end

    for i in length(s):-1:center+1
        result_r *= dag(v_k[i]) * mps_p[i]
    end

    return mps_p[center] * result_l * result_r
end

function make_beta_k(v, a, b, k, center, delta_t, nu, d1, d2, del, max_bond)
    b_del = [MPO(*(del, b[1]'')[:]), MPO(*(del, b[2]'')[:])]
    b_k_d1 = [apply(d1[1], b[k]; alg="naive", maxdim=max_bond), apply(d1[2], b[k]; alg="naive", maxdim=max_bond)]
    b_k_del_d1 = apply.(b_del, b_k_d1, maxdim=max_bond)
    b_k_d2 = [apply(d2[1], b[k]; alg="naive", maxdim=max_bond), apply(d2[2], b[k]; alg="naive", maxdim=max_bond)]

    result = apply_U_p_k_T(a[k], v[k], center)
    result += apply_U_p_k_T(-delta_t * b_k_del_d1[1], v[k], center)
    result += apply_U_p_k_T(-delta_t * b_k_del_d1[2], v[k], center)
    
    result += apply_U_p_k_T(delta_t * nu * b_k_d2[1], v[k], center)
    result += apply_U_p_k_T(delta_t * nu * b_k_d2[2], v[k], center)

    return array(result)
end

function make_beta(v, a, b, center, delta_t, nu, d1, d2, del, max_bond)
    return vcat(vec.([make_beta_k(v, a, b, 1, center, delta_t, nu, d1, d2, del, max_bond), 
                      make_beta_k(v, a, b, 2, center, delta_t, nu, d1, d2, del, max_bond)])...)
end

function apply_H_k_j(c, u_c_p, d1, center, k, j, max_bond)
    u_c_p[j][center] = ITensor(c[j], inds(u_c_p[j][center]))
    result = apply(d1[j], u_c_p[j]; alg="naive", maxdim=max_bond)
    result = apply(d1[k], result; alg="naive", maxdim=max_bond)
    result = apply_U_p_k_T(result, u_c_p[k], center)
    return result
end

function apply_H(c_vec, u_c_p, d1, center, max_bond)
    u_c_p_array = array.(getindex.(u_c_p, center))
    u_c_p_shapes = size.(u_c_p_array)
    u_c_p_lengths = length.(u_c_p_array)
    c = [c_vec[1:u_c_p_lengths[1]], c_vec[u_c_p_lengths[1]+1:length(c_vec)]]
    c = reshape.(c, u_c_p_shapes)
    result_1 = apply_H_k_j(c, u_c_p, d1, center, 1, 1, max_bond) + apply_H_k_j(c, u_c_p, d1, center, 1, 2, max_bond)
    result_2 = apply_H_k_j(c, u_c_p, d1, center, 2, 1, max_bond) + apply_H_k_j(c, u_c_p, d1, center, 2, 2, max_bond)
    result_1 = vec(array(result_1))
    result_2 = vec(array(result_2))
    # result_2 = c_vec[u_c_p_lengths[1]+1:length(c_vec)]
    return vcat(result_1, result_2)
end

function place_c_vec!(u, c, center)
    c_shapes = size.(array.(getindex.(u, center)))
    c_lenghts = prod.(c_shapes)
    c = reshape.([c[1:c_lenghts[1]], c[c_lenghts[1]+1:length(c)]], c_shapes)
    u[1][center] = ITensor(c[1], inds(u[1][center]))
    u[2][center] = ITensor(c[2], inds(u[2][center]))
end

function shape_c(u, c, center)
    c_shapes = size.(array.(getindex.(u, center)))
    c_lenghts = prod.(c_shapes)
    return reshape.([c[1:c_lenghts[1]], c[c_lenghts[1]+1:length(c)]], c_shapes)
end

function place_c_vec(u, c, center)
    c_shapes = size.(array.(getindex.(u, center)))
    c_lenghts = prod.(c_shapes)
    c = reshape.([c[1:c_lenghts[1]], c[c_lenghts[1]+1:length(c)]], c_shapes)
    u_copy = deepcopy(u)
    u_copy[1][center] = ITensor(c[1], inds(u[1][center]))
    u_copy[2][center] = ITensor(c[2], inds(u[2][center]))
    return u_copy
end

function get_c_vec(u, center)
    return vcat(vec.(array.(getindex.(u, center)))...)
end