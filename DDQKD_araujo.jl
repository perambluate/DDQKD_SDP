module DDQKD_aroujo
#=
    Compute key rate for device-dependent protocol with Araújo's method (arXiv:2211.05725)
=#
using MosekTools, SCS
using Convex
using MathOptInterface: OptimizerWithAttributes
using FastGaussQuadrature: gaussradau
using QuantumInformation, LinearAlgebra

export gaussRadau
export fullMeasurement, measurementStatistics, singleRoundEntropy


### Dictionary for solvers
SOLVERS = Dict("mosek" => Mosek, "scs" => SCS)

### Nodes and weights for Gauss-Radau quadrature
function gaussRadau(m::Integer)
    t, w = gaussradau(m)
    t = reverse(-1/2*t.+1/2)
    w = reverse(1/2*w)
    return t,w
end

### Full measurements from the CG form
function fullMeasurement(A::AbstractArray{<:Number, 4}, dim_A::Integer,
                         B::AbstractArray{<:Number, 4}, dim_B::Integer)
    #=
    Inputs:
        - A: Alice's measurements in CG form,
             i.e., for 2 inputs and 3 outcomes per input,
             A = [[A00, A01], [A10, A11]],
             and A02 = I - A00 - A01; A12 = I - A10 - A11
        - dim_A: Dimension of Alice's quantum system.
        - B: Bob's measurements in CG form similar to A
        - dim_B: Dimension of Bob's quantum system.
        
    Outputs:
        A list of operator [kron(A_xa, B_yb)]
    =#
    num_out = [size(A, 3), size(B, 3)].+1
    num_inp = [size(A, 4), size(B, 4)]
    dim = num_out.*num_inp
    dim_AB = dim_A*dim_B

    measurements = zeros(ComplexF64, dim_AB, dim_AB, dim[1]*dim[2])

    for y in 1:num_inp[2]
        for b in 1:num_out[2]
            for x in 1:num_inp[1]
                for a in 1:num_out[1]
                    i = (x-1)*2+a
                    j = (y-1)*2+b
                    op_A = zeros(ComplexF64, dim_A, dim_A)
                    op_B = zeros(ComplexF64, dim_B, dim_B)
                    op_A[:,:] = a != num_out[1] ? A[:,:,a,x] : I(dim_A)-sum(A[:,:,:,x], dims=3)
                    op_B[:,:] = b != num_out[2] ? B[:,:,b,y] : I(dim_B)-sum(B[:,:,:,y], dims=3)
                    meas_op = kron(op_A, op_B)
                    measurements[:,:,i+(j-1)*dim[1]] = meas_op
                end
            end
        end
    end

    return measurements
end

### Get the statistics of the measurements with certain state.
function measurementStatistics(state::AbstractArray{<:Number, 2},
                               measurements::AbstractArray{<:Number, 3};
                               shape::Tuple{Vararg{<:Integer}} = (),
                               trans::Bool = false)
    #=
    Inputs:
        - state: State in the form of density matrix.
        - measurements: A list of measurements ([dim, dim, num_meas])
        - shape: (Optional) The desired shape for reshaping outputs.
        - trans: (Optinal) Whether to transpose the output matrix or not.
    Outputs:
        The ststistics of measurements in list or matrix of wanted shape.
    =#
    dim = size(measurements)[end]
    Prob = zeros(dim)
    for i in 1:dim
        Prob[i] = real(tr(state*measurements[:,:,i]))
    end
    if isempty(shape)
        return Prob
    end
    Prob = reshape(Prob, shape)
    Prob = trans ? transpose(Prob) : Prob
    return Prob
end

### Feasibility check
function feasibilityCheck(dim::Integer;
                          statis_check_meas::AbstractArray{<:Number, 3} = Array{Number}(undef, 0, 0, 0),
                          observed_statis::AbstractArray{<:Number} = Array{Number}(undef, 0),
                          proj2support::Bool = false,
                          isometry::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                          solver::String = "mosek",
                          solver_args::AbstractArray{<:Pair{<:AbstractString, <:Real}, 1} = Array{Pair{String, Real}}(undef, 0))

    if proj2support
        try
            dim = size(isometry)[end]
        catch
            println(stderr, "Input 'isometry' is required for tomography.")
            return
        end
    end
    sigma = Semidefinite(dim, dim)
    lambda = Variable()
    num_check_statis = size(observed_statis)[end]
    constraints = [tr(sigma) == 1; (sigma - lambda*I(dim)) in :SDP]
    if !isempty(observed_statis)
        num_check_statis = size(observed_statis)[end]
        if proj2support
            for i in 1:num_check_statis
                operator_in_support = isometry'*statis_check_meas[:,:,i]*isometry
                constraints += [tr(operator_in_support*sigma) == observed_statis[i]]
            end
        else
            constraints += [tr(statis_check_meas[:,:,i]*sigma) == observed_statis[i]
                            for i in 1:num_check_statis]
        end
    end
    # constraints += [tr(statis_check_meas[:,:,i]*sigma) == observed_statis[i]
    #                 for i in 1:num_check_statis]
    
    problem = maximize(lambda, constraints)

    if !(solver in collect(keys(SOLVERS)))
        println(stderr, "Input 'solver' is invalid or not supported.
                        If u want to use solver other than mosek and scs,
                        plz install them and add them in the 'SOLVERS' dict in the top of the file.")
    else
        # solve!(problem, () -> SOLVERS[solver].Optimizer())
        solve!(problem, OptimizerWithAttributes(SOLVERS[solver].Optimizer, solver_args...))
    end

    println("status: ", problem.status)

    return evaluate(lambda), evaluate(sigma)
end

### Compute single round von Neumann entropy by Araújo's method
function singleRoundEntropy(num_quad::Integer, A::AbstractArray{<:Number, 4},
                            input::Integer, dim_A::Integer, dim_B::Integer;
                            statis_check_meas::AbstractArray{<:Number, 3} = Array{Number}(undef, 0, 0, 0),
                            observed_statis::AbstractArray{<:Number} = Array{Number}(undef, 0),
                            tomography::Bool = false,
                            state::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                            reduce_A::Bool = false,
                            rho_A::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                            reduce_B::Bool = false,
                            rho_B::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                            setNormConstr::Bool = false,
                            proj2support::Bool = false,
                            isometry::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                            postMap::Bool = false,
                            postIsometry::AbstractArray{<:Number, 2} = Array{Number}(undef, 0, 0),
                            solver::String = "mosek",
                            solver_args::AbstractArray{<:Pair{<:AbstractString, <:Real}, 1} = Array{Pair{String, Real}}(undef, 0))
    #=
    Inputs:
        - num_quad: Number of quadrature.
        - A: Alice's measurements in CG form.
        - dim_A: Dimension of Alice's quantum system.
        - dim_B: Dimension of Bob's quantum system.
        - statis_check_meas: A list of measurements for statistical check.
        - observed_statis: The statistics in list corresponding to statis_check_meas.
        - tomography: Whether to enforce tomography constraints or not.
        - state: The desired state for tomography constraints.
    Outputs:
        - entropy: The conditional von Neumann entropy for single round (asymptotic)
        - rho_AB: The state which gives that optimal entropy.
    =#

    ### Get Gauss-Radau quadrature
    t, w = gaussRadau(num_quad)

    dim_AB = dim_A*dim_B
    num_outcome = size(A, 3)+1
    ### Specify Alice's POVM for given input
    A_meas = zeros(dim_A, dim_A, num_outcome)
    A_meas[:,:,1:num_outcome-1] = copy(A[:,:,:,input])
    A_meas[:,:,end] = I(dim_A)-sum(A[:,:,:,input], dims=3)

    dim = dim_AB
    if proj2support & postMap
        println(stderr, "'proj2support' and 'postMap' cannot both be true")
        return
    elseif proj2support
        try
            dim = size(isometry)[end]
        catch
            println(stderr, "Input 'isometry' is required for tomography.")
            return
        end
    elseif postMap
        dim = num_outcome
    end

    ### Variables for optmization
    if tomography
        try
            sigma = copy(state)
        catch
            println(stderr, "Input 'state' is required for tomography.")
            return
        end
    elseif postMap
        sigma = Semidefinite(dim_AB, dim_AB)
    else
        sigma = Semidefinite(dim, dim)
    end

    zeta  = [[Convex.ComplexVariable(dim, dim) for _ in 1:num_outcome] for _ in 1:num_quad]
    eta   = [[HermitianSemidefinite(dim, dim) for _ in 1:num_outcome] for _ in 1:num_quad]
    theta = [[HermitianSemidefinite(dim, dim) for _ in 1:num_outcome] for _ in 1:num_quad]

    constraints = tomography ? Convex.EqConstraint[] : [tr(sigma) == 1]
    obj = 0

    for i in 1:num_quad
        coeff = w[i]/(t[i]*log(2))
        term_in_quad = 0
        for a in 1:num_outcome
            zz = zeta[i][a] + zeta[i][a]'
            zdz = (1-t[i]) * eta[i][a]
            zzd = t[i] * theta[i][a]
            if proj2support
                op_in_support = isometry' * kron(A_meas[:,:,a], I(dim_B)) * isometry
                term_in_quad += op_in_support * (zz + zdz) + zzd
            elseif postMap
                term_in_quad += A_meas[:,:,a]*(zz + zdz) + zzd
            else
                term_in_quad += kron(A_meas[:,:,a], I(dim_B))*(zz + zdz) + zzd
            end

            ### Matrixes for SDP constraints
            if postMap
                sigma_tilde = postIsometry * sigma * postIsometry'
                sigma_tilde = partialtrace(sigma_tilde, 2, [num_outcome, dim_A*dim_B])
                Gamma_1 = [sigma_tilde zeta[i][a]; zeta[i][a]' eta[i][a]]
                Gamma_2 = [sigma_tilde zeta[i][a]';zeta[i][a] theta[i][a]]
            else
                Gamma_1 = [sigma zeta[i][a]; zeta[i][a]' eta[i][a]]
                Gamma_2 = [sigma zeta[i][a]';zeta[i][a] theta[i][a]]
            end
            constraints += [Gamma_1 in :SDP; Gamma_2 in :SDP]
        end
        obj += coeff * term_in_quad
    end

    if !isempty(observed_statis)
        num_check_statis = size(observed_statis)[end]
        if proj2support
            for i in 1:num_check_statis
                operator_in_support = isometry'*statis_check_meas[:,:,i]*isometry
                constraints += [tr(operator_in_support*sigma) == observed_statis[i]]
            end
        else
            constraints += [tr(statis_check_meas[:,:,i]*sigma) == observed_statis[i]
                            for i in 1:num_check_statis]
        end
    end

    alpha_list = [3/2*max(1/t_i, 1/(1-t_i)) for t_i in t]
    # println("norm_constraints", alpha_list)

    if setNormConstr
        for i in 1:num_quad-1
            for a in 1:num_outcome
                constraints += [opnorm(zeta[i][a], 1) <= alpha_list[i]]
                constraints += [opnorm(eta[i][a], 1) <= alpha_list[i]]
                constraints += [opnorm(theta[i][a], 1) <= alpha_list[i]]
            end
        end
    end

    if reduce_A
        try
            if proj2support
                constraints += [
                    partialtrace(isometry*sigma*isometry', 2, [dim_A, dim_B]) == rho_A]
            else
                constraints += [partialtrace(sigma, 2, [dim_A, dim_B]) == rho_A]
            end
        catch
            println(stderr, "Input 'rho_A' is not defined or dim. mismatch")
            return
        end
    end

    if reduce_B
        try
            if proj2support
                constraints += [
                    partialtrace(isometry*sigma*isometry', 1, [dim_A, dim_B]) == rho_B]
            else
                constraints += [partialtrace(sigma, 1, [dim_A, dim_B]) == rho_B]
            end
        catch
            println(stderr, "Input 'rho_B' is not defined or dim. mismatch")
            return
        end
    end
    
    problem = minimize(real(tr(obj)), constraints)
    
    if !(solver in collect(keys(SOLVERS)))
        println(stderr, "Input 'solver' is invalid or not supported.
                        If u want to use solver other than mosek and scs,
                        plz install them and add them in the 'SOLVERS' dict in the top of the file.")
    else
        # solve!(problem, () -> SOLVERS[solver].Optimizer())
        solve!(problem, OptimizerWithAttributes(SOLVERS[solver].Optimizer, solver_args...))
    end
    
    println("status: ", problem.status)

    # for i in 1:num_quad
    #     println(i,"-th quadrature")
    #     println(alpha_list[i])
    #     for a in 1:num_outcome
    #         println("zeta")
    #         display(norm(evaluate(zeta[i][a])))
    #         println("eta")
    #         display(norm(evaluate(eta[i][a])))
    #         println("theta")
    #         display(norm(evaluate(theta[i][a])))
    #     end
    # end
    rho_AB = evaluate(sigma)
    opt_val = real(evaluate(tr(obj)))

    entropy = opt_val + sum(w./t)/log(2)
    return entropy, rho_AB
end

end #module