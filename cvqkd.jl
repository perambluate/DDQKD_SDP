include("./DDQKD_araujo.jl")
include("./DataSaver.jl")
using .DDQKD_aroujo, .DataSaver
using QuantumInformation, LinearAlgebra
using Combinatorics: factorial, doublefactorial
using Printf

### Quaternary entropy
function ShannonEntropy(Prob::AbstractArray{<:Real, 2})
    entropy = -Prob.*log.(Prob)
    nancheck = isnan.(entropy)
    entropy = entropy.*.!(nancheck)
    return sum(entropy)
end
function errorCorrectionCost(Prob::AbstractArray{<:Real, 2})
    #=
    num_inp, num_out = size(Prob)
    entropy = 0
    for i in 1:num_inp
        p_i = sum(Prob[i,:])
        if p_i == 0
            continue
        end
        entropy_per_inp = 0
        for j in 1:num_out
            if Prob[i,j] == 0
                continue
            end
            entropy_per_inp += -Prob[i,j]*log2(Prob[i,j]/p_i)
        end
        entropy += entropy_per_inp*p_i
    end
    =#
    H_AB = ShannonEntropy(Prob)
    H_A = ShannonEntropy(sum(Prob, dims=1))
    return H_AB - H_A
end

function coherentState(alpha::Number, dim::Integer)
    state = zeros(dim)
    for m in 1:dim
        coeff = alpha^(m-1) * exp(-norm(alpha)^2/2) / sqrt(factorial(m-1))
        state += coeff * ket(m,dim)
    end
    state = state./norm(state)
    return state
end

function thermalState(alpha::Number, dim::Integer; transmitance::Real = 1, excess_noise::Real = 0)
    mean_n = transmitance * ( alpha*conj(alpha) + excess_noise/2 )
    state = Diagonal(zeros(dim))
    for i in 1:dim
        state[i,i] += (mean_n/(1+mean_n))^(i-1) / (1+mean_n)
    end
    return state
end

function annihilation(dim::Integer)
    operator = zeros(ComplexF64, dim, dim)
    for n in 1:dim-1
        operator[n, n+1] = sqrt(n)
    end

    return operator
end

function gammaIntegral(m::Integer)
    if m % 2 == 0
        return Float64.( doublefactorial(m-1) / (2^(Int(m/2)+1)) * sqrt(pi) )
    else
        k = Int((m-1)/2)
        return factorial(k)/2
    end
end

# function supportBasis(region::Integer, dim::Integer)
#     _ket = zeros(ComplexF64, dim)
#     for m in 1:dim
#         if m == 1
#             _ket[m] = 1/4
#         else
#             m_fock = m-1
#             angle_part = 2/m_fock * sin(m_fock*pi/4) * exp(im*m_fock*(region-1)*pi/2)
#             coeff = 1/(pi*sqrt(factorial(m_fock))) * angle_part * gammaIntegral(m_fock+1)
#             _ket[m] = coeff
#         end
#     end

#     return _ket
# end

function regionMeasurement(region::Integer, dim::Integer)
    operator = zeros(dim, dim)
    for m in 1:dim
        for n in m:dim
            if m == n
                coeff = 1/(2*factorial(m-1)) * gammaIntegral(2*m-1)
                operator += coeff * ketbra(m, m, dim)
            else
                angle_part = 2/(m-n)*sin((m-n)*pi/4)*exp(im*(m-n)*(region-1)*pi/2)
                coeff = angle_part * gammaIntegral(m+n-1) /
                        (pi*factorial(m-1)*sqrt(factorial(n-1, m-1)))
                operator += coeff * ketbra(m, n, dim)
                operator += conj(coeff) * ketbra(n, m, dim)
            end
        end
    end

    return operator
end


NUM_QUAD = 10     # Number of nodes in Guass-Radau quadrature
INPUT = 1         # The specific input for key generation
                  # (for heterodyne measurement we only have one input)
# ALPHA = 0.7       # Displacement of the coherent state
N_CUTOFF = 15     # Photon number cutoff
data_alpha = Array(0.1:0.2:3.0)
data_size = length(data_alpha)
data_hAE = zeros(data_size)
data_ec_cost = zeros(data_size)
data_key_rate = zeros(data_size)
eta = 0.5
@time begin
    for i in 1:data_size
        ALPHA = data_alpha[i]
        dim_B = N_CUTOFF + 1

        # PreparedStates = [coherentState(ALPHA, dim_B), coherentState(im*ALPHA, dim_B),
        #                     coherentState(-ALPHA, dim_B), coherentState(-im*ALPHA, dim_B)]

        # posQuadOp = (creation(dim_B)' + creation(dim_B))./2
        # momQuadOp = (creation(dim_B)' - creation(dim_B)).*im./2

        ### Alice's measurements
        dim_A = 4
        A = zeros(dim_A,dim_A,3,1)
        A[:,:,1,1] = proj(ket(1,4))
        A[:,:,2,1] = proj(ket(2,4))
        A[:,:,3,1] = proj(ket(3,4))

        ### Bob's measurements
        B = zeros(ComplexF64, dim_B, dim_B, 3, 1)
        B[:,:,1,1] = regionMeasurement(1,dim_B)
        B[:,:,2,1] = regionMeasurement(2,dim_B)
        B[:,:,3,1] = regionMeasurement(3,dim_B)

        dim_AB = dim_A*dim_B

        ### Number of input/output for the shape of the matrix of measurement statistics
        num_out = [size(A, 3), size(B, 3)].+1
        num_inp = [size(A, 4), size(B, 4)]
        dim = num_out.*num_inp
        num_meas_op = prod(dim)

        # PostIsometry = zeros(ComplexF64, dim_AB*dim_A, dim_AB)
        # for i in 1:num_out[2]
        #     if i != 4
        #         global PostIsometry += kron(ket(i, num_out[2]), I(dim_A), sqrt(B[:,:,i,1]))
        #     else
        #         POVM_B = I(dim_B) - sum(B[:,:,:,1];dims=3)
        #         global PostIsometry += kron(ket(i, num_out[2]), I(dim_A), sqrt(POVM_B[:,:,1]))
        #     end
        # end
        
        # ket_ideal = 1/2* (kron(ket(1,4), PreparedStates[1]) + kron(ket(2,4), PreparedStates[2])
        #                     + kron(ket(3,4), PreparedStates[3]) + kron(ket(4,4), PreparedStates[4]))
        ket_ideal = zeros(ComplexF64, dim_AB*dim_B)
        for i in 1:4
            ket_ideal += 1/2*kron(ket(i,4), coherentState(sqrt(eta)*ALPHA*im^(i-1), dim_B), coherentState(sqrt(1-eta)*ALPHA*im^(i-1), dim_B))
        end
        noisy_state = QuantumInformation.ptrace(proj(ket_ideal), [dim_A, dim_B, dim_B], 3)

        _eigvals = eigvals(noisy_state)
        num_nozero_eigvals = sum(sqrt.(conj.(_eigvals).*_eigvals).>1e-6)
        Isometry = eigvecs(noisy_state)[:,dim_AB-num_nozero_eigvals:dim_AB]
        # for i in 1:num_out[1]
        #     for j in 1:num_out[2]
        #         Isometry[:,(i-1)*4+j] += kron(ket(i,dim_A), Bob_bases[:, j])
        #     end
        # end

        # Isometry = Matrix(qr(Isometry).Q)

        # post_state = PostIsometry * noisy_state * PostIsometry'
        # post_state = QuantumInformation.ptrace(post_state, [dim_A, dim_A, dim_B], [2,3])

        full_measurements = DDQKD_aroujo.fullMeasurement(A, dim_A, B, dim_B)
        observed_statis = DDQKD_aroujo.measurementStatistics(noisy_state, full_measurements)

        rho_A = QuantumInformation.ptrace(noisy_state, [dim_A, dim_B], 2)
        # rho_B = QuantumInformation.ptrace(noisy_state, [dim_A, dim_B], 1)

        # h_AE, rho_AB = DDQKD_aroujo.singleRoundEntropy(NUM_QUAD, A, INPUT, dim_A, dim_B;
        #                                         tomography = true, state = noisy_state,
        #                                         solver = "mosek",
        #                                         solver_args = ["INTPNT_CO_TOL_DFEAS" => 1e-7])

        h_AE, omega_AB = DDQKD_aroujo.singleRoundEntropy(NUM_QUAD, A, INPUT, dim_A, dim_B;
                                                statis_check_meas = full_measurements,
                                                observed_statis = observed_statis,
                                                reduce_A = true, rho_A = rho_A,
                                                proj2support = true, isometry = Isometry,
                                                setNormConstr = true,
                                                solver = "mosek",
                                                solver_args = ["INTPNT_CO_TOL_DFEAS" => 1e-7])

        println("h_AE ", h_AE)
        data_hAE[i] = h_AE
        rho_AB = Isometry * omega_AB * Isometry'

        # display(ptrace(rho_AB, [dim_A, dim_B], 2))
        # display(ptrace(rho_AB, [dim_A, dim_B], 1))

        Prob = DDQKD_aroujo.measurementStatistics(rho_AB, full_measurements,
                                                    shape = Tuple(dim),
                                                    trans = true)
        # display(Prob)
        ec_cost = errorCorrectionCost(Prob)
        data_ec_cost[i] = ec_cost
        # println("ec cost ", ec_cost)
        asym_rate = h_AE - ec_cost
        asym_rate = (asym_rate < 0) ? 0 : asym_rate
        # println("asymptotic rate ", asym_rate)
        data_key_rate[i] = asym_rate

    end
end

data = [data_alpha data_hAE data_ec_cost data_key_rate]
Header = "alpha, hAE, ec_cost, key_rate"
println(Header)
for i in 1:data_size
    println(join(data[i,:], "\t"))
end
#=
### Save data
out_dir = "./"
out_name = @sprintf("dm_cv-ampl_test-cutoff_%d-quad_%d-eta_%.2f.csv", N_CUTOFF, NUM_QUAD, eta)
out_path = joinpath(out_dir, out_name)
DataSaver.saveData(data, out_path; header = Header, mode = "a")
=#
