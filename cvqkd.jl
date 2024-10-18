include("./DDQKD_araujo.jl")
include("./DataSaver.jl")
include("./photonicHelper.jl")
import .DDQKD_araujo as ddqkd
import .DataSaver as saver
import .photonicHelper as photonic
using QuantumInformation, LinearAlgebra
import QuantumInformation as QI
using Printf

SOLVER = "mosek"
SOLVER_ARGS = ["INTPNT_CO_TOL_DFEAS" => 1e-7]
SAVE = false            # Whether to save the data or not
NUM_QUAD = 10           # Number of nodes in Guass-Radau quadrature
INPUT = 1               # The specific input for key generation
                        # (for heterodyne measurement we only have one input)
PARTY = 1               # The party whose outcomes are taken as a reference for the final key
N_CUTOFF = 15           # Photon number cutoff
# ALPHA = 0.7           # Displacement of the coherent state
data_alpha = Array(0.2:0.05:1.2) #[0.7]
num_alpha = length(data_alpha)
# ETA = 0.01            # The noise level during the transmission
data_eta = [1]
num_eta = length(data_eta)

### General Shannon entropy function
function ShannonEntropy(Prob::AbstractArray{<:Real, 2})
    entropy = -Prob.*log2.(Prob)
    nancheck = isnan.(entropy)
    entropy = entropy.*.!(nancheck)
    return sum(entropy)
end

function errorCorrectionCost(Prob::AbstractArray{<:Real, 2})
    H_AB = ShannonEntropy(Prob)
    H_A = ShannonEntropy(sum(Prob, dims=1))
    return H_AB - H_A
end

data_hAE = zeros(num_alpha)
data_ec_cost = zeros(num_alpha)
data_key_rate = zeros(num_alpha)

@time begin
    for j in 1:num_eta
        eta = data_eta[j]
        Threads.@threads for i in 1:num_alpha
            alpha = data_alpha[i]
            dim_B = N_CUTOFF + 1

            ### Alice's measurements
            dim_A = 4
            A = zeros(dim_A,dim_A,3,1)
            A[:,:,1,1] = proj(ket(1,4))
            A[:,:,2,1] = proj(ket(2,4))
            A[:,:,3,1] = proj(ket(3,4))

            ### Bob's measurements
            B = zeros(ComplexF64, dim_B, dim_B, 3, 1)
            B[:,:,1,1] = photonic.regionMeasurement(1,dim_B)
            B[:,:,2,1] = photonic.regionMeasurement(2,dim_B)
            B[:,:,3,1] = photonic.regionMeasurement(3,dim_B)

            dim_AB = dim_A*dim_B

            ### Number of input/output for the shape of the matrix of measurement statistics
            num_out = [size(A, 3), size(B, 3)].+1
            num_inp = [size(A, 4), size(B, 4)]
            dim = num_out.*num_inp
            num_meas_op = prod(dim)

            ideal_ket = 1/2*(kron(ket(1,4), photonic.coherentKet(alpha, dim_B))
                            + kron(ket(2,4), photonic.coherentKet(alpha*im, dim_B))
                            + kron(ket(3,4), photonic.coherentKet(-alpha, dim_B))
                            + kron(ket(4,4), photonic.coherentKet(-alpha*im, dim_B)))
            ideal_state = proj(ideal_ket)
            rho_A = QI.ptrace(ideal_state, [dim_A, dim_B], 2)
            ket_noise = zeros(ComplexF64, dim_AB*dim_B)
            for k in 1:4
                ket_noise += 1/2*kron(ket(k,4),
                            photonic.coherentKet(sqrt(eta)*alpha*im^(k-1), dim_B),
                            photonic.coherentKet(sqrt(1-eta)*alpha*im^(k-1), dim_B))
            end
            noisy_state = QI.ptrace(proj(ket_noise), [dim_A, dim_B, dim_B], 3)
            rho_B = QI.ptrace(noisy_state, [dim_A, dim_B], 1)

            # _eigvals = eigvals(noisy_state)
            # num_nonzero_eigvals = sum(sqrt.(conj.(_eigvals).*_eigvals).>1e-6)
            Isometry = eigvecs(noisy_state)[:,dim_AB-3:dim_AB]
            # for i in 1:num_out[1]
            #     for j in 1:num_out[2]
            #         Isometry[:,(i-1)*4+j] += kron(ket(i,dim_A), Bob_bases[:, j])
            #     end
            # end
            # Isometry = Matrix(qr(Isometry).Q)

            KeyMap = zeros(ComplexF64, dim_AB*dim_A, dim_AB)
            for i in 1:num_out[2]
                if i != 4
                    KeyMap += kron(ket(i, num_out[2]), I(dim_A), sqrt(B[:,:,i,1]))
                else
                    B_last = I(dim_B) - sum(B[:,:,:,1];dims=3)
                    KeyMap += kron(ket(i, num_out[2]), I(dim_A), sqrt(B_last[:,:,1]))
                end
            end


            post_state = KeyMap * noisy_state * KeyMap'
            # post_state = QI.ptrace(post_state, [dim_A, dim_A, dim_B], [2,3])

            full_measurements = ddqkd.fullMeasurement(A, dim_A, B, dim_B)
            numberOp = photonic.numberOperator(dim_B)
            posQuadOp = (photonic.creation(dim_B) + photonic.annihilation(dim_B))./sqrt(2)
            momQuadOp = (photonic.creation(dim_B) - photonic.annihilation(dim_B)).*im./sqrt(2)
            secondMomentOp = photonic.creation(dim_B)^2 + photonic.annihilation(dim_B)^2
            statis_check_meas = cat(full_measurements, kron(I(dim_A), numberOp), dims=3)
            observed_statis = ddqkd.measurementStatistics(noisy_state, statis_check_meas)
            ## Print probabilities computed by the given state and measurements
            # display(observed_statis)

            ## Use specified density matrix instead of matrix variable
            # h_AE, rho_AB = ddqkd.singleRoundEntropy(
            #                             NUM_QUAD, B, PARTY, INPUT, dim_A, dim_B;
            #                             tomography = true, state = noisy_state,
            #                             solver = SOLVER, solver_args = SOLVER_ARGS)

            
            h_AE, omega_AB = ddqkd.singleRoundEntropy(
                                        NUM_QUAD, B, PARTY, INPUT, dim_A, dim_B;
                                        statis_check_meas = statis_check_meas,
                                        observed_statis = observed_statis,
                                        reduce_A = true, rho_A = rho_A,
                                        # reduce_B = true, rho_B = rho_B,
                                        proj2support = true, isometry = Isometry,
                                        setNormConstr = true,
                                        solver = SOLVER, solver_args = SOLVER_ARGS)

            println("h_AE ", h_AE)
            data_hAE[i] = h_AE
            rho_AB = Isometry * omega_AB * Isometry'

            ## Print reduced states
            ### Print rho_A
            # display(QI.ptrace(rho_AB, [dim_A, dim_B], 2))
            ### Print rho_B
            # display(QI.ptrace(rho_AB, [dim_A, dim_B], 1))

            Prob = ddqkd.measurementStatistics(rho_AB, full_measurements,
                                                shape = Tuple(dim), trans = true)
            ## Print probabilities computed by the state after optimization
            # display(Prob)
            ec_cost = errorCorrectionCost(Prob)
            data_ec_cost[i] = ec_cost
            # println("ec cost ", ec_cost)
            asym_rate = h_AE - ec_cost
            asym_rate = (asym_rate < 0) ? 0 : asym_rate
            # println("asymptotic rate ", asym_rate)
            data_key_rate[i] = asym_rate
            
        end
        
        data = [data_alpha data_hAE data_ec_cost data_key_rate]
        Header = "alpha, hAE, ec_cost, key_rate"
        println(Header)
        for i in 1:num_alpha
            println(join(data[i,:], "\t"))
        end

        ### Save data
        if SAVE
            out_dir = "./data/dm_cv/"
            out_name = @sprintf("dm_cv-ampl_test-cutoff_%d-quad_%d-eta_%.0e.csv",
                                N_CUTOFF, NUM_QUAD, eta)
            out_path = joinpath(out_dir, out_name)
            saver.saveData(data, out_path; header = Header, mode = "a")
        end
        
    end
end
