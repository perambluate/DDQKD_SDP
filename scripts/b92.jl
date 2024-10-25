include("./DDQKD_araujo.jl")
include("./DataSaver.jl")

import .DDQKD_araujo as ddqkd
import .DataSaver as saver
import QuantumInformation as QI
using QuantumInformation, LinearAlgebra
using Printf

SOLVER = "mosek"
NTHREAD_SDP = 1
WARNING_LEVEL = 0
LOG_LEVEL = 0
SOLVER_ARGS = Dict("MSK_IPAR_NUM_THREADS" => NTHREAD_SDP,
                   "MSK_IPAR_MAX_NUM_WARNINGS" => WARNING_LEVEL,
                   "MSK_IPAR_LOG" => LOG_LEVEL)
SAVE = true
NUM_QUAD = 16   # Number of nodes in Guass-Radau quadrature
INPUT = 1       # The specific input for key generation
                # (for heterodyne measurement we only have one input)
PARTY = 0       # The party whose outcomes are taken as a reference for the final key


function h_bin(p)
    ans = - p.*log2.(p) - (1 .- p).*log2.(1 .- p)
    return ans
end

### Alice's measurements
dim_A = 2
A = zeros(dim_A,dim_A,1,1)
A[:,:,1,1] = proj([1,0])

### Bob's measurements
dim_B = 2
B = zeros(dim_B,dim_B,1,2)
B[:,:,1,1] = proj([1,0])
B[:,:,1,2] = proj(1/sqrt(2)*[1,1])
prob_Y = [1/2, 1/2]

dim_AB = dim_A*dim_B
ERR_MEAS = zeros(ComplexF64, dim_AB, dim_AB, 2)
INC_MEAS = zeros(ComplexF64, dim_AB, dim_AB, 2)
STATS_CHECK_MEAS = zeros(ComplexF64, dim_AB, dim_AB, 2)
# Error rates
ERR_MEAS[:,:,1] = kron(A[:,:,1,1], I(dim_B)-B[:,:,1,1])
ERR_MEAS[:,:,2] = kron(I(dim_A)-A[:,:,1,1], I(dim_B)-B[:,:,1,2])
# Inconclusive
INC_MEAS[:,:,1] = kron(I(dim_A), B[:,:,1,1])
INC_MEAS[:,:,2] = kron(I(dim_A), B[:,:,1,2])

STATS_CHECK_MEAS = cat(ERR_MEAS, INC_MEAS, dims=3)
STATS_CHECK_MEAS[:,:,1] = sum(ERR_MEAS, dims=3)
STATS_CHECK_MEAS[:,:,2] = sum(INC_MEAS, dims=3)

white_noise = I(dim_AB)./dim_AB
ideal_state = proj(1/sqrt(2) * (kron([1,0], [1,0]) + kron([0,1], 1/sqrt(2)*[1,1])))
# rho_A = QI.ptrace(ideal_state, [dim_A, dim_B], 2)
# display(rho_A)

visib_list = [1., .999, .99, .97, .95, .94, .93, .92, .91, .9]
N_POINT = size(visib_list, 1)
QBER_list = zeros(Float64, N_POINT)
EC_cost_list = zeros(Float64, N_POINT)
HAE_list = zeros(Float64, N_POINT)
rate_list = zeros(Float64, N_POINT)

@time begin
    Threads.@threads for i in 1:N_POINT
        visib = visib_list[i]
        noisy_state = visib*ideal_state + (1-visib)*white_noise
        rho_A = QI.ptrace(noisy_state, [dim_A, dim_B], 2)
        ObsStats = ddqkd.measurementStatistics(noisy_state, STATS_CHECK_MEAS)
        errs = ddqkd.measurementStatistics(noisy_state, ERR_MEAS)
        incs = ddqkd.measurementStatistics(noisy_state, INC_MEAS)
        inc = prob_Y' * incs
        # println("Inconclusive: $inc")
        temp = errs ./ (1 .- incs)
        qber = prob_Y' * temp
        # println("Error rate: $qber")
        QBER_list[i] = qber
        ec_cost = (1-inc)*h_bin(qber)
        ec_cost = isnan(ec_cost) ? 0 : ec_cost
        EC_cost_list[i] = ec_cost
        ObsStats = ddqkd.measurementStatistics(noisy_state, STATS_CHECK_MEAS)
        h_AE, rho_AB = ddqkd.singleRoundEntropy(
                                            NUM_QUAD, A, PARTY, INPUT,
                                            dim_A, dim_B,
                                            statis_check_meas = STATS_CHECK_MEAS,
                                            observed_statis = ObsStats,
                                            reduce_A = true, rho_A = rho_A,
                                            solver = SOLVER,
                                            solver_args = SOLVER_ARGS)
        # println("H(A|E): $h_AE")
        td = trace_distance(noisy_state, rho_AB)
        println("Trace distance between theoritical state ",
                "and numerical optimized state: $td")
        HAE_list[i] = h_AE
        rate = (1-inc)*h_AE - ec_cost
        rate_list[i] = rate > 0 ? rate : 0
    end
end

data = [visib_list HAE_list QBER_list EC_cost_list rate_list]
Header = "visibility, hAE, QBER, ec_cost, key_rate"
println(Header)
for i in 1:N_POINT
    println(join(data[i,:], "\t"))
end

### Save data
if SAVE
    out_dir = "./data"
    out_name = @sprintf("b92_wn-quad_%d-1.csv", NUM_QUAD)
    out_path = joinpath(out_dir, out_name)
    saver.saveData(data, out_path; header = Header, mode = "a")
end
