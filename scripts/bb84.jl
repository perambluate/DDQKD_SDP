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
NUM_QUAD = 10   # Number of nodes in Guass-Radau quadrature
INPUT = 1       # The specific input for key generation
                # (for heterodyne measurement we only have one input)
PARTY = 0       # The party whose outcomes are taken as a reference for the final key

function h_bin(p)
    ans = - p.*log2.(p) - (1 .- p).*log2.(1 .- p)
    return ans
end

### Alice's measurements
dim_A = 2
A = zeros(dim_A,dim_A,1,2)
A[:,:,1,1] = proj([1,0])
A[:,:,1,2] = proj(1/sqrt(2)*[1,1])

### Bob's measurements
dim_B = 2
B = zeros(dim_B,dim_B,1,2)
B[:,:,1,1] = proj([1,0])
B[:,:,1,2] = proj(1/sqrt(2)*[1,1])

dim_AB = dim_A*dim_B
STATS_CHECK_MEAS = zeros(ComplexF64, dim_AB, dim_AB, 2)
# Measurement to check Z errors
STATS_CHECK_MEAS[:,:,1] = kron(A[:,:,1,1], (I(dim_B) - B[:,:,1,1])) +
                            kron((I(dim_A) - A[:,:,1,1]), B[:,:,1,1])
# Measurement to check X errors
STATS_CHECK_MEAS[:,:,2] = kron(A[:,:,1,2], (I(dim_B) - B[:,:,1,2])) +
                            kron((I(dim_A) - A[:,:,1,2]), B[:,:,1,2])

white_noise = I(dim_AB)./dim_AB
ideal_ket = 1/sqrt(2) * [1,0,0,1]
ideal_state = proj(ideal_ket)
rho_A = QI.ptrace(ideal_state, [dim_A, dim_B], 2)

visib_list = [1., .999, .99, .95, .9, .85, .8, .79, .78, .775]
N_POINT = size(visib_list, 1)
EC_cost_list = zeros(Float64, N_POINT)
HAE_list = zeros(Float64, N_POINT)

@time begin
    Threads.@threads for i in 1:N_POINT
        visib = visib_list[i]
        noisy_state = visib*ideal_state + (1-visib)*white_noise
        ObsStats = ddqkd.measurementStatistics(noisy_state, STATS_CHECK_MEAS)
        # println("Z error: $(ObsStats[1])")
        # println("X error: $(ObsStats[2])")
        EC_cost_list[i] = h_bin(ObsStats[1])
        h_AE, rho_AB = ddqkd.singleRoundEntropy(NUM_QUAD, A, PARTY, INPUT,
                                                dim_A, dim_B,
                                                statis_check_meas = STATS_CHECK_MEAS,
                                                observed_statis = ObsStats,
                                                reduce_A = true, rho_A = rho_A,
                                                solver = SOLVER,
                                                solver_args = SOLVER_ARGS)
        # println("H(A|E): $h_AE")
        # td = trace_distance(noisy_state, rho_AB)
        # println("Trace distance between theoritical state ",
        #         "and numerical optimized state: $td")
        HAE_list[i] = h_AE
    end
end

replace!(EC_cost_list, NaN => 0.)
rate_list = HAE_list - EC_cost_list
rate_list[rate_list .< 0] .= 0.

data = [visib_list HAE_list EC_cost_list rate_list]
Header = "visibility, hAE, ec_cost, key_rate"
println(Header)
for i in 1:N_POINT
    println(join(data[i,:], "\t"))
end

### Save data
if SAVE
    out_dir = "./data"
    out_name = @sprintf("bb84_wn-quad_%d-1.csv", NUM_QUAD)
    out_path = joinpath(out_dir, out_name)
    saver.saveData(data, out_path; header = Header, mode = "a")
end