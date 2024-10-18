module photonicHelper

export coherentKet, thermalState
export annihilation, creation, numberOperator
export regionMeasurement
export displacedThermalState

using QuantumInformation, LinearAlgebra
using Combinatorics: factorial, doublefactorial
using Cuba

function coherentKet(alpha::Number, dim::Integer)
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

function creation(dim::Integer)
    return annihilation(dim)'    
end

function numberOperator(dim::Integer)
    return Diagonal(Array(0:dim-1))
end

function gammaIntegral(m::Integer)
    if m % 2 == 0
        return Float64.( doublefactorial(m-1) / (2^(Int(m/2)+1)) * sqrt(pi) )
    else
        k = Int((m-1)/2)
        return factorial(k)/2
    end
end

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

function dts_integrand(x, f, params)
    alpha, eta, xi, m, n = params
    V = eta*xi/2
    y = x[1]/(1-x[1])
    int_fun = 1/V / (1-x[1])^2 * y^(m+n+1) * exp(-y^2) *
                exp(-1/V*(y*cispi(2*x[2])-sqrt(eta)*alpha)*(y*cispi(-2*x[2])-sqrt(eta)*conj(alpha))) * 
                cispi(2*(n-m)*x[2])
    f[1], f[2] = reim(int_fun)
end

function displacedThermalState(alpha, eta, xi, dim)
    state = zeros(ComplexF64, dim, dim)
    Threads.@threads for n = 1:dim
        for m = 1:n
            int_res = cuhre(dts_integrand, 2, 2;
                            userdata = (alpha, eta, xi, m-1, n-1),
                            minevals=2e4, atol = 1e-10, rtol = 1e-5)
            # println(int_res)
            int_val = complex(int_res[1]...)
            if n == m
                state[n, n] = int_val *2 / factorial(n-1)
            else
                state[n, m] = int_val *2 / (factorial(m-1)*sqrt(factorial(n-1, m-1)))
                state[m, n] = conj(int_val) *2 / (factorial(m-1)*sqrt(factorial(n-1, m-1)))
            end
        end
    end
    return state
end

function dts_Q_function(r, theta, alpha, eta, xi)
    V = 1+eta*xi/2
    return 1/(pi*V) * exp(-1/V * abs(r*exp(im*theta) - sqrt(eta)*alpha)^2)
end

function heterPolarMeasProb(z, x, alpha, eta, xi,
                            delta_a = 0., delta_p = 0., num_polar_region = 4)
    alpha_x = alpha*cispi((x-1)*2/num_polar_region)
    ang_ub = pi/4-delta_p + (z-1)*pi/2
    ang_lb = -pi/4+delta_p + (z-1)*pi/2
    ang_ub_lb_diff = ang_ub - ang_lb
    function integrand(x, f)
        r = delta_a + x[1]/(1-x[1])
        theta = ang_ub_lb_diff * x[2] + ang_lb
        int_fun = r/(1-x[1])^2*dts_Q_function(r, theta, alpha_x, eta, xi)
        f[1], f[2] = reim(int_fun)
    end
    int_res = cuhre(integrand, 2, 2; minevals=2e4, atol = 1e-10, rtol = 1e-5)
    # println(int_res)
    int_val = complex(int_res[1]...)
    return ang_ub_lb_diff * int_val
end

function newRegionMeasurement()
end

end
