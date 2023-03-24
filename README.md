# DDQKD_SDP
Compute key rate with semidefinite program (SDP) for device-dependent protocol based on Mateus Araújo's paper (arXiv:2211.05725). The package is written in Julia with Convex (https://jump.dev/Convex.jl/stable/) as an interface for semidefinite programs. The scripts for BB84, B92, and DM CV QKD are provided as examples.

## Prerequisite
### Julia packages
#### @DDQKD_aroujo
- Convex
- MathOptInterface
- FastGaussQuadrature
- QuantumInformation
- LinearAlgebra
- SCS
- MosekTools (optional)
> - `SCS` and `MosekTools` are two of the solvers which can be used with `Convex` interface. `SCS` packages includes the executable files, while `MosekTools` is only an interface to Mosek, one needs to install Mosek separately.

#### @DataSaver
- DelimitedFiles
#### @Other scripts
- Combinatorics
- Printf

### SDP Solvers (optional)
- Mosek

## Installation
For registered Julia packages, one can using the commands
```Julia
using Pkg
Pkg.update()
Pkg.add("Convex")
```

For unregistered packages, one needs to use the package name with extension `.jl` such as `QuantumInformation.jl` instead of `QuantumInformation`.