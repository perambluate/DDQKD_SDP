# DDQKD_SDP
Compute key rate with semidefinite program (SDP) for device-dependent protocol based on
Mateus Araújo's paper (arXiv:2211.05725).
The package is written in Julia with Convex (https://jump.dev/Convex.jl/stable/) as an interface
for semidefinite programs. The scripts for BB84, B92, and DM CV QKD are provided as examples.

## Prerequisite

### Julia packages for each module

> Julia packages Installation
> For registered Julia packages, one can using the commands
> ```Julia
> using Pkg
> Pkg.update()
> Pkg.add("Convex")
> ```
> For unregistered packages, one needs to use the package name with extension `.jl`
> such as `QuantumInformation.jl` instead of `QuantumInformation`.

#### @DDQKD_araujo

- Convex
- MathOptInterface
- FastGaussQuadrature
- QuantumInformation
- LinearAlgebra
- SCS
- MosekTools (optional)

#### @DataSaver
- DelimitedFiles
#### @Other scripts
- Combinatorics
- Printf

### SDP Solvers (optional)
- Mosek

> `SCS` and `MosekTools` are two of the solvers which can be used with `Convex` interface.
> `SCS` packages includes the executable files, while `MosekTools` is only an interface to `Mosek`,
> one needs to install `Mosek` separately.

## Installation
To use this package, one can either install via `Pkg` or download this repo directly.
### Install with `Pkg`
In Julia interface type `]`, you should see
```
(v1.1) pkg>
```

Then install package with `add`
```
(v1.1) pkg> add https://github.com/perambluate/DDQKD_SDP.jl
```

### Download from GitHub
#### With `git` command
```bash
git clone https://github.com/perambluate/DDQKD_SDP.git
```
#### Download as zip
In the GitHub repo page, click `<> Code` bottom, and you will see `Download ZIP`.

## Directory structure

```
├── .gitignore
├── DataSaver.jl
├── DDQKD_araujo.jl
├── LICENSE
├── photonicHelper.jl
├── README.md
└── scripts
    ├── b92.jl
    ├── bb84.jl
    └── dmcv.jl
```

## Modules

### `DDQKD_araujo.jl`

Compute the conditional von-Neumann entropy lower bound by solving semidefinite programming[^1].
Check scripts in `scripts/` to find out how to cooperate this method to derive key rates for quantum key distribution (QKD) protocols.

[^1]: *M. Araújo et al.*, Quantum 7, 1019 (2023).

### `photonicHelper.jl`

Provide assist functions for descrete-modulated (DM)continuous-variable (CV) QKD,
including construction of creation/annihilation/number operators, coherent/thermal/displaced-thermal state,
and region measurements in photon number basis.

### `DataSaver.jl`

A helper module to simplify the data-saving process.

## Usage

Use `julia <script_name>` to run the script.

### Multi-threading

For scripts with multi-threading parallelization, e.g., multi-threading macro `Thread.@threads`,
one can use
```bash
julia --threads <num_thread> <script_name>
```
to activate multi-threading parallelization.

## Known issues and solutions

1. Sometimes Julia would crash unexpectedly (and see `Segmentation fault (core dumped)`).
> \[sol\]: One can try replacing command `julia` with `LD_LIBRARY_PATH="" julia` to get around this crash down.

2. The computation results from `dmcv.jl` seems incorrect in the scense that the optimal amplitudes, `alpha`, are much larger than the ones shown in [*J. Lin et al.*, Phys. Rev. X 9, 041064 (2019)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.041064).
