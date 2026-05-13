# GW-SDP Solver Wrapper for RiNNAL+

This repository contains a MATLAB wrapper that formulates a
Gromov-Wasserstein SDP relaxation and solves it with
[RiNNAL+](https://github.com/HouDiOpt/RiNNALplus).

The wrapper models GW-SDP as the SDP-RLT/DNN relaxation of the continuous
nonnegative quadratic program

```text
minimize    x' Q x
subject to  A x = b
            x >= 0
```

with `x = pi(:)` in MATLAB column-major order. It uses

```matlab
c = zeros(m*n,1);
B = [];
d = [];
p = [];
E = [];
```

No binary variables, complementarity constraints, or changes to `RiNNAL_plus`
are introduced.

## What Is Included

```text
gw_sdp/src/                    solver wrapper and model-building helpers
gw_sdp/tests/                  MATLAB unit tests
gw_sdp/run_demo_tiny_gaussian.m tight small Gaussian demo
gw_sdp/run_demo_gaussian.m     2D-vs-3D Gaussian point-cloud demo
gw_sdp/README.md               formulation and validation details
```

Performance comparison experiments, generated CSV files, PNG plots, MAT files,
and batch experiment scripts are intentionally not included.

## Dependency

This code depends on RiNNAL+. Clone or download RiNNAL+ separately:

```bash
git clone https://github.com/HouDiOpt/RiNNALplus.git
```

Then either copy this repository's `gw_sdp/` folder into the RiNNAL+ repository
root, or add both the RiNNAL+ root and this repository's `gw_sdp/` folder to
your MATLAB path.

## Quick Start

From the RiNNAL+ repository root in MATLAB:

```matlab
setup_path
addpath(genpath('gw_sdp'))
runtests('gw_sdp/tests')
run('gw_sdp/run_demo_tiny_gaussian.m')
run('gw_sdp/run_demo_gaussian.m')
```

The public entry point is:

```matlab
sol = gw_sdp_rinnal(C, D, alpha, beta, opts)
```

`C` and `D` are square distance or cost matrices, and `alpha`, `beta` are
probability vectors.

## Vectorized Formulation

For `x = pi(:)` and `idx(i,j) = i + (j-1)*m`, the transport constraints are
built with sparse Kronecker products:

```matlab
Arow = kron(ones(1,n), speye(m));
Acol = kron(speye(n), ones(1,m));
A_full = [Arow; Acol];
A = [Arow; Acol(1:n-1,:)];
```

`A_full` has all `m+n` marginal equations but rank `m+n-1`; RiNNAL+ receives
the reduced full-row-rank matrix `A`.

For the default squared GW loss:

```matlab
L = (kron(ones(n,n), C) - kron(D, ones(m,m))).^2;
Q = 0.5*(L + L');
```

Custom scalar losses still use the explicit scalar-loop path.

## Output Diagnostics

The returned `sol` structure includes:

- `sol.pi`: extracted transport plan
- `sol.P`: lifted SDP block
- `sol.sdp_value`: relaxation value `<Q,P>`
- `sol.gw_value`: extracted-plan value `x'Qx`
- `sol.approx_ratio_upper`: `gw_value / sdp_value` when meaningful
- `sol.rank_lift` and `sol.rank_gap`
- marginal and RLT feasibility diagnostics in `sol.diagnostics`

For tight tiny Gaussian instances, `approx_ratio_upper` should be close to 1
and `rank_gap` should be close to 0.
