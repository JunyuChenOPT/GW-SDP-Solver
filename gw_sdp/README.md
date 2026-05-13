# GW-SDP wrapper for RiNNAL+

This directory implements the Gromov-Wasserstein SDP relaxation as the
continuous equality-only SDP-RLT relaxation solved by RiNNAL+.

The public entry point is:

```matlab
sol = gw_sdp_rinnal(C, D, alpha, beta, opts)
```

Here `C` and `D` are square distance/cost matrices and `alpha`, `beta` are
probability vectors. The wrapper builds the nonnegative quadratic program

```text
minimize x' Q x
subject to A x = b
           x >= 0
```

with `x = vec(pi)` in MATLAB column-major order. It then calls:

```matlab
RiNNAL_plus(Q,c,A,b,B,d,p,r,E,par)
```

using:

```matlab
c = zeros(m*n,1);
B = [];
d = [];
p = [];
E = [];
```

No binary variables are introduced.

## Model conventions

- `idx = i + (j-1)*m` maps `pi(i,j)` to `x(idx)`.
- `A*x=b` contains all row marginals and only the first `n-1` column
  marginals, so `A` has full row rank.

With MATLAB column-major vectorization `x = pi(:)`, the marginal matrices are
built as:

```matlab
Arow = kron(ones(1,n), speye(m));
Acol = kron(speye(n), ones(1,m));

A_full = [Arow; Acol];
b_full = [alpha; beta];

A = [Arow; Acol(1:n-1,:)];
b = [alpha; beta(1:n-1)];
```

`A_full` is the mathematically complete transport system, with `m+n` rows, but
it is rank deficient because one marginal equation is redundant. RiNNAL+
receives the reduced full-row-rank matrix `A`, whose structural rank is
`m+n-1`. Diagnostics still keep `A_full` and `b_full` to check the omitted
final column marginal.

For the default squared loss, the flattened GW tensor is built by:

```matlab
L = (kron(ones(n,n), C) - kron(D, ones(m,m))).^2;
Q = 0.5*(L + L');
```

This gives
`L(idx(i,j),idx(k,l)) = (C(i,k)-D(j,l))^2`. Custom scalar loss functions still
use the explicit scalar-loop path; the wrapper does not assume custom losses
accept array inputs.

## Setup timing

`build_gw_mbqp` records lightweight setup timings in `meta.timing.build_A`,
`meta.timing.build_Q`, and `meta.timing.total_model_build`. These values measure
model construction only; they should not be interpreted as total solver speedup.

## Validation

Run from the RiNNAL+ repository root:

```matlab
setup_path
addpath(genpath('gw_sdp'))
runtests('gw_sdp/tests')
run('gw_sdp/run_demo_gaussian.m')
```

The demos print transport feasibility, nonnegativity, SDP value `<Q,P>`,
recovered objective `x'Qx`, rank, rank gap, and the ratio
`x'Qx / <Q,P>` when the denominator is positive.
