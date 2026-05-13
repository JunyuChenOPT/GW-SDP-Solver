function [Q,c,A,b,B,d,p,E,meta] = build_gw_mbqp(C,D,alpha,beta,opts)
%BUILD_GW_MBQP Build the continuous MBQP data used by RiNNAL+ for GW-SDP.

if nargin < 5 || isempty(opts)
    opts = struct();
end
opts = normalize_build_opts(opts);

[C,D,alpha,beta] = validate_gw_inputs(C,D,alpha,beta,opts.tol);
m = size(C,1);
n = size(D,1);
mn = m*n;

t_total = tic;
t0 = tic;
[Q,cost_meta] = build_gw_cost_matrix(C,D,opts.loss,opts.lossfun);
timing.build_Q = toc(t0);
t0 = tic;
[A,b,constraint_meta] = build_transport_constraints(m,n,alpha,beta);
timing.build_A = toc(t0);

c = zeros(mn,1);
B = [];
d = [];
p = [];
E = [];

meta = constraint_meta;
meta.m = m;
meta.n = n;
meta.mn = mn;
meta.alpha = alpha;
meta.beta = beta;
meta.loss = opts.loss;
meta.cost = cost_meta;
meta.timing = timing;
meta.full_row_rank = true;
meta.full_row_rank_reason = ...
    'Reduced transport matrix [Arow; Acol(1:n-1,:)] has structural rank m+n-1.';
meta.rank_check_requested = opts.check_rank;
if opts.check_rank
    meta.rank_A = rank(full(A), opts.tol);
    meta.full_row_rank = meta.rank_A == size(A,1);
    meta.full_row_rank_reason = 'Verified by dense rank(full(A), opts.tol).';
end
meta.timing.total_model_build = toc(t_total);
end

function opts = normalize_build_opts(opts)
if ~isstruct(opts)
    error('build_gw_mbqp:InvalidOpts', 'opts must be a struct.');
end
if ~isfield(opts, 'loss') || isempty(opts.loss)
    opts.loss = 'square';
end
if ~isfield(opts, 'lossfun')
    opts.lossfun = [];
end
if ~isfield(opts, 'tol') || isempty(opts.tol)
    opts.tol = 1e-6;
end
if ~isfield(opts, 'check_rank') || isempty(opts.check_rank)
    opts.check_rank = false;
end
end

function [C,D,alpha,beta] = validate_gw_inputs(C,D,alpha,beta,tol)
if ~isnumeric(C) || ndims(C) ~= 2 || size(C,1) ~= size(C,2)
    error('build_gw_mbqp:InvalidC', 'C must be a square numeric matrix.');
end
if ~isnumeric(D) || ndims(D) ~= 2 || size(D,1) ~= size(D,2)
    error('build_gw_mbqp:InvalidD', 'D must be a square numeric matrix.');
end
if isempty(C) || isempty(D)
    error('build_gw_mbqp:EmptyCost', 'C and D must be nonempty.');
end
if any(~isfinite(C(:))) || any(~isfinite(D(:)))
    error('build_gw_mbqp:NonfiniteCost', 'C and D must contain finite values.');
end

alpha = alpha(:);
beta = beta(:);
m = size(C,1);
n = size(D,1);
if length(alpha) ~= m
    error('build_gw_mbqp:AlphaDimension', 'length(alpha) must equal size(C,1).');
end
if length(beta) ~= n
    error('build_gw_mbqp:BetaDimension', 'length(beta) must equal size(D,1).');
end
if any(~isfinite(alpha)) || any(~isfinite(beta))
    error('build_gw_mbqp:NonfiniteMass', 'alpha and beta must contain finite values.');
end
if any(alpha < -tol) || any(beta < -tol)
    error('build_gw_mbqp:NegativeMass', 'alpha and beta must be nonnegative probability vectors.');
end
alpha = max(alpha,0);
beta = max(beta,0);
if abs(sum(alpha)-1) > tol
    error('build_gw_mbqp:AlphaMass', 'sum(alpha) must be 1 within tolerance.');
end
if abs(sum(beta)-1) > tol
    error('build_gw_mbqp:BetaMass', 'sum(beta) must be 1 within tolerance.');
end
alpha = alpha/sum(alpha);
beta = beta/sum(beta);
end
