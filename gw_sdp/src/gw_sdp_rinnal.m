function sol = gw_sdp_rinnal(C, D, alpha, beta, opts)
%GW_SDP_RINNAL Solve the GW-SDP relaxation through RiNNAL+.
%
%   sol = GW_SDP_RINNAL(C,D,alpha,beta,opts) builds the continuous
%   equality-only MBQP whose SDP-RLT relaxation is the GW-SDP relaxation,
%   then calls RiNNAL_plus with p=[], B=[], d=[], and E=[].

if nargin < 5 || isempty(opts)
    opts = struct();
end
opts = normalize_solver_opts(opts);

if exist('RiNNAL_plus', 'file') ~= 2
    error('gw_sdp_rinnal:MissingRiNNAL', ...
        'RiNNAL_plus is not on the MATLAB path. Run setup_path from the RiNNAL+ root first.');
end

[Q,c,A,b,B,d,p,E,meta] = build_gw_mbqp(C,D,alpha,beta,opts);

r = [];
if isfield(opts, 'rank')
    r = opts.rank;
end

par = opts.par;
if ~isfield(par, 'tol')
    par.tol = opts.tol;
end

[fval, Xsol, info] = RiNNAL_plus(Q,c,A,b,B,d,p,r,E,par);

sol = extract_gw_solution(Xsol,meta.m,meta.n,Q,meta,opts);
sol.fval = fval;
sol.info = info;
sol.meta = meta;
sol.model.Q = Q;
sol.model.c = c;
sol.model.A = A;
sol.model.b = b;
sol.model.B = B;
sol.model.d = d;
sol.model.p = p;
sol.model.E = E;
end

function opts = normalize_solver_opts(opts)
if ~isstruct(opts)
    error('gw_sdp_rinnal:InvalidOpts', 'opts must be a struct.');
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
if ~isfield(opts, 'par') || isempty(opts.par)
    opts.par = struct();
end
if ~isfield(opts, 'project_transport') || isempty(opts.project_transport)
    opts.project_transport = false;
end
end
