function sol = extract_gw_solution(Xsol,m,n,Q,meta,opts)
%EXTRACT_GW_SOLUTION Extract transport and lifted matrix from RiNNAL+ output.

if nargin < 6 || isempty(opts)
    opts = struct();
end
opts = normalize_extract_opts(opts);

if ~isstruct(Xsol) || ~isfield(Xsol, 'RR1')
    error('extract_gw_solution:MissingRR1', 'Xsol must contain the field RR1.');
end

Y = Xsol.RR1;
mn = m*n;
if size(Y,1) ~= mn+1 || size(Y,2) ~= mn+1
    error('extract_gw_solution:RR1Dimension', ...
        'Xsol.RR1 must be a %d-by-%d matrix.', mn+1, mn+1);
end

x_raw = Y(2:end,1);
P = Y(2:end,2:end);
pi_raw = reshape(x_raw,m,n);
diagnostics_raw = gw_diagnostics(pi_raw,P,Q,meta.alpha,meta.beta,meta,opts.tol);

sol.Y = Y;
sol.P = P;
sol.pi = pi_raw;
sol.x = x_raw;
sol.sdp_value = diagnostics_raw.sdp_value;
sol.gw_value = diagnostics_raw.gw_value;
sol.approx_ratio_upper = diagnostics_raw.approx_ratio_upper;
sol.rank_lift = diagnostics_raw.rank_lift;
sol.rank_gap = diagnostics_raw.rank_gap;
sol.diagnostics = diagnostics_raw;

if opts.project_transport
    [pi_projected, projection_info] = project_transport_ipfp(pi_raw,meta.alpha,meta.beta,opts.tol);
    sol.pi_raw = pi_raw;
    sol.x_raw = x_raw;
    sol.diagnostics_raw = diagnostics_raw;
    sol.pi_projected = pi_projected;
    sol.projection_info = projection_info;
    sol.pi = pi_projected;
    sol.x = pi_projected(:);
    sol.gw_value = full(sol.x'*Q*sol.x);
    sol.approx_ratio_upper = ratio_if_meaningful(sol.gw_value,sol.sdp_value,opts.tol);
    sol.diagnostics = gw_diagnostics(sol.pi,P,Q,meta.alpha,meta.beta,meta,opts.tol);
end
end

function opts = normalize_extract_opts(opts)
if ~isfield(opts, 'tol') || isempty(opts.tol)
    opts.tol = 1e-6;
end
if ~isfield(opts, 'project_transport') || isempty(opts.project_transport)
    opts.project_transport = false;
end
end

function [pi_projected,info] = project_transport_ipfp(pi_raw,alpha,beta,tol)
max_iter = 2000;
pi_projected = max(real(pi_raw),0);
pi_projected(~isfinite(pi_projected)) = 0;
if sum(pi_projected(:)) <= 0
    pi_projected = alpha(:)*beta(:)';
else
    pi_projected = pi_projected + eps*(alpha(:)*beta(:)');
end

for iter = 1:max_iter
    row_sum = pi_projected*ones(size(pi_projected,2),1);
    pi_projected = bsxfun(@times,pi_projected,alpha(:)./max(row_sum,realmin));
    col_sum = pi_projected'*ones(size(pi_projected,1),1);
    pi_projected = bsxfun(@times,pi_projected,(beta(:)./max(col_sum,realmin))');
    err = max([norm(pi_projected*ones(size(pi_projected,2),1)-alpha(:),inf), ...
        norm(pi_projected'*ones(size(pi_projected,1),1)-beta(:),inf)]);
    if err <= tol
        break
    end
end

info.iter = iter;
info.feas_inf = err;
end

function r = ratio_if_meaningful(num,den,tol)
if isfinite(num) && isfinite(den) && den > max(tol,0)
    r = num/den;
else
    r = NaN;
end
end
