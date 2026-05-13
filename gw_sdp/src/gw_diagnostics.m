function diagnostics = gw_diagnostics(pi,P,Q,alpha,beta,meta,tol)
%GW_DIAGNOSTICS Compute feasibility and relaxation diagnostics.

if nargin < 7 || isempty(tol)
    tol = 1e-6;
end
alpha = alpha(:);
beta = beta(:);
[m,n] = size(pi);
mn = m*n;
x = pi(:);

if size(P,1) ~= mn || size(P,2) ~= mn
    error('gw_diagnostics:PDimension', 'P must be numel(pi)-by-numel(pi).');
end
if size(Q,1) ~= mn || size(Q,2) ~= mn
    error('gw_diagnostics:QDimension', 'Q must be numel(pi)-by-numel(pi).');
end

row_res = pi*ones(n,1)-alpha;
col_res = pi'*ones(m,1)-beta;

diagnostics.row_feas_inf = norm(row_res,inf);
diagnostics.col_feas_inf = norm(col_res,inf);
diagnostics.full_marginal_inf = max(diagnostics.row_feas_inf,diagnostics.col_feas_inf);
diagnostics.min_pi = min(pi(:));
diagnostics.min_P = min(P(:));
diagnostics.sdp_value = full(sum(sum(Q.*P)));
diagnostics.gw_value = full(x'*Q*x);
diagnostics.approx_ratio_upper = ratio_if_meaningful( ...
    diagnostics.gw_value,diagnostics.sdp_value,tol);

Y = [1,x';x,P];
diagnostics.rank_lift = rank(full(Y),tol);
diagnostics.rank_gap = norm(P-x*x','fro')/(1+norm(P,'fro'));

if size(Y,1) <= 250
    diagnostics.psd_min_eig = min(eig(full(0.5*(Y+Y'))));
else
    diagnostics.psd_min_eig = NaN;
end

den = 1+norm(x);
row_mar = zeros(m,1);
for i = 1:m
    a = meta.row_indicators(:,i);
    row_mar(i) = norm(P*a-alpha(i)*x)/den;
end
col_mar = zeros(n,1);
for j = 1:n
    a = meta.col_indicators(:,j);
    col_mar(j) = norm(P*a-beta(j)*x)/den;
end

diagnostics.mar_row_residuals = row_mar;
diagnostics.mar_col_residuals = col_mar;
diagnostics.mar_row_residual = max(row_mar);
diagnostics.mar_col_residual = max(col_mar);
diagnostics.mar_residual = max([diagnostics.mar_row_residual, diagnostics.mar_col_residual]);

if isfield(meta, 'A_full') && isfield(meta, 'b_full')
    diagnostics.full_Ax_inf = norm(meta.A_full*x-meta.b_full,inf);
end
end

function r = ratio_if_meaningful(num,den,tol)
if isfinite(num) && isfinite(den) && den > max(tol,0)
    r = num/den;
else
    r = NaN;
end
end
