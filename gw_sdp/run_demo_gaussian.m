%% Small Gaussian point-cloud GW-SDP demo through RiNNAL+.
clear;
clc;

gwroot = fileparts(mfilename('fullpath'));
addpath(genpath(gwroot));
if exist('setup_path', 'file') == 2
    setup_path;
end

% rng(1);
m = 20;
n = 20;
X = randn(m,2);
Y = 0.75*randn(n,3) + 0.25;
C = pairwise_euclidean_distances(X);
D = pairwise_euclidean_distances(Y);
alpha = ones(m,1)/m;
beta = ones(n,1)/n;

opts = struct();
opts.rank = 8;
opts.tol = 1e-5;
opts.par = struct();
opts.par.tol = opts.tol;
opts.par.verbose = 1;
opts.par.maxtime = 1500;
% opts.par.BBmaxiter = 100;

sol = gw_sdp_rinnal(C,D,alpha,beta,opts);

fprintf('\nRecovered pi:\n');
disp(sol.pi);
print_gw_summary(sol);

function print_gw_summary(sol)
fprintf('row_feas_inf        = %.3e\n', sol.diagnostics.row_feas_inf);
fprintf('col_feas_inf        = %.3e\n', sol.diagnostics.col_feas_inf);
fprintf('min_pi              = %.3e\n', sol.diagnostics.min_pi);
fprintf('min_P               = %.3e\n', sol.diagnostics.min_P);
fprintf('sdp_value <Q,P>     = %.12e\n', sol.sdp_value);
fprintf('gw_value x''Qx       = %.12e\n', sol.gw_value);
fprintf('approx_ratio_upper  = %.12e\n', sol.approx_ratio_upper);
fprintf('rank_lift           = %d\n', sol.rank_lift);
fprintf('rank_gap            = %.3e\n', sol.rank_gap);
fprintf('mar_residual        = %.3e\n', sol.diagnostics.mar_residual);
fprintf('max_residual        = %.3e\n', max([sol.info.pfeas, sol.info.dfeas, sol.info.comp]));
end

function C = pairwise_euclidean_distances(Z)
sqnorm = sum(Z.^2,2);
C2 = bsxfun(@plus,sqnorm,sqnorm') - 2*(Z*Z');
C = sqrt(max(C2,0));
end
