%% Tiny Gaussian point-cloud GW-SDP demo through RiNNAL+.
% This instance is small enough to solve quickly and is expected to return a
% nearly rank-one SDP solution, so gw_value / sdp_value should be close to 1.
clear;
clc;

gwroot = fileparts(mfilename('fullpath'));
addpath(genpath(gwroot));
if exist('setup_path', 'file') == 2
    setup_path;
end

rng(2);
m = 3;
n = 3;
X = randn(m,2);
Y = 0.75*randn(n,3) + 0.25;
C = pairwise_euclidean_distances(X);
D = pairwise_euclidean_distances(Y);
alpha = ones(m,1)/m;
beta = ones(n,1)/n;

opts = struct();
opts.rank = 10;
opts.tol = 1e-7;
opts.par = struct();
opts.par.tol = opts.tol;
opts.par.verbose = 1;
opts.par.maxtime = 180;
opts.par.BBmaxiter = 200;
opts.par.PGmaxiter = 5;
opts.par.SSN_frequence = 1;
opts.par.useSSN = 1;

t_solve = tic;
sol = gw_sdp_rinnal(C,D,alpha,beta,opts);
solve_time = toc(t_solve);

fprintf('\n2D source Gaussian points X:\n');
disp(X);
fprintf('3D target Gaussian points Y:\n');
disp(Y);
fprintf('Recovered pi:\n');
disp(sol.pi);
print_gw_summary(sol, solve_time);

if abs(sol.approx_ratio_upper - 1) > 1e-5 || sol.rank_gap > 1e-5
    error('run_demo_tiny_gaussian:NonTightSolution', ...
        'Expected a nearly rank-one tight solution, but ratio or rank_gap is too large.');
end

function print_gw_summary(sol, solve_time)
max_residual = max([sol.info.pfeas, sol.info.dfeas, sol.info.comp]);
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
fprintf('max_residual        = %.3e\n', max_residual);
fprintf('solve_time_sec      = %.3f\n', solve_time);
if isfield(sol.meta,'timing')
    fprintf('build_A_sec         = %.3e\n', sol.meta.timing.build_A);
    fprintf('build_Q_sec         = %.3e\n', sol.meta.timing.build_Q);
    fprintf('model_build_sec     = %.3e\n', sol.meta.timing.total_model_build);
end
end

function C = pairwise_euclidean_distances(Z)
sqnorm = sum(Z.^2,2);
C2 = bsxfun(@plus,sqnorm,sqnorm') - 2*(Z*Z');
C = sqrt(max(C2,0));
end
