%% Tiny GW-SDP demo through RiNNAL+.
clear;
clc;

gwroot = fileparts(mfilename('fullpath'));
addpath(genpath(gwroot));
if exist('setup_path', 'file') == 2
    setup_path;
end

C = [0 1; 1 0];
D = [0 2; 2 0];
alpha = [0.5; 0.5];
beta = [0.5; 0.5];

opts = struct();
opts.rank = 3;
opts.tol = 1e-6;
opts.par = struct();
opts.par.tol = opts.tol;
opts.par.verbose = 1;
opts.par.maxtime = 60;

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
end
