function tests = test_gw_sdp_tiny_rank1_feasible
tests = functiontests(localfunctions);
end

function testRankOneProductCouplingDiagnostics(testCase)
C = [0 1; 1 0];
D = [0 2; 2 0];
alpha = [0.4; 0.6];
beta = [0.25; 0.75];
opts = struct('tol',1e-8);
[Q,~,~,~,~,~,~,~,meta] = build_gw_mbqp(C,D,alpha,beta,opts);

pi = alpha*beta';
x = pi(:);
P = x*x';
d = gw_diagnostics(pi,P,Q,alpha,beta,meta,1e-8);

verifyLessThanOrEqual(testCase,d.row_feas_inf,1e-14);
verifyLessThanOrEqual(testCase,d.col_feas_inf,1e-14);
verifyLessThanOrEqual(testCase,d.full_Ax_inf,1e-14);
verifyLessThanOrEqual(testCase,d.mar_residual,1e-14);
verifyLessThanOrEqual(testCase,d.rank_gap,1e-14);
verifyEqual(testCase,d.rank_lift,1);
verifyTrue(testCase,meta.full_row_rank);
verifyFalse(testCase,meta.rank_check_requested);
verifyTrue(testCase,isfield(meta,'timing'));
verifyTrue(testCase,isfield(meta.timing,'build_Q'));
verifyTrue(testCase,isfield(meta.timing,'build_A'));
end

function testOptionalDenseRankCheck(testCase)
C = [0 1; 1 0];
D = [0 2; 2 0];
alpha = [0.4; 0.6];
beta = [0.25; 0.75];
opts = struct('tol',1e-8,'check_rank',true);
[~,~,A,~,~,~,~,~,meta] = build_gw_mbqp(C,D,alpha,beta,opts);

verifyTrue(testCase,meta.full_row_rank);
verifyTrue(testCase,meta.rank_check_requested);
verifyEqual(testCase,meta.rank_A,size(A,1));
end

function testTinySolverSmoke(testCase)
assumeTrue(testCase,exist('RiNNAL_plus','file') == 2, ...
    'RiNNAL_plus is not on the MATLAB path.');

C = [0 1; 1 0];
D = [0 2; 2 0];
alpha = [0.5; 0.5];
beta = [0.5; 0.5];
opts = struct();
opts.rank = 3;
opts.tol = 1e-5;
opts.par = struct('tol',opts.tol,'verbose',0,'maxtime',30);

sol = gw_sdp_rinnal(C,D,alpha,beta,opts);

verifyTrue(testCase,isfinite(sol.sdp_value));
verifyTrue(testCase,isfinite(sol.gw_value));
verifyTrue(testCase,isfield(sol.diagnostics,'row_feas_inf'));
verifyTrue(testCase,isfield(sol.diagnostics,'col_feas_inf'));
verifyTrue(testCase,isfield(sol.diagnostics,'rank_gap'));
verifyTrue(testCase,isfield(sol.diagnostics,'approx_ratio_upper'));
verifyEqual(testCase,size(sol.pi),[2,2]);
verifyEqual(testCase,size(sol.P),[4,4]);
end
