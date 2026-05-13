function [value,Q] = gw_objective(C,D,pi,loss,lossfun)
%GW_OBJECTIVE Compute x'Qx for a transport plan pi.

if nargin < 4 || isempty(loss)
    loss = 'square';
end
if nargin < 5
    lossfun = [];
end
[Q,~] = build_gw_cost_matrix(C,D,loss,lossfun);
x = pi(:);
if length(x) ~= size(Q,1)
    error('gw_objective:DimensionMismatch', ...
        'numel(pi) must equal size(C,1)*size(D,1).');
end
value = full(x'*Q*x);
end
