function [Q,meta] = build_gw_cost_matrix(C,D,loss,lossfun)
%BUILD_GW_COST_MATRIX Build the flattened GW quadratic cost matrix.

if nargin < 3 || isempty(loss)
    loss = 'square';
end
if nargin < 4
    lossfun = [];
end
if ~isnumeric(C) || ndims(C) ~= 2 || size(C,1) ~= size(C,2)
    error('build_gw_cost_matrix:InvalidC', 'C must be a square numeric matrix.');
end
if ~isnumeric(D) || ndims(D) ~= 2 || size(D,1) ~= size(D,2)
    error('build_gw_cost_matrix:InvalidD', 'D must be a square numeric matrix.');
end
if any(~isfinite(C(:))) || any(~isfinite(D(:)))
    error('build_gw_cost_matrix:NonfiniteCost', 'C and D must contain finite values.');
end

m = size(C,1);
n = size(D,1);
mn = m*n;
switch lower(char(loss))
    case 'square'
        L = (kron(ones(n,n), C) - kron(D, ones(m,m))).^2;
        cost_build_method = 'kron_square_loss';
    case 'custom'
        L = zeros(mn,mn);
        for j = 1:n
            for i = 1:m
                s = i + (j-1)*m;
                for l = 1:n
                    for k = 1:m
                        t = k + (l-1)*m;
                        L(s,t) = evaluate_loss(C(i,k),D(j,l),loss,lossfun);
                    end
                end
            end
        end
        cost_build_method = 'loop_custom_loss';
    otherwise
        error('build_gw_cost_matrix:UnknownLoss', 'Unsupported loss: %s.', char(loss));
    end

Q = 0.5*(L+L');
meta.L = L;
meta.loss = loss;
meta.m = m;
meta.n = n;
meta.idx = @(i,j) i + (j-1)*m;
meta.cost_build_method = cost_build_method;
end

function v = evaluate_loss(a,b,loss,lossfun)
switch lower(char(loss))
    case 'square'
        v = (a-b)^2;
    case 'custom'
        if isempty(lossfun) || ~isa(lossfun, 'function_handle')
            error('build_gw_cost_matrix:MissingLossfun', ...
                'lossfun must be a function handle when loss is ''custom''.');
        end
        v = lossfun(a,b);
        if ~isscalar(v) || ~isnumeric(v) || ~isfinite(v)
            error('build_gw_cost_matrix:InvalidLossValue', ...
                'lossfun must return a finite numeric scalar.');
        end
    otherwise
        error('build_gw_cost_matrix:UnknownLoss', 'Unsupported loss: %s.', char(loss));
end
end
