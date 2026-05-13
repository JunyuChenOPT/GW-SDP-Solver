function [A,b,meta] = build_transport_constraints(m,n,alpha,beta)
%BUILD_TRANSPORT_CONSTRAINTS Build full-row-rank transport equalities.
%
%   A contains all m row marginals and the first n-1 column marginals.
%   meta.A_full contains all m+n marginal constraints for diagnostics.

validate_dimensions(m,n);
alpha = alpha(:);
beta = beta(:);
if length(alpha) ~= m
    error('build_transport_constraints:AlphaDimension', 'length(alpha) must equal m.');
end
if length(beta) ~= n
    error('build_transport_constraints:BetaDimension', 'length(beta) must equal n.');
end

Arow = kron(sparse(ones(1,n)), speye(m));
Acol = kron(speye(n), sparse(ones(1,m)));

A_full = [Arow; Acol];
b_full = [alpha; beta];

A = [Arow; Acol(1:n-1,:)];
b = [alpha; beta(1:n-1)];

meta.A_full = A_full;
meta.b_full = b_full;
meta.Arow = Arow;
meta.Acol = Acol;
meta.A_solver = A;
meta.b_solver = b;
meta.row_indicators = Arow';
meta.col_indicators = Acol';
meta.row_constraint_ids = 1:m;
meta.included_col_ids = 1:n-1;
meta.omitted_col_ids = n;
meta.A_form = 'kron_reduced_transport';
meta.idx = @(i,j) i + (j-1)*m;
end

function validate_dimensions(m,n)
if ~isscalar(m) || ~isscalar(n) || m ~= round(m) || n ~= round(n) || m < 1 || n < 1
    error('build_transport_constraints:InvalidDimensions', ...
        'm and n must be positive integer scalars.');
end
end
