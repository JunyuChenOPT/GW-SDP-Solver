function tests = test_build_transport_constraints
tests = functiontests(localfunctions);
end

function testFullRowRankAndShape(testCase)
m = 3;
n = 4;
alpha = [0.2; 0.3; 0.5];
beta = [0.1; 0.2; 0.3; 0.4];
[A,b,meta] = build_transport_constraints(m,n,alpha,beta);

verifyEqual(testCase,size(A),[m+n-1, m*n]);
verifyEqual(testCase,size(b),[m+n-1, 1]);
verifyEqual(testCase,rank(full(A)),m+n-1);
verifyEqual(testCase,size(meta.A_full),[m+n, m*n]);
verifyEqual(testCase,rank(full(meta.A_full)),m+n-1);
verifyEqual(testCase,meta.A_form,'kron_reduced_transport');
verifyEqual(testCase,meta.omitted_col_ids,n);
end

function testKroneckerMarginalIdentities(testCase)
sizes = [2 3; 3 2; 1 4; 4 1];
for row = 1:size(sizes,1)
    m = sizes(row,1);
    n = sizes(row,2);
    pi = reshape(1:m*n,m,n)/(m*n);
    x = pi(:);

    Arow = kron(sparse(ones(1,n)), speye(m));
    Acol = kron(speye(n), sparse(ones(1,m)));
    [~,~,meta] = build_transport_constraints(m,n,ones(m,1)/m,ones(n,1)/n);

    verifyLessThanOrEqual(testCase,norm(Arow*x - pi*ones(n,1),inf),1e-12);
    verifyLessThanOrEqual(testCase,norm(Acol*x - pi'*ones(m,1),inf),1e-12);
    verifyEqual(testCase,meta.Arow,Arow);
    verifyEqual(testCase,meta.Acol,Acol);
    verifyEqual(testCase,meta.row_indicators,Arow');
    verifyEqual(testCase,meta.col_indicators,Acol');
end
end

function testReducedAndFullTransportConstraints(testCase)
sizes = [2 3; 3 2; 1 4; 4 1];
for row = 1:size(sizes,1)
    m = sizes(row,1);
    n = sizes(row,2);
    alpha = probability_vector(m);
    beta = probability_vector(n);
    [A,b,meta] = build_transport_constraints(m,n,alpha',beta');
    pi0 = alpha*beta';
    x0 = pi0(:);

    verifyLessThanOrEqual(testCase,norm(meta.A_full*x0 - [alpha; beta],inf),1e-12);
    verifyLessThanOrEqual(testCase,norm(A*x0 - b,inf),1e-12);
    verifyEqual(testCase,rank(full(A)),m+n-1);
    verifyEqual(testCase,rank(full(meta.A_full)),m+n-1);
    verifyEqual(testCase,size(meta.A_full,1),m+n);
end
end

function testRltMarginalIdentity(testCase)
sizes = [2 3; 3 2; 1 4; 4 1];
for row = 1:size(sizes,1)
    m = sizes(row,1);
    n = sizes(row,2);
    alpha = probability_vector(m);
    beta = probability_vector(n);
    [A,b,meta] = build_transport_constraints(m,n,alpha,beta);
    pi0 = alpha*beta';
    x0 = pi0(:);
    P0 = x0*x0';

    verifyLessThanOrEqual(testCase,norm(A*P0 - b*x0','fro'),1e-12);
    verifyLessThanOrEqual(testCase,norm(meta.A_full*P0 - [alpha; beta]*x0','fro'),1e-12);
end
end

function testProductCouplingSatisfiesReducedAndFullConstraints(testCase)
m = 3;
n = 4;
alpha = [0.2; 0.3; 0.5];
beta = [0.1; 0.2; 0.3; 0.4];
[A,b,meta] = build_transport_constraints(m,n,alpha,beta);
pi = alpha*beta';
x = pi(:);

verifyLessThanOrEqual(testCase,norm(A*x-b,inf),1e-14);
verifyLessThanOrEqual(testCase,norm(meta.A_full*x-meta.b_full,inf),1e-14);
end

function testColumnMajorIndexing(testCase)
m = 3;
n = 4;
alpha = ones(m,1)/m;
beta = ones(n,1)/n;
[~,~,meta] = build_transport_constraints(m,n,alpha,beta);

verifyEqual(testCase,meta.idx(1,1),1);
verifyEqual(testCase,meta.idx(3,1),3);
verifyEqual(testCase,meta.idx(1,2),4);
verifyEqual(testCase,meta.idx(2,4),11);
end

function v = probability_vector(n)
v = (1:n)';
v = v/sum(v);
end
