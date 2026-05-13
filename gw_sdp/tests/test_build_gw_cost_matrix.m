function tests = test_build_gw_cost_matrix
tests = functiontests(localfunctions);
end

function testSquareLossColumnMajorTensor(testCase)
C = [0 2; 2 0];
D = [0 5; 5 0];
[Q,meta] = build_gw_cost_matrix(C,D);

verifyEqual(testCase,size(Q),[4,4]);
verifyEqual(testCase,Q,Q','AbsTol',1e-14);
verifyEqual(testCase,meta.cost_build_method,'kron_square_loss');

s = meta.idx(2,1);
t = meta.idx(1,2);
verifyEqual(testCase,s,2);
verifyEqual(testCase,t,3);
verifyEqual(testCase,Q(s,t),(C(2,1)-D(1,2))^2,'AbsTol',1e-14);
end

function testVectorizedLMatchesLoopDefinition(testCase)
m = 2;
n = 3;
C = [0 1.5; 2.5 0.25];
D = [0 2 4; 3 0 5; 6 7 0.5];

L_loop = zeros(m*n,m*n);
for j = 1:n
    for i = 1:m
        s = i + (j-1)*m;
        for l = 1:n
            for k = 1:m
                t = k + (l-1)*m;
                L_loop(s,t) = (C(i,k)-D(j,l))^2;
            end
        end
    end
end
L_kron = (kron(ones(n,n), C) - kron(D, ones(m,m))).^2;
Q_expected = 0.5*(L_kron + L_kron');
[Q,meta] = build_gw_cost_matrix(C,D,'square',[]);

verifyLessThanOrEqual(testCase,norm(L_loop - L_kron,'fro'),1e-12);
verifyLessThanOrEqual(testCase,norm(meta.L - L_loop,'fro'),1e-12);
verifyLessThanOrEqual(testCase,norm(Q - Q_expected,'fro'),1e-12);
end

function testCustomLoss(testCase)
C = [0 1; 1 0];
D = [0 3; 3 0];
lossfun = @(a,b) abs(a-b);
[Q,meta] = build_gw_cost_matrix(C,D,'custom',lossfun);

s = meta.idx(1,2);
t = meta.idx(2,1);
verifyEqual(testCase,Q(s,t),2,'AbsTol',1e-14);
verifyEqual(testCase,meta.cost_build_method,'loop_custom_loss');
end
