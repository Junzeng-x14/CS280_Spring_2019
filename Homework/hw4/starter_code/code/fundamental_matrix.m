function [F, res_err] = fundamental_matrix(corespts)
% normalization
N = size(corespts,1);
x1 = corespts(:,1:2);
x2 = corespts(:,3:4);
T1 = normalizeMatrix(x1);
T2 = normalizeMatrix(x2);

% homogenous augmented points (x, y, 1)
X1 = [x1, ones(N,1)]';
X2 = [x2, ones(N,1)]';

% normalization
normalizedX1 = (T1*X1)';
normalizedX2 = (T2*X2)';

x1 = normalizedX1(:,1);
y1 = normalizedX1(:,2);
x2 = normalizedX2(:,1);
y2 = normalizedX2(:,2) ;
A = [x1.*x2, y1.*x2, x2, x1.*y2, y1.*y2, y2, x1, y1, ones(N,1)];

% first svd
[U1,S1,V1] = svd(A);
f_star = V1(:,end);
F_star = reshape(f_star, [3,3])';

% second svd
[U2,S2,V2] = svd(F_star) ;
S(3,3) = 0;
F=U2*S2*V2';

% denormalization
F = T2'*F*T1;

% residual computation
den = sqrt(sum((F'*X2).^2,1))';
d12 = abs(diag(X1'*F*X2))./den;
clear den;

den = sqrt(sum((F'*X1).^2,1))';
d21 = abs(diag(X2'*F*X1))./den;

res_err = sum(d12.^2+d21.^2)/(2*N);
end