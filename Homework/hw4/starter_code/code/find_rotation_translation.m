function [R, t] = find_rotation_translation(E)
[U, S, V] = svd (E) ;
Z = diag ( [ 1 , 1 , 0 ] ) ;
theta = pi/2;
% rotation along z axis with 90 degrees
Rplus90 = [cos(theta), -sin(theta), 0;...
	sin(theta), cos(theta), 0;...
	0, 0, 1];
theta = -pi/2;
Rminus90 = [cos(theta), -sin(theta), 0;...
	sin(theta), cos(theta), 0;...
	0, 0, 1];
% translation
t{1} = U(:,end) ;
t{2} = -U(:,end) ;

% rotation
r{1} = U*Rplus90*V';
r{2} = -U*Rplus90*V';
r{3} = U*Rminus90*V';
r{4} = -U*Rminus90*V';

% keep those with +1 determinant
index = 0;
for i = 1:4
	if (det(r{i})) > 0
		index = index +1;
		R{index} = r{i};
	end
end


end