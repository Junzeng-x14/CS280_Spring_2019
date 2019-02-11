function [s,phi]= rot_to_ax_phi(R)
phi=acos(0.5*(trace(R)-1)); % angle of rotation
[V,D]=eig(R);
	for i=1:3
		if abs(D(i,i)-1)<1e-4
		s=V(:,i); % axis of rotation
		end
	end
end