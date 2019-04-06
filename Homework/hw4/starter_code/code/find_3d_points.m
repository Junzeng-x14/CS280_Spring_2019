function [points_3d, errs] = find_3d_points(corespts, P1, P2)
N = size(corespts,1) ;
X1 = corespts(:,1:2);
X2 = corespts(:,3:4);
points_3d = zeros(N,3);
dist = [];
for k = 1:N
	x1 = X1(k,1);
	y1 = X1(k,2);
	x2 = X2(k,1);
	y2 = X2(k,2);
	
	A = [x1*P1(3,1)-P1(1,1), x1*P1(3,2)-P1(1,2), x1*P1(3,3)-P1(1,3);...
		 y1*P1(3,1)-P1(2,1), y1*P1(3,2)-P1(2,2), y1*P1(3,3)-P1(2,3);...
		 x2*P2(3,1)-P2(1,1), x2*P2(3,2)-P2(1,2), x2*P2(3,3)-P2(1,3);...
		 y2*P2(3,1)-P2(2,1), y2*P2(3,2)-P2(2,2), y2*P2(3,3)-P2(2,3)];
	 b = [x1*P1(3,4)-P1(1,4);y1*P1(3,4)-P1(2,4);x2*P2(3,4)-P2(1,4);y2*P2(3,4)-P2(2,4)];
	 X = A\(-b);
	 points_3d(k,:) = X;
end
proj1 = P1*[ points_3d, ones(N,1)]';
proj1 = proj1./proj1(3,:);
proj1 = proj1';

proj2 = P2*[ points_3d, ones(N,1)]';
proj2 = proj2./proj2(3,:);
proj2 = proj2';

dist1 = sqrt((X1(:,1)-proj1(:,1)).^2+(X1(:,2)-proj1(:,2)).^2);
dist2 = sqrt((X1(:,1)-proj2(:,1)).^2+(X1(:,2)-proj2(:,2)).^2);

errs = mean([dist1;dist2]);
end