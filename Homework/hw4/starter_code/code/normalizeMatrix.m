function T = normalizeMatrix(pts)
% use 2d points to calculate the normalized matrix
aver = mean(pts)';
new_p(:,1) = pts(:,1)-aver(1);
new_p(:,2) = pts(:,2)-aver(2);
dist = sqrt(new_p(:,1).^2+new_p(:,2).^2);
aver_dist = mean(dist);
scale = sqrt(2)/aver_dist;
% construct matrix T
T = [scale, 0, -scale*aver(1);...
	 0, scale, -scale*aver(2);...
	 0, 0, 1];
end