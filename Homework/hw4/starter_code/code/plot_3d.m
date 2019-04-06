function plot_3d(points, R, t)
figure
% first camera
plotCamera ('Location',[0,0,0],'Orientation',eye(3),'Opacity',0,'Size',0.1);
hold on

% second camera
C2 = -pinv(R)*t;
plotCamera('Location',C2','Orientation',R,'Opacity',0,'Size',0.1);
hold on

% plot 3D points
scatter3(points(:,1),points(:,2),points(:,3),'b+');
hold on

xlim ([min(points(:,1)), max(points(:,1))]);
ylim ([min(points(:,2)), max(points(:,2))]);
zlim ([-0.5, max(points(:,3))]);
xlabel('x');
ylabel('y');
zlabel('z');
grid minor
end