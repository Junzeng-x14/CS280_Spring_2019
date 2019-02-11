% chosen unit vector to rotate about
s=[0.3,-0.4,sqrt(1-0.3^2-0.4^2)]';

% corresponding skew symmetic matrix
S=[0,-s(3),s(2);s(3),0,-s(1);-s(2),s(1),0];

% chosen inital point with inital magnitude
A=[0.4,-0.4,sqrt(1-0.4^2-0.4^2)]';

% matrix of roation angles
theta=[0,pi/12,pi/8,pi/6,pi/4,pi/2,pi,1.5*pi];

% initilization of matrix for the points after rotation
B=zeros(3,8);

% initialization of roation matrix
R=zeros(3,3);

for i=1:8
	% roation matrix
	R=eye(3)+sin(theta(i))*S+(1-cos(theta(i)))*S*S;
	B(:,i)=R*A;
end

% plotting
plot3(A(1),A(2),A(3),'rx'); % plot initial point
hold on;
plot3(B(1,:),B(2,:),B(3,:),'go'); % plot points after rotation
hold on;

% plot the axis
P1 = [0,0,0];
P2 = [s(1),s(2),s(3)];
pts = [P1; P2];
line(pts(:,1), pts(:,2), pts(:,3))
xlabel('x');
ylabel('y');
zlabel('z');
legend('initial point','after rotation','axis');
grid on;