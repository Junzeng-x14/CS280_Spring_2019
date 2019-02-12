% Image read
A=imread('../images/the_flagellation.jpg');
B=imread('../images/blank.jpg');
% u matrix
u = [624,623,677,676;301,566,577,85]
% v matrix
v1 = [1,1,50,50;1,50,50,1]

%% Affine method
H1 = affine_solve(u,v1)
for i=620:680
	for j=82:581
		point1=round(H1*[i,j,1]');
		B(point1(1)+300,point1(2)+600,:)=A(i,j,:);
	end
end
imshow(B);

B=imread('../images/blank.jpg');

%% Homography method
H2 = homography_solve(u,v1)
for i = 600:700
	for j = 82:581
		point2=(H2*[i,j,1]');
		pos2(1,1)=round(point2(1)/point2(3));
		pos2(1,2)=round(point2(2)/point2(3));
		B(pos2(1,1)+500,pos2(1,2)+1000,:)=A(i,j,:);
	end
end
imshow(B);