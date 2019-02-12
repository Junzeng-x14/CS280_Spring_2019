% Image read
A = imread('../images/computer_screen.png');
B = imread('../images/blank.jpg');
% u matrix
u = [03,300,799,548;634,862,901,682]
% v matrix
v1 = [1,1,100,100;1,150,150,1]
%% Affine method
H1 = affine_solve(u,v1)
for i = 100:1000
	for j = s500:1000
		point1 = round(H1*[i,j,1]');
		B(point1(1)+300,point1(2)+600,:) = A(i,j,:);
	end
end
imshow(B);

B=imread('../images/blank.jpg');

%% Homography method
H2 = homography_solve(u,v1)
for i = 200:1000
	for j = 600:950
		point2 = (H2*[i,j,1]');
		pos2(1,1) = round(point2(1)/point2(3));
		pos2(1,2) = round(point2(2)/point2(3));
		B(pos2(1,1)+400,pos2(1,2)+1000,:) = sA(i,j,:);
	end
end
imshow(B);