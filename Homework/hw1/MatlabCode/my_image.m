% Image read
A = imread('../images/my_image.jpg');
B = imread('../images/my_photo.jpg');
% u matrix
u = [1,1,1440,1440;1,1080,1080,1]
% v matrix
v1 = [330,348,454,436;262,330,380,312];
v2 = [230,242,290,278;854,895,897,860];
v3 = [200,212,260,248;1054,1095,1097,1060];
v4 = [380,392,440,428;1054,1095,1097,1060];
v5 = [300,312,360,348;1184,1225,1227,1190];

%% Affine method
H1 = affine_solve(u,v1)
H2 = affine_solve(u,v2)
H3 = affine_solve(u,v3)
H4 = affine_solve(u,v4)
H5 = affine_solve(u,v5)

for i = 1:1440
	for j = 1:1080
		point1 = round(H1*[i,j,1]');
		point2 = round(H2*[i,j,1]');
		point3 = round(H3*[i,j,1]');
		point4 = round(H4*[i,j,1]');
		point5 = round(H5*[i,j,1]');	
		A(point1(1),point1(2),:) = B(i,j,:);
		A(point2(1),point2(2),:) = B(i,j,:);
		A(point3(1),point3(2),:) = B(i,j,:);
		A(point4(1),point4(2),:) = B(i,j,:);
		A(point5(1),point5(2),:) = B(i,j,:);
	end
end

imshow(A);
saveas(gcf,'my_image_affine.jpg');

% Image read
A = imread('../images/my_image.jpg');
B = imread('../images/my_photo.jpg');

%% Homography method
H1 = homography_solve(u,v1)
H2 = homography_solve(u,v2)
for i = 1:1440
	for j = 1:1080
		point1 = (H1*[i,j,1]');
		pos1(1,1) = round(point1(1)/point1(3));
		pos1(1,2) = round(point1(2)/point1(3));
		
		point2 = (H2*[i,j,1]');
		pos2(1,1) = round(point2(1)/point2(3));
		pos2(1,2) = round(point2(2)/point2(3));
		
		point3 = (H3*[i,j,1]');
		pos3(1,1) = round(point3(1)/point3(3));
		pos3(1,2) = round(point3(2)/point3(3));
		
		point4 = (H4*[i,j,1]');
		pos4(1,1) = round(point4(1)/point4(3));
		pos4(1,2) = round(point4(2)/point4(3));
		
		point5 = (H5*[i,j,1]');
		pos5(1,1) = round(point5(1)/point5(3));
		pos5(1,2) = round(point5(2)/point5(3));
		
		A(pos1(1,1),pos1(1,2),:) = B(i,j,:);
		A(pos2(1,1),pos2(1,2),:) = B(i,j,:);
		A(pos3(1,1),pos3(1,2),:) = B(i,j,:);
		A(pos4(1,1),pos4(1,2),:) = B(i,j,:);
		A(pos5(1,1),pos5(1,2),:) = B(i,j,:);
		
	end
end

imshow(A);
saveas(gcf,'my_image_homography.jpg');