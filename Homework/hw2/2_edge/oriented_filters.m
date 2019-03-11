function [mag,theta] = oriented_filter(I)
% compute magnitude and theta of image I

sigma=2;
[row,col]=size(I(:,:,1));
result_x=zeros(row,col); % the gradient on x axis R,G,B channels combined)
result_y=zeros(row,col); % the gradient on y axis R,G,B channels combined)
result_1=zeros(row,col); % the gradient at 45 degree
result_2=zeros(row,col); % the gradient at 135 degree
mag=zeros(row,col);
theta=zeros(row,col); %in radius

% define the filters
Dx=[1,-1];
Dy=[1,-1]';
D1=[0,1;-1,0];
D2=[1,0;0,-1];

% first convolve with finite difference filter then convolve with the gaussian filter
diff_x=imgaussfilt(convn(I,Dx,'same'),sigma);
diff_y=imgaussfilt(convn(I,Dy,'same'),sigma);
diff_1=imgaussfilt(convn(I,D1,'same'),sigma);
diff_2=imgaussfilt(convn(I,D2,'same'),sigma);

% compute the L2-norm over the R,G&B channels
for i=1:row
	for j=1:col
		result_x(i,j)=sqrt(diff_x(i,j,1)^2+diff_x(i,j,2)^2+diff_x(i,j,3)^2);
		result_y(i,j)=sqrt(diff_y(i,j,1)^2+diff_y(i,j,2)^2+diff_y(i,j,3)^2);
		result_1(i,j)=sqrt(diff_1(i,j,1)^2+diff_1(i,j,2)^2+diff_1(i,j,3)^2);
		result_2(i,j)=sqrt(diff_2(i,j,1)^2+diff_2(i,j,2)^2+diff_2(i,j,3)^2);
	end
end

% Compute the magnitude and orientation after filtering the image with
% multiple oriented filters, the largest value is considered the magnitude
% at that point, and the orientation of the corresponding filter is
% considered the orientation at that point
threshold=12; % threshold for mag
for i=1:row
	for j=1:col
		value=[result_x(i,j),result_y(i,j),result_1(i,j),result_2(i,j)];
		degree=[0,atan(pi/2),atan(pi/4),atan(3*pi/4)];
		[mag(i,j),index]=max(value);
		theta(i,j)=degree(index);
		if mag(i,j)<threshold
			theta(i,j)=0;
			mag(i,j)=0;
		end
	end
enssd