function [mag,theta] = difference_filter(I)
% compute magnitude and theta of image I

[row,col]=size(I(:,:,1));
result_x=zeros(row,col); % the gradient on x axis(R,G,B channels combined)
result_y=zeros(row,col); % the gradient on y axis(R,G,B channels combined)
mag=zeros(row,col);
theta=zeros(row,col); % in radius
% define the filters

Dx=[1,-1];
Dy=[1,-1]';
% convolve with the filters

diff_x=convn(I,Dx); %the derivative on x axis(R,G,B channels)
diff_y=convn(I,Dy); %the derivative on y axis(R,G,B channels)

% compute the L2-norm over the R,G,B channels
for i=1:row
	for j=1:col
		result_x(i,j)=sqrt(diff_x(i,j,1)^2+diff_x(i,j,2)^2+diff_x(i,j,3)^2);
		result_y(i,j)=sqrt(diff_y(i,j,1)^2+diff_y(i,j,2)^2+diff_y(i,j,3)^2);
	end
end

threshold=20;
%compute the magintude and orientation
for i=1:row
	for j=1:col
		mag(i,j)=sqrt(result_x(i,j)^2+result_y(i,j)^2);
		theta(i,j)=atan(result_y(i,j)/result_x(i,j));
		if mag(i,j)<threshold
			theta(i,j)=0;
			mag(i,j)=0;
		end
	end
end