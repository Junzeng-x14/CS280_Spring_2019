clc;
close all;
%% Compute the orientation map
I = imread('image.jpg');
[mag,theta] = difference_filter(I);
h = figure;
imshow(mag);
saveas(h,'edges.png');

%% Original histogram
h = figure;
hist(theta(theta > 0.001),100);
saveas(h,'histogram_original.png');

%% Rotate image to vertical
J1 = imrotate(I,(pi-1.6)/pi*180,'bilinear','crop');
h = figure;
imshow(J1);
saveas(h,'image_vertical.png');
[mag1,theta1] = difference_filter(J1);
h = figure;
histogram(theta1(theta1 > 0.001),100);
saveas(h,'histogram_vertical.png');

%% Rotate image to horizontal
J2 = imrotate(I,(pi-1.6)/pi*180+90,'bilinear','crop');
h = figure;
imshow(J2);
saveas(h,'image_horizontal.png');
[mag2,theta2] = difference_filter(J2);
h = figure;
histogram(theta2(theta2 > 0.001),100);
saveas(h,'histogram_horizontal.png');