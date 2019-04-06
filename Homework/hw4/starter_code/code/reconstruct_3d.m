function [points, P1, P2] = reconstruct_3d(name)
% Homework 2: 3D reconstruction from two Views
% This function takes as input the name of the image pairs (i.e. 'house' or
% 'library') and returns the 3D points as well as the camera matrices...but
% some functions are missing.

% NOTES
% (1) The code has been written so that it can be easily understood. It has 
% not been written for efficiency.
% (2) Don't make changes to this main function since I will run my
% reconstruct_3d.m and not yours. I only want from you the missing
% functions and they should be able to run without crashing with my
% reconstruct_3d.m
% (3) Keep the names of the missing functions as they are defined here,
% otherwise things will crash

close all;
clc;

%% ------- Load images, K matrices and matches ----------------------------
data_dir = ['../data/' name];

% images
I1 = imread([data_dir '/' name '1.jpg']);
I2 = imread([data_dir '/' name '2.jpg']);

% K matrices
qq = load([data_dir '/' name '1_K.mat']); K1 = qq.K;clear qq;
qq = load([data_dir '/' name '2_K.mat']); K2 = qq.K;clear qq;

% corresponding points
matches = load([data_dir '/' name '_matches.txt']); 
% this is a N x 4 where:
% matches(i,1:2) is a point in the first image
% matches(i,3:4) is the corresponding point in the second image


% visualize matches (disable or enable this whenever you want)
if true
    figure;
    imshow([I1 I2]); hold on;
    plot(matches(:,1), matches(:,2), '+r');
    plot(matches(:,3)+size(I1,2), matches(:,4), '+r');
    line([matches(:,1) matches(:,3) + size(I1,2)]', matches(:,[2 4])', 'Color', 'r');
end
% -------------------------------------------------------------------------
%% --------- Find fundamental matrix --------------------------------------

% F        : the 3x3 fundamental matrix,
% res_err  : mean squared distance between points in the two images and their
% their corresponding epipolar lines
[F, res_err] = fundamental_matrix(matches); % <------------------------------------- You write this one!

fprintf('Residual in F = %f',res_err);

E = K2'*F*K1; % the essential matrix

% -------------------------------------------------------------------------
%% ---------- Rotation and translation of camera 2 ------------------------

% R : cell array with the possible rotation matrices of second camera
% t : cell array of the possible translation vectors of second camera
[R, t] = find_rotation_translation(E);% <------------------------------------- You write this one!


% Find R2 and t2 from R,t such that largest number of points lie in front
% of the image planes of the two cameras

P1 = K1*[eye(3) zeros(3,1)];

% the number of points in front of the image planes for all combinations
num_points = zeros(length(t),length(R)); 

% the reconstruction error for all combinations
errs = inf(length(t),length(R));

for ti = 1:length(t)
    t2 = t{ti};
    for ri = 1:length(R)
        R2 = R{ri};
        
        P2 = K2*[R2 t2];
        
        [points_3d, errs(ti,ri)] = find_3d_points(matches, P1, P2); %<---------------------- You write this one!
        
        Z1 = points_3d(:,3);
        Z2 = R2(3,:)*points_3d'+t2(3);Z2 = Z2';
        num_points(ti,ri) = sum(Z1>0 & Z2>0);
        
                
    end
end

[ti ri] = find(num_points == max(max(num_points)));

j = 1; % pick one out the best combinations

fprintf('Reconstruction error = %f',errs(ti(j),ri(j)));

t2 = t{ti(j)}; R2 = R{ri(j)};
P2 = K2*[R2 t2];

% compute the 3D points with the final P2
points = find_3d_points(matches, P1, P2); % <---------------------------------------------- You have already written this one!

%% -------- plot points and centers of cameras ----------------------------


plot_3d(points , R2 , t2); % <-------------------------------------------------------------- You write this one!



end
