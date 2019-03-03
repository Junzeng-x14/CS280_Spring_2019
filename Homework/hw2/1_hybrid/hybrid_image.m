function iml2=hybrid_image(im1, im2, cutoff_low, cutoff_high)
% for the low-pass filter, cutoff_low chosen to be 6
low_freq_image = imgaussfilt(im1,cutoff_low);

% for the high-pass filter, cutoff_high chosen to be 16
high_freq_image = im2-imgaussfilt(im2,cutoff_high);

% combine the two images together
iml2=low_freq_image+high_freq_image;
end