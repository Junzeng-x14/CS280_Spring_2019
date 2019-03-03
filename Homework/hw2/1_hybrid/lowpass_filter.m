function result_image1=lowpass_filter(image1, sigma1)
result_image1 = imgaussfilt(image1, sigma1);
end