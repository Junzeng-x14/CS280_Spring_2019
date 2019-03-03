function result_image2=highpass_filter(image2, sigma2)
result_image2 = image2-imgaussfilt(image2, sigma2);
end