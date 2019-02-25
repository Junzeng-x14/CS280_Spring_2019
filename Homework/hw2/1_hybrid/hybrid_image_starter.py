import matplotlib.pyplot as plt
from align_image_code import align_images

# First load images

# high sf
im1 = plt.imread('./malik.png')/255.
im1 = im1[:,:,:3]

# low sf
im2 = plt.imread('./papadimitriou.png')/255
im2 = im2[:,:,:3]

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = arbitrary_value_1
sigma2 = arbitrary_value_2
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid)
plt.show()