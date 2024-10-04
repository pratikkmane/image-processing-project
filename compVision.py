#first importing the necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#now we will load the image Lena.png using its path and convert it to grayscale
def load_image(image_path):
    img = Image.open(image_path).convert('L')
    return np.array(img).astype(float) / 255.0

if __name__ == "__main__":
    image_path = 'Lena.png'
    original = load_image(image_path)

#Question 1:

#downsampling the image by half in both dimensions
def downsample(img):
    return img[::2, ::2]
#upsampling the image by doubling its size and inserting empty pixels.
def upsample(img):
    h, w = img.shape
    upsampled = np.zeros((h*2, w*2))
    upsampled[::2, ::2] = img
    return upsampled

#displaying multiple images in a single figure
def display_images(images, titles):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#downsampling the image twice
downsampled_once = downsample(original)
downsampled_twice = downsample(downsampled_once)

display_images([original, downsampled_once, downsampled_twice],
               ['Original', 'Downsampled Once', 'Downsampled Twice'])

#upsampling twice on the downsampled image
upsampled_once = upsample(downsampled_twice)
upsampled_twice = upsample(upsampled_once)

display_images([original, downsampled_twice, upsampled_once, upsampled_twice],
               ['Original', 'Downsampled Twice', 'Upsampled Once', 'Upsampled Twice'])
#---------------------------------------------------------------------------------------------------------------------
#Question 2:

#creating a 2D Gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2*np.pi*sigma**2)) *
                     np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

#applying Gaussian smoothing to the image."
def myGaussianSmoothing(I, k, s):
    #creating the gaussian kernel
    kernel = gaussian_kernel(k, s)

    #padding the image
    pad = k // 2
    padded_img = np.pad(I, ((pad, pad), (pad, pad)), mode='edge')

    #applying convolution
    smoothed = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            smoothed[i, j] = np.sum(padded_img[i:i+k, j:j+k] * kernel)

    return smoothed

#testing with different kernel sizes
kernel_sizes = [3, 5, 7, 11, 51]
sigma = 1
smoothed_images_k = [myGaussianSmoothing(original, k, sigma) for k in kernel_sizes]
display_images([original] + smoothed_images_k,
               ['Original'] + [f'k={k}, σ=1' for k in kernel_sizes])

#testing with different sigma values
sigmas = [0.1, 1, 2, 3, 5]
kernel_size = 11
smoothed_images_s = [myGaussianSmoothing(original, kernel_size, s) for s in sigmas]
display_images([original] + smoothed_images_s,
               ['Original'] + [f'k=11, σ={s}' for s in sigmas])
#---------------------------------------------------------------------------------------------------------------------
#Question 3:

#applying median filtering to the image
def myMedianFilter(I, k):
    pad = k // 2
    padded_img = np.pad(I, ((pad, pad), (pad, pad)), mode='edge')
    filtered = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            filtered[i, j] = np.median(padded_img[i:i+k, j:j+k])
    return filtered

#applying gaussian smoothing and median filtering after upsampling once
gaussian_smoothed_once = myGaussianSmoothing(upsampled_once, 11, 1)
median_filtered_once = myMedianFilter(upsampled_once, 11)

display_images([original, downsampled_twice, upsampled_once, gaussian_smoothed_once, median_filtered_once],
               ['Original', 'Downsampled Twice', 'Upsampled Once', 'Gaussian Smoothed', 'Median Filtered'])

#applying gaussian smoothing and median filtering after upsampling twice
gaussian_smoothed_twice = myGaussianSmoothing(upsampled_twice, 11, 1)
median_filtered_twice = myMedianFilter(upsampled_twice, 11)

display_images([original, downsampled_twice, upsampled_twice, gaussian_smoothed_twice, median_filtered_twice],
               ['Original', 'Downsampled Twice', 'Upsampled Twice', 'Gaussian Smoothed', 'Median Filtered'])
#---------------------------------------------------------------------------------------------------------------------
#Question 4:

#adding gaussian noise to the image.
def add_gaussian_noise(img, mean=0, std=0.1):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

#adding thresholded noise to the image.
def add_thresholded_noise(img, threshold=0.2):
    noise = np.random.normal(0, 0.1, img.shape)
    thresholded_noise = np.where(noise > threshold, 1, 0)
    noisy_img = img + thresholded_noise
    return np.clip(noisy_img, 0, 1)

#adding gaussian noise
noisy_gaussian = add_gaussian_noise(original)

#applying gaussian smoothing and median filtering to noisy image
gaussian_smoothed = myGaussianSmoothing(noisy_gaussian, 5, 1)
median_filtered = myMedianFilter(noisy_gaussian, 5)

display_images([original, noisy_gaussian, gaussian_smoothed, median_filtered],
               ['Original', 'Noisy (Gaussian)', 'Gaussian Smoothed', 'Median Filtered'])

#adding thresholded noise
noisy_thresholded = add_thresholded_noise(original)

#applying gaussian smoothing and median filtering to thresholded noisy image
gaussian_smoothed_thresh = myGaussianSmoothing(noisy_thresholded, 5, 1)
median_filtered_thresh = myMedianFilter(noisy_thresholded, 5)

display_images([original, noisy_thresholded, gaussian_smoothed_thresh, median_filtered_thresh],
               ['Original', 'Noisy (Thresholded)', 'Gaussian Smoothed', 'Median Filtered'])
#---------------------------------------------------------------------------------------------------------------------
#Question 5:

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def convolve2D(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    result = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            result[i,j] = np.sum(image[i:i+m, j:j+n] * kernel)
    return result

#adding sobel filter and defining sobel kernels
def mySobelFilter(I):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    #applying convolution
    Ix = convolve2D(I, Gx)
    Iy = convolve2D(I, Gy)

    #calculating magnitude and orientation
    mag = np.sqrt(Ix**2 + Iy**2)
    ori = np.arctan2(Iy, Ix)

    #normalizing magnitude to [0, 1]
    mag = normalize(mag)

    return mag, ori

def hsv_to_rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m)

#visualizing edge orientation using HSV color mapping
def visualize_orientation(ori):
    h = (ori + np.pi) / (2 * np.pi) * 360
    s = np.ones_like(ori) 
    v = np.ones_like(ori)  

    height, width = ori.shape
    rgb_image = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            rgb_image[i, j] = hsv_to_rgb(h[i, j], s[i, j], v[i, j])

    return rgb_image

def visualize_sobel(mag, ori):
    h = (ori + np.pi) / (2 * np.pi) * 360
    s = mag
    v = mag

    height, width = mag.shape
    rgb_image = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            rgb_image[i, j] = hsv_to_rgb(h[i, j], s[i, j], v[i, j])

    return rgb_image

#applying sobel filter
I = normalize(original)
mag, ori = mySobelFilter(I)

#visualizing orientation as an image
orientation_image = visualize_orientation(ori)

#time to visualize results
rgb_image = visualize_sobel(mag, ori)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(I, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(mag, cmap='gray')
axs[1].set_title('Sobel Magnitude')
axs[1].axis('off')

axs[2].imshow(orientation_image)
axs[2].set_title('Edge Orientation')
axs[2].axis('off')

axs[3].imshow(rgb_image)
axs[3].set_title('Sobel Visualization (HSV)')
axs[3].axis('off')

plt.tight_layout()
plt.show()

print(f"Magnitude shape: {mag.shape}")
print(f"Orientation shape: {ori.shape}")
print(f"Magnitude range: [{mag.min():.4f}, {mag.max():.4f}]")
print(f"Orientation range: [{ori.min():.4f}, {ori.max():.4f}]")