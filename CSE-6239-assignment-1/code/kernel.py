

import numpy as np
from PIL import Image, ImageDraw
input_image = Image.open("B1.jpg")
input_pixels = input_image.load()


# def gaussian_kernel(size=21, sigma=3):
"""Returns a 2D Gaussian kernel.
    Parameters
    ----------
    size : float, the kernel size (will be square)

    sigma : float, the sigma Gaussian parameter

    Returns
    -------
    out : array, shape = (size, size)
      an array with the centered gaussian kernel
    """
sigma = 1
#size = (2*sigma + 1)
size = 5

x = np.linspace(- (size // 2), size // 2)
x /= np.sqrt(2)*sigma
x2 = x**2
kernel = np.exp(- x2[:, None] - x2[None, :])
test = kernel / kernel.sum()
print(test)


kernel = test

# Middle of the kernel
offset = len(kernel) // 2

# Create output image
output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

# Compute convolution between intensity and kernels
for x in range(offset, input_image.width - offset):
    for y in range(offset, input_image.height - offset):
        acc = [0, 0, 0]
        for a in range(len(kernel)):
            for b in range(len(kernel)):
                xn = x + a - offset
                yn = y + b - offset
                pixel = input_pixels[xn, yn]
                acc[0] += pixel[0] * kernel[a][b]
                acc[1] += pixel[1] * kernel[a][b]
                acc[2] += pixel[2] * kernel[a][b]

        draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

output_image.save("output_gaussian.png")
