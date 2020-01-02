# Some imports
from matplotlib import pyplot as plt
import cv2
# for display
from skimage import data  # for loading example data
from skimage.color import rgb2gray  # for converting to grayscale

# Load image, convert to grayscale and show it
image = rgb2gray(data.astronaut())


def draw_image_histogram(image, channels, color='k'):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
