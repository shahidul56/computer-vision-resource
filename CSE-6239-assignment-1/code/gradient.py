import numpy as np
from numpy import arctan2, fliplr, flipud
import matplotlib.pyplot as plt
import cv2


def gradient(image, same_size=False):
    """ Computes the Gradients of the image separated pixel difference

    Gradient of X is computed using the filter 
        [-1, 0, 1]
    Gradient of X is computed using the filter 
        [[1,
          0,
          -1]]
    Parameters
    ----------
    image: image of shape (imy, imx)
    same_size: boolean, optional, default is True
        If True, boundaries are duplicated so that the gradients
        has the same size as the original image.
        Otherwise, the gradients will have shape (imy-2, imx-2)

    Returns
    -------
    (Gradient X, Gradient Y), two numpy array with the same shape as image
        (if same_size=True)
    """
    sy, sx = image.shape
    if same_size:
        gx = np.zeros(image.shape)
        gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
        gx[:, 0] = -image[:, 0] + image[:, 1]
        gx[:, -1] = -image[:, -2] + image[:, -1]

        gy = np.zeros(image.shape)
        gy[1:-1, :] = image[:-2, :] - image[2:, :]
        gy[0, :] = image[0, :] - image[1, :]
        gy[-1, :] = image[-2, :] - image[-1, :]

    else:
        gx = np.zeros((sy-2, sx-2))
        gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

        gy = np.zeros((sy-2, sx-2))
        gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]

    return gx, gy


def magnitude_orientation(gx, gy):
    """ Computes the magnitude and orientation matrices from the gradients gx gy
    Parameters
    ----------
    gx: gradient following the x axis of the image
    gy: gradient following the y axis of the image

    Returns 
    -------
    (magnitude, orientation)


    """

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 360

    return magnitude, orientation


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread("Q_3.jpg"), cv2.COLOR_BGR2GRAY)

    gx, gy = gradient(image)
    # print('==================value gx========')
    # # print(gx)
    # plt.hist(gx)
    # plt.show()
    # print('==================value gy========')
    # # print(gy)
    # plt.title('')
    # plt.hist(gy)
    # plt.show()

    magnitude, orientation = magnitude_orientation(gx, gy)
    print('==================magnitude========')
    print(magnitude)
    plt.title('Magnitude')
    plt.hist(magnitude)
    #plt.imsave('magnitude.png', magnitude, format='png')
    plt.show()
    print('==================orienftation========')
    print(orientation)
    plt.title('orientation')
    plt.hist(orientation)
    #plt.imsave('orientation.png', orientation, format='png')
    plt.show()
