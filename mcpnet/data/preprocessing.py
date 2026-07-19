"""Preprocessing utilities —."""

import numpy as np


def standardize(image):
    """
    Standardize mean and standard deviation of each channel and z-dimension.

    Args:
        image (np.array): input image, shape (dim_x, dim_y, dim_z)

    Returns:
        standardized_image (np.array): standardized version of input image
    """
    standardized_image = np.zeros(image.shape)

    for c in range(image.shape[2]):
        image_slice = image[:, :, c]
        centered = image_slice - np.mean(image)
        if np.std(centered) != 0:
            centered_scaled = centered / np.std(centered)
            standardized_image[:, :, c] = centered_scaled
        else:
            standardized_image[:, :, c] = centered

    return standardized_image
