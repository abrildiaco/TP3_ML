import numpy as np


def normalize_image(X):
    """
    Normalizes image pixel values by dividing them by 255.

    This scales pixel intensities from the original range [0, 255]
    to the range [0, 1].

    Arguments:
        X (np.ndarray): Array containing the images.

    Returns:
        X_normalized (np.ndarray): Array containing the normalized images.
    """
    return X.astype(np.float16) / 255.0
