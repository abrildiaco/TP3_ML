import numpy as np


def dataset_summary(X, y, max_sample = 10000, random_state = 42):
    """
    Prints an exploratory summary of the image dataset.

    Pixel statistics such as mean and standard deviation are computed on a
    random sample to avoid memory issues with large datasets.

    Arguments:
        X (np.ndarray): Array containing the images.
        y (np.ndarray): Array containing the labels.
        max_sample (int): Maximum number of images used to estimate pixel statistics.
        random_state (int): Seed used to make the random sample reproducible.

    Returns:
        classes (np.ndarray): Classes found in the dataset.
        counts (np.ndarray): Number of images corresponding to each class.
    """
    classes, counts = np.unique(y, return_counts=True)

    n_images = X.shape[0]
    n_sample = min(max_sample, n_images)

    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n_images, size=n_sample, replace=False)
    X_sample = X[sample_indices].astype(np.float32)

    print("Dataset summary")
    print("-" * 40)
    print(f"Number of images: {n_images}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    print()

    print("Pixel values")
    print("-" * 40)
    print(f"Minimum: {X.min()}")
    print(f"Maximum: {X.max()}")
    print(f"Approximate mean: {X_sample.mean():.4f}")
    print(f"Approximate standard deviation: {X_sample.std():.4f}")
    print(f"Statistics computed on {n_sample} images")
    print()

    print("Classes")
    print("-" * 40)
    print(f"Number of classes: {len(classes)}")
    print(f"Classes found: {classes}")
    print()

    print("Dataset balance")
    print("-" * 40)
    print(f"Smallest class: Class {classes[counts.argmin()]} ({counts.min()} images)")
    print(f"Largest class: Class {classes[counts.argmax()]} ({counts.max()} images)")
    print(f"Max/min ratio: {counts.max() / counts.min():.2f}")

    return classes, counts
