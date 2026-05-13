import numpy as np
import matplotlib.pyplot as plt


def plot_class_distribution(classes, counts, xtick_step = 2):
    """
    Plots the number of images available for each class.

    Arguments:
        classes (np.ndarray): Classes found in the dataset.
        counts (np.ndarray): Number of images corresponding to each class.
        xtick_step (int): Frequency used to display x-axis tick labels.

    Returns:
        None
    """
    plt.figure(figsize = (9, 4))

    plt.bar(
        classes,
        counts,
        color="#082450",
        edgecolor="black",
        alpha=0.85
    )

    plt.title("Image distribution by class", fontsize = 14)
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.xticks(classes[::xtick_step])
    plt.tight_layout()
    plt.show()


def plot_images(X, y, indices = [0, 1, 2]):
    """
    Displays selected dataset images together with their labels.

    Arguments:
        X (np.ndarray): Array containing the images.
        y (np.ndarray): Array containing the labels.
        indices (list): Indices of the images to display.

    Returns:
        None
    """
    plt.figure(figsize = (12, 4))

    for i, idx in enumerate(indices):
        img = X[idx].reshape(28, 28)

        plt.subplot(1, len(indices), i + 1)
        plt.imshow(img, cmap = "gray")
        plt.title(f"Index: {idx}\nClass: {y[idx]}", fontsize=11)
        plt.axis("off")

    plt.suptitle("Dataset image examples\n", fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_loss_history(history):
    """
    Plots the evolution of the cross-entropy loss during training.

    Arguments:
        history (dict): Dictionary containing training and validation loss history.

    Returns:
        None
    """
    plt.figure(figsize = (8, 4))

    plt.plot(history["train_loss"], label = "Training loss", color = "#082450")

    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.plot(history["val_loss"], label = "Validation loss", color = "#AA1D22")

    plt.title("Cross-entropy loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrix, title = "Confusion matrix", normalize = False):
    """
    Plots a confusion matrix.

    Arguments:
        matrix (np.ndarray): Confusion matrix.
        title (str): Plot title.
        normalize (bool): Whether to normalize values by true class counts.

    Returns:
        None
    """
    matrix_to_plot = matrix.astype(float)

    plt.figure(figsize = (8, 7))
    plt.imshow(matrix_to_plot, cmap = "Blues", aspect = "auto")
    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
