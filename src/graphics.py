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


def plot_loss_history(history, model_name = "M1"):
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

    plt.title(f"{model_name} Cross-entropy loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(matrices, titles = None, normalize = False):
    """
    Plots one or more confusion matrices side by side.

    Arguments:
        matrices (np.ndarray or list): Confusion matrix or list of confusion matrices.
        titles (str or list): Title or list of titles for each confusion matrix.
        normalize (bool): Whether to normalize values by true class counts.

    Returns:
        None
    """
    if not isinstance(matrices, list):
        matrices = [matrices]

    if titles is None:
        titles = ["Confusion matrix"] * len(matrices)

    if isinstance(titles, str):
        titles = [titles]

    n_matrices = len(matrices)

    plt.figure(figsize = (7 * n_matrices, 6))

    for i, matrix in enumerate(matrices):
        matrix_to_plot = matrix.astype(float)

        if normalize:
            row_sums = matrix_to_plot.sum(axis = 1, keepdims = True)
            matrix_to_plot = matrix_to_plot / np.maximum(row_sums, 1)

        plt.subplot(1, n_matrices, i + 1)
        plt.imshow(matrix_to_plot, cmap = "Blues", aspect = "auto")
        plt.title(titles[i])
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def plot_m1_m2_comparison(results_dict, title = "M1 vs M2 validation performance"):
    """
    Plots a comparison between models using classification metrics and cross-entropy.

    Arguments:
        results_dict (dict): Dictionary with model names as keys and metric dictionaries as values.
        title (str): Plot title.

    Returns:
        None
    """
    model_names = list(results_dict.keys())

    accuracy_values = [results_dict[name]["accuracy"] for name in model_names]
    f1_values = [results_dict[name]["f1_macro"] for name in model_names]
    ce_values = [results_dict[name]["cross_entropy"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize = (11, 4))

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, accuracy_values, width = width, label = "Accuracy", color = "#082450")
    plt.bar(x + width / 2, f1_values, width = width, label = "F1 Macro", color = "#095F2A")
    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Accuracy and F1 Macro")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(model_names, ce_values, color = "#AA1D22")
    plt.ylabel("Cross-Entropy")
    plt.title("Cross-Entropy")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_final_models_comparison(results_M0, results_M1, results_M2, results_M3):
    model_names = [
        "M0 - Base NumPy",
        "M1 - Best NumPy",
        "M2 - PyTorch M1",
        "M3 - Best PyTorch"
    ]

    results = [
        results_M0,
        results_M1,
        results_M2,
        results_M3
    ]

    accuracy_values = [result["accuracy"] for result in results]
    f1_values = [result["f1_score"] for result in results]
    ce_values = [result["cross_entropy"] for result in results]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize = (13, 4))

    plt.subplot(1, 2, 1)
    plt.bar(x - width / 2, accuracy_values, width = width, label = "Accuracy", color = "#082450")
    plt.bar(x + width / 2, f1_values, width = width, label = "F1", color = "#105F1D")
    plt.xticks(x, model_names, rotation = 20, ha = "right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Test Accuracy and F1")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(model_names, ce_values, color = "#AA1D22")
    plt.xticks(rotation = 20, ha = "right")
    plt.ylabel("Cross-Entropy")
    plt.title("Test Cross-Entropy")

    plt.tight_layout()
    plt.show()