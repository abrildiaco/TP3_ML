import numpy as np
import pandas as pd
from src.metrics import evaluate_model

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


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, n_classes, epochs = 50, batch_size = None):
    """
    Trains and evaluates a model.

    Arguments:
        model (MLP): Model to train.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per mini-batch.

    Returns:
        results (dict): Dictionary containing history, metrics, and training cost.
    """
    history = model.fit(
        X_train,
        y_train,
        X_val = X_val,
        y_val = y_val,
        epochs = epochs,
        batch_size = batch_size,
        verbose = False
    )

    train_metrics = evaluate_model(model, X_train, y_train, n_classes)
    val_metrics = evaluate_model(model, X_val, y_val, n_classes)

    results = {
        "model": model,
        "history": history,
        "epochs_trained": history["epochs_trained"],
        "updates": history["updates"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }

    return results


def advanced_comparison_table(results_dict):
    """
    Displays a styled comparison table for advanced training strategies.

    Arguments:
        results_dict (dict): Dictionary containing model results.

    Returns:
        None
    """

    rows = []

    for model_name, results in results_dict.items():
        row = {
            "Model": model_name,
            "Epochs": results["epochs_trained"],
            "Updates": results["updates"],
            "Train Accuracy.": results["train_metrics"]["accuracy"],
            "Val. Accuracy.": results["val_metrics"]["accuracy"],
            "Train Cross-Entropy": results["train_metrics"]["cross_entropy"],
            "Val. Cross-Entropy": results["val_metrics"]["cross_entropy"],
            "Train F1": results["train_metrics"]["f1_macro"],
            "Val. F1": results["val_metrics"]["f1_macro"]
        }

        rows.append(row)

    table = pd.DataFrame(rows)

    styled_table = table.style \
        .hide(axis = "index") \
        .format({
            "Train Accuracy.": "{:.2f}",
            "Val. Accuracy.": "{:.2f}",
            "Train Cross-Entropy": "{:.2f}",
            "Val. Cross-Entropy": "{:.2f}",
            "Train F1": "{:.2f}",
            "Val. F1": "{:.2f}"
        }) \
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])

    display(styled_table)
