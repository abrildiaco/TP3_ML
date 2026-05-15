import numpy as np
import pandas as pd
from src.metrics import evaluate_model

try:
    from IPython.display import display
except ImportError:
    display = print

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


def average_epoch_time(history):
    """
    Computes the average epoch time from a training history.

    Arguments:
        history (dict): Dictionary containing training history.

    Returns:
        avg_epoch_time (float): Average seconds per epoch.
    """
    if "avg_epoch_time" in history:
        return history["avg_epoch_time"]

    epoch_time = history.get("epoch_time", [])

    if len(epoch_time) == 0:
        return 0.0

    return float(np.mean(epoch_time))


def print_training_summary(model_name, history, train_metrics, val_metrics):
    """
    Prints a compact summary for a trained model.

    Arguments:
        model_name (str): Name shown in the output.
        history (dict): Dictionary containing training history.
        train_metrics (dict): Metrics computed on the training set.
        val_metrics (dict): Metrics computed on the validation set.

    Returns:
        None
    """
    final_train_loss = history["train_loss"][-1] if len(history["train_loss"]) > 0 else np.nan
    final_val_loss = history["val_loss"][-1] if len(history["val_loss"]) > 0 else np.nan
    avg_time = average_epoch_time(history)
    batches = history["batches_per_epoch"][-1] if len(history.get("batches_per_epoch", [])) > 0 else np.nan
    chunks = history["chunks_per_epoch"][-1] if len(history.get("chunks_per_epoch", [])) > 0 else np.nan

    print()
    print(f"{model_name} summary")
    print("-" * 60)
    print(f"Epochs trained: {history['epochs_trained']}")
    print(f"Updates: {history['updates']}")
    print(f"Batches per epoch: {batches}")
    print(f"Chunks per epoch: {chunks}")
    print(f"Average epoch time: {avg_time:.2f}s")
    print(f"Total training time: {history['training_time']:.2f}s")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(
        "Train metrics: "
        f"accuracy={train_metrics['accuracy']:.4f}, "
        f"cross_entropy={train_metrics['cross_entropy']:.4f}, "
        f"f1_macro={train_metrics['f1_macro']:.4f}"
    )
    print(
        "Validation metrics: "
        f"accuracy={val_metrics['accuracy']:.4f}, "
        f"cross_entropy={val_metrics['cross_entropy']:.4f}, "
        f"f1_macro={val_metrics['f1_macro']:.4f}"
    )


def results_table(results_dict):
    """
    Builds a comparison table from trained model results.

    Arguments:
        results_dict (dict): Dictionary containing model results.

    Returns:
        table (pd.DataFrame): Comparison table.
    """
    rows = []

    for model_name, results in results_dict.items():
        history = results["history"]
        row = {
            "Model": model_name,
            "Epochs": results["epochs_trained"],
            "Updates": results["updates"],
            "Batches/Epoch": history["batches_per_epoch"][-1],
            "Chunks/Epoch": history["chunks_per_epoch"][-1],
            "Avg. Epoch Time (s)": average_epoch_time(history),
            "Total Time (s)": history["training_time"],
            "Train Accuracy": results["train_metrics"]["accuracy"],
            "Val. Accuracy": results["val_metrics"]["accuracy"],
            "Train Cross-Entropy": results["train_metrics"]["cross_entropy"],
            "Val. Cross-Entropy": results["val_metrics"]["cross_entropy"],
            "Train F1": results["train_metrics"]["f1_macro"],
            "Val. F1": results["val_metrics"]["f1_macro"]
        }

        rows.append(row)

    return pd.DataFrame(rows)


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, n_classes, epochs = 50,
                             batch_size = None, model_name = "Model", verbose = True):
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
        verbose = verbose
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

    print_training_summary(model_name, history, train_metrics, val_metrics)

    return results


def advanced_comparison_table(results_dict):
    """
    Displays a styled comparison table for advanced training strategies.

    Arguments:
        results_dict (dict): Dictionary containing model results.

    Returns:
        None
    """

    table = results_table(results_dict)

    styled_table = table.style \
        .hide(axis = "index") \
        .format({
            "Avg. Epoch Time (s)": "{:.2f}",
            "Total Time (s)": "{:.2f}",
            "Train Accuracy": "{:.2f}",
            "Val. Accuracy": "{:.2f}",
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

    return table
