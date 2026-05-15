import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Computes classification accuracy.

    Arguments:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.

    Returns:
        accuracy (float): Proportion of correctly classified samples.
    """
    accuracy = np.mean(y_true == y_pred)
    return accuracy


def cross_entropy_score(y_true, y_proba):
    """
    Computes multiclass cross-entropy.

    Arguments:
        y_true (np.ndarray): True class labels.
        y_proba (np.ndarray): Predicted class probabilities.

    Returns:
        loss (float): Cross-entropy loss.
    """
    epsilon = 1e-12
    y_true = y_true.astype(int)
    # Clip probabilities to avoid log(0)
    y_proba = np.clip(y_proba, epsilon, 1.0 - epsilon)
    
    # Keep only the probability assigned to the true class
    loss = -np.mean(np.log(y_proba[np.arange(len(y_true)), y_true]))

    return loss


def confusion_matrix(y_true, y_pred, n_classes):
    """
    Computes the confusion matrix for multiclass classification.

    Arguments:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        n_classes (int): Number of classes.

    Returns:
        matrix (np.ndarray): Confusion matrix.
    """
    matrix = np.zeros((n_classes, n_classes), dtype = int)

    for true_label, pred_label in zip(y_true.astype(int), y_pred.astype(int)):
        # Increment the count for the corresponding true and predicted class
        matrix[true_label, pred_label] += 1

    return matrix


def f1_score_macro(y_true, y_pred, n_classes):
    """
    Computes macro-averaged F1-score.

    Macro F1 computes the F1-score for each class and then averages them,
    giving the same weight to every class.

    Arguments:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        n_classes (int): Number of classes.

    Returns:
        macro_f1 (float): Macro-averaged F1-score.
    """
    matrix = confusion_matrix(y_true, y_pred, n_classes)

    # Values needed to compute precision and recall by class
    true_positives = np.diag(matrix)
    false_positives = np.sum(matrix, axis = 0) - true_positives # Sum over columns minus true positives
    false_negatives = np.sum(matrix, axis = 1) - true_positives # Sum over rows minus true positives

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / np.maximum(true_positives + false_negatives, 1)

    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12) # Avoid division by zero
    mean_f1 = np.mean(f1_scores)

    return mean_f1


def evaluate_model(model, X, y, n_classes):
    """
    Evaluates a trained classification model.

    Arguments:
        model (MLP): Trained model.
        X (np.ndarray): Input data.
        y (np.ndarray): True class labels.
        n_classes (int): Number of classes.

    Returns:
        results (dict): Dictionary containing performance metrics.
    """
    # First get probabilities and then convert them into class predictions
    y_proba = model.predict_proba(X)
    y_pred = np.argmax(y_proba, axis = 1)

    results = {
        "accuracy": accuracy_score(y, y_pred),
        "cross_entropy": cross_entropy_score(y, y_proba),
        "f1_macro": f1_score_macro(y, y_pred, n_classes),
        "confusion_matrix": confusion_matrix(y, y_pred, n_classes)
    }

    return results


def performance_report_table(train_results, val_results):
    """
    Displays a pandas table with performance metrics for training and validation.

    Arguments:
        train_results (dict): Metrics computed on the training set.
        val_results (dict): Metrics computed on the validation set.

    Returns:
        None
    """
    import pandas as pd

    report = pd.DataFrame(
        {
            "Train": [
                train_results["accuracy"],
                train_results["cross_entropy"],
                train_results["f1_macro"]
            ],
            "Validation": [
                val_results["accuracy"],
                val_results["cross_entropy"],
                val_results["f1_macro"]
            ]
        },
        index = [
            "Accuracy",
            "Cross-Entropy",
            "F1-Score Macro"
        ]
    )
    styled_table = report.style \
        .format({
            "Train": "{:.3f}",
            "Validation": "{:.3f}",
        }) \

    display(styled_table)
