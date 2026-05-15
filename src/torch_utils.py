import torch
import numpy as np
from src.metrics import accuracy_score, cross_entropy_score, confusion_matrix, f1_score

def train_torch_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, device, verbose = False):
    """
    Trains a PyTorch model.

    Arguments:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        loss_fn: Loss function.
        optimizer: PyTorch optimizer.
        epochs (int): Number of training epochs.
        device: Device used for computation.
        verbose (bool): Whether to print metrics after each epoch.

    Returns:
        history (dict): Training history.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Move batch to selected device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            # Backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.shape[0]

            y_pred = torch.argmax(logits, dim = 1)
            correct += (y_pred == y_batch).sum().item()
            total += y_batch.shape[0]

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate_torch_full_metrics(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"train loss: {train_loss:.4f} - "
                f"val loss: {val_loss:.4f} - "
                f"train acc: {train_accuracy:.4f} - "
                f"val acc: {val_accuracy:.4f}"
            )

    return history


def evaluate_torch_full_metrics(model, dataloader, n_classes, device):
    """
    Evaluates a PyTorch model using the same metrics as the NumPy models.

    Arguments:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader with evaluation data.
        n_classes (int): Number of classes.
        device: Device used for computation.

    Returns:
        results (dict): Dictionary containing accuracy, cross-entropy, F1 macro and confusion matrix.
    """
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            y_proba = torch.softmax(logits, dim = 1)
            y_pred = torch.argmax(y_proba, dim = 1)

            y_true_all.append(y_batch.cpu().numpy())
            y_pred_all.append(y_pred.cpu().numpy())
            y_proba_all.append(y_proba.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "cross_entropy": cross_entropy_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred, n_classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes)
    }

    return results