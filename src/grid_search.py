import numpy as np


def build_param_combinations(param_grid):
    """
    Builds all hyperparameter combinations from a parameter grid.

    Arguments:
        param_grid (dict): Dictionary where each key is a hyperparameter name and each value is a list of candidate values.

    Returns:
        combinations (list): List of dictionaries, where each dictionary contains one hyperparameter combination.
    """
    combinations = [{}]

    for param_name, param_values in param_grid.items():
        new_combinations = []

        for combination in combinations:
            for param_value in param_values:
                new_combination = combination.copy()
                new_combination[param_name] = param_value
                new_combinations.append(new_combination)

        combinations = new_combinations

    return combinations


def grid_search(model_class, param_grid, X_train, y_train, X_val, y_val, epochs = 100, batch_size = None, verbose = True):
    """
    Performs grid search over a set of hyperparameter combinations.

    Each model is trained on the training set and evaluated using the final
    validation cross-entropy loss. The best model is selected as the one with
    the lowest validation loss.

    Arguments:
        model_class (class): Model class to instantiate.
        param_grid (dict): Dictionary with hyperparameter names and candidate values.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        epochs (int): Number of training epochs for each model.
        batch_size (int): Number of samples per mini-batch. If None, full-batch gradient descent is used.
        verbose (bool): Whether to print progress during grid search.

    Returns:
        best_params (dict): Hyperparameter combination with the lowest validation loss.
        best_score (float): Best validation loss obtained.
        results (list): List with the validation loss and parameters for each combination.
    """
    best_score = np.inf
    best_params = None
    results = []

    combinations = build_param_combinations(param_grid)

    for i, params in enumerate(combinations):
        if verbose:
            print(f"Training combination {i + 1}/{len(combinations)}")
            print(params)

        model = model_class(**params)

        history = model.fit(X_train, y_train, X_val = X_val, y_val = y_val, epochs = epochs,
                            batch_size = batch_size)

        val_score = history["val_loss"][-1]

        results.append({
            "params": params,
            "val_loss": val_score,
            "history": history
        })

        # Update best score and parameters if current model is better
        if val_score < best_score:
            best_score = val_score
            best_params = params

    return best_params, best_score, results