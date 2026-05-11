import numpy as np

def split_class_indices(class_indices, train_size, val_size):
    """
    Splits indices from a single class into train, validation, and test indices.

    Arguments:
        class_indices (np.ndarray): Indices belonging to one class.
        train_size (float): Proportion assigned to the training set.
        val_size (float): Proportion assigned to the validation set.
        test_size (float): Proportion assigned to the test set.

    Returns:
        train_class_indices (np.ndarray): Training indices for the class.
        val_class_indices (np.ndarray): Validation indices for the class.
        test_class_indices (np.ndarray): Test indices for the class.
    """
    n_class = len(class_indices)

    n_train = int(n_class * train_size)
    n_val = int(n_class * val_size)
    n_test = n_class - n_train - n_val

    train_end = n_train
    val_end = train_end + n_val
    test_end = val_end + n_test

    train_class_indices = class_indices[:train_end]
    val_class_indices = class_indices[train_end:val_end]
    test_class_indices = class_indices[val_end:test_end]

    return train_class_indices, val_class_indices, test_class_indices



def train_val_test_split(X, y, train_size = 0.70, val_size = 0.15, random_state = 42):
    """
    Splits the dataset into training, validation, and test sets using NumPy.

    The split is stratified to preserve approximately the same class
    proportions in each subset.

    Arguments:
        X (np.ndarray): Array containing the images.
        y (np.ndarray): Array containing the labels.
        train_size (float): Proportion of the dataset used for training.
        val_size (float): Proportion of the dataset used for validation.
        test_size (float): Proportion of the dataset used for testing.
        random_state (int): Seed used to make the split reproducible.

    Returns:
        X_train (np.ndarray): Images in the training set.
        X_val (np.ndarray): Images in the validation set.
        X_test (np.ndarray): Images in the test set.
        y_train (np.ndarray): Labels in the training set.
        y_val (np.ndarray): Labels in the validation set.
        y_test (np.ndarray): Labels in the test set.
    """

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_value in classes:
        class_indices = np.where(y == class_value)[0]
        rng.shuffle(class_indices)

        train_class_indices, val_class_indices, test_class_indices = split_class_indices(class_indices, train_size, val_size)

        train_indices.extend(train_class_indices)
        val_indices.extend(val_class_indices)
        test_indices.extend(test_class_indices)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_summary(train, val, test):
    """
    Prints a summary of the train, validation, and test partitions.

    Arguments:
        X_train (np.ndarray): Images in the training set.
        X_val (np.ndarray): Images in the validation set.
        X_test (np.ndarray): Images in the test set.
        y_train (np.ndarray): Labels in the training set.
        y_val (np.ndarray): Labels in the validation set.
        y_test (np.ndarray): Labels in the test set.

    Returns:
        None
    """
    
    total = len(train[0]) + len(val[0]) + len(test[0])

    print("Split summary")
    print("-" * 40)
    print(f"Train: {len(train[0])} images ({len(train[0]) / total * 100:.1f}%)")
    print(f"Validation: {len(val[0])} images ({len(val[0]) / total * 100:.1f}%)")
    print(f"Test: {len(test[0])} images ({len(test[0]) / total * 100:.1f}%)")
    print()
    print(f"X_train shape: {train[0].shape}")
    print(f"X_val shape: {val[0].shape}")
    print(f"X_test shape: {test[0].shape}")
    print(f"y_train shape: {train[1].shape}")
    print(f"y_val shape: {val[1].shape}")
    print(f"y_test shape: {test[1].shape}")
