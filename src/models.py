import numpy as np
import time
from torch import nn

class MLP:
    def __init__(self, input_size, hidden_size, output_size, hidden_nodes, hidden_activation = "relu",
                 output_activation = "softmax", learning_rate = 0.1, optimizer = "gd", lr_schedule = None, 
                 lr_min = 1e-4, lr_decay = 0.95, l2_lambda = 0.0, early_stopping = False, patience = 10,
                 min_delta = 1e-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, seed = 42,
                 full_batch_chunk_size = 32000):        
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_nodes = hidden_nodes #list of hidden layer sizes
        self.output_size = output_size
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        self.learning_rate = learning_rate
        self.random_state = seed
        self.rng = np.random.default_rng(seed)

        # ================== Advanced atributes ==================
        self.optimizer = optimizer
        
        # --------------- Learning rate scheduling ---------------
        self.lr_schedule = lr_schedule
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        
        # ---------- Regularization (L2/Early Stopping) ----------
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # ------------------------- Adam -------------------------
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon 
        self.adam_m = None
        self.adam_v = None
        self.adam_t = 0
        self.full_batch_chunk_size = full_batch_chunk_size

        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initializes weights and biases for all network layers.

        Weights are initialized using He initialization, which is suitable for
        ReLU activations. Biases are initialized to zero.

        Arguments:
            None

        Returns:
            parameters (dict): Dictionary containing weights and biases.
        """
        rng = np.random.default_rng(self.random_state)
        parameters = {}

        # Initialize parameters between input layer and first hidden layer
        dim_in = self.input_size
        dim_out = self.hidden_nodes[0]
        parameters["W1"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out)).astype(np.float32)
        parameters["b1"] = np.zeros((1, dim_out), dtype = np.float32)

        # Initialize parameters for the next hidden layers
        for layer in range(2, len(self.hidden_nodes) + 1):
            dim_in = self.hidden_nodes[layer - 2]
            dim_out = self.hidden_nodes[layer - 1]

            # loc = 0.0 is the mean of the normal distribution, scale = sqrt(2 / dim_in) is the standard deviation
            parameters[f"W{layer}"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out)).astype(np.float32)
            parameters[f"b{layer}"] = np.zeros((1, dim_out), dtype = np.float32)

        # Initialize parameters between last hidden layer and output layer
        output_layer = len(self.hidden_nodes) + 1
        dim_in = self.hidden_nodes[-1]
        dim_out = self.output_size
        parameters[f"W{output_layer}"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out)).astype(np.float32)
        parameters[f"b{output_layer}"] = np.zeros((1, dim_out), dtype = np.float32)
        
        return parameters
    
    def _relu(self, A):
        """
        Applies the ReLU activation function.

        Arguments:
            A (np.ndarray): Linear transformation output.

        Returns:
            Z (np.ndarray): Activated output.
        """
        return np.maximum(0, A)
    
    def _softmax(self, A):
        """
        Applies the softmax activation function.

        Arguments:
            A (np.ndarray): Linear transformation output from the output layer.

        Returns:
            probabilities (np.ndarray): Class probabilities for each sample.
        """
        z_shifted = A - np.max(A, axis = 1, keepdims = True) # Shift for numerical stability
        exp_z = np.exp(z_shifted)
        probabilities = exp_z / np.sum(exp_z, axis = 1, keepdims = True)

        return probabilities
    
    def _one_hot_encode(self, y):
        """
        Converts class labels into one-hot encoded vectors.

        Arguments:
            y (np.ndarray): Array containing integer class labels.

        Returns:
            y_one_hot (np.ndarray): One-hot encoded labels.
        """
        y = y.astype(int)
        n_samples = len(y)

        row_indices = np.arange(n_samples)
        class_indices = y

        y_one_hot = np.zeros((n_samples, self.output_size), dtype = np.float32)
        y_one_hot[row_indices, class_indices] = 1.0

        return y_one_hot
        
    def _activation(self, A, activation):
        """
        Applies the specified activation function.

        Arguments:
            A (np.ndarray): Linear transformation output.
            activation (str): Activation function to apply ("relu" or "softmax").

        Returns:
            activated_A (np.ndarray): Activated output.
        """
        if activation == "relu":
            return self._relu(A)
        elif activation == "softmax":
            return self._softmax(A)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
    def _relu_derivate(self, A):
        """
        Computes the derivative of the ReLU activation function.

        Arguments:
            A (np.ndarray): Linear transformation output.

        Returns:
            derivative (np.ndarray): ReLU derivative evaluated at A.
        """
        return (A > 0).astype(np.float32)
        
    def _cross_entropy_loss(self, y, y_proba):
        """
        Computes the multiclass cross-entropy loss.

        Arguments:
            y (np.ndarray): Array containing true class labels.
            y_proba (np.ndarray): Predicted class probabilities.

        Returns:
            loss (float): Cross-entropy loss.
        """
        epsilon = 1e-12
        y_one_hot = self._one_hot_encode(y)
        y_proba = np.clip(y_proba, epsilon, 1.0 - epsilon)

        loss = -np.mean(np.sum(y_one_hot * np.log(y_proba), axis = 1))

        return loss

    def _get_learning_rate(self, epoch, epochs):
        """
        Computes the learning rate for the current epoch.

        Arguments:
            epoch (int): Current epoch.
            epochs (int): Total number of epochs.

        Returns:
            learning_rate (float): Learning rate for the current epoch.
        """
        if self.lr_schedule is None:
            return self.learning_rate

        if self.lr_schedule == "linear":
            progress = epoch / max(epochs - 1, 1)
            learning_rate = self.learning_rate * (1 - progress)

            return max(learning_rate, self.lr_min)

        if self.lr_schedule == "exponential":
            learning_rate = self.learning_rate * (self.lr_decay ** epoch)

            return max(learning_rate, self.lr_min)
        
        raise ValueError(f"Unsupported learning rate schedule: {self.lr_schedule}")
    
    def _resolve_chunk_size(self, n_samples, chunk_size):
        """
        Validates and bounds the chunk size used for memory-safe passes.

        Arguments:
            n_samples (int): Number of samples to process.
            chunk_size (int): Maximum number of samples per chunk.

        Returns:
            chunk_size (int): Valid chunk size.
        """
        if n_samples <= 0:
            raise ValueError("Cannot process an empty dataset.")

        if chunk_size is None:
            return n_samples

        chunk_size = int(chunk_size)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        return min(chunk_size, n_samples)
    
    def _iter_batches(self, X, y, batch_size = None, shuffle = True):
        """
        Iterates over mini-batches from the dataset.

        Arguments:
            X (np.ndarray): Input data.
            y (np.ndarray): Array containing true class labels.
            batch_size (int): Number of samples per mini-batch.
            shuffle (bool): Whether to shuffle samples before batching.

        Returns:
            Iterator containing mini-batches.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            self.rng.shuffle(indices)

        if batch_size is None:
            batch_size = n_samples

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield X[batch_indices], y[batch_indices]
    
    def _create_batches(self, X, y, batch_size = None, shuffle = True):
        """
        Creates mini-batches from the dataset.

        Arguments:
            X (np.ndarray): Input data.
            y (np.ndarray): Array containing true class labels.
            batch_size (int): Number of samples per mini-batch.
            shuffle (bool): Whether to shuffle samples before batching.

        Returns:
            batches (list): List containing mini-batches.
        """
        return list(self._iter_batches(X, y, batch_size = batch_size, shuffle = shuffle))
    
    def _iter_chunks(self, X, y, chunk_size):
        """
        Iterates over contiguous chunks without changing full-batch semantics.

        Arguments:
            X (np.ndarray): Input data.
            y (np.ndarray): Array containing true class labels.
            chunk_size (int): Maximum number of samples per chunk.

        Returns:
            Iterator containing chunks.
        """
        n_samples = X.shape[0]

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)

            yield X[start:end], y[start:end]
    
    def _l2_penalty(self, n_samples):
        """
        Computes the L2 regularization penalty.

        Arguments:
            n_samples (int): Number of samples.

        Returns:
            penalty (float): L2 regularization penalty.
        """
        penalty = 0.0
        n_layers = len(self.hidden_nodes) + 1

        for layer in range(1, n_layers + 1):
            W = self.parameters[f"W{layer}"]
            penalty += np.sum(W ** 2)

        return (self.l2_lambda / (2 * n_samples)) * penalty
    
    def _add_l2_gradients(self, gradients, n_samples):
        """
        Adds L2 regularization terms to weight gradients.

        Arguments:
            gradients (dict): Dictionary containing gradients.
            n_samples (int): Number of samples.

        Returns:
            gradients (dict): Gradients updated with L2 terms.
        """
        n_layers = len(self.hidden_nodes) + 1

        for layer in range(1, n_layers + 1):
            gradients[f"dW{layer}"] += (self.l2_lambda / n_samples) * self.parameters[f"W{layer}"]

        return gradients
    
    def foward_pass(self, X):
        """
        Performs forward propagation through the network.

        Arguments:
            X (np.ndarray): Input data with shape (n_samples, input_size).

        Returns:
            y_proba (np.ndarray): Predicted class probabilities.
            pre_activations (dict): Dictionary of pre-activation values for each layer.
            activations (dict): Dictionary of activated outputs for each layer.
        """
        pre_activations = {}
        activations = {"Z0":X}
        n_layers = len(self.hidden_nodes) + 1 # Total layers = hidden layers + output layer

        for l in range(1, n_layers + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]

            # Calculate pre-activation values
            Z_prev = activations[f"Z{l-1}"]
            A_l = Z_prev @ W + b

            # Calculate activated output
            activation = self.hidden_activation if l < n_layers else self.output_activation
            Z_l = self._activation(A_l, activation)

            # Update values
            pre_activations[f"A{l}"] = A_l
            activations[f"Z{l}"] = Z_l
    
        y_proba = activations[f"Z{n_layers}"]
        return y_proba, pre_activations, activations
    
    def backward_pass(self, X, y, pre_activations, activations):
        """
        Performs backpropagation and computes gradients for all parameters.

        Arguments:
            X (np.ndarray): Input data with shape (n_samples, input_size).
            y (np.ndarray): Array containing true class labels.
            pre_activations (dict): Dictionary of pre-activation values for each layer.
            activations (dict): Dictionary of activated outputs for each layer.

        Returns:
            gradients (dict): Dictionary containing gradients for weights and biases.
        """
        gradients = {}
        n_samples = X.shape[0]
        n_layers = len(self.hidden_nodes) + 1 # Total layers = hidden layers + output layer

        # Encode labels
        y_one_hot = self._one_hot_encode(y)

        # Output layer gradients
        Z_L = activations[f"Z{n_layers}"]
        dA = (Z_L - y_one_hot) / n_samples # The error signal for the output layer is the difference between 
                                            # predicted probabilities and one-hot labels
        
        for layer in range(n_layers, 0, -1):
            Z_prev = activations[f"Z{layer - 1}"]
            W = self.parameters[f"W{layer}"]

            # Compute gradients
            gradients[f"dW{layer}"] = Z_prev.T @ dA
            gradients[f"db{layer}"] = np.sum(dA, axis = 0, keepdims = True)

            # Compute error signal for the previous layer
            if layer > 1: # No need to compute dA for the input layer
                dZ_prev = dA @ W.T
                A_prev = pre_activations[f"A{layer - 1}"]
                dA = dZ_prev * self._relu_derivate(A_prev)

        return gradients

    def _update_parameters_gd(self, gradients, learning_rate):
        """
        Updates weights and biases using gradient descent.

        Arguments:
            gradients (dict): Dictionary containing gradients for weights and biases.
            learning_rate (float): Learning rate used for the update.

        Returns:
            None
        """
        n_layers = len(self.hidden_nodes) + 1

        for layer in range(1, n_layers + 1):
            self.parameters[f"W{layer}"] -= learning_rate * gradients[f"dW{layer}"]
            self.parameters[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"]
    
    def _initialize_adam(self):
        """
        Initializes Adam first and second moment estimates.

        Arguments:
            None

        Returns:
            None
        """
        self.adam_m = {}
        self.adam_v = {}
        n_layers = len(self.hidden_nodes) + 1

        for layer in range(1, n_layers + 1):
            self.adam_m[f"dW{layer}"] = np.zeros_like(self.parameters[f"W{layer}"])
            self.adam_m[f"db{layer}"] = np.zeros_like(self.parameters[f"b{layer}"])
            self.adam_v[f"dW{layer}"] = np.zeros_like(self.parameters[f"W{layer}"])
            self.adam_v[f"db{layer}"] = np.zeros_like(self.parameters[f"b{layer}"])
    
    def _update_parameters_adam(self, gradients, learning_rate):
        """
        Updates weights and biases using the Adam optimizer.

        Arguments:
            gradients (dict): Dictionary containing gradients for weights and biases.
            learning_rate (float): Learning rate used for the update.

        Returns:
            None
        """
        if self.adam_m is None or self.adam_v is None:
            self._initialize_adam()

        self.adam_t += 1
        n_layers = len(self.hidden_nodes) + 1

        for layer in range(1, n_layers + 1):
            for parameter_type in ["W", "b"]:
                parameter_key = f"{parameter_type}{layer}"
                gradient_key = f"d{parameter_type}{layer}"

                self.adam_m[gradient_key] = self.beta1 * self.adam_m[gradient_key] + (1 - self.beta1) * gradients[gradient_key]
                self.adam_v[gradient_key] = self.beta2 * self.adam_v[gradient_key] + (1 - self.beta2) * (gradients[gradient_key] ** 2)

                m_corrected = self.adam_m[gradient_key] / (1 - self.beta1 ** self.adam_t)
                v_corrected = self.adam_v[gradient_key] / (1 - self.beta2 ** self.adam_t)

                self.parameters[parameter_key] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
    
    def _update_parameters(self, gradients, learning_rate):
        """
        Updates weights and biases using the selected optimizer.

        Arguments:
            gradients (dict): Dictionary containing gradients for weights and biases.
            learning_rate (float): Learning rate used for the update.

        Returns:
            None
        """
        if self.optimizer == "gd":
            self._update_parameters_gd(gradients, learning_rate)

        elif self.optimizer == "adam":
            self._update_parameters_adam(gradients, learning_rate)

        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _compute_loss(self, y, y_proba):
        """
        Computes cross-entropy loss.

        Arguments:
            y (np.ndarray): Array containing true class labels.
            y_proba (np.ndarray): Predicted class probabilities.

        Returns:
            loss (float): Computed loss.
        """
        loss = self._cross_entropy_loss(y, y_proba)

        return loss
    
    def _compute_batch_loss_and_gradients(self, X_batch, y_batch):
        """
        Computes loss and gradients for a batch without updating parameters.

        Arguments:
            X_batch (np.ndarray): Batch input data.
            y_batch (np.ndarray): Batch labels.

        Returns:
            batch_loss (float): Loss computed on the batch.
            gradients (dict): Gradients computed on the batch.
        """
        y_proba, pre_activations, activations = self.foward_pass(X_batch)

        batch_loss = self._compute_loss(y_batch, y_proba)
        gradients = self.backward_pass(X_batch, y_batch, pre_activations, activations)

        return batch_loss, gradients
    
    def _train_batch(self, X_batch, y_batch, learning_rate):
        """
        Trains the model on one batch.

        Arguments:
            X_batch (np.ndarray): Batch input data.
            y_batch (np.ndarray): Batch labels.
            learning_rate (float): Learning rate used for the update.

        Returns:
            batch_loss (float): Loss computed on the batch before updating parameters.
        """
        batch_loss, gradients = self._compute_batch_loss_and_gradients(X_batch, y_batch)

        if self.l2_lambda > 0:
            gradients = self._add_l2_gradients(gradients, len(y_batch))

        self._update_parameters(gradients, learning_rate)

        return batch_loss
    
    def _train_full_batch_chunked(self, X, y, learning_rate):
        """
        Performs one full-batch update while processing data in chunks.

        The gradients from all chunks are accumulated into the exact
        full-dataset average before a single parameter update is applied.

        Arguments:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            learning_rate (float): Learning rate used for the update.

        Returns:
            train_loss (float): Average training loss for the epoch.
            n_updates (int): Number of parameter updates.
            n_batches (int): Number of full batches.
            n_chunks (int): Number of chunks used to build the full batch.
        """
        n_samples = X.shape[0]
        chunk_size = self._resolve_chunk_size(n_samples, self.full_batch_chunk_size)
        train_loss = 0.0
        accumulated_gradients = None
        n_chunks = 0

        for X_chunk, y_chunk in self._iter_chunks(X, y, chunk_size):
            chunk_loss, chunk_gradients = self._compute_batch_loss_and_gradients(X_chunk, y_chunk)
            chunk_weight = len(y_chunk) / n_samples
            train_loss += chunk_loss * chunk_weight

            if accumulated_gradients is None:
                accumulated_gradients = {
                    key: value * chunk_weight
                    for key, value in chunk_gradients.items()
                }
            else:
                for key, value in chunk_gradients.items():
                    accumulated_gradients[key] += value * chunk_weight

            n_chunks += 1

        if self.l2_lambda > 0:
            accumulated_gradients = self._add_l2_gradients(accumulated_gradients, n_samples)
            train_loss += self._l2_penalty(n_samples)

        self._update_parameters(accumulated_gradients, learning_rate)

        return train_loss, 1, 1, n_chunks
    
    def _train_epoch(self, X, y, batch_size, learning_rate):
        """
        Trains the model for one epoch.

        Arguments:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            batch_size (int): Number of samples per mini-batch.
            learning_rate (float): Learning rate used for the epoch.

        Returns:
            train_loss (float): Average training loss for the epoch.
            n_updates (int): Number of parameter updates performed during the epoch.
            n_batches (int): Number of batches used during the epoch.
            n_chunks (int): Number of chunks used during the epoch.
        """
        if batch_size is None:
            return self._train_full_batch_chunked(X, y, learning_rate)

        train_loss = 0.0
        n_updates = 0
        n_batches = 0

        for X_batch, y_batch in self._iter_batches(X, y, batch_size = batch_size, shuffle = True):
            batch_loss = self._train_batch(X_batch, y_batch, learning_rate)
            train_loss += batch_loss * len(y_batch) / X.shape[0]
            n_updates += 1
            n_batches += 1

        if self.l2_lambda > 0:
            train_loss += self._l2_penalty(X.shape[0])

        return train_loss, n_updates, n_batches, n_batches
    
    def _validate(self, X_val, y_val):
        """
        Computes validation loss.

        Arguments:
            X_val (np.ndarray): Validation data.
            y_val (np.ndarray): Validation labels.

        Returns:
            val_loss (float): Validation loss.
        """
        n_samples = X_val.shape[0]
        chunk_size = self._resolve_chunk_size(n_samples, self.full_batch_chunk_size)
        val_loss = 0.0

        for X_chunk, y_chunk in self._iter_chunks(X_val, y_val, chunk_size):
            y_val_proba, _, _ = self.foward_pass(X_chunk)
            chunk_loss = self._compute_loss(y_chunk, y_val_proba)
            val_loss += chunk_loss * len(y_chunk) / n_samples

        if self.l2_lambda > 0:
            val_loss += self._l2_penalty(len(y_val))

        return val_loss
    
    def _save_best_parameters(self):
        """
        Saves a copy of current model parameters.

        Arguments:
            None

        Returns:
            parameters_copy (dict): Copy of current parameters.
        """
        return {key: value.copy() for key, value in self.parameters.items()}
    
    def _check_early_stopping(self, val_loss, best_val_loss, epochs_without_improvement):
        """
        Updates early stopping state based on validation loss.

        Arguments:
            val_loss (float): Current validation loss.
            best_val_loss (float): Best validation loss observed so far.
            epochs_without_improvement (int): Number of epochs without improvement.

        Returns:
            should_stop (bool): Whether training should stop.
            best_val_loss (float): Updated best validation loss.
            epochs_without_improvement (int): Updated patience counter.
            improved (bool): Whether validation loss improved.
        """
        improved = val_loss < best_val_loss - self.min_delta

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        should_stop = epochs_without_improvement >= self.patience

        return should_stop, best_val_loss, epochs_without_improvement, improved
    
    def fit(self, X, y, X_val = None, y_val = None, epochs = 10, batch_size = None, verbose = True):
        """
        Trains the neural network using backpropagation.

        Arguments:
            X (np.ndarray): Training data with shape (n_samples, input_size).
            y (np.ndarray): Array containing true class labels.
            X_val (np.ndarray): Validation data with shape (n_samples, input_size).
            y_val (np.ndarray): Array containing validation labels.
            epochs (int): Number of training iterations.
            batch_size (int): Number of samples per mini-batch. If None, full-batch gradient descent is used.
            verbose (bool): Whether to print the loss during training.

        Returns:
            history (dict): Dictionary containing training history, validation history, and training time.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epochs_trained": 0,
            "updates": 0,
            "batches_per_epoch": [],
            "chunks_per_epoch": [],
            "epoch_time": [],
            "avg_epoch_time": 0.0,
            "training_time": 0.0,
            "stopped_early": False
        }
        
        best_val_loss = np.inf
        best_parameters = None
        epochs_without_improvement = 0
        start_time = time.perf_counter()

        for epoch in range(epochs):
            epoch_start_time = time.perf_counter()

            # Compute learning rate for currente epoch
            learning_rate = self._get_learning_rate(epoch, epochs)
            train_loss, n_updates, n_batches, n_chunks = self._train_epoch(X, y, batch_size, learning_rate)

            # Update history
            history["train_loss"].append(train_loss)
            history["learning_rate"].append(learning_rate)
            history["epochs_trained"] = epoch + 1
            history["updates"] += n_updates
            history["batches_per_epoch"].append(n_batches)
            history["chunks_per_epoch"].append(n_chunks)

            # Validate on validation set if provided
            if X_val is not None and y_val is not None:
                # Compute validation loss and update history
                val_loss = self._validate(X_val, y_val)
                history["val_loss"].append(val_loss)

                # Check for early stopping
                if self.early_stopping:
                    should_stop, best_val_loss, epochs_without_improvement, improved = self._check_early_stopping(
                        val_loss, best_val_loss, epochs_without_improvement)

                    if improved:
                        best_parameters = self._save_best_parameters()

                    if should_stop:
                        if best_parameters is not None:
                            self.parameters = best_parameters

                        epoch_end_time = time.perf_counter()
                        history["epoch_time"].append(epoch_end_time - epoch_start_time)
                        history["training_time"] = epoch_end_time - start_time
                        history["avg_epoch_time"] = float(np.mean(history["epoch_time"]))
                        history["stopped_early"] = True

                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")

                        break

            epoch_end_time = time.perf_counter()
            history["epoch_time"].append(epoch_end_time - epoch_start_time)
            history["training_time"] = epoch_end_time - start_time
            history["avg_epoch_time"] = float(np.mean(history["epoch_time"]))

            if verbose:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - lr: {learning_rate:.6f} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - time: {history['epoch_time'][-1]:.2f}s - avg: {history['avg_epoch_time']:.2f}s - batches: {n_batches} - chunks: {n_chunks}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - lr: {learning_rate:.6f} - train loss: {train_loss:.4f} - time: {history['epoch_time'][-1]:.2f}s - avg: {history['avg_epoch_time']:.2f}s - batches: {n_batches} - chunks: {n_chunks}")

        return history
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for each sample.

        Arguments:
            X (np.ndarray): Input data with shape (n_samples, input_size).

        Returns:
            y_proba (np.ndarray): Predicted class probabilities.
        """
        n_samples = X.shape[0]
        chunk_size = self._resolve_chunk_size(n_samples, self.full_batch_chunk_size)

        if chunk_size == n_samples:
            y_proba, _, _ = self.foward_pass(X)
            return y_proba

        y_proba_chunks = []

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            y_proba_chunk, _, _ = self.foward_pass(X[start:end])
            y_proba_chunks.append(y_proba_chunk)

        return np.vstack(y_proba_chunks)
    
    def predict(self, X):
        """
        Predicts the most likely class for each sample.

        Arguments:
            X (np.ndarray): Input data with shape (n_samples, input_size).

        Returns:
            y_pred (np.ndarray): Predicted class labels.
        """
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis = 1)

        return y_pred


# PyTorch model ----------------------------------------------------------------------------------------------------
class TorchMLP(nn.Module):
    """
    Multi-layer perceptron implemented with PyTorch.

    Arguments:
        input_size (int): Number of input features.
        hidden_nodes (list): Number of neurons in each hidden layer.
        output_size (int): Number of output classes.
        activation (str): Activation function to use in hidden layers.
        dropout_rate (float): Dropout probability after each hidden activation.
    """

    def __init__(self, input_size, hidden_nodes, output_size, activation = "relu", dropout_rate = 0.0):
        super().__init__()

        activation_functions = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,
            "gelu": nn.GELU
        }

        if activation not in activation_functions:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = []
        previous_size = input_size

        for hidden_size in hidden_nodes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(activation_functions[activation]())

            if dropout_rate > 0:
                layers.append(nn.Dropout(p = dropout_rate))

            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        """
        Performs the forward pass.

        Arguments:
            X (torch.Tensor): Input tensor.

        Returns:
            logits (torch.Tensor): Raw class scores.
        """
        logits = self.network(X)

        return logits