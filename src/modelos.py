import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, hidden_nodes,
                 hidden_activation = "relu", output_activation = "softmax", seed = 42):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_nodes = hidden_nodes #list of hidden layer sizes
        self.output_size = output_size
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

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
        parameters["W1"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out))
        parameters["b1"] = np.zeros((1, dim_out))

        # Initialize parameters for the next hidden layers
        for layer in range(1, len(self.hidden_nodes)):
            dim_in = self.input_size[layer - 1]
            dim_out = self.input_size[layer]

            parameters[f"W{layer}"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out))
            parameters[f"b{layer}"] = np.zeros((1, dim_out))

        # Initialize parameters between last hidden layer and output layer
        dim_in = self.input_size
        dim_out = self.output_size
        parameters[f"W{len(self.hidden_nodes)}"] = rng.normal(loc = 0.0, scale = np.sqrt(2 / dim_in), size = (dim_in, dim_out))
        parameters[f"b{len(self.hidden_nodes)}"] = np.zeros((1, dim_out))
        
        return parameters
    
    def _relu(self, Z):
       """
        Applies the ReLU activation function.

        Arguments:
            Z (np.ndarray): Linear transformation output.

        Returns:
            A (np.ndarray): Activated output.
        """
       return np.maximum(0, Z)
    
    def _softmax(self, Z):
        """
        Applies the softmax activation function.

        Arguments:
            Z (np.ndarray): Linear transformation output from the output layer.

        Returns:
            probabilities (np.ndarray): Class probabilities for each sample.
        """
        z_shifted = Z - np.max(Z, axis = 1, keepdims = True) # Shift for numerical stability
        exp_z = np.exp(z_shifted)
        probabilities = exp_z / np.sum(exp_z, axis = 1, keepdims = True)

        return probabilities
    
    def foward_pass(self, X):
        """
        Performs forward propagation through the network.

        Arguments:
            X (np.ndarray): Input data with shape (n_samples, input_size).

        Returns:
            y_proba (np.ndarray): Predicted class probabilities.
            a_z_vals (dict): Intermediate values needed for backpropagation.
        """
        A = 