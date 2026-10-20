import numpy as np

# **Activation Functions**
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# **Loss Functions**
def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))

# **Accuracy Calculation**
def accuracy_score(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=0) == np.argmax(y_pred, axis=0))

# **Forward Propagation**
def forward_propagation(X, parameters):
    activations = {"A0": X}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    
    activation_funcs = parameters.get('activation_functions', ['relu'] * (L-1) + ['softmax'])
    
    for l in range(1, L + 1):
        Z = np.dot(parameters[f"W{l}"], activations[f"A{l-1}"]) + parameters[f"b{l}"]
        
        if activation_funcs[l-1].lower() == 'relu':
            activations[f"A{l}"] = relu(Z)
        elif activation_funcs[l-1].lower() == 'sigmoid':
            activations[f"A{l}"] = sigmoid(Z)
        elif activation_funcs[l-1].lower() == 'tanh':
            activations[f"A{l}"] = np.tanh(Z)
        elif activation_funcs[l-1].lower() == 'softmax':
            activations[f"A{l}"] = softmax(Z)
        elif activation_funcs[l-1].lower() == 'linear':
            activations[f"A{l}"] = Z
        elif activation_funcs[l-1].lower() == 'leaky_relu':
            activations[f"A{l}"] = leaky_relu(Z)
        else:
            raise ValueError(f"Unsupported activation function: {activation_funcs[l-1]}")
            
        # Store the pre-activation values for backprop
        activations[f"Z{l}"] = Z
        
    return activations

# **Backward Propagation**
def back_propagation(y, activations, parameters):
    m = y.shape[1]
    gradients = {}
    L = len([key for key in parameters.keys() if key.startswith('W')])
    
    activation_funcs = parameters.get('activation_functions', ['relu'] * (L-1) + ['softmax'])
    
    dZ = activations[f"A{L}"] - y
    
    for l in reversed(range(1, L + 1)):
        gradients[f"dW{l}"] = (1 / m) * np.dot(dZ, activations[f"A{l-1}"].T)
        gradients[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            if activation_funcs[l-2].lower() == 'relu':
                dA = np.dot(parameters[f"W{l}"].T, dZ)
                dZ = dA * (activations[f"A{l-1}"] > 0)
            elif activation_funcs[l-2].lower() == 'sigmoid':
                dA = np.dot(parameters[f"W{l}"].T, dZ)
                dZ = dA * (activations[f"A{l-1}"] * (1 - activations[f"A{l-1}"]))
            elif activation_funcs[l-2].lower() == 'tanh':
                dA = np.dot(parameters[f"W{l}"].T, dZ)
                dZ = dA * (1 - np.power(activations[f"A{l-1}"], 2))
            elif activation_funcs[l-2].lower() == 'linear':
                dZ = np.dot(parameters[f"W{l}"].T, dZ)
            elif activation_funcs[l-2].lower() == 'leaky_relu':
                dA = np.dot(parameters[f"W{l}"].T, dZ)
                dZ = dA * leaky_relu_derivative(activations[f"A{l-1}"])
            else:
                raise ValueError(f"Fonction d'activation non supportée : {activation_funcs[l-2]}")
                
    return gradients

# **Prediction**
def predict(X, parameters):
    # Récupérer et appliquer la normalisation
    X_mean = parameters.pop('X_mean', 0)
    X_std = parameters.pop('X_std', 1)
    X_normalized = (X - X_mean) / X_std
    
    activations = forward_propagation(X_normalized, parameters)
    L = len(parameters) // 2
    
    parameters['X_mean'] = X_mean
    parameters['X_std'] = X_std
    
    return np.argmax(activations[f"A{L}"], axis=0)