import numpy as np
import pickle
import sys
import os
from ia_functions import *
from parsing import *
from load_dataset import *



# **Affichage de l'aide**
def printHelp() -> None:
    """Affiche l'aide du programme."""
    print("USAGE")
    print("     ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE\n")
    print("DESCRIPTION")
    print("     --train Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.")
    print("     --predict Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.")
    print("     --save Save neural network into SAVEFILE. Only works in train mode.\n")
    print("     LOADFILE File containing an artificial neural network")
    print("     FILE File containing chessboards")

# **Network Initialization**
def initialize_parameters(input_size, output_size, hidden_layers, activation_functions=None):
    """
    Initialisation améliorée avec la méthode He et conversion explicite en entiers
    """
    # Conversion explicite en entiers
    layers = [int(input_size)] + [int(layer) for layer in hidden_layers] + [int(output_size)]
    parameters = {}
    
    for i in range(1, len(layers)):
        # Calcul de l'écart-type pour l'initialisation
        std = np.sqrt(2.0 / layers[i-1])
        
        # Création des matrices de poids et biais avec dimensions en entiers
        parameters[f"W{i}"] = np.random.randn(int(layers[i]), int(layers[i-1])) * std
        parameters[f"b{i}"] = np.zeros((int(layers[i]), 1))
    
    # Gestion des fonctions d'activation
    if not activation_functions:
        parameters['activation_functions'] = ['leaky_relu'] * (len(hidden_layers)) + ['softmax']
    else:
        parameters['activation_functions'] = activation_functions
        
    return parameters

# **Parameter Update**
def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * gradients[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * gradients[f"db{l}"]
    return parameters

# **Training**
def train_network_simple(X, y, input_size, output_size, hidden_layers, learning_rate, epochs, batch_size, activation_functions=None):
    """
    Version simplifiée sans multiprocessing pour débogage
    """
    # Conversion explicite en entiers pour les dimensions
    input_size = int(input_size)
    output_size = int(output_size)
    hidden_layers = [int(layer) for layer in hidden_layers]
    batch_size = int(batch_size)
    epochs = int(epochs)

    # Normalisation des données
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    
    # Initialisation des paramètres
    parameters = initialize_parameters(input_size, output_size, hidden_layers, activation_functions)
    
    # Variables pour le early stopping
    best_val_loss = float('inf')
    best_parameters = None
    patience = 15
    patience_counter = 0
    
    # Séparation train/validation avec indices entiers
    val_size = int(0.1 * X_normalized.shape[1])
    X_train = X_normalized[:, :-val_size]
    y_train = y[:, :-val_size]
    X_val = X_normalized[:, -val_size:]
    y_val = y[:, -val_size:]
    
    # Historique pour le plotting
    losses = []
    accuracies = []
    
    # S'assurer que n_batches est un entier
    n_batches = int(np.ceil(X_train.shape[1] / batch_size))
    
    try:
        for epoch in range(epochs):
        # for epoch in tqdm(range(epochs), desc="Training epochs"):
            epoch_loss = 0
            # Mélanger les données
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            y_train_shuffled = y_train[:, permutation]
            
            # Training par batches
            for batch in range(n_batches):
                # Convertir explicitement les indices en entiers
                start_idx = int(batch * batch_size)
                end_idx = int(min((batch + 1) * batch_size, X_train.shape[1]))
                
                X_batch = X_train_shuffled[:, start_idx:end_idx]
                y_batch = y_train_shuffled[:, start_idx:end_idx]
                
                # Forward propagation
                activations = forward_propagation(X_batch, parameters)
                batch_loss = categorical_cross_entropy(y_batch, activations[f"A{len(parameters)//2}"])
                
                # Backward propagation
                gradients = back_propagation(y_batch, activations, parameters)
                
                # Update parameters avec learning rate décroissant
                current_lr = learning_rate * (1.0 / (1.0 + 0.01 * epoch))
                parameters = update_parameters(parameters, gradients, current_lr)
                
                epoch_loss += batch_loss
            
            # Moyenne de la loss sur l'epoch
            epoch_loss /= n_batches
            losses.append(epoch_loss)
            
            # Validation
            val_activations = forward_propagation(X_val, parameters)
            val_loss = categorical_cross_entropy(y_val, val_activations[f"A{len(parameters)//2}"])
            val_accuracy = accuracy_score(y_val, val_activations[f"A{len(parameters)//2}"])
            accuracies.append(val_accuracy)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_parameters = parameters.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        
    best_parameters['X_mean'] = X_mean
    best_parameters['X_std'] = X_std
    
    return best_parameters

# **Save/Load Model**
def save_model(parameters, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(parameters, file)
        print(f"\n✅ Modèle sauvegardé avec succès dans: {file_path}")
        print(f"   Taille du fichier: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde du modèle: {e}", file=sys.stderr)
        sys.exit(84)

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def main():
    if len(sys.argv) < 2 or "--help" in sys.argv:
        printHelp()
        sys.exit(0)
    
    save_file = None
    if "--save" in sys.argv:
        save_index = sys.argv.index("--save")
        if save_index + 1 < len(sys.argv):
            save_file = sys.argv[save_index + 1]
            sys.argv.pop(save_index)
            sys.argv.pop(save_index)
    
    if len(sys.argv) < 4:
        printHelp()
        sys.exit(1)
    
    mode = sys.argv[1]
    model_file = sys.argv[2]
    data_files = sys.argv[3:]

    if mode == "--train":
        X, y = load_multiple_datasets(data_files)

        config = load_config("basic_network.conf")

        input_size = config["input_size"]
        output_size = config["output_size"]
        hidden_layers = config["hidden_layers"]
        learning_rate = config["learning_rate"]
        batch_size = config.get("batch_size", 32)
        epochs = config["epochs"]
        
        activation_functions = config.get("activations_functions", 
            ['relu'] * (len(hidden_layers)) + ['softmax']) 
        
        print("\nConfiguration du réseau:")
        print(f"Tailles des couches: {[input_size] + hidden_layers + [output_size]}")
        print(f"Fonctions d'activation: {activation_functions}")
        print(f"Learning rate: {learning_rate}")
        print(f"Nombre d'epochs: {epochs}")

        parameters = train_network_simple(
            X, y, 
            input_size, 
            output_size, 
            hidden_layers, 
            learning_rate, 
            epochs,
            batch_size,
            activation_functions
        )

        save_path = save_file if save_file else model_file
        save_model(parameters, save_path)

        print(f"\n=== Modèle sauvegardé avec succès ===")
        print(f"Chemin : {save_path}")
        print("\nVous pouvez maintenant fermer la fenêtre du graphique.")
    
    elif mode == "--predict":
        try:
            parameters = load_model(model_file)
            X = load_prediction_data(data_files)
            predictions = predict(X, parameters)
            
            label_map = {
                0: "Check White",
                1: "Check Black",
                2: "Nothing",
                3: "Stalemate",
                4: "Checkmate White",
                5: "Checkmate Black"
            }
            
            for pred in predictions:
                print(label_map.get(pred, "Unknown"))
                
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}", file=sys.stderr)
            sys.exit(84)

    else:
        print("Invalid mode. Use --train or --predict.")
        sys.exit(84)

# **Main Execution**
if __name__ == "__main__":
    main()