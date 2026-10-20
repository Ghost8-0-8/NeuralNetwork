import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import os
from typing import List
from my_torch_analyzer import (
    load_multiple_datasets,
    train_network_simple,
    save_model
)

def print_help():
    """Affiche l'aide du programme de benchmark."""
    print("USAGE")
    print("     ./my_benchmark.py [--save SAVEFILE] FILE...\n")
    print("DESCRIPTION")
    print("     Compare different neural network configurations on the dataset(s).")
    print("     --save  Save all trained models and the best performing model")
    print("     FILE    One or more files containing training data\n")

def run_benchmark(data_files: List[str], save_file: str = None):
    """Exécute le benchmark complet"""
    
    print("\nChargement des données d'entraînement...")
    X, y = load_multiple_datasets(data_files)
    print(f"Données chargées: {X.shape[1]} exemples")
    
    configs = [
        {
            "input_size": 64,
            "hidden_layers": [256, 128, 64, 32, 16, 8, 4],
            "output_size": 6,
            "learning_rate": 0.01,
            "epochs": 50,
            "batch_size": 32,
            "activations_functions": ['relu']*7 + ['softmax'],
            "name": "ReLU"
        },
        {
            "input_size": 64,
            "hidden_layers": [256, 128, 64, 32, 16, 8, 4],
            "output_size": 6,
            "learning_rate": 0.01,
            "epochs": 50,
            "batch_size": 32,
            "activations_functions": ['linear']*7 + ['softmax'],
            "name": "Linear"
        },
        {
            "input_size": 64,
            "hidden_layers": [256, 128, 64, 32, 16, 8, 4],
            "output_size": 6,
            "learning_rate": 0.01,
            "epochs": 50,
            "batch_size": 32,
            "activations_functions": ['tanh']*7 + ['softmax'],
            "name": "Tanh"
        },
        {
            "input_size": 64,
            "hidden_layers": [256, 128, 64, 32, 16, 8, 4],
            "output_size": 6,
            "learning_rate": 0.01,
            "epochs": 50,
            "batch_size": 32,
            "activations_functions": ['sigmoid']*7 + ['softmax'],
            "name": "Sigmoid"
        }
    ]
    
    # Configuration du style des graphiques
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (loss_plot, acc_plot) = plt.subplots(1, 2, figsize=(15, 5))
    
    all_results = []
    best_model = None
    best_accuracy = -1
    best_config = None
    
    # Test de chaque configuration
    for config in configs:
        print(f"\nTest de la configuration: {config['name']}")
        print("Architecture:", end=" ")
        print(f"{config['input_size']} -> {' -> '.join(map(str, config['hidden_layers']))} -> {config['output_size']}")
        print(f"Fonctions d'activation: {config['activations_functions']}")
        
        try:
            # Entraînement du modèle
            training_history = {
                'losses': [],
                'accuracies': []
            }
            
            def history_callback(epoch, loss, accuracy):
                training_history['losses'].append(loss)
                training_history['accuracies'].append(accuracy)
            
            parameters = train_network_simple(
                X, y,
                config["input_size"],
                config["output_size"],
                config["hidden_layers"],
                config["learning_rate"],
                config["epochs"],
                config["batch_size"],
                config["activations_functions"],
                history_callback
            )
            
            # Vérifier si l'entraînement a produit des résultats valides
            if training_history['losses'] and training_history['accuracies']:
                # Filtrer les valeurs inf/nan
                current_losses = [l for l in training_history['losses'] if not np.isinf(l) and not np.isnan(l)]
                current_accuracies = [a for a in training_history['accuracies'] if not np.isinf(a) and not np.isnan(a)]
                
                if current_losses and current_accuracies:
                    epochs = range(1, len(current_losses) + 1)
                    
                    # Tracer les courbes
                    loss_plot.plot(epochs, current_losses, label=config["name"], linewidth=2)
                    acc_plot.plot(epochs, current_accuracies, label=config["name"], linewidth=2)
                    
                    final_accuracy = current_accuracies[-1]
                    final_loss = current_losses[-1]
                    
                    # Sauvegarder le modèle
                    if save_file:
                        base_name = os.path.splitext(save_file)[0]
                        model_save_path = f"{base_name}_{config['name'].lower()}.nn"
                        save_model(parameters, model_save_path)
                        print(f"Modèle sauvegardé dans: {model_save_path}")
                    
                    # Mettre à jour le meilleur modèle
                    if final_accuracy > best_accuracy:
                        best_accuracy = final_accuracy
                        best_model = parameters
                        best_config = config["name"]
                    
                    all_results.append({
                        "name": config["name"],
                        "final_accuracy": final_accuracy,
                        "final_loss": final_loss
                    })
                    
                    print(f"Final Loss: {final_loss:.4f}")
                    print(f"Final Accuracy: {final_accuracy:.4f}")
                else:
                    print(f"Warning: Pas de métriques valides pour {config['name']}")
            else:
                print(f"Warning: Pas de métriques pour {config['name']}")
                
        except Exception as e:
            print(f"Erreur lors du test de {config['name']}: {str(e)}")
            continue
    
    # Configuration des graphiques
    if all_results:
        loss_plot.set_title('Evolution de la Loss', pad=20, size=12, weight='bold')
        loss_plot.set_xlabel('Epochs')
        loss_plot.set_ylabel('Loss')
        loss_plot.legend()
        loss_plot.grid(True)
        
        acc_plot.set_title('Evolution de l\'Accuracy', pad=20, size=12, weight='bold')
        acc_plot.set_xlabel('Epochs')
        acc_plot.set_ylabel('Accuracy')
        acc_plot.legend()
        acc_plot.grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        
        # Sauvegarder les résultats
        with open('benchmark_results.txt', 'w') as f:
            f.write("Résultats du Benchmark\n")
            f.write("====================\n\n")
            
            for result in all_results:
                f.write(f"\nConfiguration: {result['name']}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Loss finale: {result['final_loss']:.4f}\n")
                f.write(f"Accuracy finale: {result['final_accuracy']:.4f}\n")
            
            if best_config:
                f.write("\nMeilleure Configuration\n")
                f.write("=====================\n")
                f.write(f"Type: {best_config}\n")
                f.write(f"Accuracy: {best_accuracy:.4f}\n")
        
        # Sauvegarder le meilleur modèle
        if save_file and best_model:
            best_model_path = f"{os.path.splitext(save_file)[0]}_best.nn"
            save_model(best_model, best_model_path)
            print(f"\nMeilleur modèle sauvegardé dans: {best_model_path}")
        
        print("\n=== Résultats du Benchmark ===")
        if best_config:
            print(f"Meilleure configuration: {best_config}")
            print(f"Meilleure accuracy: {best_accuracy:.4f}")
        print(f"\nGraphiques sauvegardés dans: benchmark_results.png")
        print(f"Résultats détaillés dans: benchmark_results.txt")
    else:
        print("\nAucun résultat valide n'a été obtenu pendant le benchmark.")
    
    plt.show()

if __name__ == "__main__":
    if "--help" in sys.argv or len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    save_file = None
    if "--save" in sys.argv:
        save_index = sys.argv.index("--save")
        if save_index + 1 >= len(sys.argv):
            print("Error: Missing filename after --save")
            sys.exit(84)
        save_file = sys.argv[save_index + 1]
        sys.argv.pop(save_index)
        sys.argv.pop(save_index)

    data_files = sys.argv[1:]
    if not data_files:
        print("Error: No data files provided")
        sys.exit(84)

    try:
        run_benchmark(data_files, save_file)
    except KeyboardInterrupt:
        print("\nBenchmark interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du benchmark: {e}", file=sys.stderr)
        sys.exit(84)