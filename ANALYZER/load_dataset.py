import os
import sys
import numpy as np
import multiprocessing
from typing import List, Tuple
import random
from parsing import *

def process_file_chunk(args):
    """
    Traite un chunk de fichiers en parallèle
    """
    file_paths, output_dir = args
    lines = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                file_lines = f.readlines()
                lines.extend((line.strip(), i) for i, line in enumerate(file_lines))
        except Exception as e:
            print(f"\nErreur lors de la lecture de {file_path}: {e}", file=sys.stderr)
            continue
            
    return lines

def shuffle_dataset_files(file_paths: List[str], output_dir: str = "temp_shuffled_data") -> List[str]:
    """
    Version avec multiprocessing pour le mélange des fichiers
    """
    os.makedirs(output_dir, exist_ok=True)
    n_processes = multiprocessing.cpu_count() - 1
    print(f"\nUtilisation de {n_processes} processus pour le shuffle")
    
    # Diviser les fichiers en chunks pour les processus
    chunk_size = len(file_paths) // n_processes + 1
    file_chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    
    # Préparer les arguments pour chaque processus
    process_args = [(chunk, output_dir) for chunk in file_chunks]
    
    # Créer un pool de processus et traiter les fichiers en parallèle
    all_lines = []
    total_lines = 0
    
    print("\nLecture parallèle des fichiers...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(process_file_chunk, process_args)
        
        # Combiner tous les résultats
        for chunk_lines in results:
            total_lines += len(chunk_lines)
            all_lines.extend(chunk_lines)
    
    if not all_lines:
        print("Erreur: Aucune donnée n'a été chargée!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nMélange de {total_lines} lignes...")
    random.shuffle(all_lines)
    
    # Créer le fichier de sortie mélangé
    output_file = os.path.join(output_dir, "shuffled_data.txt")
    print(f"Écriture du fichier mélangé: {output_file}")
    
    with open(output_file, 'w') as f:
        for line, _ in all_lines:
            f.write(line + '\n')
    
    return [output_file]

def load_prediction_data(file_paths):
    """
    Charge les données pour la prédiction
    """
    X = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Erreur : Le fichier {file_path} n'existe pas.", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    fen = line.strip()
                    X.append(fen_to_input(fen))
                    
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not X:
        print("Erreur: Aucune donnée n'a été chargée!", file=sys.stderr)
        sys.exit(1)
        
    return np.array(X).T

def load_config(config_file):
    """
    Charge la configuration à partir d'un fichier texte.
    Assure que epochs est un entier.
    """
    config = {}
    with open(config_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if key == 'activations_functions':
                config[key] = [v.strip() for v in value.split(',')]
            elif key == 'epochs':
                config[key] = int(float(value)) 
            elif ',' in value:
                config[key] = [int(v) for v in value.split(',')]
            elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                config[key] = float(value)
            else:
                config[key] = value

    if 'activations_functions' in config:
        valid_activations = {'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu'}
        for act in config['activations_functions']:
            if act.lower() not in valid_activations:
                raise ValueError(f"Fonction d'activation non supportée : {act}")
        
        expected_length = len(config.get('hidden_layers', [])) + 1
        if len(config['activations_functions']) != expected_length:
            raise ValueError(f"Le nombre de fonctions d'activation ({len(config['activations_functions'])}) "
                           f"doit correspondre au nombre de couches ({expected_length})")

    return config

def load_multiple_datasets(file_paths: List[str], shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset and shuffle
    """
    label_map = {
        "Check White": 0,
        "Check Black": 1,
        "Nothing": 2,
        "Stalemate": 3,
        "Checkmate White": 4,
        "Checkmate Black": 5
    }
    
    if shuffle:
        temp_dir = "temp_shuffled_data"
        file_paths = shuffle_dataset_files(file_paths, temp_dir)
    
    try:
        stats = {label: 0 for label in label_map}
        invalid_lines = 0
        total_lines = 0
        
        X, y = [], []
        print("\nChargement des données...")
        
        for file_path in file_paths:
                if not os.path.exists(file_path):
                    print(f"\nErreur : Le fichier {file_path} n'existe pas.")
                    continue
                    
                try:
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                    total_lines += len(lines)
                    
                    valid_entries = 0
                    print(f"\nTraitement de {file_path}")
                    print(f"Nombre de lignes : {len(lines)}")
                    
                    for line in lines:
                        fen, label = validate_fen_line(line, label_map)
                        if fen is not None and label is not None:
                            X.append(fen_to_input(fen))
                            y.append(label)
                            stats[list(label_map.keys())[label]] += 1
                            valid_entries += 1
                        else:
                            invalid_lines += 1
                            
                    print(f"Entrées valides : {valid_entries}")
                    
                except Exception as e:
                    print(f"\nErreur lors du traitement de {file_path}: {e}")
        
        if not X:
            print("\nErreur: Aucune donnée valide n'a été chargée!")
            sys.exit(1)
        
        X = np.array(X).T
        y = np.eye(len(label_map))[y].T
        
        if shuffle:
            print("\nMélange final des données...")
            permutation = np.random.permutation(X.shape[1])
            X = X[:, permutation]
            y = y[:, permutation]
        
        print("\nStatistiques de chargement:")
        print(f"Total des lignes lues: {total_lines}")
        print(f"Total des positions valides chargées: {X.shape[1]}")
        print(f"Lignes invalides ignorées: {invalid_lines}")
        print("\nDistribution des labels:")
        for label, count in stats.items():
            percentage = (count / X.shape[1]) * 100 if X.shape[1] > 0 else 0
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return X, y
        
    finally:
        # Nettoyage des fichiers temporaires
        if shuffle:
            try:
                for file_path in file_paths:
                    try:
                        os.remove(file_path)
                    except:
                        pass
                os.rmdir(temp_dir)
            except:
                pass
