import numpy as np
import sys
import os

def printHelp() -> None:
    """Affiche l'aide du programme."""
    print("USAGE")
    print("     ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n")
    print("DESCRIPTION")
    print("     config_file_i       Configuration file containing description of a neural network we want to generate.")
    print("     nb_i                Number of neural networks to generate based on the configuration file.")


def parse_conf_file(config_file):
    """
    Lit et parse le fichier .conf
    """
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            key, value = line.split('=')
            key, value = key.strip(), value.strip()
            
            if key == 'activations_functions':
                config[key] = [v.strip() for v in value.split(',')]
            elif ',' in value:
                config[key] = [int(v) for v in value.split(',')]
            elif value.replace('.', '', 1).isdigit():
                config[key] = float(value) if '.' in value else int(value)
            else:
                config[key] = value
    
    # Verifie le nb de hidden layer avec le nb de fontion d'activation
    if 'activations_functions' in config:
        expected_length = len(config['hidden_layers']) + 1 
        actual_length = len(config['activations_functions'])
        if actual_length != expected_length:
            raise ValueError(f"Number of activation functions ({actual_length}) must match number of layers ({expected_length})")
    
    return config

def get_activation_function(name):
    """
    Recupere la fonction d'activation
    """
    functions = {
        'relu': lambda x: np.maximum(0, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'tanh': lambda x: np.tanh(x),
        'linear': lambda x: x
    }
    
    if name.lower() not in functions:
        raise ValueError(f"Unsupported activation function: {name}")
    
    return functions[name.lower()]

def generate_network_from_config(config, output_file):
    """
    Genere le reseau de neurone
    """
    dimensions = [config['input_size']] + config['hidden_layers'] + [config['output_size']]
    parameters = {}
    
    # Stock les poids et les biais
    for i in range(1, len(dimensions)):
        parameters[f'W{i}'] = np.random.randn(dimensions[i], dimensions[i-1]) * 0.01
        parameters[f'b{i}'] = np.zeros((dimensions[i], 1))
    
    # Stock fonctions d'activation
    if 'activations_functions' in config:
        parameters['activation_functions'] = config['activations_functions']
    
    with open(output_file, 'w') as f:
        f.write(str(parameters))
    
    print(f"Network generated and saved to {output_file}")


def main():
    """
    Main du generator.
    """
    try:
        args = sys.argv[1:]
        if '--help' in args:
            printHelp()
            return
        
        if len(args) < 2 or len(args) % 2 != 0:
            print("Error: Invalid number of arguments. Use --help for usage instructions.")
            sys.exit(84)
        
        for i in range(0, len(args), 2):
            config_file = args[i]
            try:
                nb_networks = int(args[i + 1])
                if nb_networks <= 0:
                    raise ValueError("Number of networks must be a positive integer.")
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(84)
            
            if not os.path.exists(config_file):
                print(f"Error: Configuration file '{config_file}' not found.")
                sys.exit(84)
            
            try:
                config = parse_conf_file(config_file)
            except Exception as e:
                print(f"Error reading configuration file '{config_file}': {e}")
                sys.exit(84)
            
            for j in range(nb_networks):
                output_file = f"{os.path.splitext(config_file)[0]}_{j + 1}.nn"
                generate_network_from_config(config, output_file)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(84)

if __name__ == "__main__":
    main()