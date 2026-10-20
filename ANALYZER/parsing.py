import numpy as np

# **FEN to Input Vector**
def fen_to_input(fen):
    """
    conversion FEN avec normalisation
    """
    piece_encoding = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # Blanc
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Noir
    }
    
    board_vector = []
    fen_board = fen.split(' ')[0]
    
    for char in fen_board:
        if char.isdigit():
            board_vector.extend([0] * int(char))
        elif char in piece_encoding:
            # Normalisation entre -1 et 1
            board_vector.append(piece_encoding[char] / 6.0)
    
    # Compléter à 64 cases
    board_vector.extend([0] * (64 - len(board_vector)))
    return np.array(board_vector, dtype=np.float32)

def validate_fen_line(line, label_map):
    """
    Valide une ligne contenant un FEN et une étiquette.
    Renvoie le FEN et son label si valide, sinon None.
    """
    try:
        parts = line.strip().split(' ')
        
        if len(parts) < 7:
            print(f"Warning: Invalid FEN format in line: {line}")
            return None, None
        
        label_str = parts[-1]
        if len(parts) >= 8 and f"{parts[-2]} {parts[-1]}" in label_map:
            label_str = f"{parts[-2]} {parts[-1]}"
            fen = ' '.join(parts[:-2])
        else:
            fen = ' '.join(parts[:-1])
        
        if label_str not in label_map:
            print(f"Warning: Unknown label '{label_str}'")
            return None, None
        
        fen_parts = fen.split(' ')
        if len(fen_parts) != 6:
            print(f"Warning: Invalid FEN structure: {fen}")
            return None, None
            
        return fen, label_map[label_str]
    
    except Exception as e:
        print(f"Error processing line: {line}. Error: {e}")
        return None, None