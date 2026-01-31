"""
Data preprocessing module for chess AI.

This module handles:
1. PGN file parsing and filtering
2. Board encoding (119 planes AlphaZero style)
3. Move indexing (0-4095)
4. Dataset creation and saving
"""

import os
import io
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import chess
import chess.pgn
import h5py
from pathlib import Path

try:
    from datasets import load_from_disk, Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    Dataset = None

logger = logging.getLogger(__name__)

# Move encoding: 
# - Regular moves: from_square * 64 + to_square (4096 possibilities)
# - Promotions: encoded separately using promotion piece type
# Total: 4096 regular + 4 promotions per from/to = effectively 4096 (promotions handled via legal move masking)
MAX_MOVES = 4096


def load_pgn_file(path: str, min_elo: int = 2600) -> List[chess.pgn.Game]:
    """
    Load and filter PGN games by minimum ELO rating.
    
    Args:
        path: Path to PGN file
        min_elo: Minimum ELO rating for both players
        
    Returns:
        List of chess.pgn.Game objects that meet the criteria
        
    Raises:
        ValueError: If path is invalid or file cannot be read
    """
    if not os.path.exists(path):
        raise ValueError(f"PGN file not found: {path}")
    
    games = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Extract ELO ratings
                white_elo = game.headers.get("WhiteElo", "")
                black_elo = game.headers.get("BlackElo", "")
                
                # Try to parse ELOs
                try:
                    white_elo_int = int(white_elo) if white_elo.isdigit() else 0
                    black_elo_int = int(black_elo) if black_elo.isdigit() else 0
                except (ValueError, AttributeError):
                    continue
                
                # Filter by minimum ELO
                if white_elo_int >= min_elo and black_elo_int >= min_elo:
                    # Filter out unusual terminations
                    termination = game.headers.get("Termination", "").lower()
                    is_unusual = any(x in termination for x in [
                        "time", "abandoned", "rules", "illegal", 
                        "forfeit", "infraction"
                    ])
                    
                    if not is_unusual:
                        games.append(game)
                        
    except Exception as e:
        logger.error(f"Error loading PGN file {path}: {e}")
        raise
    
    logger.info(f"Loaded {len(games)} games from {path} (min_elo={min_elo})")
    return games


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode chess board to neural network input (119 planes, 8x8).
    
    Board Encoding (119 planes):
    - 12 planes: current position (6 piece types × 2 colors)
    - 12 planes: previous position (for move history)
    - 4 planes: castling rights
    - 1 plane: en passant
    - 1 plane: color to move
    - 1 plane: move count
    - 2 planes: repetition counters
    - 86 planes: additional features (optional, reserved for future use)
    
    Args:
        board: python-chess Board object
        
    Returns:
        numpy array of shape (119, 8, 8)
        
    Raises:
        ValueError: if board is invalid
    """
    if not isinstance(board, chess.Board):
        raise ValueError(f"Expected chess.Board, got {type(board)}")
    
    try:
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Piece types: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                      chess.ROOK, chess.QUEEN, chess.KING]
        
        # Planes 0-11: Current position (6 piece types × 2 colors)
        plane_idx = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_types:
                piece_mask = board.pieces_mask(piece_type, color)
                for square in chess.scan_forward(piece_mask):
                    row, col = divmod(square, 8)
                    # Flip row for correct orientation (rank 0 = bottom)
                    planes[plane_idx, 7 - row, col] = 1.0
                plane_idx += 1
        
        # Planes 12-23: Previous position (if available)
        # For now, we'll use zeros (can be enhanced with move history)
        # This would require maintaining board history
        
        # Planes 24-27: Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[24, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[25, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[26, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[27, :, :] = 1.0
        
        # Plane 28: En passant square
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            planes[28, 7 - row, col] = 1.0
        
        # Plane 29: Color to move (1.0 for white, 0.0 for black)
        if board.turn == chess.WHITE:
            planes[29, :, :] = 1.0
        
        # Plane 30: Move count (normalized)
        # Full move number is stored in board.fullmove_number
        planes[30, :, :] = min(board.fullmove_number / 100.0, 1.0)
        
        # Planes 31-32: Repetition counters
        # python-chess doesn't directly expose this, so we approximate
        # by checking if position is repeated
        if board.is_repetition(2):
            planes[31, :, :] = 1.0
        if board.is_repetition(3):
            planes[32, :, :] = 1.0
        
        # Planes 33-118: Reserved for additional features
        # Could include: piece-square tables, attack maps, etc.
        
        return planes
        
    except Exception as e:
        logger.error(f"Board encoding failed: {e}")
        raise


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Convert a chess move to an index (0-4095).
    
    Encoding: from_square * 64 + to_square
    Note: Promotions are encoded the same way (promotion piece info is implicit
    in the move itself and will be handled during legal move generation).
    
    Args:
        move: chess.Move object
        board: chess.Board object (for validation)
        
    Returns:
        Integer index in range [0, 4095]
        
    Raises:
        ValueError: if move is invalid
    """
    if not isinstance(move, chess.Move):
        raise ValueError(f"Expected chess.Move, got {type(move)}")
    
    if move.from_square is None or move.to_square is None:
        raise ValueError("Move must have from_square and to_square")
    
    # Basic encoding: from_square * 64 + to_square
    # This works for regular moves, castling, and promotions
    # The actual promotion piece is part of the move object and will be
    # preserved when converting back via legal move matching
    move_idx = move.from_square * 64 + move.to_square
    
    # Clamp to valid range
    return min(move_idx, MAX_MOVES - 1)


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Convert an index back to a chess move.
    
    This function finds the legal move that matches the index.
    For promotions, it will return the first matching legal move
    (typically queen promotion, which is most common).
    
    Args:
        index: Integer index in range [0, 4095]
        board: chess.Board object (for move validation)
        
    Returns:
        chess.Move object, or None if index is invalid or no matching legal move
    """
    if index < 0 or index >= MAX_MOVES:
        return None
    
    from_square = index // 64
    to_square = index % 64
    
    if from_square >= 64 or to_square >= 64:
        return None
    
    # Find matching legal move
    # Check all legal moves and find one that matches this index
    for move in board.legal_moves:
        if move.from_square == from_square and move.to_square == to_square:
            return move
    
    return None


def process_game_from_datasets(game_data: Dict) -> List[Dict]:
    """
    Process a game from datasets format (with moves string).
    
    Args:
        game_data: Dictionary with game data from datasets (must have 'moves' key)
        
    Returns:
        List of position dictionaries
    """
    import chess.pgn
    from io import StringIO
    
    # Reconstruct game from moves string
    moves_str = game_data.get('moves', '')
    if not moves_str:
        return []
    
    # Parse moves
    pgn_string = f"[Event \"?\"]\n[Site \"?\"]\n[Date \"?\"]\n[Round \"?\"]\n[White \"?\"]\n[Black \"?\"]\n[Result \"*\"]\n\n{moves_str}"
    game = chess.pgn.read_game(StringIO(pgn_string))
    
    if game is None:
        return []
    
    # Add headers from game_data
    for key, value in game_data.items():
        if key != 'moves' and value:
            game.headers[key] = str(value)
    
    return process_game(game)


def process_game(game: chess.pgn.Game) -> List[Dict]:
    """
    Process a single game and extract training positions.
    
    Args:
        game: chess.pgn.Game object
        
    Returns:
        List of dictionaries, each containing:
        - position: encoded board (119, 8, 8)
        - move: move index (0-4095)
        - value: game outcome from current player's perspective (-1, 0, 1)
        - move_number: move number in game
        - player: chess.WHITE or chess.BLACK
    """
    board = game.board()
    positions = []
    
    # Determine game outcome
    result = game.headers.get("Result", "*")
    if result == "1-0":
        white_outcome = 1.0
        black_outcome = -1.0
    elif result == "0-1":
        white_outcome = -1.0
        black_outcome = 1.0
    else:  # Draw
        white_outcome = 0.0
        black_outcome = 0.0
    
    move_number = 0
    prev_board = None
    
    for node in game.mainline():
        move = node.move
        
        # Encode current position
        position = encode_board(board)
        
        # Store previous position for history (if available)
        if prev_board is not None:
            prev_position = encode_board(prev_board)
            # Copy previous position planes (12-23)
            position[12:24] = prev_position[0:12]
        
        # Get move index
        try:
            move_idx = move_to_index(move, board)
        except ValueError:
            logger.warning(f"Invalid move in game: {move}")
            board.push(move)
            continue
        
        # Determine value from current player's perspective
        if board.turn == chess.WHITE:
            value = white_outcome
        else:
            value = black_outcome
        
        positions.append({
            'position': position,
            'move': move_idx,
            'value': value,
            'move_number': move_number,
            'player': board.turn
        })
        
        # Update board
        prev_board = board.copy()
        board.push(move)
        move_number += 1
    
    return positions


def save_dataset(data: List[Dict], output_path: str):
    """
    Save processed dataset to HDF5 format.
    
    Args:
        data: List of position dictionaries
        output_path: Path to save HDF5 file
        
    Raises:
        IOError: if file cannot be written
    """
    if not data:
        raise ValueError("Cannot save empty dataset")
    
    try:
        # Prepare arrays
        positions = np.stack([d['position'] for d in data])
        moves = np.array([d['move'] for d in data], dtype=np.int32)
        values = np.array([d['value'] for d in data], dtype=np.float32)
        move_numbers = np.array([d['move_number'] for d in data], dtype=np.int32)
        players = np.array([d['player'] for d in data], dtype=np.int32)
        
        # Save to HDF5
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('positions', data=positions, compression='gzip')
            f.create_dataset('moves', data=moves, compression='gzip')
            f.create_dataset('values', data=values, compression='gzip')
            f.create_dataset('move_numbers', data=move_numbers, compression='gzip')
            f.create_dataset('players', data=players, compression='gzip')
            
            # Store metadata
            f.attrs['num_samples'] = len(data)
            f.attrs['board_shape'] = positions.shape[1:]
            f.attrs['max_moves'] = MAX_MOVES
        
        logger.info(f"Saved {len(data)} positions to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving dataset to {output_path}: {e}")
        raise


def get_project_root(script_path: Optional[str] = None) -> str:
    """
    Get project root directory.
    
    Args:
        script_path: Path to current script (optional, auto-detected if None)
        
    Returns:
        Absolute path to project root
    """
    if script_path is None:
        # Try to find project root from common entry points
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            script_path = frame.f_back.f_globals.get('__file__', __file__)
        else:
            script_path = __file__
    
    # Normalize path
    script_path = os.path.abspath(script_path)
    
    # If it's a file, get its directory
    if os.path.isfile(script_path):
        script_dir = os.path.dirname(script_path)
    else:
        script_dir = script_path
    
    # Determine how many levels to go up based on current location
    # If we're in chess-ai/data/, go up 2 levels
    # If we're in chess-ai/scripts/, go up 2 levels
    # If we're in chess-ai/, go up 1 level
    
    # Check if we're already at project root (has data/ and chess-ai/ directories)
    if os.path.exists(os.path.join(script_dir, "data")) and os.path.exists(os.path.join(script_dir, "chess-ai")):
        return script_dir
    
    # Check if we're in chess-ai/ subdirectory
    if "chess-ai" in script_dir:
        # Go up until we find project root
        current = script_dir
        while current != os.path.dirname(current):  # Not at root
            if os.path.exists(os.path.join(current, "data")) and os.path.exists(os.path.join(current, "chess-ai")):
                return current
            current = os.path.dirname(current)
        # Fallback: go up 2 levels from chess-ai subdirectory
        return os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Fallback: assume we need to go up 2 levels
    return os.path.abspath(os.path.join(script_dir, "../.."))


def get_dataset_path(dataset_name: str = "chess_dataset", script_path: Optional[str] = None) -> str:
    """
    Get path to dataset in data/processed folder.
    
    Args:
        dataset_name: Name of dataset folder
        script_path: Path to current script (optional)
        
    Returns:
        Absolute path to dataset
    """
    project_root = get_project_root(script_path)
    dataset_path = os.path.join(project_root, "data", "processed", dataset_name)
    return dataset_path


def load_dataset_from_datasets(dataset_path: str) -> 'Dataset':
    """
    Load dataset from HuggingFace datasets format.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        HuggingFace Dataset object
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets")
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded dataset from {dataset_path}: {len(dataset)} samples")
        logger.info(f"Dataset columns: {dataset.column_names}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset from {dataset_path}: {e}")
        raise


def load_dataset(input_path: str, format: str = 'auto') -> Dict[str, np.ndarray]:
    """
    Load processed dataset from HDF5 or datasets format.
    
    Args:
        input_path: Path to dataset file or directory
        format: Format type ('hdf5', 'datasets', or 'auto' for auto-detect)
        
    Returns:
        Dictionary with keys: 'positions', 'moves', 'values', 'move_numbers', 'players'
        (for HDF5) or Dataset object (for datasets format)
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Dataset path not found: {input_path}")
    
    # Auto-detect format
    if format == 'auto':
        if os.path.isdir(input_path):
            # Check if it's a datasets directory
            if os.path.exists(os.path.join(input_path, "dataset_info.json")):
                format = 'datasets'
            else:
                format = 'hdf5'
        elif input_path.endswith('.h5') or input_path.endswith('.hdf5'):
            format = 'hdf5'
        else:
            format = 'datasets'
    
    if format == 'datasets':
        # Return Dataset object directly
        return load_dataset_from_datasets(input_path)
    
    # HDF5 format
    try:
        with h5py.File(input_path, 'r') as f:
            data = {
                'positions': np.array(f['positions']),
                'moves': np.array(f['moves']),
                'values': np.array(f['values']),
                'move_numbers': np.array(f['move_numbers']),
                'players': np.array(f['players'])
            }
            
            logger.info(f"Loaded dataset from {input_path}: {len(data['positions'])} samples")
            return data
            
    except Exception as e:
        logger.error(f"Error loading dataset from {input_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test board encoding
    board = chess.Board()
    encoded = encode_board(board)
    print(f"Encoded board shape: {encoded.shape}")
    print(f"Encoded board dtype: {encoded.dtype}")
    
    # Test move encoding
    move = list(board.legal_moves)[0]
    move_idx = move_to_index(move, board)
    print(f"Move {move} -> index {move_idx}")
    
    # Test index to move
    recovered = index_to_move(move_idx, board)
    print(f"Index {move_idx} -> move {recovered}")
