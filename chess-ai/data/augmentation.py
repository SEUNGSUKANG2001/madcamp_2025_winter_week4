"""
Chess-specific data augmentation module.

Unlike Go, chess is not fully symmetric due to castling rules and pawn structure.
This module implements safe augmentation techniques that preserve game legality.
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import chess
import chess.pgn

from data.preprocessing import encode_board, move_to_index, index_to_move

logger = logging.getLogger(__name__)


def is_symmetric_position(board: chess.Board) -> bool:
    """
    Check if a position can be safely flipped horizontally.
    
    Horizontal flip is safe only if:
    1. No castling rights exist (or they're symmetric)
    2. No en passant target
    3. Position is not in endgame (where symmetry matters less)
    
    Args:
        board: chess.Board object
        
    Returns:
        True if position can be safely flipped
    """
    # Cannot flip if castling rights exist (not symmetric)
    if (board.has_kingside_castling_rights(chess.WHITE) or
        board.has_queenside_castling_rights(chess.WHITE) or
        board.has_kingside_castling_rights(chess.BLACK) or
        board.has_queenside_castling_rights(chess.BLACK)):
        return False
    
    # Cannot flip if en passant is available (square position matters)
    if board.ep_square is not None:
        return False
    
    # Generally safe for middlegame positions
    # Endgame positions might have asymmetric king positions
    if board.fullmove_number < 20:
        return True
    
    # For later positions, be more conservative
    return False


def flip_board_horizontal(board: chess.Board) -> chess.Board:
    """
    Create a horizontally flipped version of the board.
    
    This swaps the a-file and h-file, b-file and g-file, etc.
    Note: This is only valid for positions without castling or en passant.
    
    Args:
        board: chess.Board object
        
    Returns:
        New chess.Board object with flipped position
    """
    flipped = board.copy()
    
    # Create a new empty board
    new_board = chess.Board()
    new_board.clear()
    
    # Flip all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Calculate flipped square: file 0->7, 1->6, etc.
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            flipped_file = 7 - file
            flipped_square = chess.square(flipped_file, rank)
            new_board.set_piece_at(flipped_square, piece)
    
    # Copy other properties
    new_board.turn = board.turn
    new_board.fullmove_number = board.fullmove_number
    new_board.halfmove_clock = board.halfmove_clock
    
    return new_board


def flip_move_horizontal(move: chess.Move) -> chess.Move:
    """
    Flip a move horizontally.
    
    Args:
        move: chess.Move object
        
    Returns:
        Horizontally flipped chess.Move
    """
    if move.from_square is None or move.to_square is None:
        return move
    
    # Flip squares
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    flipped_from = chess.square(7 - from_file, from_rank)
    flipped_to = chess.square(7 - to_file, to_rank)
    
    return chess.Move(flipped_from, flipped_to, promotion=move.promotion)


def invert_color(board: chess.Board) -> chess.Board:
    """
    Invert the colors of all pieces (swap white and black).
    
    This creates a position from the opponent's perspective.
    Note: This changes the turn, so the value labels must also be inverted.
    
    Args:
        board: chess.Board object
        
    Returns:
        New chess.Board object with inverted colors
    """
    inverted = board.copy()
    
    # Create new empty board
    new_board = chess.Board()
    new_board.clear()
    
    # Invert all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Swap color
            inverted_color = not piece.color
            inverted_piece = chess.Piece(piece.piece_type, inverted_color)
            new_board.set_piece_at(square, inverted_piece)
    
    # Invert turn
    new_board.turn = not board.turn
    
    # Copy other properties
    new_board.fullmove_number = board.fullmove_number
    new_board.halfmove_clock = board.halfmove_clock
    
    # Invert castling rights (swap white and black)
    new_castling = chess.BB_EMPTY
    if board.has_kingside_castling_rights(chess.WHITE):
        new_castling |= chess.BB_H8  # White's rights become black's
    if board.has_queenside_castling_rights(chess.WHITE):
        new_castling |= chess.BB_A8
    if board.has_kingside_castling_rights(chess.BLACK):
        new_castling |= chess.BB_H1  # Black's rights become white's
    if board.has_queenside_castling_rights(chess.BLACK):
        new_castling |= chess.BB_A1
    new_board.castling_rights = new_castling
    
    # Invert en passant (if exists)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        # En passant is always on rank 3 or 6, invert to opposite rank
        new_ep_rank = 5 if ep_rank == 2 else 2
        new_ep_square = chess.square(ep_file, new_ep_rank)
        new_board.ep_square = new_ep_square
    
    return new_board


def augment_position(board: chess.Board, move: chess.Move) -> List[Tuple[chess.Board, chess.Move, float]]:
    """
    Generate augmented versions of a position and move.
    
    Returns list of (augmented_board, augmented_move, value_multiplier) tuples.
    The value_multiplier is used to adjust the game outcome value:
    - 1.0 for original position
    - -1.0 for color-inverted position (perspective change)
    
    Args:
        board: chess.Board object
        move: chess.Move object
        
    Returns:
        List of (board, move, value_multiplier) tuples
    """
    augmentations = []
    
    # Original position
    augmentations.append((board.copy(), move, 1.0))
    
    # Color inversion (always safe)
    inverted_board = invert_color(board)
    inverted_move = chess.Move(move.from_square, move.to_square, promotion=move.promotion)
    # Note: Move squares don't change, but perspective does
    augmentations.append((inverted_board, inverted_move, -1.0))
    
    # Horizontal flip (only if safe)
    if is_symmetric_position(board):
        flipped_board = flip_board_horizontal(board)
        flipped_move = flip_move_horizontal(move)
        augmentations.append((flipped_board, flipped_move, 1.0))
        
        # Combined: flip + invert
        flipped_inverted = invert_color(flipped_board)
        flipped_inverted_move = flip_move_horizontal(inverted_move)
        augmentations.append((flipped_inverted, flipped_inverted_move, -1.0))
    
    return augmentations


def augment_game_data(position: np.ndarray, move_idx: int, value: float, 
                     board: chess.Board) -> List[Dict]:
    """
    Augment a single training sample.
    
    Args:
        position: Encoded board position (33, 8, 8)
        move_idx: Move index
        value: Game outcome value
        board: chess.Board object (for move recovery)
        
    Returns:
        List of augmented training samples
    """
    # Recover move from index
    move = index_to_move(move_idx, board)
    if move is None:
        return [{'position': position, 'move': move_idx, 'value': value}]
    
    # Get augmentations
    augmentations = augment_position(board, move)
    
    augmented_samples = []
    for aug_board, aug_move, value_mult in augmentations:
        try:
            aug_position = encode_board(aug_board)
            aug_move_idx = move_to_index(aug_move, aug_board)
            aug_value = value * value_mult
            
            augmented_samples.append({
                'position': aug_position,
                'move': aug_move_idx,
                'value': aug_value
            })
        except Exception as e:
            logger.warning(f"Failed to augment position: {e}")
            continue
    
    return augmented_samples if augmented_samples else [
        {'position': position, 'move': move_idx, 'value': value}
    ]


def sample_opening_variation(game: chess.pgn.Game, max_depth: int = 10) -> chess.pgn.Game:
    """
    Sample a variation from the opening phase of a game.
    
    This creates diversity by exploring different move orders in the opening.
    Currently returns the mainline, but could be extended to sample variations.
    
    Args:
        game: chess.pgn.Game object
        max_depth: Maximum depth to sample variations
        
    Returns:
        Modified game with sampled variation
    """
    # For now, return mainline
    # Future: could sample from game.variations
    return game


if __name__ == "__main__":
    # Test augmentation
    logging.basicConfig(level=logging.INFO)
    
    board = chess.Board()
    move = list(board.legal_moves)[0]
    
    print("Original position:")
    print(board)
    print(f"Move: {move}")
    
    augmentations = augment_position(board, move)
    print(f"\nGenerated {len(augmentations)} augmentations")
    
    for i, (aug_board, aug_move, mult) in enumerate(augmentations):
        print(f"\nAugmentation {i+1} (value_mult={mult}):")
        print(aug_board)
        print(f"Move: {aug_move}")
