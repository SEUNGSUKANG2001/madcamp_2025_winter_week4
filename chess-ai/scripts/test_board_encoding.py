#!/usr/bin/env python3
"""
Test to verify board encoding perspective.
This checks if the board encoding is from the correct player's perspective.
"""

import os
import sys
import chess
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from data.preprocessing import encode_board

def test_perspective():
    """Test if board encoding changes based on whose turn it is."""
    
    # Create a simple position
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # White plays e4
    
    print("=" * 80)
    print("Testing Board Encoding Perspective")
    print("=" * 80)
    
    print("\nPosition after 1.e4:")
    print(board)
    print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Encode board
    encoded = encode_board(board)
    
    # Check plane 29 (color to move)
    color_plane = encoded[29, 0, 0]
    print(f"\nPlane 29 (color to move): {color_plane}")
    print(f"Expected: 0 (Black to move)")
    
    # Check if white pawn on e4 is encoded
    # e4 is square index 28 (row=3, col=4 in 0-indexed)
    # In flipped coordinates: row=7-3=4, col=4
    white_pawn_plane = encoded[0]  # Plane 0 is white pawns
    print(f"\nWhite pawn plane (plane 0) at e4 (row=4, col=4):")
    print(f"Value: {white_pawn_plane[4, 4]}")
    print(f"Expected: 1.0")
    
    print("\n" + "=" * 80)
    print("ISSUE IDENTIFIED:")
    print("=" * 80)
    print("""
The board encoding is ALWAYS from White's perspective, even when it's Black's turn!

During TRAINING:
- All positions are encoded from White's perspective
- The model learns: "When it's my turn (plane 29=1), I should play these moves"

During INFERENCE for Black:
- Board is still encoded from White's perspective
- But we're asking the model to play as Black
- The model sees: plane 29=0 (Black to move)
- But the board still shows White's pieces in planes 0-5!

This is a FUNDAMENTAL BUG! 

When playing as Black, we should:
1. Flip the board (Black's pieces in planes 0-5, White in 6-11)
2. Or always encode from current player's perspective

AlphaGo/AlphaZero always encodes from CURRENT PLAYER's perspective!
""")

if __name__ == "__main__":
    test_perspective()
