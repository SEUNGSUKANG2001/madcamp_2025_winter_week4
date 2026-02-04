#!/usr/bin/env python3
"""
Test the value labeling in our training data.
This will help verify if the labeling bug exists.
"""

import os
import sys
import chess
import chess.pgn
import io

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from data.preprocessing import encode_board, move_to_index

def test_value_labeling():
    """
    Create a simple game and check value labels.
    """
    # Create a simple game: e4 e5 Qh5 Nc6 Qxf7# (Scholar's mate)
    pgn = """
[Event "Test Game"]
[White "White Player"]
[Black "Black Player"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Qxf7# 1-0
"""
    
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()
    
    # Game outcome
    result = game.headers.get("Result", "*")
    if result == "1-0":
        white_outcome = 1.0
        black_outcome = -1.0
    elif result == "0-1":
        white_outcome = -1.0
        black_outcome = 1.0
    else:
        white_outcome = 0.0
        black_outcome = 0.0
    
    print("=" * 80)
    print("Testing Value Labeling in Training Data")
    print("=" * 80)
    print(f"\nGame: Scholar's Mate (White wins)")
    print(f"White outcome: {white_outcome}, Black outcome: {black_outcome}\n")
    
    move_number = 0
    for node in game.mainline():
        move = node.move
        
        print(f"\n{'-' * 80}")
        print(f"Move {move_number + 1}: {board.san(move)}")
        print(f"Board before move:")
        print(board)
        print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        # Current labeling (POTENTIALLY WRONG)
        if board.turn == chess.WHITE:
            value = white_outcome
        else:
            value = black_outcome
        
        print(f"\nCurrent labeling: value = {value}")
        print(f"Interpretation: This position/move combo will lead to {'White winning' if value > 0 else 'Black winning' if value < 0 else 'a draw'}")
        
        # What it should be (position after move from moving player's perspective)
        # Move the board to see result
        board_after = board.copy()
        board_after.push(move)
        
        # Value should be from perspective of player who just moved
        if board.turn == chess.WHITE:  # White just moved
            correct_value = white_outcome
            print(f"Correct value (White's perspective after move): {correct_value}")
        else:  # Black just moved
            correct_value = black_outcome
            print(f"Correct value (Black's perspective after move): {correct_value}")
        
        print(f"\nBoard after move:")
        print(board_after)
        
        # Check if this makes sense
        if move_number == 4:  # The checkmate move (Qxf7#)
            print(f"\n⚠️  This is the CHECKMATE move by White!")
            print(f"   Current labeling gives value: {value} (White wins)")
            print(f"   This is actually CORRECT for this move!")
            print(f"   White plays checkmate → White wins (value = 1.0)")
        
        if move_number == 3:  # Black's last move (Nc6)
            print(f"\n⚠️  This is Black's move BEFORE getting checkmated!")
            print(f"   Current labeling gives value: {value} (Black loses)")
            print(f"   Is Nc6 a BAD move? Let's think:")
            print(f"   - Black plays Nc6")
            print(f"   - This leads to Black losing the game")
            print(f"   - So labeling this as -1.0 (bad for Black) makes sense!")
            print(f"   - UNLESS Nc6 was the ONLY move available")
        
        board.push(move)
        move_number += 1
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
The value labeling assigns:
- Value = game outcome from current player's perspective

This means:
- Every White move in a White-win game gets value +1.0
- Every Black move in a White-win game gets value -1.0

PROBLEM:
- This doesn't distinguish between good and bad moves!
- ALL White moves in winning games are labeled as good
- ALL Black moves in losing games are labeled as bad
- The model can't learn move quality, only correlation with outcome

The model learns:
"If I'm White and will win the game, play any of these moves"
NOT: "This move is objectively strong"

SOLUTION:
- Use position evaluation instead of game outcome
- Or use outcome but filter for high-quality games where every move is good
- Or use outcome with temporal discounting (moves closer to end matter more)
""")

if __name__ == "__main__":
    test_value_labeling()
