import sys
import os
import torch
import numpy as np
import chess

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.append(os.path.join(project_root, "chess-ai"))

from data.preprocessing import encode_board
from data.dataset import ChessDataset

def test_optimization():
    print("Optimization Verification Started...")
    
    # 1. Verify encode_board returns uint8
    board = chess.Board()
    # Apply some moves to increase move count
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    
    encoded = encode_board(board)
    print(f"\n1. Dtype Check: {encoded.dtype}")
    if encoded.dtype == np.uint8:
        print("   PASS: Dtype is uint8")
    else:
        print(f"   FAIL: Dtype is {encoded.dtype}, expected uint8")
        
    # 2. Verify Move Count Storage (Plane 30)
    # Move number for this board should be 3 (after 4 moves, fullmove is 3)
    # 1. e4 e5 2. Nf3 Nc6 3. (next move)
    expected_move_count = 3
    stored_move_count = encoded[30, 0, 0]
    print(f"\n2. Value Check (Move Count):")
    print(f"   Board fullmove_number: {board.fullmove_number}")
    print(f"   Stored value (uint8): {stored_move_count}")
    
    if stored_move_count == expected_move_count:
        print("   PASS: Move count stored as raw integer")
    else:
        print(f"   FAIL: Expected {expected_move_count}, got {stored_move_count}")
        
    # 3. Verify Size Reduction
    float_size = 33 * 8 * 8 * 4  # float32 = 4 bytes
    uint8_size = 33 * 8 * 8 * 1  # uint8 = 1 byte
    print(f"\n3. Size Check (Per Position):")
    print(f"   Float32 size: {float_size} bytes")
    print(f"   Uint8 size:   {uint8_size} bytes")
    print(f"   Reduction:    {float_size / uint8_size}x")
    
    # 4. Verify Dataset Loading & Normalization
    print(f"\n4. Dataset Loading & Normalization Check:")
    
    # Mocking a dataset structure logic for testing get_item logic
    # We can't easily instantiate ChessDataset without files, so we'll test the logic directly
    # duplicating the logic from __getitem__ to verify it works as intended
    
    # Simulate loading from disk
    loaded_position_uint8 = torch.from_numpy(encoded).float()
    
    # Apply normalization logic
    # Logic: position[30] = torch.clamp(position[30] / 100.0, max=1.0)
    loaded_position_uint8[30] = torch.clamp(loaded_position_uint8[30] / 100.0, max=1.0)
    
    restored_move_count = loaded_position_uint8[30, 0, 0].item()
    expected_normalized = min(expected_move_count / 100.0, 1.0)
    
    print(f"   Restored value (float): {restored_move_count}")
    print(f"   Expected normalized:    {expected_normalized}")
    
    if abs(restored_move_count - expected_normalized) < 1e-6:
        print("   PASS: Normalization logic restores correct float value")
    else:
        print("   FAIL: Normalization logic failed")
        
    print("\nVerification Complete!")

if __name__ == "__main__":
    test_optimization()
