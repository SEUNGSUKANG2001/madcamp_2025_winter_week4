import os
import sys
import logging
import torch
import torch.nn.functional as F
import chess
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to import modules
# Since this script is in scripts/, we need to go up one level to reach the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from models.architecture import HybridChessNet
from data.preprocessing import encode_board, index_to_move, move_to_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    try:
        checkpoint_path = os.path.join(project_root, 'checkpoints', 'checkpoint_epoch_5.pt')
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found at {checkpoint_path}")
            return False

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Initialize model
        model = HybridChessNet(
            input_channels=33,
            cnn_channels=256,
            num_res_blocks=8,
            transformer_embed_dim=512,
            transformer_num_heads=8,
            transformer_num_blocks=4,
            num_moves=4096,
            dropout=0.1
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def get_move():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    fen = data.get('fen')
    
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400

    try:
        board = chess.Board(fen)
        
        # Check game over conditions
        if board.is_game_over():
            return jsonify({'game_over': True, 'result': board.result()})

        # Encode board
        encoded_board = encode_board(board)
        x = torch.from_numpy(encoded_board).unsqueeze(0).float().to(device) # Add batch dim
        
        # Inference
        with torch.no_grad():
            policy_logits, value = model(x)
            
        # Get best legal move using VALUE-BASED selection
        # Evaluate each legal move by simulating it and checking the value
        legal_moves = list(board.legal_moves)
        
        # Get best legal move using POLICY NETWORK
        legal_moves = list(board.legal_moves)
        
        # Get policy from current position
        with torch.no_grad():
            policy_logits, value = model(x)
        
        # Apply softmax to get probabilities
        policy_probs = F.softmax(policy_logits, dim=-1)[0]  # Shape: (4096,)
        
        # Evaluate all legal moves
        move_evaluations = []
        best_move = None
        best_prob = float('-inf')  # Initialize to negative infinity for MAXIMUM search
        
        logger.info(f"Evaluating {len(legal_moves)} legal moves using POLICY network...")
        
        for move in legal_moves:
            # Get move index
            try:
                move_idx = move_to_index(move, board)
                prob = policy_probs[move_idx].item()
                
                move_evaluations.append({
                    'move': move.uci(),
                    'value': prob  # Store policy probability
                })
                
                if prob > best_prob:  # Select move with MAXIMUM probability
                    best_prob = prob
                    best_move = move
                    
            except Exception as e:
                logger.warning(f"Could not evaluate move {move}: {e}")
                continue
        
        if best_move is None:
            logger.error("No legal move found!")
            import random
            best_move = random.choice(legal_moves)
            best_prob = 0.0
        
        # Renormalize probabilities for legal moves only (so they sum to 1.0)
        total_prob = sum([e['value'] for e in move_evaluations])
        if total_prob > 0:
            for eval_item in move_evaluations:
                eval_item['value'] = eval_item['value'] / total_prob
            best_prob = best_prob / total_prob
        
        # Sort evaluations by probability (descending)
        move_evaluations.sort(key=lambda x: x['value'], reverse=True)
        
        # DEBUG: Log top 5 moves with probabilities
        logger.info("="*60)
        logger.info(f"Top 5 move evaluations (POLICY probabilities):")
        for i, eval_item in enumerate(move_evaluations[:5]):
            logger.info(f"  {i+1}. {eval_item['move']}: prob={eval_item['value']:.6f}")
        logger.info(f"Selected move: {best_move.uci()} with probability: {best_prob:.6f}")
        logger.info("="*60)

        # Apply move
        board.push(best_move)
        
        return jsonify({
            'fen': board.fen(),
            'move': best_move.uci(),
            'value': best_prob,
            'move_evaluations': move_evaluations,  # Add all evaluations
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        })

    except Exception as e:
        logger.error(f"Error processing move: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/new_game', methods=['POST'])
def new_game():
    board = chess.Board()
    return jsonify({'fen': board.fen()})

if __name__ == '__main__':
    if load_model():
        app.run(debug=True, port=5000)
    else:
        print("Failed to start application due to model loading error.")
