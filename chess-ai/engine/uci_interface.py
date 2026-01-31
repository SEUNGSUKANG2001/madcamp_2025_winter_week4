"""
UCI (Universal Chess Interface) protocol implementation.
"""

import logging
import sys
from typing import Optional
import chess
import chess.engine

from ensemble.multi_agent import EnsembleEngine, ChessAgent
from models.architecture import HybridChessNet
import torch

logger = logging.getLogger(__name__)


class UCIEngine:
    """
    UCI-compatible chess engine.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize UCI engine.
        
        Args:
            model_path: Path to model checkpoint (optional)
        """
        self.board = chess.Board()
        self.engine = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load neural network model."""
        model = HybridChessNet()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create ensemble (simplified: single agent for now)
        agent = ChessAgent(model, style='positional')
        self.engine = EnsembleEngine([agent])
    
    def uci_loop(self):
        """Main UCI communication loop."""
        while True:
            try:
                command = input().strip()
                if not command:
                    continue
                
                self.process_command(command)
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    def process_command(self, command: str):
        """Process a UCI command."""
        parts = command.split()
        
        if command == "uci":
            self.send_id()
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            self.board = chess.Board()
        elif command.startswith("position"):
            self.set_position(command)
        elif command.startswith("go"):
            self.search(command)
        elif command == "quit":
            sys.exit(0)
        else:
            logger.warning(f"Unknown command: {command}")
    
    def send_id(self):
        """Send engine identification."""
        print("id name UltimateChess v1.0")
        print("id author Chess AI Team")
        print("option name Hash type spin default 512 min 16 max 32768")
        print("option name Threads type spin default 1 min 1 max 512")
        print("uciok")
    
    def set_position(self, command: str):
        """Set board position from UCI command."""
        parts = command.split()
        
        if "startpos" in parts:
            self.board = chess.Board()
            moves_start = parts.index("startpos") + 1
        elif "fen" in parts:
            fen_start = parts.index("fen") + 1
            fen_end = fen_start + 6
            fen = " ".join(parts[fen_start:fen_end])
            self.board = chess.Board(fen)
            moves_start = fen_end if fen_end < len(parts) else len(parts)
        else:
            return
        
        # Apply moves
        if moves_start < len(parts) and parts[moves_start] == "moves":
            for move_str in parts[moves_start + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except Exception as e:
                    logger.error(f"Invalid move: {move_str}")
    
    def search(self, command: str):
        """Perform search and return best move."""
        if self.engine is None:
            # Fallback: random legal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
            else:
                return
        else:
            # Use ensemble to select move
            best_move = self.engine.select_move(self.board)
        
        # Send best move
        print(f"bestmove {best_move.uci()}")


if __name__ == "__main__":
    # Run UCI engine
    logging.basicConfig(level=logging.WARNING)  # Reduce logging for UCI
    
    engine = UCIEngine()
    engine.uci_loop()
