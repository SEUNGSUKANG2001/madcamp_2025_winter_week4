"""
Adaptive MCTS with reasoning and verification (DeepSeek-R1 style).
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import chess

from mcts.mcts import MCTS
from data.preprocessing import encode_board

logger = logging.getLogger(__name__)


class CriticalityAnalyzer:
    """
    Analyzes position criticality to determine computation budget.
    """
    
    def __init__(self):
        """Initialize criticality analyzer."""
        pass
    
    def analyze(self, board: chess.Board) -> float:
        """
        Analyze position criticality.
        
        Returns score in [0, 1] where:
        - 0.0: Low criticality (opening, quiet position)
        - 1.0: High criticality (tactical, endgame, critical moment)
        
        Args:
            board: Chess board position
            
        Returns:
            Criticality score
        """
        score = 0.0
        
        # Move number factor (endgame is more critical)
        move_number = board.fullmove_number
        if move_number > 40:
            score += 0.3
        
        # Material imbalance (uneven positions are critical)
        material_diff = self._material_difference(board)
        score += min(abs(material_diff) / 10.0, 0.3)
        
        # King safety (exposed king is critical)
        king_safety = self._king_safety(board)
        score += (1.0 - king_safety) * 0.2
        
        # Tactical complexity (checks, captures, threats)
        tactical_score = self._tactical_complexity(board)
        score += tactical_score * 0.2
        
        return min(score, 1.0)
    
    def _material_difference(self, board: chess.Board) -> float:
        """Calculate material difference."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_material = sum(
            piece_values.get(piece.piece_type, 0)
            for square in chess.SQUARES
            if (piece := board.piece_at(square)) and piece.color == chess.WHITE
        )
        
        black_material = sum(
            piece_values.get(piece.piece_type, 0)
            for square in chess.SQUARES
            if (piece := board.piece_at(square)) and piece.color == chess.BLACK
        )
        
        return white_material - black_material
    
    def _king_safety(self, board: chess.Board) -> float:
        """Calculate king safety (1.0 = safe, 0.0 = exposed)."""
        # Simplified: check if king is in center or exposed
        king_square = board.king(board.turn)
        if king_square is None:
            return 0.5
        
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)
        
        # Kings in center are less safe
        center_distance = abs(rank - 3.5) + abs(file - 3.5)
        safety = min(center_distance / 7.0, 1.0)
        
        return safety
    
    def _tactical_complexity(self, board: chess.Board) -> float:
        """Estimate tactical complexity."""
        score = 0.0
        
        # Check if in check
        if board.is_check():
            score += 0.5
        
        # Count captures available
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        score += min(len(captures) / 10.0, 0.3)
        
        # Count checks available
        checks = [m for m in board.legal_moves if board.gives_check(m)]
        score += min(len(checks) / 5.0, 0.2)
        
        return min(score, 1.0)


class MoveVerifier:
    """
    Verifies candidate moves for tactical and positional soundness.
    """
    
    def verify(self, board: chess.Board, move: chess.Move) -> Tuple[float, List[str]]:
        """
        Verify a move and return score and reasons.
        
        Args:
            board: Current board position
            move: Move to verify
            
        Returns:
            Tuple of (score, reasons) where score is in [0, 1]
        """
        reasons = []
        score = 1.0
        
        # Make move temporarily
        board.push(move)
        
        # Check 1: Hanging pieces
        if self._has_hanging_pieces(board):
            score -= 0.3
            reasons.append("Leaves pieces hanging")
        
        # Check 2: King safety
        if self._weakens_king_safety(board, move):
            score -= 0.2
            reasons.append("Weakens king safety")
        
        # Check 3: Material balance
        if self._loses_material(board, move):
            score -= 0.4
            reasons.append("Loses material")
        
        # Check 4: Development (opening)
        if board.fullmove_number < 15:
            if not self._improves_development(board, move):
                score -= 0.1
                reasons.append("Doesn't improve development")
        
        board.pop()
        
        return max(score, 0.0), reasons
    
    def _has_hanging_pieces(self, board: chess.Board) -> bool:
        """Check if position has hanging pieces."""
        # Simplified: check if any pieces are under attack
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                if attackers:
                    return True
        return False
    
    def _weakens_king_safety(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move weakens king safety."""
        # Simplified: check if king is more exposed
        king_square = board.king(board.turn)
        if king_square is None:
            return False
        
        attackers = board.attackers(not board.turn, king_square)
        return len(attackers) > 2
    
    def _loses_material(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move loses material."""
        # Simplified: check if we're giving up material
        if board.is_capture(move):
            # Check if capture is favorable
            return False  # Would need deeper analysis
        
        return False
    
    def _improves_development(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move improves piece development."""
        # Simplified: moving pieces from back rank is good
        from_rank = chess.square_rank(move.from_square)
        return from_rank in [0, 7]  # Back rank


class AdaptiveMCTS(MCTS):
    """
    Adaptive MCTS that adjusts simulation budget based on position criticality.
    """
    
    def __init__(
        self,
        model,
        base_simulations: int = 800,
        max_simulations: int = 10000,
        c_puct: float = 1.5,
        device: str = 'cuda'
    ):
        """
        Initialize adaptive MCTS.
        
        Args:
            model: Neural network model
            base_simulations: Base number of simulations
            max_simulations: Maximum simulations for critical positions
            c_puct: UCB exploration constant
            device: Device to run model on
        """
        super().__init__(model, base_simulations, c_puct, device=device)
        self.base_simulations = base_simulations
        self.max_simulations = max_simulations
        self.criticality_analyzer = CriticalityAnalyzer()
        self.move_verifier = MoveVerifier()
    
    def search(self, board: chess.Board, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        """
        Perform adaptive MCTS search.
        
        Args:
            board: Current board position
            temperature: Temperature for move selection
            
        Returns:
            Tuple of (best_move, policy_distribution)
        """
        # Analyze criticality
        criticality = self.criticality_analyzer.analyze(board)
        
        # Adjust simulation budget
        move_number = board.fullmove_number
        if move_number < 15:
            # Opening: low budget
            num_simulations = int(self.base_simulations * 0.5)
        elif move_number > 40:
            # Endgame: high budget if critical
            num_simulations = int(self.base_simulations + 
                                (self.max_simulations - self.base_simulations) * criticality)
        else:
            # Middlegame: adaptive
            num_simulations = int(self.base_simulations + 
                                (self.max_simulations - self.base_simulations) * criticality * 0.5)
        
        # Set simulation budget
        self.num_simulations = min(num_simulations, self.max_simulations)
        
        # Perform search
        move, policy = super().search(board, temperature)
        
        # Verify move
        score, reasons = self.move_verifier.verify(board, move)
        if score < 0.5:
            logger.warning(f"Move verification failed: {reasons}")
            # Could re-search or select alternative move
        
        return move, policy


class ReasoningEngine:
    """
    Multi-path reasoning engine with verification.
    """
    
    def __init__(self, mcts: AdaptiveMCTS):
        """
        Initialize reasoning engine.
        
        Args:
            mcts: Adaptive MCTS instance
        """
        self.mcts = mcts
        self.verifier = MoveVerifier()
    
    def think(self, board: chess.Board, num_paths: int = 5) -> chess.Move:
        """
        Think through multiple reasoning paths.
        
        Args:
            board: Current board position
            num_paths: Number of reasoning paths to explore
            
        Returns:
            Best verified move
        """
        verified_plans = []
        
        for _ in range(num_paths):
            # Generate candidate plan
            try:
                move, _ = self.mcts.search(board, temperature=1.0)
                
                # Verify plan
                score, reasons = self.verifier.verify(board, move)
                
                if score >= 0.5:  # Acceptable threshold
                    verified_plans.append((move, score, reasons))
                else:
                    logger.debug(f"Rejected plan: {reasons}")
            except Exception as e:
                logger.error(f"Error in reasoning path: {e}")
                continue
        
        if not verified_plans:
            # Fallback: return any legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            raise ValueError("No legal moves available")
        
        # Select best verified plan
        best_plan = max(verified_plans, key=lambda x: x[1])
        return best_plan[0]
