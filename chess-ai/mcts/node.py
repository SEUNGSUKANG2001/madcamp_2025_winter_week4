"""
MCTS Node implementation.
"""

import logging
from typing import Dict, Optional
import numpy as np
import chess
import chess.engine

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    MCTS Node representing a game state.
    """
    
    def __init__(
        self,
        board: chess.Board,
        parent: Optional['MCTSNode'] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ):
        """
        Initialize MCTS node.
        
        Args:
            board: Chess board state
            parent: Parent node
            move: Move that led to this state
            prior: Prior probability from policy network
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        # Statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        
        # Children
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        
        # Virtual loss for parallel search
        self.virtual_loss = 0
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children expanded)."""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if node represents a terminal game state."""
        return (self.board.is_checkmate() or 
                self.board.is_stalemate() or
                self.board.is_insufficient_material() or
                self.board.is_seventyfive_moves() or
                self.board.is_fivefold_repetition())
    
    def get_value(self) -> float:
        """Get current value estimate (with virtual loss)."""
        if self.visit_count + self.virtual_loss == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / (self.visit_count + self.virtual_loss)
    
    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """
        Select child using UCB formula.
        
        UCB(s,a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant
            
        Returns:
            Selected child node
        """
        best_score = float('-inf')
        best_child = None
        
        parent_visits = self.visit_count
        
        for move, child in self.children.items():
            # UCB formula
            q_value = child.get_value()
            prior = child.prior
            visit_count = child.visit_count + child.virtual_loss
            
            if visit_count == 0:
                u_value = float('inf')  # Unexplored nodes get highest priority
            else:
                u_value = c_puct * prior * np.sqrt(parent_visits) / (1 + visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child is None:
            raise ValueError("No children available for selection")
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray, legal_moves: list):
        """
        Expand node by creating children.
        
        Args:
            policy_probs: Policy probabilities from neural network (4096,)
            legal_moves: List of legal moves from this position
        """
        # Normalize policy over legal moves only
        legal_move_indices = []
        legal_move_probs = []
        
        for move in legal_moves:
            # Convert move to index (simplified - would need proper mapping)
            # For now, use a simple hash-based approach
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < len(policy_probs):
                legal_move_indices.append(move_idx)
                legal_move_probs.append(policy_probs[move_idx])
        
        if not legal_move_probs:
            return
        
        # Normalize
        total_prob = sum(legal_move_probs)
        if total_prob > 0:
            legal_move_probs = [p / total_prob for p in legal_move_probs]
        else:
            # Uniform prior if no policy support
            legal_move_probs = [1.0 / len(legal_moves)] * len(legal_moves)
        
        # Create children
        for move, prob in zip(legal_moves, legal_move_probs):
            child_board = self.board.copy()
            child_board.push(move)
            
            child = MCTSNode(
                board=child_board,
                parent=self,
                move=move,
                prior=prob
            )
            self.children[move] = child
    
    def update(self, value: float):
        """
        Update node statistics with new value.
        
        Args:
            value: Value estimate from evaluation
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def add_virtual_loss(self):
        """Add virtual loss for parallel search."""
        self.virtual_loss += 1
    
    def remove_virtual_loss(self):
        """Remove virtual loss after evaluation."""
        self.virtual_loss = max(0, self.virtual_loss - 1)
    
    def get_policy_distribution(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy distribution from visit counts.
        
        Args:
            temperature: Temperature for softmax (1.0 = normal, <1.0 = sharper)
            
        Returns:
            Policy distribution over moves
        """
        if not self.children:
            return np.array([])
        
        # Get visit counts
        moves = list(self.children.keys())
        visit_counts = np.array([self.children[move].visit_count for move in moves])
        
        # Apply temperature
        if temperature == 0:
            # Deterministic: return one-hot
            policy = np.zeros(len(moves))
            policy[np.argmax(visit_counts)] = 1.0
        else:
            # Softmax with temperature
            visit_probs = visit_counts ** (1.0 / temperature)
            policy = visit_probs / visit_probs.sum()
        
        return policy
