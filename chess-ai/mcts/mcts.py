"""
Monte Carlo Tree Search implementation for chess.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import chess
import torch

from mcts.node import MCTSNode
from data.preprocessing import encode_board, index_to_move

logger = logging.getLogger(__name__)


class MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = 'cuda'
    ):
        """
        Initialize MCTS.
        
        Args:
            model: Neural network model for policy and value prediction
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB
            dirichlet_alpha: Dirichlet noise parameter for root
            dirichlet_epsilon: Weight of Dirichlet noise
            device: Device to run model on
        """
        self.model = model
        self.model.eval()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device
        
        self.root: Optional[MCTSNode] = None
    
    def search(self, board: chess.Board, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        """
        Perform MCTS search and return best move and policy.
        
        Args:
            board: Current chess board position
            temperature: Temperature for move selection
            
        Returns:
            Tuple of (best_move, policy_distribution)
        """
        # Create root node
        self.root = MCTSNode(board=board.copy())
        
        # Expand root with policy
        policy_probs = self._get_policy(board)
        legal_moves = list(board.legal_moves)
        
        # Add Dirichlet noise to root policy (for exploration)
        if len(legal_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                move_idx = move.from_square * 64 + move.to_square
                if move_idx < len(policy_probs):
                    policy_probs[move_idx] = (
                        (1 - self.dirichlet_epsilon) * policy_probs[move_idx] +
                        self.dirichlet_epsilon * noise[i]
                    )
        
        self.root.expand(policy_probs, legal_moves)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            self._simulate()
        
        # Get policy distribution from visit counts
        policy_dist = self.root.get_policy_distribution(temperature=temperature)
        
        # Select best move
        if len(self.root.children) == 0:
            # Fallback: select random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = np.random.choice(legal_moves)
                policy_dist = np.ones(len(legal_moves)) / len(legal_moves)
            else:
                raise ValueError("No legal moves available")
        else:
            # Select move with highest visit count
            moves = list(self.root.children.keys())
            visit_counts = [self.root.children[move].visit_count for move in moves]
            best_move_idx = np.argmax(visit_counts)
            best_move = moves[best_move_idx]
        
        return best_move, policy_dist
    
    def _simulate(self) -> float:
        """
        Perform a single MCTS simulation.
        
        Returns:
            Value estimate for the simulation
        """
        # Selection: traverse from root to leaf
        node = self._select()
        
        # Check if terminal
        if node.is_terminal():
            value = self._get_terminal_value(node.board)
            self._backpropagate(node, value)
            return value
        
        # Expansion and evaluation
        value = self._expand_and_evaluate(node)
        
        # Backpropagation
        self._backpropagate(node, value)
        
        return value
    
    def _select(self) -> MCTSNode:
        """
        Select a leaf node by traversing the tree.
        
        Returns:
            Leaf node
        """
        node = self.root
        
        while not node.is_leaf():
            node = node.select_child(c_puct=self.c_puct)
        
        return node
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node and evaluate position.
        
        Args:
            node: Leaf node to expand
            
        Returns:
            Value estimate
        """
        # Check if terminal
        if node.is_terminal():
            return self._get_terminal_value(node.board)
        
        # Get policy and value from neural network
        policy_probs, value = self._evaluate_position(node.board)
        
        # Expand node
        legal_moves = list(node.board.legal_moves)
        node.expand(policy_probs, legal_moves)
        
        return value
    
    def _evaluate_position(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Evaluate position using neural network.
        
        Args:
            board: Chess board position
            
        Returns:
            Tuple of (policy_probs, value)
        """
        # Encode board
        position = encode_board(board)
        position_tensor = torch.from_numpy(position).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.model(position_tensor)
        
        # Convert to numpy
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value = value.cpu().item()
        
        return policy_probs, value
    
    def _get_terminal_value(self, board: chess.Board) -> float:
        """
        Get value for terminal position.
        
        Args:
            board: Terminal chess board
            
        Returns:
            Value: +1 for win, -1 for loss, 0 for draw
        """
        if board.is_checkmate():
            # Current player lost (checkmated)
            return -1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        else:
            # Draw by repetition or 75-move rule
            return 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            node: Node to start backpropagation from
            value: Value to propagate
        """
        # Value is from current player's perspective
        # When backpropagating, we need to flip perspective
        current_value = value
        
        while node is not None:
            node.update(current_value)
            current_value = -current_value  # Flip for opponent's perspective
            node = node.parent
    
    def _get_policy(self, board: chess.Board) -> np.ndarray:
        """
        Get policy from neural network.
        
        Args:
            board: Chess board position
            
        Returns:
            Policy probabilities (4096,)
        """
        policy_probs, _ = self._evaluate_position(board)
        return policy_probs
    
    def update_root(self, move: chess.Move):
        """
        Update root after making a move.
        
        Args:
            move: Move that was played
        """
        if self.root is not None and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None  # New root has no parent
        else:
            # Create new root
            board = self.root.board.copy() if self.root else chess.Board()
            board.push(move)
            self.root = MCTSNode(board=board)
