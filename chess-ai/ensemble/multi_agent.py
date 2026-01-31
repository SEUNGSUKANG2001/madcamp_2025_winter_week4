"""
Multi-agent ensemble with style-specialized models.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import chess
import torch

from models.architecture import HybridChessNet
from mcts.mcts import MCTS

logger = logging.getLogger(__name__)


class StyleClassifier:
    """
    Classifies games by playing style.
    """
    
    @staticmethod
    def classify_game(game_data: Dict) -> str:
        """
        Classify a game by style.
        
        Args:
            game_data: Game data dictionary
            
        Returns:
            Style label: 'aggressive', 'defensive', or 'positional'
        """
        # Simplified classification based on game characteristics
        # In practice, would analyze move patterns, tempo, sacrifices, etc.
        
        # Placeholder: return balanced style
        return 'positional'


class ContextAnalyzer:
    """
    Analyzes game context to determine which agent to use.
    """
    
    def analyze(self, board: chess.Board) -> Dict[str, float]:
        """
        Analyze current game context.
        
        Args:
            board: Current board position
            
        Returns:
            Dictionary with context features
        """
        context = {
            'opening_phase': board.fullmove_number < 15,
            'middlegame': 15 <= board.fullmove_number < 40,
            'endgame': board.fullmove_number >= 40,
            'material_advantage': self._has_material_advantage(board),
            'king_exposed': self._is_king_exposed(board),
            'complex_position': self._is_complex(board)
        }
        
        return context
    
    def _has_material_advantage(self, board: chess.Board) -> bool:
        """Check if current player has material advantage."""
        # Simplified
        return False
    
    def _is_king_exposed(self, board: chess.Board) -> bool:
        """Check if king is exposed."""
        king_square = board.king(board.turn)
        if king_square is None:
            return False
        
        attackers = board.attackers(not board.turn, king_square)
        return len(attackers) > 0
    
    def _is_complex(self, board: chess.Board) -> bool:
        """Check if position is complex."""
        # Simplified: many legal moves = complex
        return len(list(board.legal_moves)) > 20


class ChessAgent:
    """
    Individual chess agent with a specific playing style.
    """
    
    def __init__(
        self,
        model: HybridChessNet,
        style: str,
        num_simulations: int = 800,
        device: str = 'cuda'
    ):
        """
        Initialize agent.
        
        Args:
            model: Neural network model
            style: Playing style ('aggressive', 'defensive', 'positional')
            num_simulations: MCTS simulations
            device: Device to run on
        """
        self.model = model
        self.style = style
        self.mcts = MCTS(model, num_simulations=num_simulations, device=device)
    
    def select_move(self, board: chess.Board) -> Tuple[chess.Move, float]:
        """
        Select move using MCTS.
        
        Args:
            board: Current board position
            
        Returns:
            Tuple of (move, confidence)
        """
        move, policy = self.mcts.search(board)
        
        # Confidence is max policy probability
        confidence = float(np.max(policy)) if len(policy) > 0 else 0.5
        
        return move, confidence


class EnsembleEngine:
    """
    Multi-agent ensemble that combines specialized agents.
    """
    
    def __init__(
        self,
        agents: List[ChessAgent],
        context_analyzer: Optional[ContextAnalyzer] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            agents: List of specialized agents
            context_analyzer: Context analyzer for agent selection
        """
        self.agents = agents
        self.context_analyzer = context_analyzer or ContextAnalyzer()
    
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select move using ensemble.
        
        Args:
            board: Current board position
            
        Returns:
            Best move
        """
        # Analyze context
        context = self.context_analyzer.analyze(board)
        
        # Select appropriate agent(s)
        if context['opening_phase']:
            # Use positional agent in opening
            agent = self._get_agent_by_style('positional')
        elif context['material_advantage'] and context['king_exposed']:
            # Use defensive agent when ahead but king exposed
            agent = self._get_agent_by_style('defensive')
        elif not context['material_advantage']:
            # Use aggressive agent when behind
            agent = self._get_agent_by_style('aggressive')
        else:
            # Use weighted ensemble
            return self._weighted_voting(board, context)
        
        if agent is None:
            # Fallback to first agent
            agent = self.agents[0]
        
        move, _ = agent.select_move(board)
        return move
    
    def _get_agent_by_style(self, style: str) -> Optional[ChessAgent]:
        """Get agent by style."""
        for agent in self.agents:
            if agent.style == style:
                return agent
        return None
    
    def _weighted_voting(self, board: chess.Board, context: Dict) -> chess.Move:
        """
        Use weighted voting from multiple agents.
        
        Args:
            board: Current board position
            context: Game context
            
        Returns:
            Best move
        """
        # Get moves from all agents
        candidate_moves = {}
        
        for agent in self.agents:
            try:
                move, confidence = agent.select_move(board)
                weight = self._get_agent_weight(agent, context)
                
                if move not in candidate_moves:
                    candidate_moves[move] = 0.0
                
                candidate_moves[move] += weight * confidence
            except Exception as e:
                logger.error(f"Error getting move from {agent.style} agent: {e}")
                continue
        
        if not candidate_moves:
            # Fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            raise ValueError("No legal moves available")
        
        # Select move with highest weighted score
        best_move = max(candidate_moves, key=candidate_moves.get)
        return best_move
    
    def _get_agent_weight(self, agent: ChessAgent, context: Dict) -> float:
        """Get weight for agent based on context."""
        # Simplified weighting
        if agent.style == 'aggressive' and not context['material_advantage']:
            return 1.5
        elif agent.style == 'defensive' and context['king_exposed']:
            return 1.5
        elif agent.style == 'positional' and context['opening_phase']:
            return 1.5
        
        return 1.0
