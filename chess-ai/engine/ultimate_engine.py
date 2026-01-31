"""
Ultimate chess engine integrating all components.
"""

import logging
from typing import Optional, Dict
import chess
import torch

from models.architecture import HybridChessNet
from mcts.reasoning_mcts import AdaptiveMCTS, ReasoningEngine
from ensemble.multi_agent import EnsembleEngine, ChessAgent, ContextAnalyzer

logger = logging.getLogger(__name__)


class UltimateChessEngine:
    """
    Complete chess engine with all advanced features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ultimate chess engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load models
        self.models = {}
        self.agents = []
        self.ensemble = None
        
        self._load_models()
        self._create_ensemble()
    
    def _load_models(self):
        """Load all models."""
        model_config = self.config.get('model', {})
        
        # Base model
        base_path = model_config.get('base_path')
        if base_path:
            model = HybridChessNet()
            checkpoint = torch.load(base_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.models['base'] = model
        
        # Ensemble models
        ensemble_paths = model_config.get('ensemble_paths', [])
        for path in ensemble_paths:
            model = HybridChessNet()
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.models[f'ensemble_{len(self.models)}'] = model
    
    def _create_ensemble(self):
        """Create ensemble of agents."""
        search_config = self.config.get('search', {})
        base_sims = search_config.get('base_simulations', 800)
        max_sims = search_config.get('max_simulations', 10000)
        
        # Create agents
        if 'base' in self.models:
            # Base agent
            mcts = AdaptiveMCTS(
                self.models['base'],
                base_simulations=base_sims,
                max_simulations=max_sims,
                device=str(self.device)
            )
            agent = ChessAgent(self.models['base'], style='positional', 
                            num_simulations=base_sims, device=str(self.device))
            self.agents.append(agent)
        
        # Create ensemble
        if self.agents:
            self.ensemble = EnsembleEngine(self.agents, ContextAnalyzer())
    
    def analyze_position(
        self,
        board: chess.Board,
        time_left: Optional[float] = None,
        history: Optional[list] = None
    ) -> Dict:
        """
        Analyze position and return evaluation.
        
        Args:
            board: Chess board position
            time_left: Time remaining in seconds
            history: Move history
            
        Returns:
            Analysis dictionary
        """
        if self.ensemble is None:
            return {'error': 'Engine not initialized'}
        
        # Select move
        move = self.ensemble.select_move(board)
        
        # Get evaluation (simplified)
        evaluation = 0.0  # Would get from model
        
        return {
            'move': move,
            'evaluation': evaluation,
            'depth': 0  # Would track search depth
        }
    
    def select_move(
        self,
        board: chess.Board,
        constraints: Optional[Dict] = None
    ) -> chess.Move:
        """
        Select best move.
        
        Args:
            board: Current board position
            constraints: Optional constraints (time, depth, etc.)
            
        Returns:
            Best move
        """
        if self.ensemble is None:
            # Fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0]
            raise ValueError("No legal moves available")
        
        return self.ensemble.select_move(board)
    
    def get_analysis(self) -> Dict:
        """
        Get detailed position analysis.
        
        Returns:
            Analysis dictionary
        """
        # Placeholder
        return {
            'evaluation': 0.0,
            'best_move': None,
            'variation': []
        }
