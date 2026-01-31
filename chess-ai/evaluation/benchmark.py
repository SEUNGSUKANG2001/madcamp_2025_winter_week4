"""
Comprehensive evaluation suite for chess AI.

Includes ELO estimation, puzzle performance, and style analysis.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import chess

logger = logging.getLogger(__name__)


class ELOCalculator:
    """
    Calculate ELO ratings from match results.
    """
    
    @staticmethod
    def calculate_elo(match_results: Dict[str, int]) -> float:
        """
        Calculate ELO estimate from match results.
        
        Args:
            match_results: Dictionary with 'wins', 'losses', 'draws'
            
        Returns:
            Estimated ELO rating
        """
        wins = match_results.get('wins', 0)
        losses = match_results.get('losses', 0)
        draws = match_results.get('draws', 0)
        
        total = wins + losses + draws
        if total == 0:
            return 0.0
        
        score = (wins + draws * 0.5) / total
        
        # Simplified ELO calculation (would use proper Bayesian method)
        # Assuming opponent ELO of 3000
        opponent_elo = 3000
        expected_score = 1 / (1 + 10 ** ((opponent_elo - 3000) / 400))
        
        # Calculate rating difference
        rating_diff = 400 * np.log10(score / (1 - score)) if score > 0 and score < 1 else 0
        
        estimated_elo = opponent_elo + rating_diff
        
        return estimated_elo


class PuzzleBenchmark:
    """
    Benchmark engine on chess puzzles.
    """
    
    def __init__(self, engine):
        """
        Initialize puzzle benchmark.
        
        Args:
            engine: Chess engine to test
        """
        self.engine = engine
    
    def evaluate_puzzle(self, puzzle_fen: str, solution: str) -> bool:
        """
        Evaluate engine on a single puzzle.
        
        Args:
            puzzle_fen: FEN string of puzzle position
            solution: UCI notation of solution move
            
        Returns:
            True if engine finds correct solution
        """
        board = chess.Board(puzzle_fen)
        
        try:
            move = self.engine.select_move(board)
            return move.uci() == solution
        except Exception as e:
            logger.error(f"Error evaluating puzzle: {e}")
            return False
    
    def evaluate_puzzles(self, puzzles: List[Dict]) -> Dict[str, float]:
        """
        Evaluate engine on multiple puzzles.
        
        Args:
            puzzles: List of puzzle dictionaries with 'fen' and 'solution'
            
        Returns:
            Dictionary with success rates by rating range
        """
        results = {}
        
        for puzzle in puzzles:
            rating_range = puzzle.get('rating_range', 'unknown')
            if rating_range not in results:
                results[rating_range] = {'correct': 0, 'total': 0}
            
            is_correct = self.evaluate_puzzle(puzzle['fen'], puzzle['solution'])
            results[rating_range]['total'] += 1
            if is_correct:
                results[rating_range]['correct'] += 1
        
        # Calculate success rates
        success_rates = {}
        for rating_range, stats in results.items():
            if stats['total'] > 0:
                success_rates[rating_range] = stats['correct'] / stats['total']
        
        return success_rates


class PerformanceProfiler:
    """
    Profile engine performance (speed, memory, etc.).
    """
    
    def __init__(self, engine):
        """
        Initialize profiler.
        
        Args:
            engine: Chess engine to profile
        """
        self.engine = engine
    
    def profile(self, num_positions: int = 100) -> Dict[str, float]:
        """
        Profile engine performance.
        
        Args:
            num_positions: Number of positions to test
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        positions = []
        for _ in range(num_positions):
            board = chess.Board()
            # Randomize position (simplified)
            positions.append(board)
        
        # Measure speed
        start_time = time.time()
        nodes_searched = 0
        
        for board in positions:
            try:
                move = self.engine.select_move(board)
                nodes_searched += 1000  # Placeholder: would track actual nodes
            except Exception as e:
                logger.error(f"Error profiling: {e}")
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        nps = nodes_searched / elapsed_time if elapsed_time > 0 else 0
        positions_per_sec = num_positions / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'nodes_per_second': nps,
            'positions_per_second': positions_per_sec,
            'average_time_per_move': elapsed_time / num_positions if num_positions > 0 else 0
        }


def generate_evaluation_report(
    elo_results: Dict,
    puzzle_results: Dict,
    performance_stats: Dict
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        elo_results: ELO calculation results
        puzzle_results: Puzzle benchmark results
        performance_stats: Performance profiling results
        
    Returns:
        Formatted report string
    """
    report = """
=== UltimateChess Evaluation Report ===

ELO Ratings:
"""
    
    for opponent, elo in elo_results.items():
        report += f"- vs {opponent}: Estimated ELO {elo:.0f}\n"
    
    report += "\nPuzzle Performance:\n"
    for rating_range, success_rate in puzzle_results.items():
        report += f"- Rating {rating_range}: {success_rate*100:.1f}%\n"
    
    report += "\nPerformance:\n"
    report += f"- NPS: {performance_stats.get('nodes_per_second', 0):,.0f}\n"
    report += f"- Positions/sec: {performance_stats.get('positions_per_second', 0):,.0f}\n"
    report += f"- Avg time/move: {performance_stats.get('average_time_per_move', 0):.3f}s\n"
    
    return report
