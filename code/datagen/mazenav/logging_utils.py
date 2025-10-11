#!/usr/bin/env python3
"""
Logging utilities for separating logging concerns from business logic.
Also implements Logging configuration.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                  console_output: bool = True) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(module_name)


# Convenience function for quick setup
def quick_setup(verbose: bool = False) -> logging.Logger:
    """
    Quick setup for common logging scenarios.
    
    Args:
        verbose: If True, use DEBUG level, otherwise INFO
    
    Returns:
        Configured logger
    """
    level = "DEBUG" if verbose else "INFO"
    return setup_logging(log_level=level)



class ProgressTracker:
    """Handles progress logging for long-running operations."""
    
    def __init__(self, total: int, name: str = "Progress", log_every: int = 100):
        self.total = total
        self.name = name
        self.log_every = log_every
        self.current = 0
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress and log at intervals."""
        self.current += increment
        
        if self.current % self.log_every == 0 or self.current == self.total:
            if message:
                self.logger.info(f"{self.name}: {self.current}/{self.total} - {message}")
            else:
                self.logger.info(f"{self.name}: {self.current}/{self.total}")


@contextmanager
def log_execution_time(operation_name: str):
    """Context manager to log execution time of operations."""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info(f"Starting {operation_name}...")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed {operation_name} in {elapsed:.2f}s")


@dataclass
class SearchResult:
    """Container for search algorithm results."""
    is_accepted: bool
    beam_solution: Optional[list]
    beam_cost: Optional[float]
    astar_solution: Optional[list]
    astar_cost: Optional[float]
    maze_data: Optional[dict]
    
    def summary(self) -> str:
        """Return a brief summary of the search result."""
        if not self.is_accepted:
            return "Not accepted (beam search failed)"
        
        if self.astar_solution == self.beam_solution:
            return f"Same solution (length: {len(self.beam_solution)})"
        else:
            return f"Different solutions - A*: {len(self.astar_solution)}, Beam: {len(self.beam_solution)}"


@dataclass
class MazeSearchStats:
    """Handles statistics tracking without logging concerns."""
    
    def __init__(self):
        self.total_generation_trials = 0
        self.both_same_solution = 0
        self.both_different_solution = 0
        self.total_astar_solutions = 0
        self.cost_differences = []
        self.accepted_mazes = []
    
    def update(self, search_result: SearchResult):
        """Update statistics based on search result."""
        self.total_generation_trials += 1
        
        if search_result.astar_solution is not None:
            self.total_astar_solutions += 1
        
        if search_result.is_accepted:
            self.accepted_mazes.append(search_result.maze_data)
            
            if search_result.astar_solution == search_result.beam_solution:
                self.both_same_solution += 1
            else:
                self.both_different_solution += 1
                if search_result.astar_cost is not None and search_result.beam_cost is not None:
                    self.cost_differences.append((search_result.astar_cost, search_result.beam_cost))
    
    def get_summary(self) -> dict:
        """Return comprehensive statistics summary."""
        total_beam_solutions = len(self.accepted_mazes)
        astar_only_solutions = self.total_astar_solutions - self.both_same_solution - self.both_different_solution
        
        return {
            'total_generation_trials': self.total_generation_trials,
            'accepted_mazes': total_beam_solutions,
            'acceptance_rate': 100 * total_beam_solutions / self.total_generation_trials if self.total_generation_trials > 0 else 0,
            'astar_only_solutions': astar_only_solutions,
            'astar_success_rate': 100 * self.total_astar_solutions / self.total_generation_trials if self.total_generation_trials > 0 else 0,
            'beam_success_rate': 100 * total_beam_solutions / self.total_generation_trials if self.total_generation_trials > 0 else 0,
            'both_same_solution': self.both_same_solution,
            'both_different_solution': self.both_different_solution,
            'cost_differences': self.cost_differences
        }


def log_search_result(logger: logging.Logger, search_result: SearchResult, trial_num: int, total_trials: int):
    """Log details about a search result."""
    if not search_result.is_accepted:
        logger.debug(f"Trial {trial_num}/{total_trials}: Rejected - beam search failed")
    else:
        if search_result.astar_solution == search_result.beam_solution:
            logger.debug(f"Trial {trial_num}/{total_trials}: ✓ Same solution (length: {len(search_result.beam_solution)})")
        else:
            logger.warning(f"Trial {trial_num}/{total_trials}: ⚠ Different solutions - A*: {len(search_result.astar_solution)}, Beam: {len(search_result.beam_solution)}")


def log_final_summary(logger: logging.Logger, stats: MazeSearchStats):
    """Log comprehensive final summary."""
    summary = stats.get_summary()
    
    logger.info("=" * 60)
    logger.info("BATCH MAZE SEARCH SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Total mazes generated: {summary['total_generation_trials']}")
    logger.info(f"Accepted mazes (beam found solution): {summary['accepted_mazes']}")
    logger.info(f"Acceptance rate: {summary['acceptance_rate']:.1f}%")
    logger.info(f"A* found solution but beam didn't: {summary['astar_only_solutions']}")
    logger.info(f"A* success rate: {summary['astar_success_rate']:.1f}%")
    logger.info(f"Beam success rate: {summary['beam_success_rate']:.1f}%")
    
    if summary['accepted_mazes'] > 0:
        logger.info("Solution comparison (accepted mazes only):")
        logger.info(f"  Both found same solution: {summary['both_same_solution']} ({100*summary['both_same_solution']/summary['accepted_mazes']:.1f}%)")
        logger.info(f"  Both found different solutions: {summary['both_different_solution']} ({100*summary['both_different_solution']/summary['accepted_mazes']:.1f}%)")
        
        if summary['cost_differences']:
            logger.info("Cost analysis (when both found solutions):")
            astar_costs = [cost[0] for cost in summary['cost_differences']]
            beam_costs = [cost[1] for cost in summary['cost_differences']]
            cost_ratios = [beam_cost/astar_cost for astar_cost, beam_cost in summary['cost_differences']]
            
            logger.info(f"  Average A* cost: {sum(astar_costs)/len(astar_costs):.2f}")
            logger.info(f"  Average beam cost: {sum(beam_costs)/len(beam_costs):.2f}")
            logger.info(f"  Average cost ratio (beam/A*): {sum(cost_ratios)/len(cost_ratios):.2f}")
            logger.info(f"  Max cost ratio: {max(cost_ratios):.2f}")
            logger.info(f"  Min cost ratio: {min(cost_ratios):.2f}")
