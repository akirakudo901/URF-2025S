# Author: Akira Kudo
# Created: 2025/11/10
# Last Updated: 2025/11/14

"""
This file aims to extend the Maze class defined in generate_maze.py, to cover Sokoban problems.

A Sokoban problem is defined as follows:
- A map is provided, with randomly generated walls, boxes and goal tiles.
- The user can move UP, DOWN, LEFT or RIGHT in the map, starting at the 'start' tile. They cannot move into walls.
- The user can input a movement command into a box, pushing it in the same direction. The user cannot
  do this if the box will end up inside a wall or another box.
- The user wins if they have every goal tile covered with a box.

Here's the generation protocol for a Sokoban problem from SearchFormer:
- A 7 × 7 grid was sampled and two additional wall cells were added as obstacles to the interior of the map. 
- Two docks, boxes, and the worker locations were randomly placed. 
- If the sampled task is solvable by A*, then the task was admitted to the dataset.

The heuristic used for A* will be as follows:
- First match every box to the closest dock.
- Compute the sum of all Manhattan distances between each box and dock pair.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Set
import random

import numpy as np
import scipy
import os
import sys

# Add parent directory to path to import astar_beam_search
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from astar_beam_search import AStarBeamSearch

@dataclass
class Sokoban:
    """Container for a Sokoban puzzle with navigation methods."""
    
    size: int
    start_pos: Tuple[int, int]
    walls: Tuple[Tuple[int, int], ...]
    boxes: Tuple[Tuple[int, int], ...]  # Initial box positions
    goals: Tuple[Tuple[int, int], ...]
    
    def __post_init__(self):
        """Validate Sokoban data after initialization and ensure all tuples are sorted."""
        # Ensure walls is a sorted tuple of tuples
        if isinstance(self.walls, list) or isinstance(self.walls, tuple):
            self.walls = tuple(sorted(tuple(tuple(wall) for wall in self.walls)))
        
        # Ensure boxes is a sorted tuple of tuples
        if isinstance(self.boxes, list) or isinstance(self.boxes, tuple):
            self.boxes = tuple(sorted(tuple(tuple(box) for box in self.boxes)))
        
        # Ensure goals is a sorted tuple of tuples
        if isinstance(self.goals, list) or isinstance(self.goals, tuple):
            self.goals = tuple(sorted(tuple(tuple(goal) for goal in self.goals)))
        
        # Validate boxes using the validation method
        self._validate_boxes(self.boxes)
        
        # Validate start position is valid (not a wall, within bounds, without boxes)
        if not self.is_valid_position(self.start_pos):
            raise ValueError(f"Start position {self.start_pos} is not valid")
        
        # Validate start position doesn't overlap with initial boxes
        if self.start_pos in self.boxes:
            raise ValueError(f"Start position {self.start_pos} cannot overlap with a box")
        
        # Validate all goals are on valid positions (not walls, within bounds)
        for goal in self.goals:
            if not self.is_valid_position(goal):
                raise ValueError(f"Goal position {goal} is not valid")
    
    def is_valid_position(self, pos: Tuple[int, int], boxes: Optional[Set[Tuple[int, int]]] = None) -> bool:
        """
        Check if a position is valid (within bounds, not a wall, and optionally not occupied by a box).
        
        Args:
            pos: Position as (x, y)
            boxes: Optional set of box positions to check against. If None, checks validity without regard to boxes.
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size): # Check bounds
            return False
        if pos in self.walls: # Check if it's a wall
            return False
        if boxes is not None and pos in boxes: # Check if there's a box (if boxes is provided)
            return False
        return True
    
    def get_all_valid_positions(self, boxes: Optional[Set[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """
        Get all valid (non-wall) positions in the Sokoban puzzle.
        
        Args:
            boxes: Optional set of box positions to exclude. If None, returns all valid positions without regard to boxes.
            
        Returns:
            List of all valid positions
        """
        valid_positions = []
        for y in range(self.size):
            for x in range(self.size):
                pos = (x, y)
                if self.is_valid_position(pos, boxes=boxes):
                    valid_positions.append(pos)
        return valid_positions
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Compute Manhattan distance between two positions.
        
        Args:
            pos1: First position as (x, y)
            pos2: Second position as (x, y)
            
        Returns:
            Manhattan distance between the two positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def heuristic(self, 
                  boxes: Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]], 
                  goals: Optional[Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]] = None
                  ) -> float:
        """
        Compute heuristic value for A* search by matching boxes to closest goals.
        
        The heuristic matches every box to the closest dock (goal) and computes
        the sum of all Manhattan distances between each box-dock pair.
        Uses the Hungarian algorithm for Balanced Assignment to find the minimum total distance.
        
        Args:
            boxes: Box positions (list, tuple, or set)
            goals: Optional goal positions (list, tuple, or set). If None, uses self.goals.
            
        Returns:
            Sum of Manhattan distances between matched box-goal pairs (minimum cost matching)
        """
        def solve_assignment_hungarian(docks: List[Tuple[int, int]], 
                                       boxes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """
            Solve the assignment problem in O(n^3) using the Hungarian algorithm (optimal solution).
            
            Args:
                docks: List of dock positions as (x, y) tuples
                boxes: List of box positions as (x, y) tuples
            
            Returns:
                List of pairs (dock_index, box_index) representing the optimal assignment
            
            Raises:
                ValueError: If docks and boxes have different lengths
            """
            if len(docks) != len(boxes):
                raise ValueError(f"Docks and boxes must have the same length. "
                                 f"Got {len(docks)} docks and {len(boxes)} boxes.")

            n = len(docks)

            # Build cost matrix: cost[i, j] = manhattan distance from dock i to box j
            cost_matrix = np.zeros((n, n), dtype=int)
            for i, dock in enumerate(docks):
                for j, box in enumerate(boxes):
                    cost_matrix[i, j] = self.manhattan_distance(dock, box)
            
            # Solve using Hungarian algorithm
            row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            # Return as list of tuples (dock_index, box_index)
            assignment = [(int(row_indices[i]), int(col_indices[i])) 
                          for i in range(len(row_indices))]
            return assignment

        if goals is None:
            goals = self.goals

        boxes_list = list(boxes)
        goals_list = list(goals)
        
        # If number of boxes doesn't match number of goals, return a large value
        if len(boxes_list) != len(goals_list):
            return float('inf')

        n = len(boxes_list)

        if n == 0:
            return 0.0
        elif n == 1:
            return self.manhattan_distance(boxes_list[0], goals_list[0])
        else:
            total_distance = 0.0
            assignment = solve_assignment_hungarian(goals_list, boxes_list)
            for dock_index, box_index in assignment:
                total_distance += self.manhattan_distance(goals_list[dock_index], boxes_list[box_index])
            return total_distance
    
    def _validate_boxes(self, boxes: Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]) -> None:
        """
        Validate that a box configuration is valid.
        
        Args:
            boxes: Box positions to validate (list, tuple, or set)
            
        Raises:
            ValueError: If box configuration is invalid
        """
        # Convert to set for validation
        boxes_set = set(boxes)
        
        # Check number of boxes matches number of goals
        if len(boxes_set) != len(self.goals):
            raise ValueError(f"Number of boxes ({len(boxes_set)}) must equal number of goals ({len(self.goals)})")
        
        # Check boxes don't overlap with each other (set automatically handles this, but check length)
        if len(boxes) != len(boxes_set):
            raise ValueError("Boxes cannot overlap with each other")
        
        # Check all boxes are on valid positions (not walls, within bounds)
        for box in boxes_set:
            if not self.is_valid_position(box):
                raise ValueError(f"Box position {box} is not valid")
    
    def _move_results_in_new_state(self, pos: Tuple[int, int], boxes: Set[Tuple[int, int]], direction: Tuple[int, int]) -> bool:
        """
        Check if moving in the given direction from the current position with the given boxes results in a new state.
        
        Args:
            pos: Current position as (x, y)
            boxes: Set of current box positions
            direction: Direction to move as (dx, dy)
            
        Returns:
            True if the move results in a different state, False if it stays the same or is invalid
        """
        dx, dy = direction
        new_pos = (pos[0] + dx, pos[1] + dy)
        
        # Check if new position is valid (within bounds, not a wall)
        if not self.is_valid_position(new_pos):
            return False
        
        # Check if new position is blocked by a box
        if new_pos not in boxes:
            return True # If not, valid move
        else:
            # Try to push the box
            box_behind_pos = (new_pos[0] + dx, new_pos[1] + dy)
            # Check if position behind box is valid (within bounds, not a wall, not a box)
            return self.is_valid_position(box_behind_pos, boxes=boxes)
    
    def _compute_resulting_state(self, pos: Tuple[int, int], boxes: Set[Tuple[int, int]], direction: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]]:
        """
        Compute the resulting state after moving in the given direction.
        
        Args:
            pos: Current position as (x, y)
            boxes: Set of current box positions
            direction: Direction to move as (dx, dy)
            
        Returns:
            Tuple of (new_position, new_boxes_tuple) where new_boxes_tuple is sorted
        """
        dx, dy = direction
        new_pos = (pos[0] + dx, pos[1] + dy)
        
        # Check if new position has a box
        if new_pos in boxes:
            # Push the box
            box_behind_pos = (new_pos[0] + dx, new_pos[1] + dy)
            # Create new boxes set with the pushed box
            new_boxes = boxes.copy()
            new_boxes.remove(new_pos)
            new_boxes.add(box_behind_pos)
        else:
            # No box to push
            new_boxes = boxes.copy()
        
        # Convert to sorted tuple for consistency
        new_boxes_tuple = tuple(sorted(new_boxes))
        return (new_pos, new_boxes_tuple)
    
    def get_neighbors(self, pos: Tuple[int, int], boxes: Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]) -> List[Tuple[Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...]], float]]:
        """
        Get valid neighboring states (up, down, left, right) considering box positions and pushing.
        
        Args:
            pos: Current position as (x, y)
            boxes: Box positions (list, tuple, or set). Must be a valid configuration (validated before use).
            
        Returns:
            List of ((new_position, new_boxes), step_cost) tuples representing the resulting states
        """
        # Convert to set and validate
        boxes_set = set(boxes)
        self._validate_boxes(boxes_set)
        
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
        
        for direction in directions:
            # Check if move results in a new state
            if not self._move_results_in_new_state(pos, boxes_set, direction):
                continue
            
            # Compute the resulting state
            new_pos, new_boxes = self._compute_resulting_state(pos, boxes_set, direction)
            
            # Add the resulting state as a neighbor
            neighbors.append(((new_pos, new_boxes), 1.0))
        
        return neighbors
    
    def is_goal(self, boxes: Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]) -> bool:
        """
        Check if the puzzle is solved (all goal tiles are covered by boxes).
        
        Args:
            boxes: Box positions to check (list, tuple, or set).
            
        Returns:
            True if every goal has exactly one box at that position, False otherwise
        """
        boxes_set = set(boxes)
        goals_set = set(self.goals)
        return goals_set == boxes_set
    
    def to_string_format(self, add_special_tokens: bool = True, bos: str="bos", eos: str="eos", boxes: Optional[Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]] = None) -> str:
        """
        Convert Sokoban configuration to string format for training.
        
        Args:
            add_special_tokens: Whether to add bos/eos tokens
            bos: String put at beginning, default: 'bos'
            eos: String put at end, default: 'eos'
            boxes: Optional box positions (list, tuple, or set). If provided, validated and used as "box".
                   Initial boxes are always included as "ini_box".
            
        Returns:
            String representation of the Sokoban puzzle
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(bos)
        
        # Add start position
        tokens.extend(["start", str(self.start_pos[0]), str(self.start_pos[1])])
        
        # Add goal positions (sorted for consistency)
        for goal_x, goal_y in self.goals:
            tokens.extend(["goal", str(goal_x), str(goal_y)])
        
        # Add wall positions (sorted for consistency)
        for wall_x, wall_y in self.walls:
            tokens.extend(["wall", str(wall_x), str(wall_y)])
        
        # Add initial box positions as "ini_box" (sorted for consistency)
        for box_x, box_y in self.boxes:
            tokens.extend(["ini_box", str(box_x), str(box_y)])
        
        # Add current box positions as "box" if provided (validated and sorted)
        if boxes is not None:
            # Validate boxes
            boxes_set = set(boxes)
            self._validate_boxes(boxes_set)
            # Add boxes in sorted order
            for box_x, box_y in sorted(boxes_set):
                tokens.extend(["box", str(box_x), str(box_y)])
        
        if add_special_tokens:
            tokens.append(eos)
        
        return " ".join(tokens)
    
    def hash_state(self, boxes: Optional[Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]] = None) -> int:
        """
        Generate a unique hash for a Sokoban state (configuration with optional box positions).
        
        Args:
            boxes: Optional box positions (list, tuple, or set). If provided, validated and included in hash.
                   Initial boxes are always included in hash as "ini_box".
        
        Returns:
            Integer hash representing the Sokoban state
        """
        # Create hash string from Sokoban components
        walls_str = '_'.join([f'{w[0]}_{w[1]}' for w in self.walls])
        ini_boxes_str = '_'.join([f'{b[0]}_{b[1]}' for b in self.boxes])
        goals_str = '_'.join([f'{g[0]}_{g[1]}' for g in self.goals])
        
        # Include initial boxes in hash
        sokoban_str = f"{self.size}_{self.start_pos[0]}_{self.start_pos[1]}_{walls_str}_ini_{ini_boxes_str}_{goals_str}"
        
        # Include current boxes in hash if provided
        if boxes is not None:
            # Validate boxes
            boxes_set = set(boxes)
            self._validate_boxes(boxes_set)
            # Add boxes in sorted order
            boxes_str = '_'.join([f'{b[0]}_{b[1]}' for b in sorted(boxes_set)])
            sokoban_str = f"{sokoban_str}_box_{boxes_str}"
        
        # Use Python's built-in hash function for a compact representation
        return hash(sokoban_str)
    
    def __hash__(self) -> int:
        """
        Generate a unique hash for this Sokoban configuration (initial state only).
        Uses initial boxes as part of the hash.
        
        Returns:
            Integer hash representing the Sokoban configuration
        """
        # Hash the initial configuration (without current boxes)
        return self.hash_state(boxes=None)
    
    def visualize(self, pos, boxes=None):  # pos: Tuple[int, int], boxes: Optional[Union[List[Tuple[int, int]], Tuple[Tuple[int, int], ...], Set[Tuple[int, int]]]] = None -> str
        """
        Create a visual representation of the Sokoban puzzle state.
        
        Args:
            pos: Current player position as (x, y)
            boxes: Optional box positions (list, tuple, or set). If None, uses self.boxes.
            
        Returns:
            ASCII representation of the Sokoban puzzle state
        """
        # Determine which boxes to use
        if boxes is not None:
            boxes_set = set(boxes)
        else:
            boxes_set = set(self.boxes)
        
        # Create grid with borders (size+2 x size+2)
        grid = [[' ' for _ in range(self.size+2)] for _ in range(self.size+2)]
        
        # Place border walls
        for i in range(self.size+2):
            grid[i][0] = '#'
            grid[0][i] = '#'
            grid[i][self.size+1] = '#'
            grid[self.size+1][i] = '#'
        
        # Place walls from the puzzle
        for wall in self.walls:
            grid[wall[1]+1][wall[0]+1] = '#'
        
        # Place goals (docks) first
        goals_set = set(self.goals)
        for goal in self.goals:
            grid[goal[1]+1][goal[0]+1] = '⏺'
        
        # Place boxes - check if box is on a goal
        for box in boxes_set:
            box_x, box_y = box[0]+1, box[1]+1
            if box in goals_set:
                # Box is on a goal - use ⌾ symbol
                grid[box_y][box_x] = '⌾'
            else:
                # Box is not on a goal - use ▫️ symbol
                grid[box_y][box_x] = '▫️'
        
        # Place player position (overrides everything except walls)
        if self.is_valid_position(pos):
            player_x, player_y = pos[0]+1, pos[1]+1
            # Only place player if it's not a wall
            if grid[player_y][player_x] != '#':
                grid[player_y][player_x] = 'P'
        
        # Convert grid to string
        lines = []
        for row in grid:
            lines.append(''.join(row))
        
        return '\n'.join(lines)


def generate_sokoban_random(size: int, M: int, N: int) -> Sokoban:
    """
    Generate a random Sokoban puzzle with specified number of walls, goals, and boxes.
    
    Args:
        size: Size of the map (size x size grid)
        M: Number of walls to generate
        N: Number of goal tiles and boxes (must be equal)
    
    Returns:
        Sokoban instance with randomly generated configuration
    
    Raises:
        ValueError: If there are not enough available positions for walls, goals, boxes, and start position
    """
    # Calculate total number of cells
    total_cells = size * size
    
    # Check if we have enough cells for all elements
    required_cells = M + N + N # M walls + N goals + N boxes
    if required_cells > total_cells:
        raise ValueError(f"Not enough cells: need {required_cells} cells but only have {total_cells} cells")
    
    # Generate all possible positions
    all_positions = [(x, y) for x in range(size) for y in range(size)]
    
    # Step 1: Generate M walls (non-overlapping, within bounds)
    if M > len(all_positions):
        raise ValueError(f"Cannot place {M} walls: only {len(all_positions)} positions available")
    walls = random.sample(all_positions, M)
    walls_set = set(walls)
    
    # Step 2: Generate N goal tiles (non-overlapping with walls, within bounds)
    available_for_goals = [pos for pos in all_positions if pos not in walls_set]
    if N > len(available_for_goals):
        raise ValueError(f"Cannot place {N} goals: only {len(available_for_goals)} positions available after placing walls")
    goals = random.sample(available_for_goals, N)
    goals_set = set(goals)
    
    # Step 3: Generate N boxes (non-overlapping with each other, walls, goals, within bounds)
    available_for_boxes = [pos for pos in available_for_goals if pos not in goals_set]
    if N > len(available_for_boxes):
        raise ValueError(f"Cannot place {N} boxes: only {len(available_for_boxes)} positions available after placing walls and goals")
    boxes = random.sample(available_for_boxes, N)
    boxes_set = set(boxes)
    
    # Step 4: Pick a random starting position (not a wall, not a box, but can be on a goal)
    available_for_start = [pos for pos in all_positions if pos not in walls_set and pos not in boxes_set]
    if len(available_for_start) == 0:
        raise ValueError("Cannot find a valid start position: no positions available that are not walls or boxes")
    start_pos = random.choice(available_for_start)
    
    # Create and return Sokoban instance
    return Sokoban(
        size=size,
        start_pos=start_pos,
        walls=tuple(sorted(walls)),
        boxes=tuple(sorted(boxes)),
        goals=tuple(sorted(goals))
    )

def sokobans_to_npz(sokobans: List[Sokoban], filename: str) -> None:
    """
    Convert a list of Sokoban objects to NPZ format and save to file.
    
    Args:
        sokobans: List of Sokoban objects to save
        filename: Path to save the NPZ file
    """
    num_sokobans = len(sokobans)
    
    # Initialize arrays for each field
    sizes = np.zeros(num_sokobans, dtype=np.int8)
    start_positions = np.zeros((num_sokobans, 2), dtype=np.int8)
    
    # Find the maximum number of walls, boxes, and goals across all sokobans to determine array sizes
    max_walls = max(len(sokoban.walls) for sokoban in sokobans) if sokobans else 0
    max_boxes = max(len(sokoban.boxes) for sokoban in sokobans) if sokobans else 0
    max_goals = max(len(sokoban.goals) for sokoban in sokobans) if sokobans else 0
    
    # Initialize arrays with -1 as padding value (invalid coordinate)
    walls = np.full((num_sokobans, max_walls, 2), -1, dtype=np.int8)
    boxes = np.full((num_sokobans, max_boxes, 2), -1, dtype=np.int8)
    goals = np.full((num_sokobans, max_goals, 2), -1, dtype=np.int8)
    
    # Fill arrays with sokoban data
    for i, sokoban in enumerate(sokobans):
        sizes[i] = sokoban.size
        start_positions[i] = [sokoban.start_pos[0], sokoban.start_pos[1]]
        
        # Fill walls array
        for j, wall in enumerate(sokoban.walls):
            walls[i, j] = [wall[0], wall[1]]
        
        # Fill boxes array
        for j, box in enumerate(sokoban.boxes):
            boxes[i, j] = [box[0], box[1]]
        
        # Fill goals array
        for j, goal in enumerate(sokoban.goals):
            goals[i, j] = [goal[0], goal[1]]
    
    # Save to NPZ file
    np.savez_compressed(filename, 
             sizes=sizes,
             start_positions=start_positions,
             walls=walls,
             boxes=boxes,
             goals=goals)


def npz_to_sokobans(filename: str) -> List[Sokoban]:
    """
    Load Sokoban puzzles from NPZ format and recreate Sokoban objects.
    
    Args:
        filename: Path to the NPZ file
        
    Returns:
        List of Sokoban objects reconstructed from the NPZ data
    """
    # Load NPZ file
    data = np.load(filename)
    
    sizes = data['sizes']
    start_positions = data['start_positions']
    walls = data['walls']
    boxes = data['boxes']
    goals = data['goals']
    
    num_sokobans = len(sizes)
    sokobans = []
    
    # Reconstruct each sokoban
    for i in range(num_sokobans):
        size = int(sizes[i])
        start_pos = (int(start_positions[i, 0]), int(start_positions[i, 1]))
        
        # Extract walls, filtering out padding values (-1)
        sokoban_walls = []
        for wall_coord in walls[i]:
            if wall_coord[0] != -1 and wall_coord[1] != -1:  # Skip padding
                sokoban_walls.append((int(wall_coord[0]), int(wall_coord[1])))
        
        # Extract boxes, filtering out padding values (-1)
        sokoban_boxes = []
        for box_coord in boxes[i]:
            if box_coord[0] != -1 and box_coord[1] != -1:  # Skip padding
                sokoban_boxes.append((int(box_coord[0]), int(box_coord[1])))
        
        # Extract goals, filtering out padding values (-1)
        sokoban_goals = []
        for goal_coord in goals[i]:
            if goal_coord[0] != -1 and goal_coord[1] != -1:  # Skip padding
                sokoban_goals.append((int(goal_coord[0]), int(goal_coord[1])))
        
        # Create Sokoban object
        sokoban = Sokoban(size=size, start_pos=start_pos, walls=tuple(sokoban_walls), boxes=tuple(sokoban_boxes), goals=tuple(sokoban_goals))
        sokobans.append(sokoban)
    
    return sokobans


def main():
    """
    Demonstration of solving a randomly generated Sokoban puzzle using A* Beam Search.
    """
    # Generate a small Sokoban puzzle for demonstration
    SIZE = 5
    NUM_WALLS = 2
    NUM_BOXES = 2
    
    print("=" * 60)
    print("Sokoban Puzzle Solver Demonstration")
    print("=" * 60)
    print()
    
    # Generate random Sokoban puzzle
    print(f"Generating random Sokoban puzzle (size={SIZE}, walls={NUM_WALLS}, boxes={NUM_BOXES})...")
    sokoban = generate_sokoban_random(size=SIZE, M=NUM_WALLS, N=NUM_BOXES)
    
    print("\nInitial Puzzle State:")
    print(sokoban.visualize(sokoban.start_pos, sokoban.boxes))
    print(f"Start position: {sokoban.start_pos}")
    print(f"Initial boxes: {sokoban.boxes}")
    print(f"Goals: {sokoban.goals}")
    print(f"Walls: {sokoban.walls}")
    print()
    
    # Define wrapper functions for AStarBeamSearch
    # State format: (position, boxes_tuple) where boxes_tuple is a sorted tuple
    initial_state = (sokoban.start_pos, sokoban.boxes)
    
    def next_states(state):
        """
        Get neighboring states from a given state.
        
        Args:
            state: Tuple of (position, boxes_tuple)
            
        Returns:
            List of ((new_position, new_boxes), step_cost) tuples
        """
        pos, boxes = state
        return sokoban.get_neighbors(pos, boxes)
    
    def heuristic_fn(state):
        """
        Compute heuristic value for a state.
        
        Args:
            state: Tuple of (position, boxes_tuple)
            
        Returns:
            Heuristic value (sum of Manhattan distances for optimal box-goal matching)
        """
        pos, boxes = state
        return sokoban.heuristic(boxes)
    
    def is_goal_fn(state):
        """
        Check if a state is the goal state.
        
        Args:
            state: Tuple of (position, boxes_tuple)
            
        Returns:
            True if all goals are covered by boxes, False otherwise
        """
        pos, boxes = state
        return sokoban.is_goal(boxes)
    
    # Create A* Beam Search instance
    beam_search = AStarBeamSearch(
        next_states=next_states,
        heuristic_fn=heuristic_fn,
        is_goal_fn=is_goal_fn,
        cost_fn=None  # Use additive costs (g + step_cost)
    )
    
    # Run normal A* search first
    print("Running Normal A* Search...")
    print("-" * 60)
    astar_solution, astar_cost = beam_search.solve_normal_astar(
        start_state=initial_state,
        max_steps=None
    )
    
    print(f"A* Solution found: {astar_solution is not None}")
    if astar_solution:
        print(f"A* Solution path length: {len(astar_solution)} steps")
        print(f"A* Total cost: {astar_cost}")
        print(f"A* Number of states explored: {len(astar_solution)}")
        print()
        
        # Display solution path visualization
        print("Solution Path Visualization:")
        print("-" * 60)
        for step_idx, state in enumerate(astar_solution):
            pos, boxes = state
            print(f"Step {step_idx}:")
            print(sokoban.visualize(pos, boxes))
            print()
    else:
        print("A* could not find a solution within the step limit.")
    print()
    
    # Run beam search with small beam size for comparison
    BEAM_SIZE = 30
    print(f"Running Beam Search (beam_size={BEAM_SIZE})...")
    print("-" * 60)
    beam_solution, state_matrix, backpointer_matrix, beam_cost = beam_search.solve(
        start_state=initial_state,
        beam_size=BEAM_SIZE,
        max_steps=100000
    )
    
    print(f"Beam Search Solution found: {beam_solution is not None}")
    if beam_solution:
        print(f"Beam Search Solution path length: {len(beam_solution)} steps")
        print(f"Beam Search Total cost: {beam_cost}")
        print(f"State matrix shape: {len(state_matrix)} x {len(state_matrix[0]) if state_matrix else 0}")
        print(f"Backpointer matrix shape: {len(backpointer_matrix)} x {len(backpointer_matrix[0]) if backpointer_matrix else 0}")
    else:
        print("Beam Search could not find a solution within the step limit.")
    print()
    
    # Comparison summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    if astar_solution and beam_solution:
        print(f"A* path length: {len(astar_solution)}, Beam path length: {len(beam_solution)}")
        print(f"A* cost: {astar_cost}, Beam cost: {beam_cost}")
        print(f"Both found solutions: Yes")
        print(f"Solutions are identical: {astar_solution == beam_solution}")
    elif astar_solution:
        print("Only A* found a solution")
    elif beam_solution:
        print("Only Beam Search found a solution")
    else:
        print("Neither method found a solution within the step limit")
    print()

if __name__ == "__main__":
    main()