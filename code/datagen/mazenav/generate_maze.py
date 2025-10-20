#!/usr/bin/env python3
"""
Generate a textual 10x10 maze with randomly placed walls, start, and end positions.
The maze uses a 0-based index grid where walls are specified as a list of coordinates.
"""

from dataclasses import dataclass
import random
import numpy as np
from typing import List, Tuple, Optional

@dataclass
class Maze:
    """Container for a maze with navigation methods."""
    
    size: int
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    walls: Tuple[Tuple[int, int], ...]
    
    def __post_init__(self):
        """Validate maze data after initialization and ensure walls are sorted."""
        # Ensure walls is a sorted tuple of tuples
        if isinstance(self.walls, list) or isinstance(self.walls, tuple):
            # Sort walls by coordinate before storing
            self.walls = tuple(sorted(tuple(tuple(wall) for wall in self.walls)))
        
        # Validate start & end positions are valid
        if not self.is_valid_position(self.start_pos):
            raise ValueError(f"Start position {self.start_pos} is not valid")
        if not self.is_valid_position(self.end_pos):
            raise ValueError(f"End position {self.end_pos} is not valid")
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighboring positions (up, down, left, right).
        
        Args:
            pos: Current position as (x, y)
            
        Returns:
            List of (neighbor_position, step_cost) tuples
        """
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            nx, ny = new_pos
            # Check bounds and walls
            if (0 <= nx < self.size and 0 <= ny < self.size and 
                new_pos not in self.walls):
                neighbors.append((new_pos, 1.0))  # (state, step_cost)
        return neighbors
    
    def manhattan_distance(self, pos: Tuple[int, int]) -> float:
        """
        Heuristic: Manhattan distance to goal.
        
        Args:
            pos: Position as (x, y)
            
        Returns:
            Manhattan distance to the goal position
        """
        return abs(pos[0] - self.end_pos[0]) + abs(pos[1] - self.end_pos[1])
    
    def is_goal(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is the goal.
        
        Args:
            pos: Position as (x, y)
            
        Returns:
            True if position is the goal, False otherwise
        """
        return pos == self.end_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            pos: Position as (x, y)
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        return (0 <= x < self.size and 0 <= y < self.size and pos not in self.walls)
    
    def get_all_valid_positions(self) -> List[Tuple[int, int]]:
        """
        Get all valid (non-wall) positions in the maze.
        
        Returns:
            List of all valid positions
        """
        valid_positions = []
        for y in range(self.size):
            for x in range(self.size):
                pos = (x, y)
                if self.is_valid_position(pos):
                    valid_positions.append(pos)
        return valid_positions
    
    def visualize(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """
        Create a visual representation of the maze using ASCII characters.
        
        Args:
            path: Optional path from start to end, showing direction arrows
            
        Returns:
            ASCII representation of the maze
        """
        return visualize_maze(self, path)
    
    def to_string_format(self, add_special_tokens: bool = True, bos: str="bos", eos: str="eos") -> str:
        """
        Convert maze data to string format for training.
        
        Args:
            add_special_tokens: Whether to add bos/eos tokens
            bos: String put at beginning, default: 'bos'
            eos: String put at end, default: 'eos'
            
        Returns:
            String representation of the maze
        """
        tokens = []
        
        if add_special_tokens:
            tokens.append(bos)
        
        # Add start position
        tokens.extend(["start", str(self.start_pos[0]), str(self.start_pos[1])])
        # Add goal position  
        tokens.extend(["goal", str(self.end_pos[0]), str(self.end_pos[1])])
        # Add wall positions
        for wall_x, wall_y in self.walls:
            tokens.extend(["wall", str(wall_x), str(wall_y)])
        
        if add_special_tokens:
            tokens.append(eos)
        
        return " ".join(tokens)


def generate_maze_recursive_backtracking(size: int = 10) -> Maze:
    """
    Generate a proper maze using recursive backtracking algorithm.
    Creates a perfect maze with exactly one path between any two points.
    
    Args:
        size: Size of the maze (size x size)
    
    Returns:
        Maze instance
    """
    # Initialize maze grid: True = wall, False = path
    # Start with all walls
    maze = [[True for _ in range(size)] for _ in range(size)]
    
    # Directions: up, down, left, right
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    def is_valid(x: int, y: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= x < size and 0 <= y < size
    
    def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid unvisited neighbors (2 steps away)."""
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and maze[ny][nx]:  # If it's still a wall (unvisited)
                neighbors.append((nx, ny))
        return neighbors
    
    def carve_path(x: int, y: int):
        """Recursively carve paths through the maze."""
        maze[y][x] = False  # Mark as path
        
        # Get random unvisited neighbors
        neighbors = get_neighbors(x, y)
        random.shuffle(neighbors)
        
        for nx, ny in neighbors:
            if maze[ny][nx]:  # If still a wall
                # Carve the wall between current and neighbor
                wall_x, wall_y = (x + nx) // 2, (y + ny) // 2
                maze[wall_y][wall_x] = False
                # Recursively carve from neighbor
                carve_path(nx, ny)
    
    # Start carving from a random odd position (to ensure proper grid)
    start_x = random.randrange(1, size, 2)
    start_y = random.randrange(1, size, 2)
    carve_path(start_x, start_y)
    
    # Convert maze grid to wall list & find all path positions for start/end selection
    walls = []
    path_positions = []
    for y in range(size):
        for x in range(size):
            if maze[y][x]:  # If it's a wall
                walls.append((x, y))
            else:
                path_positions.append((x, y))
    
    # Randomly select start and end positions
    start_pos, end_pos = random.sample(path_positions, 2)
    
    return Maze(size=size, start_pos=start_pos, end_pos=end_pos, walls=walls)

def generate_maze_random(size: int = 10, wall_density: float = 0.4) -> Maze:
    """
    Generate a maze with random walls (original algorithm).
    
    Args:
        size: Size of the maze (size x size)
        wall_density: Fraction of cells that should be walls (0.0 to 1.0)
    
    Returns:
        Maze instance
    """
    # Calculate number of walls to place
    total_cells = size * size
    num_walls = int(total_cells * wall_density)
    
    # Generate random wall positions
    all_positions = [(x, y) for x in range(size) for y in range(size)]
    walls = random.sample(all_positions, num_walls)
    
    # Generate start and end positions (not walls), choosing both at once without replacement
    available_positions = [pos for pos in all_positions if pos not in walls]
    start_pos, end_pos = random.sample(available_positions, 2)
    
    return Maze(size=size, start_pos=start_pos, end_pos=end_pos, walls=walls)

def format_maze_output(maze: Maze) -> str:
    """
    Format the maze data as a textual representation.
    
    Args:
        maze: Maze object to format
    
    Returns:
        Formatted string representation
    """
    output_lines = []
    
    # Add walls
    output_lines.append("Walls:")
    for wall in sorted(maze.walls):
        output_lines.append(f"  ({wall[0]}, {wall[1]})")
    
    # Add start and end positions
    output_lines.append(f"Start: ({maze.start_pos[0]}, {maze.start_pos[1]})")
    output_lines.append(f"End: ({maze.end_pos[0]}, {maze.end_pos[1]})")
    
    return "\n".join(output_lines)

def visualize_maze(maze: Maze, path: Optional[List[Tuple[int, int]]] = None) -> str:
    """
    Create a visual representation of the maze using ASCII characters.
    
    Args:
        maze: Maze object to visualize
        path: Optional path from start to end, showing direction arrows
    
    Returns:
        ASCII representation of the maze
    """
    # Create grid encircled by walls
    grid = [[' ' for _ in range(maze.size+2)] for _ in range(maze.size+2)]
    
    # Place walls
    for i in range(maze.size+2):
        grid[i][0] = '#'
        grid[0][i] = '#'
        grid[i][maze.size+1] = '#'
        grid[maze.size+1][i] = '#'
    
    for wall in maze.walls:
        grid[wall[1]+1][wall[0]+1] = '#'
    
    # Add path visualization if provided
    if path:
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            
            # Choose direction symbol
            if dx == 1:
                direction_symbol = '>'  # Right
            elif dx == -1:
                direction_symbol = '<'  # Left
            elif dy == -1:
                direction_symbol = '^'  # Up
            elif dy == 1:
                direction_symbol = 'v'  # Down
            else:
                direction_symbol = '.'  # No movement (shouldn't happen)
            
            # Place direction symbol in grid (offset by border)
            grid[current_pos[1]+1][current_pos[0]+1] = direction_symbol
    
    # Place start and end (override path symbols at start/end positions)
    grid[maze.start_pos[1]+1][maze.start_pos[0]+1] = 'S'
    grid[maze.end_pos[1]  +1][maze.end_pos[0]  +1] = 'E'
    
    # Convert to string
    lines = []
    for row in grid:
        lines.append(''.join(row))
    
    return '\n'.join(lines)

# =============================================================================
# CONVENIENCE FUNCTIONS FOR MAZE STORAGE AND COMPRESSION
# =============================================================================

def mazes_to_npz(mazes: List[Maze], filename: str) -> None:
    """
    Convert a list of Maze objects to NPZ format and save to file.
    
    Args:
        mazes: List of Maze objects to save
        filename: Path to save the NPZ file
    """
    num_mazes = len(mazes)
    
    # Initialize arrays for each field
    sizes = np.zeros(num_mazes, dtype=np.int8)
    start_positions = np.zeros((num_mazes, 2), dtype=np.int8)
    end_positions = np.zeros((num_mazes, 2), dtype=np.int8)
    
    # Find the maximum number of walls across all mazes to determine array size
    max_walls = max(len(maze.walls) for maze in mazes) if mazes else 0
    
    # Initialize walls array with -1 as padding value (invalid coordinate)
    walls = np.full((num_mazes, max_walls, 2), -1, dtype=np.int8)
    
    # Fill arrays with maze data
    for i, maze in enumerate(mazes):
        sizes[i] = maze.size
        start_positions[i] = [maze.start_pos[0], maze.start_pos[1]]
        end_positions[i] = [maze.end_pos[0], maze.end_pos[1]]
        
        # Fill walls array
        for j, wall in enumerate(maze.walls):
            walls[i, j] = [wall[0], wall[1]]
    
    # Save to NPZ file
    np.savez_compressed(filename, 
             sizes=sizes,
             start_positions=start_positions,
             end_positions=end_positions,
             walls=walls)


def npz_to_mazes(filename: str) -> List[Maze]:
    """
    Load mazes from NPZ format and recreate Maze objects.
    
    Args:
        filename: Path to the NPZ file
        
    Returns:
        List of Maze objects reconstructed from the NPZ data
    """
    # Load NPZ file
    data = np.load(filename)
    
    sizes = data['sizes']
    start_positions = data['start_positions']
    end_positions = data['end_positions']
    walls = data['walls']
    
    num_mazes = len(sizes)
    mazes = []
    
    # Reconstruct each maze
    for i in range(num_mazes):
        size = int(sizes[i])
        start_pos = (int(start_positions[i, 0]), int(start_positions[i, 1]))
        end_pos = (int(end_positions[i, 0]), int(end_positions[i, 1]))
        
        # Extract walls, filtering out padding values (-1)
        maze_walls = []
        for wall_coord in walls[i]:
            if wall_coord[0] != -1 and wall_coord[1] != -1:  # Skip padding
                maze_walls.append((int(wall_coord[0]), int(wall_coord[1])))
        
        # Create Maze object
        maze = Maze(size=size, start_pos=start_pos, end_pos=end_pos, walls=tuple(maze_walls))
        mazes.append(maze)
    
    return mazes


def path_to_string_format(path: List[Tuple[int, int]], 
                         add_special_tokens: bool = True) -> str:
    """
    Convert path data to the format:
    "bos plan x1 y1 plan x2 y2 ... plan xm ym eos"
    
    Args:
        path: List of position coordinates (x, y) in order
        add_special_tokens: Whether to add bos/eos tokens (default True for training)
    
    Returns:
        String representation of the path
    """
    tokens = []
    
    if add_special_tokens:
        tokens.append("bos")
    
    # Add path positions
    for x, y in path:
        tokens.extend(["plan", str(x), str(y)])
    
    if add_special_tokens:
        tokens.append("eos")
    
    return " ".join(tokens)


def main():
    """Generate and display a 10x10 maze."""
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    print("Maze Generation Comparison")
    print("=" * 60)
    
    # Generate maze using recursive backtracking (proper maze)
    print("\n1. PROPER MAZE (Recursive Backtracking Algorithm):")
    print("-" * 50)
    maze_proper = generate_maze_recursive_backtracking(size=10)
    print(format_maze_output(maze_proper))
    print("\nVisual representation:")
    print(visualize_maze(maze_proper))
    print("\nLegend: S=Start, E=End, #=Wall, .=Empty")
    
    # Demonstrate new Maze class
    print("\n\n3. NEW MAZE CLASS DEMONSTRATION:")
    print("-" * 50)
    maze = generate_maze_recursive_backtracking(size=8)
    print(f"Maze size: {maze.size}x{maze.size}")
    print(f"Start position: {maze.start_pos}")
    print(f"End position: {maze.end_pos}")
    print(f"Number of walls: {len(maze.walls)}")
    
    # Test navigation functions
    print(f"\nNavigation function tests:")
    test_pos = maze.start_pos
    print(f"Testing from position {test_pos}:")
    print(f"  Is goal? {maze.is_goal(test_pos)}")
    print(f"  Manhattan distance to goal: {maze.manhattan_distance(test_pos)}")
    neighbors = maze.get_neighbors(test_pos)
    print(f"  Valid neighbors: {[pos for pos, cost in neighbors]}")
    print(f"  Is valid position? {maze.is_valid_position(test_pos)}")
    
    # Show visualization
    print(f"\nMaze visualization:")
    print(maze.visualize())
    
    # Test string format conversion
    print(f"\nString format (for training):")
    print(maze.to_string_format())
    
    # Generate maze using random placement (original algorithm)
    print("\n\n2. RANDOM WALL PLACEMENT (Original Algorithm):")
    print("-" * 50)
    maze_random = generate_maze_random(size=10, wall_density=0.4)
    print(format_maze_output(maze_random))
    print("\nVisual representation:")
    print(visualize_maze(maze_random))
    print("\nLegend: S=Start, E=End, #=Wall, .=Empty")
    
    # Demonstrate NPZ save/load functionality
    print("\n\n4. NPZ SAVE/LOAD DEMONSTRATION:")
    print("-" * 50)
    
    # Create a few test mazes
    test_mazes = [
        generate_maze_recursive_backtracking(size=6),
        generate_maze_random(size=7, wall_density=0.3)
    ]
    
    print(f"Created {len(test_mazes)} test mazes for NPZ demonstration")
    
    # Save to NPZ
    npz_filename = "demo_mazes.npz"
    mazes_to_npz(test_mazes, npz_filename)
    print(f"Saved mazes to {npz_filename}")
    
    # Load from NPZ
    loaded_mazes = npz_to_mazes(npz_filename)
    print(f"Loaded {len(loaded_mazes)} mazes from NPZ file")
    
    # Verify integrity
    all_match = True
    for i, (original, loaded) in enumerate(zip(test_mazes, loaded_mazes)):
        match = (original.size == loaded.size and 
                original.start_pos == loaded.start_pos and 
                original.end_pos == loaded.end_pos and 
                original.walls == loaded.walls)
        print(f"Maze {i+1} integrity: {'✅ Match' if match else '❌ Mismatch'}")
        if not match:
            all_match = False
    
    print(f"NPZ save/load test: {'✅ PASSED' if all_match else '❌ FAILED'}")
    
    # Clean up
    import os
    if os.path.exists(npz_filename):
        os.remove(npz_filename)
        print(f"Cleaned up {npz_filename}")
    
    print("\n" + "=" * 60)
    print("NOTES:")
    print("- The proper maze guarantees connectivity and a single solution path")
    print("- The random placement may create isolated regions or multiple paths")
    print("- Both mazes have implicit boundary walls (not listed in coordinates)")
    print("- The new Maze class provides convenient navigation methods")
    print("- NPZ format efficiently stores maze data using int8 arrays")

if __name__ == "__main__":
    main()