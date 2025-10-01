#!/usr/bin/env python3
"""
Generate a textual 10x10 maze with randomly placed walls, start, and end positions.
The maze uses a 0-based index grid where walls are specified as a list of coordinates.
"""

import random
import sys
from typing import List, Tuple, Set

def generate_maze_recursive_backtracking(size: int = 10) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    """
    Generate a proper maze using recursive backtracking algorithm.
    Creates a perfect maze with exactly one path between any two points.
    
    Args:
        size: Size of the maze (size x size)
    
    Returns:
        Tuple of (walls, start_pos, end_pos)
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
    
    return walls, start_pos, end_pos

def generate_maze_random(size: int = 10, wall_density: float = 0.4
                        ) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
    """
    Generate a maze with random walls (original algorithm).
    
    Args:
        size: Size of the maze (size x size)
        wall_density: Fraction of cells that should be walls (0.0 to 1.0)
    
    Returns:
        Tuple of (walls, start_pos, end_pos)
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
    
    return walls, start_pos, end_pos

def format_maze_output(walls: List[Tuple[int, int]], start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> str:
    """
    Format the maze data as a textual representation.
    
    Args:
        walls: List of wall coordinates
        start_pos: Start position coordinates
        end_pos: End position coordinates
    
    Returns:
        Formatted string representation
    """
    output_lines = []
    
    # Add walls
    output_lines.append("Walls:")
    for wall in sorted(walls):
        output_lines.append(f"  ({wall[0]}, {wall[1]})")
    
    # Add start and end positions
    output_lines.append(f"Start: ({start_pos[0]}, {start_pos[1]})")
    output_lines.append(f"End: ({end_pos[0]}, {end_pos[1]})")
    
    return "\n".join(output_lines)

def visualize_maze(walls: List[Tuple[int, int]], start_pos: Tuple[int, int], end_pos: Tuple[int, int], size: int = 10) -> str:
    """
    Create a visual representation of the maze using ASCII characters.
    
    Args:
        walls: List of wall coordinates
        start_pos: Start position coordinates
        end_pos: End position coordinates
        size: Size of the maze
    
    Returns:
        ASCII representation of the maze
    """
    # Create grid encircled by walls
    grid = [[' ' for _ in range(size+2)] for _ in range(size+2)]
    
    # Place walls
    for i in range(size+2):
        grid[i][0] = '#'
        grid[0][i] = '#'
        grid[i][size+1] = '#'
        grid[size+1][i] = '#'
    
    for wall in walls:
        grid[wall[1]+1][wall[0]+1] = '#'
    
    # Place start and end
    grid[start_pos[1]+1][start_pos[0]+1] = 'S'
    grid[end_pos[1]  +1][end_pos[0]  +1] = 'E'
    
    # Convert to string
    lines = []
    for row in grid:
        lines.append(''.join(row))
    
    return '\n'.join(lines)

def main():
    """Generate and display a 10x10 maze."""
    # Set random seed for reproducibility (optional)
    # random.seed(42)
    
    print("Maze Generation Comparison")
    print("=" * 60)
    
    # Generate maze using recursive backtracking (proper maze)
    print("\n1. PROPER MAZE (Recursive Backtracking Algorithm):")
    print("-" * 50)
    walls_proper, start_pos_proper, end_pos_proper = generate_maze_recursive_backtracking(size=10)
    print(format_maze_output(walls_proper, start_pos_proper, end_pos_proper))
    print("\nVisual representation:")
    print(visualize_maze(walls_proper, start_pos_proper, end_pos_proper, size=10))
    print("\nLegend: S=Start, E=End, #=Wall, .=Empty")
    
    # Generate maze using random placement (original algorithm)
    print("\n\n2. RANDOM WALL PLACEMENT (Original Algorithm):")
    print("-" * 50)
    walls_random, start_pos_random, end_pos_random = generate_maze_random(size=10, wall_density=0.4)
    print(format_maze_output(walls_random, start_pos_random, end_pos_random))
    print("\nVisual representation:")
    print(visualize_maze(walls_random, start_pos_random, end_pos_random, size=10))
    print("\nLegend: S=Start, E=End, #=Wall, .=Empty")
    
    print("\n" + "=" * 60)
    print("NOTES:")
    print("- The proper maze guarantees connectivity and a single solution path")
    print("- The random placement may create isolated regions or multiple paths")
    print("- Both mazes have implicit boundary walls (not listed in coordinates)")

if __name__ == "__main__":
    main()