#!/usr/bin/env python3
"""
Generate a textual 10x10 maze with randomly placed walls, start, and end positions.
The maze uses a 0-based index grid where walls are specified as a list of coordinates.
"""

import random
import sys
import json
import base64
import zlib
from typing import List, Tuple, Set, Dict, Any, Optional

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

def visualize_maze(walls: List[Tuple[int, int]], start_pos: Tuple[int, int], end_pos: Tuple[int, int], size: int = 10, path: Optional[List[Tuple[int, int]]] = None) -> str:
    """
    Create a visual representation of the maze using ASCII characters.
    
    Args:
        walls: List of wall coordinates
        start_pos: Start position coordinates
        end_pos: End position coordinates
        size: Size of the maze
        path: Optional path from start to end, showing direction arrows
    
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
    grid[start_pos[1]+1][start_pos[0]+1] = 'S'
    grid[end_pos[1]  +1][end_pos[0]  +1] = 'E'
    
    # Convert to string
    lines = []
    for row in grid:
        lines.append(''.join(row))
    
    return '\n'.join(lines)

# =============================================================================
# CONVENIENCE FUNCTIONS FOR MAZE STORAGE AND COMPRESSION
# =============================================================================

def format_maze_compact(walls: List[Tuple[int, int]], start_pos: Tuple[int, int], 
                       end_pos: Tuple[int, int], size: int) -> str:
    """
    Format a maze in compact format for storage and later reconstruction : "size:walls_encoded:start:end"
    - size: Grid size as integer
    - walls_encoded: Base64-encoded JSON list of wall coordinates
    - start: Start position as "x,y"
    - end: End position as "x,y"
    
    Args:
        walls: List of wall coordinates
        start_pos: Start position coordinates
        end_pos: End position coordinates
        size: Size of the maze grid
    
    Returns:
        Compact string representation
    """
    # Encode walls as base64 JSON for compactness
    walls_json = json.dumps(walls, separators=(',', ':'))  # Compact JSON
    walls_encoded = base64.b64encode(walls_json.encode('utf-8')).decode('ascii')
    
    # Format as compact string
    compact_format = f"{size}:{walls_encoded}:{start_pos[0]},{start_pos[1]}:{end_pos[0]},{end_pos[1]}"
    
    return compact_format

def reconstruct_maze_from_compact(compact_format: str) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]:
    """
    Reconstruct maze data from compact format.
    
    Args:
        compact_format: Compact string format from format_maze_compact()
    
    Returns:
        Tuple of (walls, start_pos, end_pos, size)
    """
    parts = compact_format.split(':')
    if len(parts) != 4:
        raise ValueError("Invalid compact format")
    
    size = int(parts[0])
    
    # Decode walls
    walls_encoded = parts[1]
    walls_json = base64.b64decode(walls_encoded.encode('ascii')).decode('utf-8')
    walls = json.loads(walls_json)
    walls = [(int(x), int(y)) for x, y in walls]  # Ensure integers
    
    # Parse start and end positions
    start_x, start_y = map(int, parts[2].split(','))
    end_x, end_y = map(int, parts[3].split(','))
    
    start_pos = (start_x, start_y)
    end_pos = (end_x, end_y)
    
    return walls, start_pos, end_pos, size

def compress_maze(compact_format: str) -> str:
    """
    Compress the compact maze format further using zlib compression and base64 encoding.
    
    Compression algorithm:
    1. Take the compact format string
    2. Compress using zlib (deflate algorithm) - reduces redundant data
    3. Encode compressed bytes as base64 for safe string transmission
    
    This achieves compression by:
    - zlib/deflate removes redundancy in the JSON data and coordinate patterns
    - Base64 encoding ensures the compressed data is ASCII-safe
    - Typical compression ratios: 60-80% for maze data
    
    Args:
        compact_format: Compact string format from format_maze_compact()
    
    Returns:
        Compressed and base64-encoded string
    """
    # Compress the compact format string
    compressed_bytes = zlib.compress(compact_format.encode('utf-8'), level=9)  # Maximum compression
    
    # Encode as base64 for safe string handling
    compressed_string = base64.b64encode(compressed_bytes).decode('ascii')
    
    return compressed_string

def decompress_maze(compressed_string: str) -> str:
    """
    Decompress a compressed maze format back to the original compact format.
    
    Decompression algorithm:
    1. Decode base64 string back to compressed bytes
    2. Decompress using zlib (inflate algorithm)
    3. Decode bytes back to original compact format string
    
    Args:
        compressed_string: Compressed string from compress_maze()
    
    Returns:
        Original compact format string
    """
    # Decode base64 back to compressed bytes
    compressed_bytes = base64.b64decode(compressed_string.encode('ascii'))
    
    # Decompress using zlib
    compact_format = zlib.decompress(compressed_bytes).decode('utf-8')
    
    return compact_format

def save_maze_compressed(walls: List[Tuple[int, int]], start_pos: Tuple[int, int], 
                        end_pos: Tuple[int, int], size: int, filename: str) -> None:
    """
    Save a maze in compressed format to a file.
    
    Args:
        walls: List of wall coordinates
        start_pos: Start position coordinates
        end_pos: End position coordinates
        size: Size of the maze grid
        filename: Output filename
    """
    compact_format = format_maze_compact(walls, start_pos, end_pos, size)
    compressed_format = compress_maze(compact_format)
    
    with open(filename, 'w') as f:
        f.write(compressed_format)

def load_maze_compressed(filename: str) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]:
    """
    Load a maze from a compressed file format.
    
    Args:
        filename: Input filename
    
    Returns:
        Tuple of (walls, start_pos, end_pos, size)
    """
    with open(filename, 'r') as f:
        compressed_format = f.read().strip()
    
    compact_format = decompress_maze(compressed_format)
    return reconstruct_maze_from_compact(compact_format)

# =============================================================================
# MULTI-MAZE STORAGE AND COMPRESSION FUNCTIONS
# =============================================================================

def format_multiple_mazes_compact(mazes: List[Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]]) -> str:
    """
    Format multiple mazes in compact format for storage.
    
    Multi-maze format: "count|maze1|maze2|...|mazeN"
    - count: Number of mazes as integer
    - maze1, maze2, etc.: Individual maze compact formats (separated by |)
    
    Args:
        mazes: List of maze tuples (walls, start_pos, end_pos, size)
    
    Returns:
        Compact string representation of multiple mazes
    """
    maze_compacts = []
    for walls, start_pos, end_pos, size in mazes:
        compact = format_maze_compact(walls, start_pos, end_pos, size)
        maze_compacts.append(compact)
    
    # Join with count prefix using | as delimiter (since : is used within maze formats)
    multi_maze_format = f"{len(mazes)}|" + "|".join(maze_compacts)
    return multi_maze_format

def reconstruct_multiple_mazes_from_compact(multi_maze_format: str) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]]:
    """
    Reconstruct multiple maze data from compact format.
    
    Args:
        multi_maze_format: Compact string format from format_multiple_mazes_compact()
    
    Returns:
        List of maze tuples (walls, start_pos, end_pos, size)
    """
    parts = multi_maze_format.split('|')
    if len(parts) < 2:
        raise ValueError("Invalid multi-maze format")
    
    count = int(parts[0])
    if len(parts) != count + 1:
        raise ValueError(f"Expected {count + 1} parts, got {len(parts)}")
    
    mazes = []
    for i in range(1, count + 1):
        maze_compact = parts[i]
        walls, start_pos, end_pos, size = reconstruct_maze_from_compact(maze_compact)
        mazes.append((walls, start_pos, end_pos, size))
    
    return mazes

def save_multiple_mazes_compressed(mazes: List[Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]], filename: str) -> None:
    """
    Save multiple mazes in compressed format to a file.
    
    Args:
        mazes: List of maze tuples (walls, start_pos, end_pos, size)
        filename: Output filename
    """
    multi_maze_format = format_multiple_mazes_compact(mazes)
    compressed_format = compress_maze(multi_maze_format)
    
    with open(filename, 'w') as f:
        f.write(compressed_format)

def load_multiple_mazes_compressed(filename: str) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, int], int]]:
    """
    Load multiple mazes from a compressed file format.
    
    Args:
        filename: Input filename
    
    Returns:
        List of maze tuples (walls, start_pos, end_pos, size)
    """
    with open(filename, 'r') as f:
        compressed_format = f.read().strip()
    
    multi_maze_format = decompress_maze(compressed_format)
    return reconstruct_multiple_mazes_from_compact(multi_maze_format)

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
    
    # Test the new convenience functions
    print("\n" + "=" * 60)
    print("TESTING CONVENIENCE FUNCTIONS:")
    print("-" * 50)
    
    # Test with the proper maze
    compact_format = format_maze_compact(walls_proper, start_pos_proper, end_pos_proper, size=10)
    print(f"Compact format size: {len(compact_format)} characters")
    print(f"Compact format: {compact_format}")
    
    compressed = compress_maze(compact_format)
    print(f"Compressed size: {len(compressed)} characters")
    print(f"Compression ratio: {len(compressed)/len(compact_format):.2%}")
    print(f"Compressed data: {compressed}")
    
    decompressed = decompress_maze(compressed)
    print(f"Decompressed matches original: {decompressed == compact_format}")
    
    # Test reconstruction
    reconstructed_walls, reconstructed_start, reconstructed_end, reconstructed_size = reconstruct_maze_from_compact(decompressed)
    print(f"Reconstruction successful: {reconstructed_walls == walls_proper and reconstructed_start == start_pos_proper and reconstructed_end == end_pos_proper}")
    
    # Test multi-maze functionality
    print("\n" + "=" * 60)
    print("TESTING MULTI-MAZE FUNCTIONALITY:")
    print("-" * 50)
    
    # Generate multiple mazes for testing
    test_mazes = []
    for i in range(3):
        walls, start_pos, end_pos = generate_maze_recursive_backtracking(size=8)
        test_mazes.append((walls, start_pos, end_pos, 8))
    
    print(f"Generated {len(test_mazes)} test mazes")
    
    # Test multi-maze compact format
    multi_compact = format_multiple_mazes_compact(test_mazes)
    print(f"Multi-maze compact format size: {len(multi_compact)} characters")
    
    # Test multi-maze compression
    multi_compressed = compress_maze(multi_compact)
    print(f"Multi-maze compressed size: {len(multi_compressed)} characters")
    print(f"Multi-maze compression ratio: {len(multi_compressed)/len(multi_compact):.2%}")
    
    # Test multi-maze decompression and reconstruction
    multi_decompressed = decompress_maze(multi_compressed)
    reconstructed_mazes = reconstruct_multiple_mazes_from_compact(multi_decompressed)
    
    print(f"Multi-maze decompression successful: {multi_decompressed == multi_compact}")
    print(f"Multi-maze reconstruction successful: {len(reconstructed_mazes) == len(test_mazes)}")
    
    # Verify each maze matches
    all_match = True
    for i, (original, reconstructed) in enumerate(zip(test_mazes, reconstructed_mazes)):
        walls_match = original[0] == reconstructed[0]
        start_match = original[1] == reconstructed[1]
        end_match = original[2] == reconstructed[2]
        size_match = original[3] == reconstructed[3]
        maze_match = walls_match and start_match and end_match and size_match
        print(f"Maze {i+1} matches: {maze_match}")
        if not maze_match:
            all_match = False
    
    print(f"All mazes match: {all_match}")
    
    # Test file I/O with multiple mazes
    test_filename = "test_multiple_mazes.txt"
    save_multiple_mazes_compressed(test_mazes, test_filename)
    loaded_mazes = load_multiple_mazes_compressed(test_filename)
    
    print(f"File I/O test successful: {len(loaded_mazes) == len(test_mazes)}")
    
    # Clean up test file
    import os
    if os.path.exists(test_filename):
        os.remove(test_filename)
        print(f"Cleaned up test file: {test_filename}")

if __name__ == "__main__":
    main()