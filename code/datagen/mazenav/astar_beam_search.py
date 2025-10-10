from __future__ import annotations

from dataclasses import dataclass
import heapq
import os
import sys
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_maze import (
    generate_maze_recursive_backtracking, visualize_maze
)
    

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class BeamEntry(Generic[StateT]):
    """Container for a state with its cumulative cost and heuristic."""

    state: StateT
    g_cost: float
    h_cost: float

    @property
    def f_score(self) -> float:
        return self.g_cost + self.h_cost


class AStarBeamSearch(Generic[StateT]):
    """
    A* flavored Beam Search with custom expansion order (by cost + heuristic).

    Required callables provided via constructor:
    - next_states: (state) -> Iterable[(neighbor_state, step_cost)] OR Iterable[neighbor_state]
      If step_cost is not provided, a default step cost of 1.0 is assumed.
    - cost_fn: (path_cost_so_far, from_state, to_state) -> new_path_cost
      If you already pass step_cost in next_states, you may set cost_fn to `None` to
      indicate additive costs: new_g = g + step_cost.
    - heuristic_fn: (state) -> heuristic_value
    - is_goal_fn: (state) -> bool

    The solve() method returns (solution_path or None, state_matrix, backpointer_matrix).
    The matrices are shaped as [k, L] where L is the number of beam iterations actually run.
    """

    def __init__(
        self,
        next_states: Callable[[StateT], Iterable[Any]],
        heuristic_fn: Callable[[StateT], float],
        is_goal_fn: Callable[[StateT], bool],
        cost_fn: Optional[Callable[[float, StateT, StateT, Optional[float]], float]] = None,
    ) -> None:
        self.next_states = next_states
        self.heuristic_fn = heuristic_fn
        self.is_goal_fn = is_goal_fn
        self.cost_fn = cost_fn

    def _expand_state(self, entry: BeamEntry[StateT]) -> List[Tuple[StateT, float]]:
        """
        Expand a state using `next_states`.
        Supports two formats from `next_states(state)`:
          - Iterable[neighbor_state]
          - Iterable[(neighbor_state, step_cost)]
        Returns: list of (neighbor_state, new_g_cost)
        """
        expanded: List[Tuple[StateT, float]] = []
        for candidate in self.next_states(entry.state):
            if isinstance(candidate, tuple) and len(candidate) >= 2:
                neighbor, step_cost = candidate[0], float(candidate[1])    
            else:
                neighbor, step_cost = candidate, 1.0  # type: ignore[assignment]
            if self.cost_fn is None:
                new_g = entry.g_cost + step_cost
            else:
                new_g = self.cost_fn(entry.g_cost, entry.state, neighbor, step_cost)
            expanded.append((neighbor, new_g))
        return expanded

    def solve(
        self,
        start_state: StateT,
        beam_size: int,
        max_steps: Optional[int] = None,
        return_path: bool = True,
    ) -> Tuple[Optional[List[StateT]], List[List[Optional[StateT]]], List[List[Optional[int]]], float]:
        """
        Run beam search.

        Args:
            start_state: The initial problem state.
            beam_size: Beam width k.
            max_steps: Optional maximum number of beam iterations.
            return_path: Whether to reconstruct and return the solution path if found.

        Returns:
            (solution_path_or_None, state_matrix, backpointer_matrix, solution_cost)
            - state_matrix: shape [k, L] where L = iterations performed; entries are states or None
            - backpointer_matrix: shape [k, L]; entries are parent rank indices (int) or None
            - solution_cost: cost of the solution path, or float('inf') if no solution found
        """
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")

        # Initialize beam with start state
        start_entry = BeamEntry(state=start_state, g_cost=0.0, h_cost=self.heuristic_fn(start_state))
        beam: List[BeamEntry[StateT]] = [start_entry]

        # For path reconstruction: map (iteration_index, rank) -> (prev_iteration_rank, state)
        # Also keep best_seen to avoid revisiting identical states with higher cost
        best_g_for_state: Dict[Any, float] = {start_state: 0.0}

        # Matrices: rows=k, columns=iterations (we will append a column per iteration)
        state_matrix: List[List[Optional[StateT]]] = [[] for _ in range(beam_size)]
        backpointer_matrix: List[List[Optional[int]]] = [[] for _ in range(beam_size)]

        # Maintain auxiliary structure to reconstruct path: per-iteration list of states
        per_iter_states: List[List[StateT]] = []

        iteration: int = 0
        goal_found: Optional[Tuple[int, StateT, float]] = None  # (rank_in_beam, state, cost)

        while True:
            # Record current beam into matrices (as a column)
            sorted_beam = sorted(beam, key=lambda e: e.f_score)
            per_iter_states.append([e.state for e in sorted_beam])

            # Place states into state_matrix column-wise; pad with None if beam < k
            for row in range(beam_size):
                if row < len(sorted_beam):
                    state_matrix[row].append(sorted_beam[row].state)
                else:
                    state_matrix[row].append(None)
                backpointer_matrix[row].append(None)

            # If there are pending backpointers from the previous expansion, write them
            pending_bp: Optional[List[Optional[int]]] = getattr(self, "_pending_backpointers", None)
            if pending_bp is not None:
                col_idx = len(state_matrix[0]) - 1
                for row in range(beam_size):
                    if row < len(pending_bp):
                        backpointer_matrix[row][col_idx] = pending_bp[row]
                delattr(self, "_pending_backpointers")

            # Check for goal in the current beam
            for rank, entry in enumerate(sorted_beam):
                if self.is_goal_fn(entry.state):
                    goal_found = (rank, entry.state, entry.g_cost)
                    beam = sorted_beam
                    break

            if goal_found is not None:
                break

            # Termination by step limit
            if max_steps is not None and iteration + 1 >= max_steps:
                break

            # Expand each beam entry
            candidates: List[Tuple[BeamEntry[StateT], int]] = []  # (entry, parent_rank)
            for parent_rank, parent in enumerate(sorted_beam):
                for neighbor, new_g in self._expand_state(parent):
                    # Dominance pruning: keep best g for a state
                    prev_best = best_g_for_state.get(neighbor)
                    if prev_best is not None and prev_best <= new_g:
                        continue
                    best_g_for_state[neighbor] = new_g
                    neighbor_entry = BeamEntry(state=neighbor, g_cost=new_g, h_cost=self.heuristic_fn(neighbor))
                    candidates.append((neighbor_entry, parent_rank))

            # If no candidates, search cannot proceed
            if not candidates:
                break

            # Select top-k by f = g + h
            candidates.sort(key=lambda x: x[0].f_score)
            next_beam_with_bp = candidates[:beam_size]

            # Fill backpointers for next column we will create on the next loop
            # To align with spec, we will put the backpointer in the just-created column
            # corresponding to the next beam (iteration+1). So we defer writing until after selection.
            # Build next column now:
            # First, prepare an empty next column to be populated at the start of next loop.
            # But we also need to store backpointers for the candidates chosen now.
            next_backpointers: List[Optional[int]] = [None] * beam_size
            for idx in range(beam_size):
                if idx < len(next_beam_with_bp):
                    next_backpointers[idx] = next_beam_with_bp[idx][1]
                else:
                    next_backpointers[idx] = None

            # Advance beam
            beam = [pair[0] for pair in next_beam_with_bp]

            # To record the backpointers as column (iteration+1), we can only insert them when that column exists.
            # So we temporarily stash them and after writing the next states (next loop), we'll overwrite backpointers.
            # Simple approach: keep a stash variable.
            setattr(self, "_pending_backpointers", next_backpointers)

            iteration += 1

            # Continue loop
            continue

        # If we exit while loop without writing pending backpointers for the last appended column, ensure consistency.
        # Write pending backpointers onto the last column if available and sizes match.
        pending_bp: Optional[List[Optional[int]]] = getattr(self, "_pending_backpointers", None)
        if pending_bp is not None:
            last_col_index = len(state_matrix[0]) - 1
            if last_col_index >= 0:
                for row in range(beam_size):
                    if row < len(pending_bp):
                        backpointer_matrix[row][last_col_index] = pending_bp[row]
            # Clear the stash
            delattr(self, "_pending_backpointers")

        # If goal found and a path is requested, attempt to reconstruct a path approximation using per_iter_states
        solution_path: Optional[List[StateT]] = None
        solution_cost: float = float('inf')
        
        if goal_found is not None:
            goal_rank, goal_state, goal_cost = goal_found
            solution_cost = goal_cost
            
            if return_path:
                # We don't track full per-node parent references beyond ranks, so reconstruct using backpointers matrix
                # Walk backwards from the goal column to the first column using rank indices
                last_col = len(state_matrix[0]) - 1
                # Ensure goal is in last column; if not, adjust to the column where it was found
                # Find the column where goal_state appears
                found_col = None
                for col_idx in range(len(state_matrix[0]) - 1, -1, -1):
                    column_states = [state_matrix[row][col_idx] for row in range(beam_size)]
                    for r, s in enumerate(column_states):
                        if s == goal_state:
                            goal_rank = r
                            found_col = col_idx
                            break
                    if found_col is not None:
                        break

                if found_col is None:
                    # Fallback: return just the goal
                    solution_path = [goal_state]
                else:
                    path: List[StateT] = []
                    current_rank = goal_rank
                    for col in range(found_col, -1, -1):
                        state_at_rank = state_matrix[current_rank][col]
                        if state_at_rank is not None:
                            path.append(state_at_rank)
                        bp = backpointer_matrix[current_rank][col]
                        if bp is None:
                            # Reached the first column or missing pointer
                            break
                        current_rank = bp
                    path.reverse()
                    solution_path = path

        return solution_path, state_matrix, backpointer_matrix, solution_cost

    def solve_normal_astar(
        self,
        start_state: StateT,
        max_steps: Optional[int] = None,
        return_path: bool = True,
    ) -> Tuple[Optional[List[StateT]], float]:
        """
        Run standard A* search.
        
        Args:
            start_state: The initial problem state.
            max_steps: Optional maximum number of steps to explore.
            return_path: Whether to reconstruct and return the solution path if found.
            
        Returns:
            (solution_path_or_None, total_cost)
            - solution_path: List of states from start to goal, or None if not found
            - total_cost: Cost of the solution path, or float('inf') if not found
        """
        
        # Priority queue: (f_score, g_cost, state)
        # Using g_cost as tiebreaker for consistent ordering
        open_set: List[Tuple[float, float, StateT, Optional[StateT], float]] = []
        heapq.heappush(open_set, (0.0, 0.0, start_state))
        
        # Track best g_cost for each state (for duplicate detection)
        g_costs: Dict[StateT, float] = {start_state: 0.0}
        
        # For path reconstruction: state -> parent_state
        came_from: Dict[StateT, StateT] = {}
        
        steps = 0
        
        while open_set:
            # Check step limit
            if max_steps is not None and steps >= max_steps:
                break
                
            # Get state with lowest f_score
            f_score, g_cost, current_state = heapq.heappop(open_set)
            
            # Skip if we've found a better path to this state
            if g_cost > g_costs.get(current_state, float('inf')):
                continue
                
            # Check if goal
            if self.is_goal_fn(current_state):
                if return_path:
                    # Reconstruct path
                    path = []
                    state = current_state
                    while state is not None:
                        path.append(state)
                        state = came_from.get(state)
                    path.reverse()
                    return path, g_cost
                else:
                    return None, g_cost
            
            # Expand current state
            for neighbor, tentative_g in self._expand_state(
                BeamEntry(state=current_state, g_cost=g_cost, h_cost=0.0)
            ):  
                # Skip if we've already found a better path to this neighbor
                if tentative_g >= g_costs.get(neighbor, float('inf')):
                    continue
                    
                # Update best path to neighbor
                g_costs[neighbor] = tentative_g
                came_from[neighbor] = current_state
                
                # Calculate f_score and add to open set
                h_cost = self.heuristic_fn(neighbor)
                f_score = tentative_g + h_cost
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
            
            steps += 1
        
        # No solution found
        return None, float('inf')


def example_maze_search():
    """
    Example usage of AStarBeamSearch with a randomly generated maze.
    """
    SIZE = 20
    # Generate a small maze for demonstration
    maze = generate_maze_recursive_backtracking(size=SIZE)
    # maze = generate_maze_random(size=SIZE, wall_density=0.4)
    
    print("Generated Maze:")
    print(visualize_maze(maze))
    print(f"Start: {maze.start_pos}, End: {maze.end_pos}")
    print(f"Walls: {maze.walls}")
    print()
    
    # Create beam search instance using the maze's navigation methods
    beam_search = AStarBeamSearch(
        next_states=maze.get_neighbors,
        heuristic_fn=maze.manhattan_distance,
        is_goal_fn=maze.is_goal,
        cost_fn=None  # Use additive costs (g + step_cost)
    )
    
    # Run normal A* search first
    print("Running Normal A* Search:")
    astar_solution, astar_cost = beam_search.solve_normal_astar(
        start_state=maze.start_pos,
        max_steps=1000
    )
    print(f"A* Solution found: {astar_solution is not None}")
    if astar_solution:
        print(f"A* Solution path length: {len(astar_solution)}")
        print(f"A* Total cost: {astar_cost}")
        print(f"A* Path: {astar_solution}")
    print()
    
    # Run beam search
    print("Running Beam Search:")
    solution, state_matrix, backpointer_matrix, solution_cost = beam_search.solve(
        start_state=maze.start_pos,
        beam_size=3,
        max_steps=None
    )
    
    print("Beam Search Results:")
    print(f"Solution found: {solution is not None}")
    if solution:
        print(f"Solution path: {solution}")
        print(f"Path length: {len(solution)}")
        print(f"Solution cost: {solution_cost}")
    
    print(f"\nState matrix shape: {len(state_matrix)} x {len(state_matrix[0]) if state_matrix else 0}")
    print(f"Backpointer matrix shape: {len(backpointer_matrix)} x {len(backpointer_matrix[0]) if backpointer_matrix else 0}")
    
    # Show first few iterations of the beam
    if state_matrix and state_matrix[0]:
        print("\nFirst few beam iterations:")
        for col in range(min(3, len(state_matrix[0]))):
            print(f"Iteration {col}:")
            for row in range(len(state_matrix)):
                state = state_matrix[row][col]
                backptr = backpointer_matrix[row][col]
                print(f"  Beam[{row}]: state={state}, parent_rank={backptr}")
    
    # Compare results
    print("\nComparison:")
    if astar_solution and solution:
        print(f"A* path length: {len(astar_solution)}, Beam path length: {len(solution)}")
        print(f"A* cost: {astar_cost}, Beam cost: {solution_cost}")
        print(f"Both found the same solutions: {astar_solution == solution}")
    elif astar_solution:
        print("Only A* found a solution")
    elif solution:
        print("Only Beam search found a solution")
    else:
        print("Neither method found a solution")


if __name__ == "__main__":
    # Run single example
    example_maze_search()