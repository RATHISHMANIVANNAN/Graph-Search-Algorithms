# Graph Search Algorithms Implementation

## Overview
This project implements multiple search algorithms in a single Python script (`main.py`). The goal is to allow users to choose from various search techniques to explore a given graph and find a path to the goal node.

## Features
- **Multiple Search Algorithms:** Includes classical, heuristic-based, and optimization techniques.
- **Graph Input:** Users can define a graph structure and specify a start and goal node.
- **User Choice:** The script dynamically executes the selected search algorithm.
- **Path Visualization:** Outputs the sequence of visited nodes and the final path to the goal.

## Algorithms Implemented
### Uninformed Search Algorithms
1. **British Museum Search**
2. **Breadth-First Search (BFS)**
3. **Depth-First Search (DFS)**
4. **Branch and Bound**
5. **Branch and Bound with Extended List**

### Heuristic-Based Search Algorithms
6. **Hill Climbing**
7. **Beam Search**
8. **Best-First Search**
9. **Branch and Bound with Heuristics**
10. **A* Algorithm**
11. **AO* Algorithm**

### Oracle-Based Search Algorithms
12. **Oracle Search**
13. **Oracle Search with Heuristics**

## How It Works
1. **Initialize the Graph:** Define the nodes and their connections.
2. **Choose an Algorithm:** The user selects an algorithm to execute.
3. **Perform Search:** The selected algorithm explores the graph.
4. **Output Results:** The script prints visited nodes, paths, and final results.

## Algorithm Descriptions
### British Museum Search
- Explores all possible paths in an exhaustive manner until the goal is found.
- Uses backtracking when a path is exhausted.

### Breadth-First Search (BFS)
- Explores the graph level by level using a queue.
- Guarantees the shortest path in an unweighted graph.

### Depth-First Search (DFS)
- Explores as deep as possible using a stack before backtracking.
- Can be implemented recursively or iteratively.

### Hill Climbing
- Expands the best immediate neighbor based on heuristic values.
- May get stuck in local optima without backtracking.

### Beam Search
- Uses a fixed number of best candidates (beam width) at each level.
- Limits the number of expanded nodes to improve efficiency.

### Oracle Search
- Uses predefined knowledge of optimal paths to guide exploration.
- Can be heuristic-based for better efficiency.

### A* Algorithm
- Uses the sum of path cost and heuristic to find the most promising node.
- Guarantees the shortest path if heuristics are admissible.

### AO* Algorithm
- Works with AND-OR graphs for complex problem-solving.
- Recursively finds the best subpath using heuristic evaluation.

## Running the Code
1. Clone this repository:
   ```sh
   git clone https://github.com/RATHISHMANIVANNAN/Graph-Search-Algorithms.git
   cd search-algorithms
   ```
2. Run the Python script:
   ```sh
   python main.py
   ```
3. Follow the on-screen prompts to select an algorithm and see the results.

## Example Output
```
Enter the search algorithm: BFS
Nodes visited: [S, A, B, C, G]
Path to goal: S -> A -> G
```

## Future Improvements
- Implement additional search algorithms like IDA* and Genetic Algorithm.
- Improve visualization with graphical representation of search paths.
- Optimize performance for large-scale graphs.

## Contributions
Feel free to submit pull requests or suggest enhancements.

## License
This project is licensed under the MIT License.

