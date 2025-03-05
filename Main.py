import random 
from collections import deque 
from queue import PriorityQueue 
def bms(graph, current_node, goal, visited, path): 
# Add the current node to the path 
path.append(current_node) 
# Print the current path 
    print("Current Path:", path) 
 
    # Check if the current node is the goal 
    if current_node == goal: 
        print(f"Goal {goal} found with path: {path}") 
 
    # Mark the current node as visited 
    visited.add(current_node) 
 
    # Get unvisited neighbors 
    neighbors = [n for n in graph[current_node] if n not in visited] 
 
    if not neighbors: 
        print("No unvisited neighbors left to explore from", current_node) 
    else: 
        # Explore each neighbor randomly 
        for next_node in neighbors: 
            bms(graph, next_node, goal, visited, path) 
 
    # Backtrack: remove the current node from the path and visited set 
    path.pop() 
    visited.remove(current_node) 
     
def bfs(graph, start, goal,visited, path): 
    queue = deque([start]) 
 
    while queue: 
        current_node = queue.popleft() 
        visited.add(current_node) 
        path.append(current_node) 
         
        # Print the current node and path 
        print(f"Visited: {current_node}, Current Path: {path}") 
 
        if current_node == goal: 
            print(f"Goal {goal} found!") 
            return path  # Return the path if the goal is found 
         
        for neighbor in graph[current_node]: 
            if neighbor not in visited and neighbor not in queue: 
                queue.append(neighbor) 
 
    print("Goal not found.") 
    return None  # Goal not found 
 
def dfs(graph, start, goal, visited, path): 
    stack = [start]  # Use a stack for DFS traversal 
 
    while stack: 
        current_node = stack.pop() 
         
        # Visit the current node if not already visited 
        visited.add(current_node) 
        path.append(current_node) 
         
        print(f"Visited: {current_node}, Current Path: {path}") 
 
        if current_node == goal: 
            print(f"Goal {goal} found!") 
            return path  # Return the path if the goal is found 
 
        # Add unvisited neighbors to the stack if not already in stack 
        for neighbor in reversed(graph[current_node]): 
            if neighbor not in visited and neighbor not in stack: 
                stack.append(neighbor) 
 
    print("Goal not found.") 
    return None  # Goal not found 
 
def hill_climbing(graph, start, goal, visited, path): 
    stack = [start] 
     
    while stack: 
        # Get the current node from the stack 
        current_node = stack.pop() 
        visited.add(current_node) 
        path.append(current_node) 
         
        print(f"Visited: {current_node}, Current Path: {path}") 
 
        # Check if the current node is the goal 
        if current_node == goal: 
            print(f"Goal {goal} found!") 
            return path  # Return the path if the goal is found 
         
        # Get unvisited neighbors sorted by their heuristic values 
        neighbors = reversed(sorted(graph[current_node], key=lambda x: heuristic[x])) 
         
        for neighbor in neighbors: 
            # Only add unvisited neighbors to the stack 
            if neighbor not in visited and neighbor not in stack: 
                stack.append(neighbor) 
 
    print("Goal not found.") 
    return None  # Goal not found 
 
def beam_search(graph, start, goal, beam_width): 
    queue = deque([[start]])  # Use a queue to store paths 
 
    while queue: 
        # Limit the size of the queue to beam_width 
        if len(queue) > beam_width: 
            # Sort the current paths by the heuristic value of their last node 
            queue = deque(sorted(queue, key=lambda x: heuristic[x[-1]])[:beam_width]) 
 
        current_path = queue.popleft() 
        current_node = current_path[-1] 
 
        print(f"Visited: {current_node}, Current Path: {current_path}") 
 
        if current_node == goal: 
            print(f"Goal {goal} found with path: {current_path}") 
            return current_path  # Return the path if the goal is found 
 
        # Explore neighbors in a consistent order 
        neighbors = sorted(graph[current_node], key=lambda n: heuristic[n])  # Sort neighbors by 
heuristic 
        for neighbor in neighbors: 
            if neighbor not in current_path:  # Avoid cycles 
                new_path = current_path + [neighbor] 
                queue.append(new_path) 
 
    print("Goal not found.") 
    return None  # Goal not found 
 
def oracle_search(graph, edge_costs, start, goal, input_cost): 
    optimal_paths = [] 
 
    def explore_path(current_node, current_path, current_cost): 
        if current_cost > input_cost: 
            return 
 
        current_path.append(current_node) 
 
        if current_node == goal: 
            optimal_paths.append((list(current_path), current_cost)) 
            print(f"Found path: {current_path} with cost: {current_cost}") 
 
        for neighbor in graph[current_node]: 
            if neighbor not in current_path:  # Avoid cycles 
                explore_path(neighbor, current_path, current_cost + edge_costs.get((current_node, 
neighbor), 0)) 
 
        current_path.pop()  # Backtrack 
 
    explore_path(start, [], 0) 
 
    for path, cost in optimal_paths: 
        if cost < input_cost: 
            print(f"A shorter path found: {path} with cost: {cost}") 
 
def oracle_search_with_heuristics(graph, edge_costs, heuristic, start, goal, input_cost): 
    optimal_paths = [] 
 
    # Recursive function to explore paths 
    def explore_path(current_node, current_path, current_cost): 
        # If the current cost exceeds the input cost, terminate this path 
        if current_cost + heuristic[current_node] > input_cost: 
            return 
 
        # Add current node to the path 
        current_path.append(current_node) 
 
        # If goal is reached, add the path and its cost to optimal_paths 
        if current_node == goal: 
            total_cost = current_cost + heuristic[current_node] 
            optimal_paths.append((list(current_path), total_cost)) 
            print(f"Found path: {current_path} with cost (including heuristic): {total_cost}") 
 
        # Sort neighbors by cumulative cost + heuristic for efficient exploration 
        neighbors = sorted( 
            graph[current_node], 
            key=lambda neighbor: current_cost + edge_costs.get((current_node, neighbor), float('inf')) + 
heuristic[neighbor] 
        ) 
 
        # Recursively explore each sorted neighbor 
        for neighbor in neighbors: 
            if neighbor not in current_path:  # Avoid cycles 
                next_cost = current_cost + edge_costs.get((current_node, neighbor), 0) 
                explore_path(neighbor, current_path, next_cost) 
 
        # Backtrack to previous state 
        current_path.pop() 
 
    # Start the search from the start node 
    explore_path(start, [], 0) 
 
    # Print all optimal paths found within the input cost limit 
    for path, cost in optimal_paths: 
        if cost <= input_cost: 
            print(f"A viable path found: {path} with total cost: {cost}") 
 
 
def branch_and_bound(graph, edge_costs, start, goal): 
    queue = deque([(start, [start], 0)])  # (node, path, cost) 
    best_path, best_cost = None, float('inf') 
 
    while queue: 
        queue = deque(sorted(queue, key=lambda x: x[2]))  # Sort by path cost 
        node, path, cost = queue.popleft() 
 
        print(f"Exploring: {path} with cost: {cost}") 
 
        if node == goal: 
            best_path, best_cost = path, cost 
            print(f"Goal reached: {path} with cost: {cost}") 
            break  # Stop the search as the goal is reached with the lowest path cost 
 
        # Explore neighbors 
        for neighbor in graph[node]: 
            if neighbor not in path:  # Avoid cycles 
                new_cost = cost + edge_costs.get((node, neighbor), float('inf')) 
                if new_cost < best_cost:  # Only consider paths with a lower cost than the best found 
                    queue.append((neighbor, path + [neighbor], new_cost)) 
 
    if best_path: 
        print(f"Optimal path to goal: {best_path} with cost: {best_cost}") 
    else: 
        print("Goal not reachable.") 
 
def branch_and_bound_with_heuristic(graph, edge_costs, start, goal): 
    queue = deque([(start, [start], 0)])  # (node, path, actual cost) 
    best_path, best_cost = None, float('inf') 
 
    while queue: 
        # Sort queue by the estimated total cost (actual cost + heuristic) 
        queue = deque(sorted(queue, key=lambda x: x[2] + heuristic.get(x[0], float('inf')))) 
        node, path, cost = queue.popleft() 
 
        print(f"Exploring: {path} with actual cost: {cost} and estimated total cost: {cost + 
heuristic.get(node, 0)}") 
 
        # Stop if goal is reached with the best cost 
        if node == goal: 
            best_path, best_cost = path, cost 
            print(f"Goal reached: {path} with cost: {cost}") 
            break  # Stop as soon as goal is reached with the lowest path cost 
 
        # Explore neighbors 
        for neighbor in graph[node]: 
            if neighbor not in path:  # Avoid cycles 
                new_cost = cost + edge_costs.get((node, neighbor), float('inf')) 
                if new_cost < best_cost:  # Only add paths with costs lower than the best found 
                    queue.append((neighbor, path + [neighbor], new_cost)) 
 
    if best_path: 
        print(f"Optimal path to goal: {best_path} with total cost: {best_cost}") 
    else: 
        print("Goal not reachable.") 
 
def branch_and_bound_with_extended_list(graph, edge_costs, start, goal): 
    queue = deque([(start, [start], 0)])  # (node, path, cost) 
    best_path, best_cost = None, float('inf') 
    extended = set()  # Keeps track of fully explored nodes 
 
    while queue: 
        # Sort queue by path cost 
        queue = deque(sorted(queue, key=lambda x: x[2])) 
        node, path, cost = queue.popleft() 
 
        print(f"Exploring: {path} with cost: {cost}") 
 
        # Stop the search as soon as we reach the goal 
        if node == goal: 
            best_path, best_cost = path, cost 
            print(f"Goal reached: {path} with cost: {cost}") 
            break 
 
        # Mark the node as extended to avoid re-expanding it 
        extended.add(node) 
 
        # Explore neighbors 
        for neighbor in graph[node]: 
            if neighbor not in path and neighbor not in extended:  # Avoid cycles and revisits 
                new_cost = cost + edge_costs.get((node, neighbor), float('inf')) 
                if new_cost < best_cost:  # Only consider paths with a cost lower than the best found 
                    queue.append((neighbor, path + [neighbor], new_cost)) 
 
    if best_path: 
        print(f"Optimal path to goal: {best_path} with cost: {best_cost}") 
    else: 
        print("Goal not reachable.") 
 
def a_star_algorithm(graph, edge_costs, start, goal): 
    queue = deque([(start, [start], 0)])  # (node, path, actual cost) 
    best_path, best_cost = None, float('inf') 
    extended = set()  # Keeps track of fully explored nodes 
 
    while queue: 
        # Sort queue by estimated total cost (actual cost + heuristic) 
        queue = deque(sorted(queue, key=lambda x: x[2] + heuristic.get(x[0], float('inf')))) 
        node, path, cost = queue.popleft() 
 
        print(f"Exploring: {path} with actual cost: {cost} and estimated total cost: {cost + 
heuristic.get(node, 0)}") 
 
        # Stop the search as soon as we reach the goal 
        if node == goal: 
            best_path, best_cost = path, cost 
            print(f"Goal reached: {path} with cost: {cost}") 
            break 
 
        # Mark the node as extended to avoid re-expanding it 
        extended.add(node) 
 
        # Explore neighbors 
        for neighbor in graph[node]: 
            if neighbor not in path and neighbor not in extended:  # Avoid cycles and revisits 
                new_cost = cost + edge_costs.get((node, neighbor), float('inf')) 
                if new_cost < best_cost:  # Only consider paths with a lower cost than the best found 
                    queue.append((neighbor, path + [neighbor], new_cost)) 
 
    if best_path: 
        print(f"Optimal path to goal: {best_path} with total cost: {best_cost}") 
    else: 
        print("Goal not reachable.") 
 
def ao_star(node, ao_graph, costs, heuristic_values, solution_path={}, goal_reached=False): 
    if goal_reached: 
        return solution_path  # Stop processing if goal is already reached 
     
    print(f"Processing Node: {node}") 
    if node in solution_path: 
        return solution_path[node] 
 
    # Track best path and cost for this node 
    best_cost, best_subpath = float('inf'), None 
    for path_type, children in ao_graph.get(node, []):  # Ensure node has children 
        if path_type == 'AND': 
            path_cost = sum(costs.get((node, child), float('inf')) + heuristic_values.get(child, float('inf')) 
for child in children) 
        elif path_type == 'OR': 
            path_cost = min(costs.get((node, child), float('inf')) + heuristic_values.get(child, float('inf')) 
for child in children) 
 
        if path_cost < best_cost: 
            best_cost, best_subpath = path_cost, children 
 
    # Check if best_subpath was found 
    if best_subpath is None: 
        print(f"No valid path found from node {node}.") 
        return None 
 
    # Update heuristic value and solution path 
    heuristic_values[node] = best_cost 
    solution_path[node] = best_subpath 
    print(f"Node {node} -> Optimal Path {best_subpath} with cost: {best_cost}") 
 
    # Recursively solve for nodes in the selected path 
    for child in best_subpath: 
        if heuristic_values.get(child, float('inf')) == 0:  # Goal node detected 
            print(f"Goal node {child} reached.") 
            goal_reached = True 
            return solution_path 
        if child not in solution_path: 
            ao_star(child, ao_graph, costs, heuristic_values, solution_path, goal_reached) 
 
    return solution_path 
 
def best_first_search(graph, start, goal): 
    open_list = PriorityQueue() 
    open_list.put((heuristic[start], start)) 
    visited = set() 
 
    while not open_list.empty(): 
        _, current = open_list.get() 
         
        if current == goal: 
            print(f"Goal {goal} reached.") 
            return True 
         
        visited.add(current) 
        print(f"Exploring: {current}") 
 
        for neighbor in graph.get(current, []): 
            if neighbor not in visited: 
                open_list.put((heuristic[neighbor], neighbor)) 
     
    print("Goal not reachable.") 
    return False 
 
heuristic = { 
    'S': 6, 
    'A': 3, 
    'B': 4, 
    'C': 7, 
    'D': 2, 
    'E': 8, 
    'G': 0  # Goal node has a heuristic of 0 
} 
 
# graph 
graph = { 
    'S': ['A', 'B'], 
    'A': ['S', 'B', 'D'], 
    'B': ['S', 'A', 'C'], 
    'C': ['B', 'E'], 
    'E': ['C'], 
    'D': ['A', 'G'], 
    'G': ['D'] 
} 
 
edge_costs = { 
    ('S', 'A'): 3, 
    ('S', 'B'): 5, 
    ('A', 'D'): 3, 
    ('B', 'C'): 4, 
    ('C', 'E'): 6, 
    ('D', 'G'): 5, 
    ('A', 'B'): 4 
} 
 
 
# Start the search 
ao_graph = { 
    'A': [('OR', ['B', 'C'])], 
    'B': [('AND', ['D', 'E'])], 
    'C': [('AND', ['F'])], 
    'D': [('OR', ['G'])], 
    'E': [('OR', ['G'])], 
    'F': [('OR', ['G'])], 
    'G': []  # Goal node 
} 
 
# Edge costs 
costs = { 
    ('A', 'B'): 1, ('A', 'C'): 2, 
    ('B', 'D'): 1, ('B', 'E'): 2, 
    ('C', 'F'): 3, ('D', 'G'): 1, 
    ('E', 'G'): 2, ('F', 'G'): 1 
} 
 
# Heuristic values for each node 
heuristic_values = { 
    'A': 4, 'B': 3, 'C': 2, 
    'D': 1, 'E': 1, 'F': 1, 
    'G': 0  # Goal node 
} 
 
ch=int(input("Enter your choice:")) 
if(ch==1): 
   print("British Museum Search") 
   bms(graph, 'S', 'G', set(), []) 
elif(ch==2): 
   print("Breadth First Search") 
   result = bfs(graph, 'S', 'G', set(), []) 
   print("Final Path Found (BFS):", result) 
elif ch == 3: 
   print("Depth First Search") 
   result = dfs(graph, 'S', 'G', set(), []) 
   print("Final Path Found (DFS):", result) 
elif ch == 4: 
    print("Hill Climbing") 
    result = hill_climbing(graph, 'S', 'G', set(), []) 
    print("Final Path Found (Hill Climbing):", result) 
elif ch == 5: 
    print("Beam Search") 
    beam_width = int(input("Enter beam width: ")) 
    result = beam_search(graph, 'S', 'G', beam_width) 
    print("Final Path Found (Beam Search):", result) 
elif ch==6: 
    print("Oracle Search") 
    input_cost = 15 
    oracle_search(graph, edge_costs, 'S', 'G', input_cost) 
elif ch==7: 
    print("Oracle Search with Heuristics") 
    input_cost = 20 
    oracle_search_with_heuristics(graph, edge_costs, heuristic, 'S', 'G', input_cost) 
elif ch==8: 
    print("Branch and Bound") 
    branch_and_bound(graph, edge_costs, 'S', 'G') 
elif ch==9: 
    print("Branch and Bound with Heuristics") 
    branch_and_bound_with_heuristic(graph, edge_costs, 'S', 'G') 
elif ch==10: 
    print("Branch and Bound with Extended List") 
    branch_and_bound_with_extended_list(graph, edge_costs, 'S', 'G') 
elif ch==11: 
    print("A* Algorithm") 
    a_star_algorithm(graph, edge_costs, 'S', 'G') 
elif ch==12: 
    print("AO* Algorithm") 
    solutiosn = ao_star('A', ao_graph, costs, heuristic_values) 
    print("\nOptimal Solution Path:", solution) 
elif ch==13: 
    print("Best First Search") 
    best_first_search(graph, 'S', 'G') 
