import networkx as nx
import matplotlib.pyplot as plt
import heapq
import numpy as np
import time
import tracemalloc

def create_complex_city_graph():
    G = nx.DiGraph()
    edges = [
        ('A', 'B', 5), ('A', 'C', 2), ('B', 'C', 6), ('B', 'D', 2), ('C', 'D', 4),
        ('C', 'E', 8), ('D', 'E', 1), ('E', 'F', 5), ('D', 'G', 3), ('E', 'H', 6),
        ('F', 'I', 4), ('G', 'H', 7), ('G', 'J', 5), ('H', 'K', 9), ('I', 'K', 6),
        ('J', 'K', 8), ('A', 'J', 10), ('I', 'L', 7), ('J', 'L', 3), ('K', 'L', 2),
        ('L', 'M', 3), ('K', 'M', 1), ('H', 'N', 2), ('M', 'N', 4), ('L', 'O', 5),
        ('N', 'O', 1)
    ]
    G.add_weighted_edges_from(edges)
    return G

positions = {
    'A': (0, 1), 'B': (1, 2), 'C': (1, 0), 'D': (2, 2), 'E': (2, 0),
    'F': (3, 1), 'G': (3, 3), 'H': (4, 0), 'I': (4, 2), 'J': (5, 3),
    'K': (5, 1), 'L': (6, 2), 'M': (7, 1), 'N': (6, 0), 'O': (7, 0)
}

def heuristic(node, goal, positions):
    h = np.sqrt((positions[node][0] - positions[goal][0])**2 + (positions[node][1] - positions[goal][1])**2)
    return h

def run_algorithm(graph, start, goal, positions, algorithm):
    tracemalloc.start()
    start_time = time.time()
    came_from, visited_nodes = algorithm(graph, start, goal, positions)
    duration = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return came_from, visited_nodes, duration, current, peak

def dijkstra(graph, start, goal, positions):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {node: float('infinity') for node in graph.nodes()}
    cost_so_far[start] = 0
    visited_nodes = []

    while open_set:
        current_cost, current = heapq.heappop(open_set)
        visited_nodes.append(current)
        if current == goal:
            break
        for neighbor in graph[current]:
            new_cost = current_cost + graph[current][neighbor]['weight']
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost, neighbor))
                came_from[neighbor] = current

    return came_from, visited_nodes

def a_star(graph, start, goal, positions):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal, positions), 0, start))
    came_from = {}
    cost_so_far = {node: float('infinity') for node in graph.nodes()}
    cost_so_far[start] = 0
    visited_nodes = []

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        visited_nodes.append(current)
        if current == goal:
            break
        for neighbor in graph[current]:
            new_cost = current_g + graph[current][neighbor]['weight']
            h_value = heuristic(neighbor, goal, positions)
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + h_value
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return came_from, visited_nodes

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()
    return path

def draw_graph(graph, path, positions, goal, title):
    plt.figure(figsize=(12, 8))
    nx.draw(graph, positions, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_nodes(graph, positions, nodelist=path, node_color='green', node_size=500)
    nx.draw_networkx_edges(graph, positions, edgelist=path_edges, edge_color='green', width=2)

    edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels)

    heuristic_labels = {node: f"h={heuristic(node, goal, positions):.2f}" for node in graph.nodes()}
    for node, (x, y) in positions.items():
        plt.text(x, y-0.1, s=heuristic_labels[node], color='red')

    plt.title(title)
    plt.show()

G = create_complex_city_graph()
goal_node = 'O'
dijkstra_result, dijkstra_visited, dijkstra_duration, dijkstra_mem_current, dijkstra_mem_peak = run_algorithm(G, 'A', goal_node, positions, dijkstra)
a_star_result, a_star_visited, a_star_duration, a_star_mem_current, a_star_mem_peak = run_algorithm(G, 'A', goal_node, positions, a_star)

dijkstra_path = reconstruct_path(dijkstra_result, 'A', goal_node)
a_star_path = reconstruct_path(a_star_result, 'A', goal_node)

print("Dijkstra Algorithm:")
print("Visited Nodes:", dijkstra_visited)
print("Shortest Path:", dijkstra_path)
print("Duration:", dijkstra_duration, "seconds")
print("Memory used: Current = {} bytes, Peak = {} bytes".format(dijkstra_mem_current, dijkstra_mem_peak))

print("\nA* Algorithm:")
print("Visited Nodes:", a_star_visited)
print("Shortest Path:", a_star_path)
print("Duration:", a_star_duration, "seconds")
print("Memory used: Current = {} bytes, Peak = {} bytes".format(a_star_mem_current, a_star_mem_peak))

draw_graph(G, dijkstra_path, positions, goal_node, "Dijkstra's Algorithm Path")
draw_graph(G, a_star_path, positions, goal_node, "A* Algorithm Path")
