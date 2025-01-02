import networkx as nx
import matplotlib.pyplot as plt
import heapq
import numpy as np
import tracemalloc
def create_complex_city_graph():
    G = nx.DiGraph()
    edges = [ #edges for smaller problem
        ('A', 'B', 5), ('B', 'C', 3), ('C', 'D', 7), ('A', 'D', 10),
        ('D', 'E', 4), ('E', 'F', 11), ('B', 'E', 9), ('C', 'F', 5),
        ('B', 'G', 2), ('G', 'F', 20), ('G', 'E', 13), ('A', 'H', 1),
        ('H', 'G', 3), ('H', 'C', 4), ('F', 'I', 8), ('I', 'D', 14)
    ]
    # edges = [
    # ('A', 'B', 5), ('A', 'C', 2), ('B', 'C', 6), ('B', 'D', 2), ('C', 'D', 4),
    # ('C', 'E', 8), ('D', 'E', 1), ('E', 'F', 5), ('D', 'G', 3), ('E', 'H', 6),
    # ('F', 'I', 4), ('G', 'H', 7), ('G', 'J', 5), ('H', 'K', 9), ('I', 'K', 6),
    # ('J', 'K', 8), ('A', 'J', 10), ('I', 'L', 7), ('J', 'L', 3), ('K', 'L', 2),
    # ('L', 'M', 3), ('K', 'M', 1), ('H', 'N', 2), ('M', 'N', 4), ('L', 'O', 5),
    # ('N', 'O', 1)
    # ]
    G.add_weighted_edges_from(edges)
    return G


def draw_initial_graph(graph, goal, positions):
    pos = positions  # Node positions for visual layout
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')

    # Edge weights
    edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Node heuristic values
    heuristic_labels = {n: f"{heuristic(n, goal, positions):.2f}" for n in graph.nodes()}
    for node, (x, y) in pos.items():
        plt.text(x, y + 0.1, s=f"h={heuristic_labels[node]}", color='green', weight='bold')

    plt.title("Initial Graph: Weights and Heuristic Values to Goal ('F')")
    plt.show()

def heuristic(node, goal, positions):
    h = np.sqrt((positions[node][0] - positions[goal][0])**2 + (positions[node][1] - positions[goal][1])**2)
    return h
def dijkstra(graph, start, goal, positions, annotate):
    tracemalloc.start()
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {node: float('infinity') for node in graph.nodes()}
    cost_so_far[start] = 0
    annotate[start]['dijkstra'] = 0

    while open_set:
        current_cost, current = heapq.heappop(open_set)
        print(f"Processing Dijkstra node {current} with cost {current_cost}")

        if current == goal:
            break
        for neighbor in graph[current]:
            new_cost = current_cost + graph[current][neighbor]['weight']
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(open_set, (new_cost, neighbor))
                came_from[neighbor] = current
                annotate[neighbor]['dijkstra'] = new_cost
                print(f"Updated Dijkstra path to {neighbor}, cost {new_cost}")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Dijkstra memory usage: Current = {current} bytes, Peak = {peak} bytes")
    tracemalloc.stop()
    return came_from

def a_star(graph, start, goal, positions, annotate):
    tracemalloc.start()
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal, positions), 0, start))
    came_from = {}
    cost_so_far = {node: float('infinity') for node in graph.nodes()}
    cost_so_far[start] = 0
    annotate[start]['a_star'] = 0

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        print(f"Processing A* node {current} with f-score {current_f} (g-score: {current_g}, h-score: {heuristic(current, goal, positions)})")

        if current == goal:
            break
        for neighbor in graph[current]:
            new_cost = current_g + graph[current][neighbor]['weight']
            h_value = heuristic(neighbor, goal, positions)  # Store the heuristic value
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + h_value
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current
                annotate[neighbor]['a_star'] = new_cost
                print(f"Updated A* path to {neighbor}, g-score {new_cost}, h-score {h_value}")
    current, peak = tracemalloc.get_traced_memory()
    print(f"Dijkstra memory usage: Current = {current} bytes, Peak = {peak} bytes")
    tracemalloc.stop()
    return came_from
positions = { #positions for smaller problem
    'A': (0, 0), 'B': (1, 2), 'C': (2, 4), 'D': (3, 1),
    'E': (4, 3), 'F': (5, 5), 'G': (1, 5), 'H': (0, 3), 'I': (6, 2)
}

# positions = {
#     'A': (0, 1), 'B': (1, 2), 'C': (1, 0), 'D': (2, 2), 'E': (2, 0),
#     'F': (3, 1), 'G': (3, 3), 'H': (4, 0), 'I': (4, 2), 'J': (5, 3),
#     'K': (5, 1), 'L': (6, 2), 'M': (7, 1), 'N': (6, 0), 'O': (7, 0)
# }
G = create_complex_city_graph()
draw_initial_graph(G, 'F', positions)
annotate = {node: {'dijkstra': float('inf'), 'a_star': float('inf')} for node in G.nodes()}
dijkstra_came_from = dijkstra(G, 'A', 'F', positions, annotate)
a_star_came_from = a_star(G, 'A', 'F', positions, annotate)

# Function to reconstruct paths from the came_from dictionaries
def reconstruct_path(came_from, start, goal):
    path = []
    while goal != start:
        path.append(goal)
        goal = came_from.get(goal, start)
    path.append(start)
    path.reverse()
    return path

path_dijkstra = reconstruct_path(dijkstra_came_from, 'A', 'F')
path_a_star = reconstruct_path(a_star_came_from, 'A', 'F')

# Function to draw graph and paths
def draw_graph(graph, annotate, positions, path_dijkstra, path_a_star):
    pos = positions
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500)

    # Draw paths
    nx.draw_networkx_nodes(graph, pos, nodelist=path_dijkstra, node_color='red', node_size=500)
    nx.draw_networkx_edges(graph, pos, edgelist=list(zip(path_dijkstra[:-1], path_dijkstra[1:])), edge_color='red', width=2)
    nx.draw_networkx_nodes(graph, pos, nodelist=path_a_star, node_color='green', node_size=500)
    nx.draw_networkx_edges(graph, pos, edgelist=list(zip(path_a_star[:-1], path_a_star[1:])), edge_color='green', style='dashed', width=2)

    # Annotate nodes with costs
    for node in graph.nodes():
        x, y = pos[node]
        plt.text(x, y+0.1, s=f"D: {annotate[node]['dijkstra']}\nA*: {annotate[node]['a_star']}",
                 bbox=dict(facecolor='yellow', alpha=0.5), horizontalalignment='center')

    plt.title("Red: Dijkstra's Path, Green: A* Path with Costs")
    plt.show()

draw_graph(G, annotate, positions, path_dijkstra, path_a_star)
