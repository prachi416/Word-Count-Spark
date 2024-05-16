from pyspark import SparkContext
import heapq

sc = SparkContext.getOrCreate()

graph_1 = sc.textFile("/home/adduser/question2_1.txt")
graph_2 = sc.textFile("/home/adduser/question2_2.txt")

def parse_line(line):
    parts = line.strip().split(',')
    if len(parts) != 3:
        raise ValueError("Invalid input format. Expected format: 'start, end, weight'.")
    start, end, weight = map(int, parts)
    return start, (end, weight)

parsed_graph_1 = graph_1.map(parse_line)
parsed_graph_2 = graph_2.map(parse_line)

def merge_graphs(graph1, graph2):
    return graph1.union(graph2)

merged_graph = merge_graphs(parsed_graph_1, parsed_graph_2) \
    .groupByKey() \
    .mapValues(list) \
    .collectAsMap()

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    visited = set() 
    
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight
            if neighbor in distances and distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    
    return distances

start_node = 0  
distances = dijkstra(merged_graph, start_node)

max_distance_node = max(distances, key=distances.get)
min_distance_node = min(distances, key=distances.get)

print(f"Node {max_distance_node} has the greatest distance of {distances[max_distance_node]} from the starting node.")
print(f"Node {min_distance_node} has the least distance of {distances[min_distance_node]} from the starting node.")

sc.stop()
