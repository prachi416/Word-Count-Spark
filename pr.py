from pyspark import SparkContext
from operator import add

sc = SparkContext("local", "PageRank")

def read_network(file_path):
    network = sc.textFile(file_path)\
                .map(lambda line: line.strip().split(': '))\
                .map(lambda x: (int(x[0]), [int(link) for link in x[1][1:-1].split(', ')]))\
                .collectAsMap()
    return network

def initialize_pagerank(network):
    num_nodes = len(network)
    initial_pagerank = 1 / num_nodes
    pagerank = {node: initial_pagerank for node in network.keys()}
    return pagerank

def calculate_pagerank(network, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    pagerank = initialize_pagerank(network)
    num_nodes = len(network)
    for _ in range(max_iterations):
        new_pagerank = sc.parallelize(pagerank.items())\
                        .flatMap(lambda x: [(neighbor, x[1] / len(network[x[0]]) * damping_factor) for neighbor in network[x[0]]])\
                        .reduceByKey(add)\
                        .mapValues(lambda x: (1 - damping_factor) / num_nodes + x)\
                        .collectAsMap()
        if all(abs(new_pagerank[node] - pagerank[node]) < tolerance for node in pagerank.keys()):
            break
        pagerank = new_pagerank
    return pagerank

def find_extreme_pagerank(pagerank):
    highest_pagerank_node = max(pagerank, key=pagerank.get)
    lowest_pagerank_node = min(pagerank, key=pagerank.get)
    return highest_pagerank_node, lowest_pagerank_node

def output_results(pagerank, highest_pagerank_node, lowest_pagerank_node):
    print("PageRank Values:")
    for node, pr in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
        print(f"{node}: {pr}")
    print("Node with the highest PageRank:", highest_pagerank_node)
    print("Node with the lowest PageRank:", lowest_pagerank_node)

def main():
    file_path = '/home/adduser/question3.txt'
    network = read_network(file_path)
    pagerank = calculate_pagerank(network)
    highest_pagerank_node, lowest_pagerank_node = find_extreme_pagerank(pagerank)
    output_results(pagerank, highest_pagerank_node, lowest_pagerank_node)
    sc.stop()

if __name__ == "__main__":
    main()
