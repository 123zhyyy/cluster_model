
import networkx as nx
import community
import matplotlib.pyplot as plt
import random

# construct graph
def read_gene_network(file_path):
    graph = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            gene1, gene2 = line.strip().split()
            graph.add_edge(gene1, gene2)
    return graph

# cluster
def louvain_clustering(graph):
    partition = community.best_partition(graph)
    return partition

    
def write_communities_to_file(partition, output_file):
    with open(output_file, 'w') as file:
        for community_id, genes in group_by_community(partition).items():
            file.write(f"Community {community_id}:\n")
            file.write(", ".join(genes) + "\n\n")

def group_by_community(partition):
    community_dict = {}
    for gene, comm_id in partition.items():
        if comm_id not in community_dict:
            community_dict[comm_id] = []
        community_dict[comm_id].append(gene)
    return community_dict

# draw
def draw_cluster_graph(graph, partition):
    pos = nx.spring_layout(graph)
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(5,5))
    nx.draw_networkx_nodes(graph, pos, node_color=list(partition.values()), cmap=cmap,node_size=50)
    nx.draw_networkx_edges(graph, pos, alpha=0)
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.show()


file_path = '///gene.txt'
output_file = '///cluster.txt'

gene_network = read_gene_network(file_path)
clustering_result = louvain_clustering(gene_network)

write_communities_to_file(clustering_result, output_file)
draw_cluster_graph(gene_network, clustering_result)
