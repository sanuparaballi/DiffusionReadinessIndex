# diffusion_readiness_project/graph_utils/utils.py
# Python 3.9

"""
General utility functions for graph operations using NetworkX.
These functions can be used by various modules for tasks like:
- Extracting k-hop neighborhoods.
- Getting induced subgraphs.
- Other common graph manipulations or queries.
"""

import networkx as nx
from collections import deque # For BFS in k-hop neighborhood

# --- Graph Utility Functions ---

def get_k_hop_neighborhood_subgraph(graph, center_node, k, include_center=True):
    """
    Extracts the k-hop neighborhood of a center_node as an induced subgraph.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        center_node (any hashable): The node from which to start the k-hop search.
        k (int): The number of hops (depth) to explore. k=0 returns just the center_node
                 (if include_center=True) or an empty graph. k=1 returns the center_node
                 and its direct neighbors, and edges between them.
        include_center (bool): Whether to include the center_node in the subgraph.

    Returns:
        nx.Graph or nx.DiGraph: An induced subgraph containing nodes within k hops
                                of center_node and all edges between them from the
                                original graph. Returns an empty graph if center_node
                                is not in the graph. The type of graph returned
                                (Graph or DiGraph) matches the input graph type.
    """
    if center_node not in graph:
        # print(f"Warning: Center node {center_node} not found in graph.")
        if isinstance(graph, nx.DiGraph):
            return nx.DiGraph()
        else:
            return nx.Graph()

    if k < 0:
        # print("Warning: k (hops) cannot be negative. Returning empty graph.")
        if isinstance(graph, nx.DiGraph):
            return nx.DiGraph()
        else:
            return nx.Graph()

    # Use Breadth-First Search (BFS) to find nodes within k hops
    nodes_in_k_hop = {center_node}
    queue = deque([(center_node, 0)]) # (node, current_depth)
    visited = {center_node}

    while queue:
        current_node, depth = queue.popleft()

        if depth < k:
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_in_k_hop.add(neighbor)
                    queue.append((neighbor, depth + 1))
    
    if not include_center and k > 0 : # if k=0 and not include_center, it should be empty
        nodes_in_k_hop.discard(center_node)
    elif not include_center and k == 0:
         nodes_in_k_hop = set()


    # Create the induced subgraph from the collected nodes
    # This automatically includes all edges between these nodes that exist in the original graph.
    if isinstance(graph, nx.DiGraph):
        subgraph = graph.subgraph(nodes_in_k_hop).copy() # Use .copy() to make it mutable if needed later
    else:
        subgraph = graph.subgraph(nodes_in_k_hop).copy()
        
    return subgraph


def get_ego_network_minus_ego(graph, ego_node, radius=1):
    """
    Returns the ego network of a node (at a given radius) with the ego node removed.
    This is often used for calculating structural diversity.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        ego_node (any hashable): The central node of the ego network.
        radius (int): The radius of the ego network (default is 1 for direct neighbors).

    Returns:
        nx.Graph or nx.DiGraph: The induced subgraph of the ego_node's neighbors
                                (within `radius` hops), excluding the ego_node itself.
                                Returns an empty graph if ego_node is not in the graph.
    """
    if ego_node not in graph:
        # print(f"Warning: Ego node {ego_node} not found in graph.")
        if isinstance(graph, nx.DiGraph):
            return nx.DiGraph()
        else:
            return nx.Graph()

    # nx.ego_graph includes the center node by default.
    # We want the subgraph of its neighbors.
    
    # Get nodes in the k-hop neighborhood (excluding the center node itself)
    # Using the k_hop function with include_center=False
    neighborhood_nodes_subgraph = get_k_hop_neighborhood_subgraph(graph, ego_node, k=radius, include_center=False)
    
    return neighborhood_nodes_subgraph


def get_graph_laplacian(graph, normalized=False):
    """
    Computes the Laplacian matrix of the graph.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
                                       For DiGraph, it's typically the Laplacian of the
                                       underlying undirected graph or a specific DiGraph Laplacian.
                                       NetworkX `laplacian_matrix` uses the undirected version by default.
        normalized (bool): If True, computes the normalized Laplacian. Otherwise,
                           computes the combinatorial Laplacian.

    Returns:
        scipy.sparse.csr_matrix: The Laplacian matrix.
    """
    if normalized:
        return nx.normalized_laplacian_matrix(graph)
    else:
        return nx.laplacian_matrix(graph)

# Add other general graph utility functions as needed, for example:
# - Function to get giant component
# - Function to convert graph to specific formats if necessary for other libraries
# - Functions to calculate specific types of paths or walks not readily in NetworkX

# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing graph_utils.py...")

    # Create a sample graph for testing
    G_test = nx.Graph()
    G_test.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 7)])
    # 1 -- 2 -- 4 -- 6 -- 7
    # |  / |    |  /
    # 3 -- 5 -- +

    print("\nOriginal Test Graph:")
    print(f"Nodes: {G_test.nodes()}")
    print(f"Edges: {G_test.edges()}")

    # Test get_k_hop_neighborhood_subgraph
    center_node_test = 2
    k_hops_test = 1
    k_hop_sub = get_k_hop_neighborhood_subgraph(G_test, center_node_test, k_hops_test)
    print(f"\n{k_hops_test}-hop neighborhood subgraph around node {center_node_test} (including center):")
    print(f"Nodes: {k_hop_sub.nodes()}")
    print(f"Edges: {k_hop_sub.edges()}")
    
    center_node_test = 6
    k_hops_test = 2
    k_hop_sub_2 = get_k_hop_neighborhood_subgraph(G_test, center_node_test, k_hops_test, include_center=True)
    print(f"\n{k_hops_test}-hop neighborhood subgraph around node {center_node_test} (including center):")
    print(f"Nodes: {k_hop_sub_2.nodes()}")
    print(f"Edges: {k_hop_sub_2.edges()}")

    # Test get_ego_network_minus_ego
    ego_node_test = 4
    radius_test = 1
    ego_minus_sub = get_ego_network_minus_ego(G_test, ego_node_test, radius=radius_test)
    print(f"\nEgo network (radius {radius_test}) around node {ego_node_test} (excluding ego):")
    print(f"Nodes: {ego_minus_sub.nodes()}") # Should be {2, 5, 6}
    print(f"Edges: {ego_minus_sub.edges()}") # Should include (2,5) if it existed, (5,6)

    # Test Laplacian
    # Note: For this to run, you'd need scipy installed as nx.laplacian_matrix returns a scipy sparse matrix.
    try:
        laplacian = get_graph_laplacian(G_test)
        print(f"\nLaplacian matrix (combinatorial) for G_test (shape: {laplacian.shape}):")
        # print(laplacian.toarray()) # Can be large for big graphs
    except Exception as e:
        print(f"Could not compute Laplacian, ensure scipy is installed. Error: {e}")

    # Test with a DiGraph
    DG_test = nx.DiGraph()
    DG_test.add_edges_from([(1,2), (2,3), (1,3), (3,4)])
    center_node_dg = 1
    k_hops_dg = 1
    k_hop_sub_dg = get_k_hop_neighborhood_subgraph(DG_test, center_node_dg, k_hops_dg)
    print(f"\n{k_hops_dg}-hop neighborhood DiGraph around node {center_node_dg} (including center):")
    print(f"Nodes: {k_hop_sub_dg.nodes()}")
    print(f"Edges: {k_hop_sub_dg.edges()}")


    print("\n--- Graph Utils Test Complete ---")
