#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:39:08 2025

@author: sanup
"""


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
from collections import deque

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
        # Create an empty graph of the same type as the input
        return type(graph)()

    if k < 0:
        return type(graph)()

    # Use Breadth-First Search (BFS) to find nodes within k hops
    nodes_in_k_hop = {center_node}
    queue = deque([(center_node, 0)])  # (node, current_depth)
    visited = {center_node}

    while queue:
        current_node, depth = queue.popleft()

        if depth < k:
            # For directed graphs, successors are 'out-neighbors'
            # For undirected, neighbors() is equivalent
            neighbors_to_explore = (
                graph.successors(current_node) if graph.is_directed() else graph.neighbors(current_node)
            )
            for neighbor in neighbors_to_explore:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_in_k_hop.add(neighbor)
                    queue.append((neighbor, depth + 1))

    if not include_center:
        nodes_in_k_hop.discard(center_node)

    # Create the induced subgraph from the collected nodes
    # This automatically includes all edges between these nodes that exist in the original graph.
    return graph.subgraph(nodes_in_k_hop).copy()


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
    # This function is a convenient wrapper around get_k_hop_neighborhood_subgraph
    return get_k_hop_neighborhood_subgraph(graph, ego_node, k=radius, include_center=False)


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing graph_utils.py...")

    # Create a sample graph for testing
    G_test = nx.Graph()
    G_test.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 7)])
    # Convert node IDs to string to match our parser's behavior
    G_test_str = nx.relabel_nodes(G_test, str)

    print("\nOriginal Test Graph:")
    print(f"Nodes: {G_test_str.nodes()}")
    print(f"Edges: {G_test_str.edges()}")

    # Test get_k_hop_neighborhood_subgraph
    center_node_test = "2"
    k_hops_test = 1
    k_hop_sub = get_k_hop_neighborhood_subgraph(G_test_str, center_node_test, k_hops_test)
    print(f"\n{k_hops_test}-hop neighborhood subgraph around node {center_node_test} (including center):")
    print(f"Nodes: {sorted(list(k_hop_sub.nodes()))}")  # Expected: ['1', '2', '3', '4']
    print(f"Edges: {list(k_hop_sub.edges())}")

    center_node_test_2 = "6"
    k_hops_test_2 = 2
    k_hop_sub_2 = get_k_hop_neighborhood_subgraph(
        G_test_str, center_node_test_2, k_hops_test_2, include_center=True
    )
    print(
        f"\n{k_hops_test_2}-hop neighborhood subgraph around node {center_node_test_2} (including center):"
    )
    print(f"Nodes: {sorted(list(k_hop_sub_2.nodes()))}")  # Expected: ['2', '3', '4', '5', '6', '7']
    print(f"Edges: {list(k_hop_sub_2.edges())}")

    # Test get_ego_network_minus_ego
    ego_node_test = "4"
    radius_test = 1
    ego_minus_sub = get_ego_network_minus_ego(G_test_str, ego_node_test, radius=radius_test)
    print(f"\nEgo network (radius {radius_test}) around node {ego_node_test} (excluding ego):")
    print(f"Nodes: {sorted(list(ego_minus_sub.nodes()))}")  # Expected: ['2', '5', '6']
    print(f"Edges: {list(ego_minus_sub.edges())}")  # Expected: [('5', '6')]

    # Test with a DiGraph
    DG_test = nx.DiGraph()
    DG_test.add_edges_from([("1", "2"), ("2", "3"), ("1", "3"), ("3", "4")])
    center_node_dg = "1"
    k_hops_dg = 1
    k_hop_sub_dg = get_k_hop_neighborhood_subgraph(DG_test, center_node_dg, k_hops_dg)
    print(f"\n{k_hops_dg}-hop neighborhood DiGraph around node {center_node_dg} (successors):")
    print(f"Nodes: {sorted(list(k_hop_sub_dg.nodes()))}")  # Expected: ['1', '2', '3']
    print(f"Edges: {list(k_hop_sub_dg.edges())}")  # Expected: [('1','2'), ('1','3'), ('2','3')]

    print("\n--- Graph Utils Test Complete ---")
