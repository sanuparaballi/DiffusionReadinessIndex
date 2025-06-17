# diffusion_readiness_project/structural_indices/baselines.py
# Python 3.9

"""
Functions to calculate baseline structural graph centralities.
These will serve as benchmarks for comparison with the proposed
Diffusion Readiness Index (DRI).
"""

import networkx as nx

# --- Baseline Centrality Functions ---


def get_degree_centrality(graph, per_node=False, node=None):
    """
    Calculates degree centrality for all nodes or a specific node.
    For directed graphs, this uses out-degree centrality as a measure
    of direct influence potential.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary of centrality for all nodes.
        node (any hashable, optional): The specific node for which to calculate centrality.

    Returns:
        dict or float: Dictionary of {node: centrality} or a single float value.
    """
    if isinstance(graph, nx.DiGraph):
        # For transmission readiness, out-degree is more indicative.
        centralities = nx.out_degree_centrality(graph)
    else:  # Undirected graph
        centralities = nx.degree_centrality(graph)

    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(str(node))


def get_k_core_centrality(graph, per_node=False, node=None):
    """
    Calculates k-core number (coreness) for all nodes or a specific node.
    Uses an undirected view of the graph.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary of coreness for all nodes.
        node (any hashable, optional): The specific node.

    Returns:
        dict or int: Dictionary of {node: coreness} or a single integer value.
    """
    if graph.is_directed():
        undirected_graph = nx.Graph(graph)
    else:
        undirected_graph = graph

    core_numbers = nx.core_number(undirected_graph)

    if per_node:
        return core_numbers
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return core_numbers.get(str(node))


def get_eigenvector_centrality(graph, per_node=False, node=None, max_iter=1000, tol=1.0e-6):
    """
    Calculates eigenvector centrality.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        max_iter (int): Maximum number of iterations in power method.
        tol (float): Error tolerance for convergence.

    Returns:
        dict or float: Dictionary or single float value.
    """
    try:
        # eigenvector_centrality_numpy is generally faster and more robust.
        centralities = nx.eigenvector_centrality_numpy(graph)
    except (nx.NetworkXError, nx.NetworkXPointlessConcept) as e:
        # Fallback for graphs where it's not well-defined (e.g., empty graph)
        # or if the algorithm fails to converge.
        print(f"Eigenvector centrality computation failed: {e}. Returning zeros.")
        centralities = {n: 0.0 for n in graph.nodes()}

    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(str(node))


def get_betweenness_centrality(graph, per_node=False, node=None, k_samples=None, normalized=True):
    """
    Calculates betweenness centrality. Can be computationally expensive.
    Allows for approximation using k-samples on large graphs.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        k_samples (int, optional): Number of sample nodes for approximation.
        normalized (bool): Whether to normalize the centrality score.

    Returns:
        dict or float: Dictionary or single float value.
    """
    centralities = nx.betweenness_centrality(graph, k=k_samples, normalized=normalized)
    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(str(node))


def get_closeness_centrality(graph, per_node=False, node=None):
    """
    Calculates closeness centrality.
    For disconnected graphs, NetworkX calculates it for each node
    based on the size of its connected component.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: Dictionary or single float value.
    """
    centralities = nx.closeness_centrality(graph)
    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(str(node))


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing baselines.py...")

    G_test = nx.karate_club_graph()
    test_node = 0

    print(f"\n--- Testing on Karate Club Graph ---")

    all_degree = get_degree_centrality(G_test, per_node=True)
    print(
        f"Degree Centrality (all nodes, sample): {{k: round(v,3) for k,v in list(all_degree.items())[:5]}}"
    )
    print(f"Degree Centrality (node {test_node}): {get_degree_centrality(G_test, node=test_node):.3f}")

    all_kcore = get_k_core_centrality(G_test, per_node=True)
    print(f"\nK-Core Centrality (all nodes, sample): {{k: v for k,v in list(all_kcore.items())[:5]}}")
    print(f"K-Core Centrality (node {test_node}): {get_k_core_centrality(G_test, node=test_node)}")

    all_eigen = get_eigenvector_centrality(G_test, per_node=True)
    print(
        f"\nEigenvector Centrality (all nodes, sample): {{k: round(v,3) for k,v in list(all_eigen.items())[:5]}}"
    )
    print(
        f"Eigenvector Centrality (node {test_node}): {get_eigenvector_centrality(G_test, node=test_node):.3f}"
    )

    # Betweenness is expensive, but Karate club is small enough for full computation
    all_between = get_betweenness_centrality(G_test, per_node=True)
    print(
        f"\nBetweenness Centrality (all nodes, sample): {{k: round(v,3) for k,v in list(all_between.items())[:5]}}"
    )

    # Closeness
    all_closeness = get_closeness_centrality(G_test, per_node=True)
    print(
        f"\nCloseness Centrality (all nodes, sample): {{k: round(v,3) for k,v in list(all_closeness.items())[:5]}}"
    )

    print("\n--- Baselines Test Complete ---")
