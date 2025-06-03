# diffusion_readiness_project/structural_indices/baselines.py
# Python 3.9

"""
Functions to calculate baseline structural graph centralities.
These will serve as benchmarks for comparison with the proposed
Diffusion Readiness Index (TRI).
"""

import networkx as nx

# --- Baseline Centrality Functions ---

def get_degree_centrality(graph, per_node=False, node=None):
    """
    Calculates degree centrality for all nodes or a specific node.
    For directed graphs, this typically refers to out-degree centrality
    as a measure of direct influence potential.
    NetworkX's degree_centrality by default considers in+out degree for DiGraphs
    unless the graph is converted to undirected or specific degree types are used.
    For "transmission readiness", out-degree is often more relevant.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary of centrality for all nodes.
                         If False (and node is specified), returns centrality for that node.
        node (any hashable, optional): The specific node for which to calculate centrality.
                                       Required if per_node is False.

    Returns:
        dict or float: Dictionary of {node: centrality} if per_node is True,
                       or a float value for the specified node if per_node is False.
                       Returns None if node is not in graph when per_node is False.
    """
    if isinstance(graph, nx.DiGraph):
        # For transmission readiness, out-degree centrality is often more indicative.
        # nx.out_degree_centrality(graph) directly gives this.
        centralities = nx.out_degree_centrality(graph)
    else: # Undirected graph
        centralities = nx.degree_centrality(graph)
        
    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(node)


def get_k_core_centrality(graph, per_node=False, node=None):
    """
    Calculates k-core number (coreness) for all nodes or a specific node.
    A node's k-core number is the largest k such that it belongs to a k-core.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph. NetworkX k_core works on
                                       the undirected version of the graph.
        per_node (bool): If True, returns a dictionary of coreness for all nodes.
        node (any hashable, optional): The specific node.

    Returns:
        dict or int: Dictionary of {node: coreness} or coreness for the specified node.
                     Returns None if node is not in graph when per_node is False.
    """
    # nx.core_number creates a dictionary {node: k_core_number}
    # It works on an undirected view of the graph.
    if graph.is_directed():
        # k-core is typically defined for undirected graphs.
        # We can use the undirected version or specific DiGraph core definitions if available/needed.
        # For simplicity and common usage, using the undirected version.
        core_numbers = nx.core_number(nx.Graph(graph)) # Use undirected copy
    else:
        core_numbers = nx.core_number(graph)
        
    if per_node:
        return core_numbers
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return core_numbers.get(node)


def get_eigenvector_centrality(graph, per_node=False, node=None, max_iter=100, tol=1.0e-6):
    """
    Calculates eigenvector centrality.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary of centrality for all nodes.
        node (any hashable, optional): The specific node.
        max_iter (int): Maximum number of iterations in power method.
        tol (float): Error tolerance used to check convergence in power method.


    Returns:
        dict or float: Dictionary or single float value.
                       Returns None if node is not in graph when per_node is False.
    """
    try:
        # For DiGraphs, NetworkX eigenvector_centrality computes right eigenvector centrality
        # which corresponds to the influence a node receives. For spreading influence (what we want),
        # we often consider the left eigenvector centrality of the reversed graph, or simply
        # use it on an undirected version if the concept of "influence from connections" is key.
        # If the graph is strongly connected, left and right dominant eigenvectors are related.
        # Let's use the standard nx.eigenvector_centrality and note its interpretation.
        # For a more "spreading" focused version, one might use G.reverse().eigenvector_centrality()
        # or apply to an undirected version.
        # For now, standard NetworkX behavior:
        centralities = nx.eigenvector_centrality_numpy(graph) # Uses numpy, generally preferred
        # centralities = nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol) # Slower, pure Python
    except nx.NetworkXError as e:
        print(f"Eigenvector centrality computation failed: {e}. Returning empty dict or None.")
        # This can happen for graphs with no edges or certain structures.
        # Fallback to a dictionary of zeros for all nodes in the graph.
        centralities = {n: 0.0 for n in graph.nodes()}

    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(node)


def get_betweenness_centrality(graph, per_node=False, node=None, k_samples=None, normalized=True):
    """
    Calculates betweenness centrality. Can be computationally expensive.
    Option to use k-samples for approximation on large graphs.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        k_samples (int, optional): If not None, estimates betweenness centrality using
                                   k sample nodes. Much faster for large graphs.
        normalized (bool): Whether to normalize by 2/((n-1)(n-2)) for graphs,
                           and 1/((n-1)(n-2)) for directed graphs.

    Returns:
        dict or float: Dictionary or single float value.
                       Returns None if node is not in graph when per_node is False.
    """
    centralities = nx.betweenness_centrality(graph, k=k_samples, normalized=normalized)
    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(node)


def get_closeness_centrality(graph, per_node=False, node=None):
    """
    Calculates closeness centrality.
    Note: For graphs with disconnected components, NetworkX calculates it for each node
    based on the size of its connected component.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: Dictionary or single float value.
                       Returns None if node is not in graph when per_node is False.
    """
    # For DiGraphs, closeness centrality considers outgoing paths.
    # If a node cannot reach all others, wf_improved=True (default in later NX) handles this.
    centralities = nx.closeness_centrality(graph)
    if per_node:
        return centralities
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return centralities.get(node)

# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing baselines.py...")

    # Create a sample graph for testing
    G_test_undirected = nx.Graph()
    G_test_undirected.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6), (6, 7)])
    #  1 -- 2 -- 4 -- 6 -- 7
    #  |  / |    |  /
    #  3 -- 5 -- +
    nodes_list = list(G_test_undirected.nodes())
    test_node = nodes_list[0] if nodes_list else 1 # Pick a node for single node tests

    print(f"\n--- Testing on Undirected Graph ({G_test_undirected.number_of_nodes()} nodes, {G_test_undirected.number_of_edges()} edges) ---")
    print(f"Degree Centrality (all): {get_degree_centrality(G_test_undirected, per_node=True)}")
    print(f"Degree Centrality (node {test_node}): {get_degree_centrality(G_test_undirected, node=test_node)}")
    
    print(f"K-Core Centrality (all): {get_k_core_centrality(G_test_undirected, per_node=True)}")
    print(f"K-Core Centrality (node {test_node}): {get_k_core_centrality(G_test_undirected, node=test_node)}")

    print(f"Eigenvector Centrality (all): {get_eigenvector_centrality(G_test_undirected, per_node=True)}")
    print(f"Eigenvector Centrality (node {test_node}): {get_eigenvector_centrality(G_test_undirected, node=test_node)}")

    print(f"Betweenness Centrality (all): {get_betweenness_centrality(G_test_undirected, per_node=True)}")
    print(f"Betweenness Centrality (node {test_node}): {get_betweenness_centrality(G_test_undirected, node=test_node)}")
    
    print(f"Closeness Centrality (all): {get_closeness_centrality(G_test_undirected, per_node=True)}")
    print(f"Closeness Centrality (node {test_node}): {get_closeness_centrality(G_test_undirected, node=test_node)}")

    # Create a sample directed graph
    DG_test = nx.DiGraph()
    DG_test.add_edges_from([(1,2), (1,3), (2,3), (3,4), (4,1)]) # A cycle to ensure strong connectivity for eigenvector
    dg_nodes_list = list(DG_test.nodes())
    test_node_dg = dg_nodes_list[0] if dg_nodes_list else 1

    print(f"\n--- Testing on Directed Graph ({DG_test.number_of_nodes()} nodes, {DG_test.number_of_edges()} edges) ---")
    print(f"Out-Degree Centrality (all): {get_degree_centrality(DG_test, per_node=True)}")
    print(f"Out-Degree Centrality (node {test_node_dg}): {get_degree_centrality(DG_test, node=test_node_dg)}")

    print(f"K-Core Centrality (all, from undirected view): {get_k_core_centrality(DG_test, per_node=True)}")
    print(f"K-Core Centrality (node {test_node_dg}, from undirected view): {get_k_core_centrality(DG_test, node=test_node_dg)}")

    print(f"Eigenvector Centrality (all): {get_eigenvector_centrality(DG_test, per_node=True)}")
    print(f"Eigenvector Centrality (node {test_node_dg}): {get_eigenvector_centrality(DG_test, node=test_node_dg)}")

    print(f"Betweenness Centrality (all): {get_betweenness_centrality(DG_test, per_node=True)}")
    print(f"Closeness Centrality (all): {get_closeness_centrality(DG_test, per_node=True)}")
    
    print("\n--- Baselines Test Complete ---")
