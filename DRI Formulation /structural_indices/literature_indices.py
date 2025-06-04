# diffusion_readiness_project/structural_indices/literature_indices.py
# Python 3.9

"""
Functions to calculate structural indices from existing literature,
beyond basic centralities. These include:
- Collective Influence (CI)
- Gravity Centrality
- Forward Linear Threshold Rank (FwLTR - simplified structural version)
- Structural Diversity (SD_ego)
"""

import networkx as nx
import math
from collections import deque

# It's assumed that graph_utils.utils might be in a different path,
# so for standalone execution, a placeholder or direct import might be needed.
# For the project structure, it would be:
# from graph_utils.utils import get_k_hop_neighborhood_subgraph, get_ego_network_minus_ego
# For now, let's define a placeholder for get_ego_network_minus_ego if used in testing block.


# --- Placeholder for graph_utils for standalone testing of this file ---
# In the full project, these would be imported from graph_utils.utils
def _placeholder_get_k_hop_neighborhood_subgraph(graph, center_node, k, include_center=True):
    if center_node not in graph:
        return nx.Graph() if not graph.is_directed() else nx.DiGraph()
    nodes_in_k_hop = {center_node}
    queue = deque([(center_node, 0)])
    visited = {center_node}
    while queue:
        curr, depth = queue.popleft()
        if depth < k:
            for neighbor in graph.neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_in_k_hop.add(neighbor)
                    queue.append((neighbor, depth + 1))
    if not include_center:
        nodes_in_k_hop.discard(center_node)
    return graph.subgraph(nodes_in_k_hop).copy()


def _placeholder_get_ego_network_minus_ego(graph, ego_node, radius=1):
    return _placeholder_get_k_hop_neighborhood_subgraph(graph, ego_node, k=radius, include_center=False)


# --- End Placeholder ---


# --- Literature Index Functions ---


def get_collective_influence(graph, per_node=False, node=None, l_dist=2):
    """
    Calculates Collective Influence (CI) for nodes.
    CI_l(i) = (k_i - 1) * sum_{j in Ball(i, l)} (k_j - 1)
    where Ball(i, l) is the set of nodes at shortest distance l from node i.
    k_i is the degree of node i.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph. Degrees are used.
                                       For DiGraph, out-degree might be more relevant for spreading.
                                       This implementation will use graph.degree().
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        l_dist (int): The distance 'l' for the CI calculation.

    Returns:
        dict or float: CI scores.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("Input graph must be a NetworkX Graph or DiGraph.")

    ci_scores = {}

    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph:
        return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            ci_scores[n_i] = 0.0  # Or handle as error/None
            continue

        k_i = graph.degree(n_i)
        term1 = float(k_i - 1) if k_i > 0 else 0.0

        sum_ball_term = 0.0
        if term1 > 0 and l_dist > 0:  # Only compute sum if (k_i-1) is positive
            # Find nodes exactly at shortest distance l_dist
            # This requires shortest path calculations from n_i
            try:
                # Get all shortest path lengths from n_i
                path_lengths = nx.single_source_shortest_path_length(graph, n_i, cutoff=l_dist)
                nodes_at_l_dist = [j for j, dist in path_lengths.items() if dist == l_dist]

                for n_j in nodes_at_l_dist:
                    k_j = graph.degree(n_j)
                    sum_ball_term += float(k_j - 1) if k_j > 0 else 0.0
            except nx.NetworkXError:  # Node not in graph (should be caught above) or other issues
                sum_ball_term = 0.0  # Default if path finding fails

        ci_scores[n_i] = term1 * sum_ball_term

    if per_node:
        return ci_scores
    else:
        return ci_scores.get(node)


def get_gravity_centrality(graph, per_node=False, node=None, r_radius=None):
    """
    Calculates Gravity Centrality.
    G(i) = sum_{j != i, dist(i,j) <= r} (Coreness(i) * Coreness(j)) / (dist(i,j)^2)
    If r_radius is None, sum over all j != i.

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph. Coreness and distance are used.
                                       Distances are typically on unweighted graph.
                                       Coreness is on undirected version.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        r_radius (int, optional): The maximum radius 'r' for considering neighbors.
                                  If None, all nodes are considered (can be very slow).

    Returns:
        dict or float: Gravity Centrality scores.
    """
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError("Input graph must be a NetworkX Graph or DiGraph.")

    # K-core is typically for undirected graphs
    undirected_graph = nx.Graph(graph) if graph.is_directed() else graph
    core_numbers = nx.core_number(undirected_graph)

    gravity_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())

    if node is not None and not per_node and node not in graph:
        return None

    # Pre-calculate all-pairs shortest paths if r_radius is None (very expensive)
    # Or calculate single-source shortest paths for each node_i if r_radius is set.
    all_pairs_distances = None
    if r_radius is None and per_node:  # Only precompute if needed for all nodes and no radius limit
        try:
            # This is the most expensive part for dense graphs
            all_pairs_distances = dict(nx.all_pairs_shortest_path_length(graph))
        except Exception as e:
            print(
                f"Warning: Could not compute all-pairs shortest paths for Gravity: {e}. Will compute per node."
            )
            all_pairs_distances = None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            gravity_scores[n_i] = 0.0
            continue

        core_i = float(core_numbers.get(n_i, 0))
        if core_i == 0:  # If node has 0 coreness, its gravity contribution might be 0
            gravity_scores[n_i] = 0.0
            continue

        current_gravity_sum = 0.0

        # Determine distances from n_i
        distances_from_ni = None
        if all_pairs_distances and n_i in all_pairs_distances:
            distances_from_ni = all_pairs_distances[n_i]
        else:  # Calculate on the fly or if precomputation failed/wasn't done
            try:
                distances_from_ni = nx.single_source_shortest_path_length(
                    graph, n_i, cutoff=r_radius if r_radius is not None else None
                )
            except nx.NetworkXError:  # Node might be isolated
                distances_from_ni = {}

        for n_j in graph.nodes():
            if n_i == n_j:
                continue

            dist_ij = distances_from_ni.get(n_j, float("inf"))

            if r_radius is not None and dist_ij > r_radius:
                continue

            if dist_ij > 0 and dist_ij != float("inf"):  # Ensure reachable and not self
                core_j = float(core_numbers.get(n_j, 0))
                if core_j > 0:  # Only consider nodes with positive coreness
                    current_gravity_sum += (core_i * core_j) / (dist_ij**2)

        gravity_scores[n_i] = current_gravity_sum

    if per_node:
        return gravity_scores
    else:
        return gravity_scores.get(node)


def get_structural_diversity_ego(graph, per_node=False, node=None, radius=1):
    """
    Calculates Structural Diversity based on the number of connected components
    in the ego-network of a node (minus the ego node itself).

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        radius (int): Radius for the ego network (default 1 for direct neighbors).

    Returns:
        dict or int: Structural diversity scores.
    """
    # This function relies on get_ego_network_minus_ego from graph_utils
    # For standalone testing, we use the placeholder.
    # In project: from graph_utils.utils import get_ego_network_minus_ego
    _get_ego_net_func = _placeholder_get_ego_network_minus_ego
    # if 'get_ego_network_minus_ego' in globals() and callable(get_ego_network_minus_ego):
    #    _get_ego_net_func = get_ego_network_minus_ego

    sd_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())

    if node is not None and not per_node and node not in graph:
        return None

    for n in nodes_to_compute:
        if n not in graph:
            sd_scores[n] = 0
            continue

        ego_net_minus_ego = _get_ego_net_func(graph, n, radius=radius)

        if not ego_net_minus_ego.nodes():
            sd_scores[n] = 0  # No neighbors, so 0 components
        else:
            # For DiGraph, consider weakly connected components for diversity of connections
            # For Graph, connected_components is fine.
            if ego_net_minus_ego.is_directed():
                num_components = nx.number_weakly_connected_components(ego_net_minus_ego)
            else:
                num_components = nx.number_connected_components(ego_net_minus_ego)
            sd_scores[n] = num_components

    if per_node:
        return sd_scores
    else:
        return sd_scores.get(node)


# FwLTR is more complex as it implies running a simplified LT model.
# For a purely structural index, we need to define structural thresholds and weights.
def get_fwltr_structural(
    graph, per_node=False, node=None, activation_threshold_mode="uniform", uniform_threshold=0.5
):
    """
    Calculates a simplified, purely structural Forward Linear Threshold Rank (FwLTR).
    The idea is to simulate a local LT process where thresholds and weights are
    derived from structure. FwLTR considers the spread from a node U its direct out-neighbors.

    Args:
        graph (nx.DiGraph): Input directed graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        activation_threshold_mode (str): 'uniform' (all nodes have `uniform_threshold`),
                                         'degree_based' (e.g., threshold = 1/in_degree, or constant).
                                         For simplicity, let's start with 'uniform'.
        uniform_threshold (float): The threshold if mode is 'uniform'.

    Returns:
        dict or int: The size of the cascade initiated by (node + its out-neighbors).
    """
    if not graph.is_directed():
        raise TypeError("FwLTR is typically defined for directed graphs.")

    fwltr_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())

    if node is not None and not per_node and node not in graph:
        return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            fwltr_scores[n_i] = 0
            continue

        # Initial active set for FwLTR: node n_i and its direct out-neighbors
        initial_active_set = {n_i}.union(set(graph.successors(n_i)))

        # Simulate a simple LT process
        active_nodes = set(initial_active_set)
        newly_activated_this_step = set(initial_active_set)

        # For structural version, edge weights are uniform (1) or degree-based. Let's use uniform.
        # Thresholds are also structural.

        max_iters = graph.number_of_nodes()  # Safeguard
        current_iter = 0

        while newly_activated_this_step and current_iter < max_iters:
            current_iter += 1
            can_be_activated_next = set()

            # Consider nodes that can be influenced by the currently active set
            potential_targets = set()
            for (
                active_node
            ) in newly_activated_this_step:  # Only consider newly active for influence propagation
                for successor in graph.successors(active_node):
                    if successor not in active_nodes:
                        potential_targets.add(successor)

            newly_activated_this_step = set()  # Reset for this iteration

            for target_node in potential_targets:
                # Calculate sum of influence from active neighbors
                influence_sum = 0
                for predecessor in graph.predecessors(target_node):
                    if predecessor in active_nodes:
                        # Structural weight: e.g., 1 (unweighted), or 1/out_degree(predecessor)
                        # Let's use unweighted for simplicity (each active neighbor contributes 1)
                        influence_sum += 1

                # Determine threshold for target_node
                threshold = 0
                if activation_threshold_mode == "uniform":
                    threshold = (
                        uniform_threshold * graph.in_degree(target_node)
                        if graph.in_degree(target_node) > 0
                        else uniform_threshold
                    )
                    if threshold == 0 and graph.in_degree(target_node) > 0:
                        threshold = 1  # Min 1 active neighbor if it has in-degree
                    elif threshold == 0 and graph.in_degree(target_node) == 0:
                        threshold = float("inf")  # Cannot be activated if no in-degree
                elif (
                    activation_threshold_mode == "degree_based"
                ):  # Example: 1/in_degree (but this is a weight, not sum)
                    # A common LT threshold is a random value per node, or a fraction of its in-degree
                    # For a structural version, let's say threshold is ceil(in_degree * some_fraction)
                    # For simplicity, using uniform_threshold as a fraction of in-degree
                    in_deg = graph.in_degree(target_node)
                    threshold = math.ceil(in_deg * uniform_threshold) if in_deg > 0 else float("inf")

                if influence_sum >= threshold and target_node not in active_nodes:  # Check again to be sure
                    can_be_activated_next.add(target_node)

            newly_activated_this_step = can_be_activated_next
            active_nodes.update(newly_activated_this_step)

        fwltr_scores[n_i] = len(active_nodes)  # The total number of nodes activated

    if per_node:
        return fwltr_scores
    else:
        return fwltr_scores.get(n_i)


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing literature_indices.py...")

    # Create a sample graph for testing
    G_test = nx.karate_club_graph()  # A well-known small graph
    test_node = 0  # A specific node in Karate club

    print(
        f"\n--- Testing on Karate Club Graph ({G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges) ---"
    )

    # Test Collective Influence
    ci_all = get_collective_influence(G_test, per_node=True, l_dist=2)
    print(f"Collective Influence (l=2, all nodes, sample): {{k: v for k, v in list(ci_all.items())[:5]}}")
    print(
        f"Collective Influence (l=2, node {test_node}): {get_collective_influence(G_test, node=test_node, l_dist=2)}"
    )

    # Test Gravity Centrality
    # Using r_radius for faster testing, otherwise it's slow for all pairs.
    gravity_all = get_gravity_centrality(G_test, per_node=True, r_radius=3)
    print(
        f"Gravity Centrality (r=3, all nodes, sample): {{k: round(v,2) for k, v in list(gravity_all.items())[:5]}}"
    )
    print(
        f"Gravity Centrality (r=3, node {test_node}): {get_gravity_centrality(G_test, node=test_node, r_radius=3):.2f}"
    )

    # Test Structural Diversity
    sd_all = get_structural_diversity_ego(G_test, per_node=True, radius=1)
    print(
        f"Structural Diversity (radius=1, all nodes, sample): {{k: v for k, v in list(sd_all.items())[:5]}}"
    )
    print(
        f"Structural Diversity (radius=1, node {test_node}): {get_structural_diversity_ego(G_test, node=test_node, radius=1)}"
    )

    # Test FwLTR (Structural) - requires a DiGraph, Karate is Undirected. Create a small DiGraph.
    DG_fwltr_test = nx.DiGraph()
    DG_fwltr_test.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    # 0 -> 1 \
    #   -> 2 -> 3 -> 5
    #        -> 4 /
    test_node_fwltr = 0
    # Using a threshold like 0.6 means a node needs >60% of its active incoming neighbors (or total for unweighted)
    # For simpler structural: threshold = 1 (if any active neighbor) or threshold = in_degree (all must be active)
    # Let's try a threshold that implies at least 1 or 2 active neighbors are needed depending on in-degree.
    # For uniform_threshold=0.5, a node with in-degree 2 needs 1 active neighbor. In-degree 3 needs 2.
    fwltr_all = get_fwltr_structural(
        DG_fwltr_test, per_node=True, uniform_threshold=0.5
    )  # Threshold is 0.5 * in_degree
    print(
        f"FwLTR (structural, threshold=0.5*in_degree, all nodes, sample): {{k: v for k, v in list(fwltr_all.items())[:5]}}"
    )
    print(
        f"FwLTR (structural, node {test_node_fwltr}): {get_fwltr_structural(DG_fwltr_test, node=test_node_fwltr, uniform_threshold=0.5)}"
    )

    print("\n--- Literature Indices Test Complete ---")
