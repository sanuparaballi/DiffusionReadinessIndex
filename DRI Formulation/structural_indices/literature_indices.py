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

# Import from project modules (this assumes a certain project structure)
# In a flat structure or for testing, you might need to adjust paths.
from graph_utils.utils import get_k_hop_neighborhood_subgraph, get_ego_network_minus_ego

# --- Literature Index Functions ---


def get_collective_influence(graph, per_node=False, node=None, l_dist=2):
    """
    Calculates Collective Influence (CI) for nodes.
    CI_l(i) = (degree(i) - 1) * sum_{j in Ball(i, l)} (degree(j) - 1)
    where Ball(i, l) is the set of nodes at shortest distance l from node i.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph. Degrees are used.
                                       For DiGraph, this implementation uses total degree.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        l_dist (int): The distance 'l' for the CI calculation.

    Returns:
        dict or float: CI scores.
    """
    ci_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    # Pre-calculate all degrees to avoid repeated calls inside loops
    all_degrees = dict(graph.degree())

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            ci_scores[n_i_str] = 0.0
            continue

        k_i = all_degrees.get(n_i_str, 0)
        term1 = float(k_i - 1) if k_i > 0 else 0.0

        sum_ball_term = 0.0
        if term1 > 0 and l_dist > 0:
            try:
                path_lengths = nx.single_source_shortest_path_length(graph, n_i_str, cutoff=l_dist)
                nodes_at_l_dist = [j for j, dist in path_lengths.items() if dist == l_dist]

                for n_j in nodes_at_l_dist:
                    k_j = all_degrees.get(n_j, 0)
                    sum_ball_term += float(k_j - 1) if k_j > 0 else 0.0
            except nx.NetworkXError:
                sum_ball_term = 0.0

        ci_scores[n_i_str] = term1 * sum_ball_term

    return ci_scores if per_node else ci_scores.get(str(node))


def get_gravity_centrality(graph, per_node=False, node=None, r_radius=None):
    """
    Calculates Gravity Centrality.
    G(i) = sum_{j != i, dist(i,j) <= r} (Coreness(i) * Coreness(j)) / (dist(i,j)^2)

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        r_radius (int, optional): The maximum radius 'r'. If None, all nodes are considered.

    Returns:
        dict or float: Gravity Centrality scores.
    """
    undirected_graph = nx.Graph(graph) if graph.is_directed() else graph
    core_numbers = nx.core_number(undirected_graph)

    gravity_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            gravity_scores[n_i_str] = 0.0
            continue

        core_i = float(core_numbers.get(n_i_str, 0))
        if core_i == 0:
            gravity_scores[n_i_str] = 0.0
            continue

        current_gravity_sum = 0.0
        try:
            distances_from_ni = nx.single_source_shortest_path_length(graph, n_i_str, cutoff=r_radius)
        except (nx.NetworkXError, KeyError):
            distances_from_ni = {}

        for n_j, dist_ij in distances_from_ni.items():
            if n_i_str == n_j:
                continue

            if dist_ij > 0:
                core_j = float(core_numbers.get(n_j, 0))
                if core_j > 0:
                    current_gravity_sum += (core_i * core_j) / (dist_ij**2)

        gravity_scores[n_i_str] = current_gravity_sum

    return gravity_scores if per_node else gravity_scores.get(str(node))


def get_structural_diversity_ego(graph, per_node=False, node=None, radius=1):
    """
    Calculates Structural Diversity based on the number of connected components
    in the ego-network of a node (minus the ego node itself).

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        radius (int): Radius for the ego network.

    Returns:
        dict or int: Structural diversity scores.
    """
    sd_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    for n in nodes_to_compute:
        n_str = str(n)
        if n_str not in graph:
            sd_scores[n_str] = 0
            continue

        ego_net_minus_ego = get_ego_network_minus_ego(graph, n_str, radius=radius)

        if not ego_net_minus_ego.nodes():
            sd_scores[n_str] = 0
        else:
            if ego_net_minus_ego.is_directed():
                num_components = nx.number_weakly_connected_components(ego_net_minus_ego)
            else:
                num_components = nx.number_connected_components(ego_net_minus_ego)
            sd_scores[n_str] = num_components

    return sd_scores if per_node else sd_scores.get(str(node))


def get_fwltr_structural(graph, per_node=False, node=None, uniform_threshold_fraction=0.5):
    """
    Calculates a simplified, purely structural Forward Linear Threshold Rank (FwLTR).
    Simulates a local LT process where the initial active set is the node and its
    direct out-neighbors.

    Args:
        graph (nx.DiGraph): Input directed graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        uniform_threshold_fraction (float): Threshold as a fraction of a node's in-degree.

    Returns:
        dict or int: The size of the cascade initiated by (node + its out-neighbors).
    """
    if not graph.is_directed():
        raise TypeError("FwLTR is defined for directed graphs.")

    fwltr_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    # Pre-calculate in-degrees
    all_in_degrees = dict(graph.in_degree())

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            fwltr_scores[n_i_str] = 0
            continue

        initial_active_set = {n_i_str}.union(set(graph.successors(n_i_str)))

        active_nodes = set(initial_active_set)
        newly_activated_in_round = set(initial_active_set)

        max_iters = graph.number_of_nodes()
        for _ in range(max_iters):
            if not newly_activated_in_round:
                break

            potential_targets = set()
            for active_node in newly_activated_in_round:
                for successor in graph.successors(active_node):
                    if successor not in active_nodes:
                        potential_targets.add(successor)

            newly_activated_this_round = set()
            for target_node in potential_targets:
                in_degree = all_in_degrees.get(target_node, 0)
                if in_degree == 0:
                    continue

                threshold = math.ceil(in_degree * uniform_threshold_fraction)

                # Count active predecessors
                active_predecessors_count = sum(
                    1 for p in graph.predecessors(target_node) if p in active_nodes
                )

                if active_predecessors_count >= threshold:
                    newly_activated_this_round.add(target_node)

            active_nodes.update(newly_activated_this_round)
            newly_activated_in_round = newly_activated_this_round

        fwltr_scores[n_i_str] = len(active_nodes)

    return fwltr_scores if per_node else fwltr_scores.get(str(node))


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing literature_indices.py...")

    G_test = nx.karate_club_graph()
    test_node = 0

    print(f"\n--- Testing on Karate Club Graph ---")

    ci_all = get_collective_influence(G_test, per_node=True, l_dist=2)
    print(f"Collective Influence (l=2, node {test_node}): {ci_all.get(str(test_node), None)}")

    gravity_all = get_gravity_centrality(G_test, per_node=True, r_radius=3)
    print(f"Gravity Centrality (r=3, node {test_node}): {gravity_all.get(str(test_node), None):.2f}")

    sd_all = get_structural_diversity_ego(G_test, per_node=True, radius=1)
    print(f"Structural Diversity (r=1, node {test_node}): {sd_all.get(str(test_node), None)}")

    DG_fwltr_test = nx.DiGraph()
    DG_fwltr_test.add_edges_from(
        [("0", "1"), ("0", "2"), ("1", "3"), ("2", "3"), ("2", "4"), ("3", "5"), ("4", "5")]
    )
    test_node_fwltr = "0"
    fwltr_all = get_fwltr_structural(DG_fwltr_test, per_node=True, uniform_threshold_fraction=0.5)
    print(f"FwLTR (t_frac=0.5, node {test_node_fwltr}): {fwltr_all.get(test_node_fwltr, None)}")

    print("\n--- Literature Indices Test Complete ---")
