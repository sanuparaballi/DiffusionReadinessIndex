# diffusion_readiness_project/structural_indices/composite_dri.py
# Python 3.9

"""
Functions to calculate composite Transmission Readiness Indices (DRIs)
based on a weighted combination of multiple structural features.
Specifically, the Weighted Multi-Factor Score DRI (WMFS-DRI).
"""

import networkx as nx
import numpy as np

# For a full project, these would be imported from their respective files:
# from .baselines import get_degree_centrality, get_k_core_centrality # etc.
from .literature_indices import get_structural_diversity_ego  # etc.
from .spectral_dri import get_localized_fiedler_value  # etc.

# For standalone testing, we might need placeholder or simplified versions of these.


# --- Placeholder for feature functions (for standalone testing) ---
# In the actual project, these would call functions from baselines.py, etc.
def _placeholder_degree_centrality(graph, per_node=True, node=None):
    if isinstance(graph, nx.DiGraph):
        cent = nx.out_degree_centrality(graph)
    else:
        cent = nx.degree_centrality(graph)
    return cent if per_node else cent.get(node)


def _placeholder_k_core_centrality(graph, per_node=True, node=None):
    g_undirected = nx.Graph(graph) if graph.is_directed() else graph
    cores = nx.core_number(g_undirected)
    return cores if per_node else cores.get(node)


def _placeholder_sd_ego(graph, per_node=True, node=None, radius=1):
    # Simplified for placeholder: just count neighbors for diversity proxy
    sd_scores = {}
    nodes_to_calc = [node] if (node is not None and not per_node) else list(graph.nodes())
    for n_val in nodes_to_calc:
        if n_val not in graph:
            sd_scores[n_val] = 0
            continue
        sd_scores[n_val] = len(list(graph.neighbors(n_val)))  # Placeholder logic
    return sd_scores if per_node else sd_scores.get(node)


# --- Normalization Helper ---
def normalize_feature_dict(feature_dict, method="min_max"):
    """
    Normalizes a dictionary of feature values.

    Args:
        feature_dict (dict): {node: value}
        method (str): 'min_max' for scaling to [0,1],
                      'z_score' for standard scaling.

    Returns:
        dict: {node: normalized_value}
    """
    if not feature_dict:
        return {}

    values = np.array(list(feature_dict.values()), dtype=float)
    nodes = list(feature_dict.keys())

    if len(values) == 0:  # Should not happen if feature_dict is not empty
        return {}

    if np.all(values == values[0]):  # All values are the same
        # Min-max would be NaN if min==max. Z-score would be NaN.
        # Return 0.5 for min-max, or 0 for z-score, or handle as needed.
        # For simplicity, if all same, min-max makes them all 0 (or 0.5 if preferred).
        # Let's make them 0 if min==max for min-max to avoid division by zero.
        # Or, if min==max, all normalized values could be 0.5 or 0.
        min_val = np.min(values)
        max_val = np.max(values)
        if min_val == max_val:
            if method == "min_max":
                # If all values are same, normalized to 0.5 (or 0 or 1, depending on convention)
                # If we want it to be 0, then (v - min_val) / 1 (if max_val-min_val=0, assume divisor 1)
                # Let's set to 0 if all values are identical to avoid issues.
                return {n: 0.0 for n in nodes}

    if method == "min_max":
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:  # Avoid division by zero if all values are the same
            normalized_values = (
                np.zeros_like(values) if min_val == 0 else np.full_like(values, 0.5)
            )  # Or all 0.0
        else:
            normalized_values = (values - min_val) / (max_val - min_val)
    elif method == "z_score":
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:  # Avoid division by zero
            normalized_values = np.zeros_like(values)
        else:
            normalized_values = (values - mean_val) / std_val
    else:
        raise ValueError("Unsupported normalization method. Choose 'min_max' or 'z_score'.")

    return {node_id: normalized_values[i] for i, node_id in enumerate(nodes)}


# --- Weighted Multi-Factor Score DRI (WMFS-DRI) ---


def get_wmfs_dri(graph, feature_configs, per_node=True, node=None, normalization_method="min_max"):
    """
    Calculates a Weighted Multi-Factor Score DRI.
    DRI_WMFS(u) = w1 * Norm(F1(u)) + w2 * Norm(F2(u)) + ...

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        feature_configs (list of dicts): Configuration for each feature.
            Each dict: {'name': str, 'func': callable, 'weight': float, 'params': dict (optional)}
            'func' should take (graph, per_node=True) and return a dict {node: value}.
            'params' are additional keyword arguments for the feature function.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        normalization_method (str): Method for normalizing features ('min_max' or 'z_score').

    Returns:
        dict or float: WMFS-DRI scores.
    """
    if not feature_configs:
        raise ValueError("feature_configs cannot be empty.")

    all_node_features_normalized = {}  # {feature_name: {node: normalized_value}}

    # 1. Calculate and normalize each feature for all nodes
    for config in feature_configs:
        feature_name = config["name"]
        feature_func = config["func"]
        feature_params = config.get("params", {})

        # Calculate raw feature values for all nodes
        raw_feature_values = feature_func(graph, per_node=True, **feature_params)

        if not isinstance(raw_feature_values, dict):
            raise TypeError(f"Feature function {feature_name} did not return a dictionary.")

        # Normalize these values
        normalized_values = normalize_feature_dict(raw_feature_values, method=normalization_method)
        all_node_features_normalized[feature_name] = normalized_values

    # 2. Calculate weighted sum for each node
    wmfs_scores = {}
    nodes_in_graph = list(graph.nodes())  # Ensure consistent node iteration

    for n_id in nodes_in_graph:
        current_wmfs_score = 0.0
        for config in feature_configs:
            feature_name = config["name"]
            weight = config["weight"]

            node_feature_val = all_node_features_normalized.get(feature_name, {}).get(
                n_id, 0.0
            )  # Default to 0 if node missing
            current_wmfs_score += weight * node_feature_val
        wmfs_scores[n_id] = current_wmfs_score

    if per_node:
        return wmfs_scores
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return wmfs_scores.get(node)


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing composite_dri.py...")

    # Create a sample graph for testing
    G_test = nx.karate_club_graph()
    test_node = 0

    print(f"\n--- Testing WMFS-DRI on Karate Club Graph ---")

    # Define feature configurations
    # In a real scenario, these functions would be imported from baselines.py, etc.
    feature_set_1 = [
        {
            "name": "degree_out",
            "func": _placeholder_degree_centrality,
            "weight": 0.4,
            "params": {},
        },  # Assumes out-degree for DiGraph
        {"name": "k_core", "func": _placeholder_k_core_centrality, "weight": 0.3, "params": {}},
        {"name": "sd_ego_1hop", "func": _placeholder_sd_ego, "weight": 0.3, "params": {"radius": 1}},
    ]

    print(f"Feature Set 1 Configs: {feature_set_1}")

    # Calculate WMFS-DRI for all nodes
    wmfs_all_nodes = get_wmfs_dri(G_test, feature_set_1, per_node=True, normalization_method="min_max")
    print(
        f"\nWMFS-DRI (Set 1, min_max norm, all nodes, sample): {{k: round(v,3) for k,v in list(wmfs_all_nodes.items())[:5]}}"
    )

    # Calculate for a single node
    wmfs_single_node = get_wmfs_dri(
        G_test, feature_set_1, per_node=False, node=test_node, normalization_method="min_max"
    )
    print(f"WMFS-DRI (Set 1, min_max norm, node {test_node}): {wmfs_single_node:.3f}")

    # Test with z_score normalization
    wmfs_all_zscore = get_wmfs_dri(G_test, feature_set_1, per_node=True, normalization_method="z_score")
    print(
        f"\nWMFS-DRI (Set 1, z_score norm, all nodes, sample): {{k: round(v,3) for k,v in list(wmfs_all_zscore.items())[:5]}}"
    )
    wmfs_single_zscore = get_wmfs_dri(
        G_test, feature_set_1, per_node=False, node=test_node, normalization_method="z_score"
    )
    print(f"WMFS-DRI (Set 1, z_score norm, node {test_node}): {wmfs_single_zscore:.3f}")

    # Example with a directed graph
    DG_test = nx.DiGraph()
    DG_test.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    test_node_dg = 0

    # Note: _placeholder_degree_centrality uses out-degree for DiGraphs
    wmfs_dg_all = get_wmfs_dri(DG_test, feature_set_1, per_node=True)
    print(
        f"\nWMFS-DRI (Set 1, DiGraph, all nodes, sample): {{k: round(v,3) for k,v in list(wmfs_dg_all.items())[:5]}}"
    )

    print("\n--- Composite DRI Test Complete ---")
