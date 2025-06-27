# diffusion_readiness_project/structural_indices/composite_tri.py
# Python 3.9

"""
Functions to calculate composite Diffusion Readiness Indices (DRIs)
based on a weighted combination of multiple structural features.
Specifically, the Weighted Multi-Factor Score DRI (WMFS-DRI).
"""

import numpy as np
import networkx as nx

# For a full project, these would be imported from their respective files:
# from .baselines import get_degree_centrality, get_k_core_centrality
# from .literature_indices import get_structural_diversity_ego
# from .spectral_tri import get_localized_fiedler_value
# For standalone testing, we include placeholder functions.


def _placeholder_degree_centrality(graph, per_node=True, **kwargs):
    if isinstance(graph, nx.DiGraph):
        cent = nx.out_degree_centrality(graph)
    else:
        cent = nx.degree_centrality(graph)
    return cent if per_node else cent.get(list(cent.keys())[0])


def _placeholder_k_core_centrality(graph, per_node=True, **kwargs):
    g_undirected = nx.Graph(graph) if graph.is_directed() else graph
    cores = nx.core_number(g_undirected)
    return cores if per_node else cores.get(list(cores.keys())[0])


def _placeholder_sd_ego(graph, per_node=True, node=None, **kwargs):
    sd_scores = {}
    nodes_to_calc = [node] if (node is not None and not per_node) else list(graph.nodes())
    for n_val in nodes_to_calc:
        sd_scores[str(n_val)] = len(list(graph.neighbors(n_val)))
    return sd_scores if per_node else sd_scores.get(list(sd_scores.keys())[0])


# --- Normalization Helper ---


def normalize_feature_dict(feature_dict, method="min_max"):
    """
    Normalizes a dictionary of feature values.

    Args:
        feature_dict (dict): {node: value}
        method (str): 'min_max' for scaling to [0,1], 'z_score' for standard scaling.

    Returns:
        dict: {node: normalized_value}
    """
    if not feature_dict:
        return {}

    nodes = list(feature_dict.keys())
    values = np.array([feature_dict[n] for n in nodes], dtype=float)

    if len(values) == 0:
        return {}

    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:
        return {n: 0.5 for n in nodes}  # All values are the same, assign a neutral 0.5

    if method == "min_max":
        normalized_values = (values - min_val) / (max_val - min_val)
    elif method == "z_score":
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
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
            Each dict: {'name': str, 'func': callable, 'weight': float, 'params': dict}
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        normalization_method (str): Method for normalizing features.

    Returns:
        dict or float: WMFS-DRI scores.
    """
    if not feature_configs:
        raise ValueError("feature_configs cannot be empty.")

    all_node_features_normalized = {}

    for config in feature_configs:
        feature_name = config["name"]
        feature_func = config["func"]
        feature_params = config.get("params", {})

        raw_feature_values = feature_func(graph, per_node=True, **feature_params)

        if not isinstance(raw_feature_values, dict):
            raise TypeError(f"Feature function {feature_name} did not return a dictionary.")

        normalized_values = normalize_feature_dict(raw_feature_values, method=normalization_method)
        all_node_features_normalized[feature_name] = normalized_values

    wmfs_scores = {}
    nodes_in_graph = list(graph.nodes())

    for n_id in nodes_in_graph:
        n_id_str = str(n_id)
        current_wmfs_score = 0.0
        for config in feature_configs:
            feature_name = config["name"]
            weight = config["weight"]

            node_feature_val = all_node_features_normalized.get(feature_name, {}).get(n_id_str, 0.0)
            current_wmfs_score += weight * node_feature_val
        wmfs_scores[n_id_str] = current_wmfs_score

    if per_node:
        return wmfs_scores
    else:
        if node is None:
            raise ValueError("Node must be specified if per_node is False.")
        return wmfs_scores.get(str(node))


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing composite_dri.py...")

    G_test = nx.karate_club_graph()
    test_node = 0

    print(f"\n--- Testing WMFS-DRI on Karate Club Graph ---")

    feature_set_1 = [
        {"name": "degree_out", "func": _placeholder_degree_centrality, "weight": 0.4, "params": {}},
        {"name": "k_core", "func": _placeholder_k_core_centrality, "weight": 0.3, "params": {}},
        {"name": "sd_ego_1hop", "func": _placeholder_sd_ego, "weight": 0.3, "params": {"radius": 1}},
    ]

    print(f"Feature Set 1 Configs: {[(c['name'], c['weight']) for c in feature_set_1]}")

    wmfs_all_nodes = get_wmfs_dri(G_test, feature_set_1, per_node=True, normalization_method="min_max")
    print(f"WMFS-DRI (min_max norm, node {test_node}): {wmfs_all_nodes.get(str(test_node)):.3f}")

    wmfs_all_zscore = get_wmfs_dri(G_test, feature_set_1, per_node=True, normalization_method="z_score")
    print(f"WMFS-DRI (z_score norm, node {test_node}): {wmfs_all_zscore.get(str(test_node)):.3f}")

    print("\n--- Composite DRI Test Complete ---")
