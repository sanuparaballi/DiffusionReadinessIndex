# diffusion_readiness_project/etm_calculator/etm_functions.py
# Python 3.9

"""
Functions to calculate Effective Transmission Metrics (ETMs):
- Average Depth of Transmission (ADT)
- Maximum Cascade Subgraph Size (MCSS)
- High Neighborhood Reach (HNR_k)

These metrics are calculated based on observed or simulated cascade data.
"""

import networkx as nx
from collections import defaultdict
import numpy as np

# --- ETM Calculation Functions ---


def _build_cascade_graph(cascade_events):
    """
    Helper function to build a directed graph for a single cascade from its events.
    Events are expected to be (source, target, step/time) tuples.
    """
    cascade_graph = nx.DiGraph()
    if not cascade_events:
        return cascade_graph

    edges = [(event[0], event[1]) for event in cascade_events]
    cascade_graph.add_edges_from(edges)

    all_nodes_in_cascade = set()
    for u, v, *_ in cascade_events:
        all_nodes_in_cascade.add(u)
        all_nodes_in_cascade.add(v)
    cascade_graph.add_nodes_from(list(all_nodes_in_cascade))

    return cascade_graph


def calculate_adt(target_node_cascades, target_node):
    """
    Calculates the Average Depth of Transmission (ADT) for a target_node.
    Depth is the longest shortest path from the target_node to any other
    node it influenced within that specific cascade.

    Args:
        target_node_cascades (list): A list of individual cascades. Each cascade is
                                     represented by a list of propagation events
                                     [(parent, child, step), ...].
        target_node (any hashable): The node for which ADT is being calculated.

    Returns:
        float: The Average Depth of Transmission.
    """
    all_cascade_depths = []
    if not target_node_cascades:
        return 0.0

    for cascade_events in target_node_cascades:
        if not cascade_events:
            continue

        cascade_graph = _build_cascade_graph(cascade_events)

        if target_node not in cascade_graph:
            continue

        max_depth_for_this_cascade = 0
        try:
            descendants = nx.descendants(cascade_graph, target_node)
            if not descendants:
                # If a node spreads but only one hop (no multi-hop descendants), depth is 1.
                # If a node infects anyone, its out_degree in cascade > 0.
                if cascade_graph.out_degree(target_node) > 0:
                    max_depth_for_this_cascade = 1
            else:
                influence_subgraph = cascade_graph.subgraph(descendants.union({target_node}))
                path_lengths = nx.shortest_path_length(influence_subgraph, source=target_node)
                max_depth_for_this_cascade = max(path_lengths.values()) if path_lengths else 0

        except (nx.NetworkXError, KeyError):
            max_depth_for_this_cascade = 0

        all_cascade_depths.append(max_depth_for_this_cascade)

    return np.mean(all_cascade_depths) if all_cascade_depths else 0.0


def calculate_mcss(target_node_cascades, target_node):
    """
    Calculates the average influence subgraph size for a target_node.
    This is defined as 1 (for the node itself) + the number of nodes it
    (directly or indirectly) infects in a cascade, averaged over all cascades.

    Args:
        target_node_cascades (list): List of cascades where target_node spread.
        target_node (any hashable): The node for which MCSS is being calculated.

    Returns:
        float: The average size of the influence subgraph caused by target_node.
    """
    all_cascade_subgraph_sizes = []
    if not target_node_cascades:
        return 0.0

    for cascade_events in target_node_cascades:
        if not cascade_events:
            continue

        cascade_graph = _build_cascade_graph(cascade_events)

        if target_node not in cascade_graph:
            continue

        current_subgraph_size = 0
        try:
            descendants = nx.descendants(cascade_graph, target_node)
            current_subgraph_size = 1 + len(descendants)
        except (nx.NetworkXError, KeyError):
            current_subgraph_size = 0

        all_cascade_subgraph_sizes.append(current_subgraph_size)

    return np.mean(all_cascade_subgraph_sizes) if all_cascade_subgraph_sizes else 0.0


def calculate_hnr_k(target_node_cascades, target_node, graph, k_hop=1):
    """
    Calculates the High Neighborhood Reach (HNR_k) for a target_node.
    Average fraction of a target_node's k-hop static graph neighbors
    that are activated in cascades where the target_node was a spreader.

    Args:
        target_node_cascades (list): List of cascades where target_node spread.
        target_node (any hashable): The node for which HNR_k is calculated.
        graph (nx.DiGraph): The static underlying social/interaction graph.
        k_hop (int): The number of hops to define the neighborhood.

    Returns:
        float: The average HNR_k value (between 0 and 1).
    """
    if target_node not in graph:
        return 0.0

    # Get k-hop successors from the static graph
    try:
        path_lengths = nx.single_source_shortest_path_length(graph, target_node, cutoff=k_hop)
        k_hop_neighbors = {neighbor for neighbor, dist in path_lengths.items() if 0 < dist <= k_hop}
    except (nx.NetworkXError, KeyError):
        return 0.0

    if not k_hop_neighbors or not target_node_cascades:
        return 0.0

    all_hnr_k_values = []
    for cascade_events in target_node_cascades:
        if not cascade_events:
            continue

        cascade_graph = _build_cascade_graph(cascade_events)
        if target_node not in cascade_graph:
            continue

        nodes_influenced_in_cascade = set()
        try:
            descendants = nx.descendants(cascade_graph, target_node)
            nodes_influenced_in_cascade = descendants.union({target_node})
        except (nx.NetworkXError, KeyError):
            pass

        reached_k_hop_neighbors = k_hop_neighbors.intersection(nodes_influenced_in_cascade)
        fraction_reached = len(reached_k_hop_neighbors) / len(k_hop_neighbors)
        all_hnr_k_values.append(fraction_reached)

    return np.mean(all_hnr_k_values) if all_hnr_k_values else 0.0


def preprocess_cascade_data_for_node_etms(cascade_data, all_nodes_in_graph):
    """
    Preprocesses raw cascade data into a node-centric format for ETM calculation.

    Args:
        cascade_data (any): Can be a flat list of CasFlow events (dicts) or a
                            dict of {seed: [list_of_sim_events]} from IC sims.
        all_nodes_in_graph (set or list): All unique node IDs in the static graph.

    Returns:
        dict: {node_id: [list_of_cascades_where_node_spreads]}
    """
    node_etm_input_data = defaultdict(list)

    if isinstance(cascade_data, list) and cascade_data and isinstance(cascade_data[0], dict):
        # Case 1: CasFlow flat event list
        cascades_by_id = defaultdict(list)
        for event in cascade_data:
            if all(k in event for k in ("cascade_id", "parent", "target")):
                cascades_by_id[event["cascade_id"]].append(
                    (str(event["parent"]), str(event["target"]), event.get("infection_time_rel", 0))
                )

        for node_id in all_nodes_in_graph:
            node_id_str = str(node_id)
            for _, events_in_cascade in cascades_by_id.items():
                if any(event[0] == node_id_str for event in events_in_cascade):
                    node_etm_input_data[node_id_str].append(events_in_cascade)

    elif isinstance(cascade_data, dict):
        # Case 2: IC Simulation output {seed_node: list_of_sim_event_lists}
        # This structure is already node-centric for the seed nodes.
        for seed_node, list_of_simulations_for_seed in cascade_data.items():
            node_etm_input_data[str(seed_node)].extend(list_of_simulations_for_seed)

    return dict(node_etm_input_data)


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing etm_functions.py...")

    G_static = nx.DiGraph()
    G_static.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "E"), ("B", "F"), ("C", "G")])
    print(f"Static Graph: Nodes: {G_static.nodes()}, Edges: {G_static.edges()}")

    cascade1_events = [("A", "B", 1), ("A", "C", 1), ("B", "E", 2), ("C", "G", 2)]
    cascade2_events = [("A", "D", 1)]
    cascade3_events_A = [("A", "B", 1)]
    cascade4_events_Z = [("Z", "A", 1), ("A", "B", 2)]

    node_A_cascades = [cascade1_events, cascade2_events, cascade3_events_A, cascade4_events_Z]

    adt_A = calculate_adt(node_A_cascades, "A")
    print(
        f"\nADT for node A: {adt_A:.2f}"
    )  # Expected: C1 depth=2, C2 depth=1, C3 depth=1, C4 depth=1. Avg=(2+1+1+1)/4=1.25
    mcss_A = calculate_mcss(node_A_cascades, "A")
    print(
        f"MCSS for node A: {mcss_A:.2f}"
    )  # C1 size=5, C2 size=2, C3 size=2, C4 size=2. Avg=(5+2+2+2)/4=2.75
    hnr1_A = calculate_hnr_k(node_A_cascades, "A", G_static, k_hop=1)
    print(
        f"HNR_1 for node A: {hnr1_A:.2f}"
    )  # Neighbors={B,C,D}. C1 reaches B,C (2/3). C2 reaches D (1/3). C3 reaches B (1/3). C4 reaches B (1/3). Avg = 0.42

    node_B_cascades = [cascade1_events]
    adt_B = calculate_adt(node_B_cascades, "B")
    print(f"\nADT for node B (based on its role in cascade1): {adt_B:.2f}")  # Expected: 1.0
    mcss_B = calculate_mcss(node_B_cascades, "B")
    print(f"MCSS for node B: {mcss_B:.2f}")  # Expected: 2.0
    hnr1_B = calculate_hnr_k(node_B_cascades, "B", G_static, k_hop=1)
    print(f"HNR_1 for node B: {hnr1_B:.2f}")  # Neighbors={E,F}. Reaches E (1/2). Avg=0.5

    print("\n--- ETM Functions Test Complete ---")
