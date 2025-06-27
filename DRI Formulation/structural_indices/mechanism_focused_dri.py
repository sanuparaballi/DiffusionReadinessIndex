# diffusion_readiness_project/structural_indices/mechanism_focused_dri.py
# Python 3.9

"""
Functions to calculate mechanism-focused Transmission Readiness Indices (DRIs).
Specifically, the Bridging Potential Index (BPI-DRI).
BPI-DRI aims to quantify a node's ability to connect otherwise
disparate parts of its local or semi-local environment.
"""

import networkx as nx
import numpy as np  # For potential normalization if needed, or averaging
from collections import deque  # For _placeholder_get_ego_network_minus_ego

# --- Dependency Imports ---
# In a full project, these would be imported from other project files.
# For standalone testing, placeholders are included below.
from graph_utils.utils import get_ego_network_minus_ego
from structural_indices.literature_indices import get_structural_diversity_ego
from structural_indices.composite_dri import normalize_feature_dict


# For standalone testing:
def _placeholder_get_ego_network_minus_ego(graph, ego_node, radius=1):
    """Simplified k-hop subgraph extraction for local features, excluding ego."""
    if ego_node not in graph:
        return nx.Graph() if not graph.is_directed() else nx.DiGraph()

    nodes_in_k_hop = {ego_node}
    queue = deque([(ego_node, 0)])  # Using deque from collections
    visited = {ego_node}
    neighbor_nodes = set()

    while queue:
        current_node, depth = queue.popleft()
        if depth < radius:  # Explore up to radius
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor != ego_node:  # Add to neighbor_nodes if not ego
                        neighbor_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
                elif (
                    neighbor != ego_node and neighbor not in neighbor_nodes
                ):  # If visited but not added (e.g. ego's direct neighbor)
                    neighbor_nodes.add(neighbor)

    # Subgraph of neighbors only
    return graph.subgraph(neighbor_nodes).copy()


def _placeholder_structural_diversity_ego(graph, per_node=False, node=None, radius=1):
    """Placeholder for structural diversity."""
    sd_scores = {}
    nodes_to_calc = [node] if (node is not None and not per_node) else list(graph.nodes())
    for n_val in nodes_to_calc:
        if n_val not in graph:
            sd_scores[n_val] = 0
            continue
        ego_net_minus_ego = _placeholder_get_ego_network_minus_ego(graph, n_val, radius=radius)
        if not ego_net_minus_ego.nodes():
            sd_scores[n_val] = 0
        else:
            if ego_net_minus_ego.is_directed():
                sd_scores[n_val] = nx.number_weakly_connected_components(ego_net_minus_ego)
            else:
                sd_scores[n_val] = nx.number_connected_components(ego_net_minus_ego)
    return sd_scores if per_node else sd_scores.get(n_val)


def get_local_clustering_coefficient(graph, per_node=True, node=None):
    """
    Calculates the local clustering coefficient for nodes. A node with low LCC
    has neighbors that are not well-connected among themselves, making the
    node a potential local bridge.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: LCC scores.
    """
    # For this metric, an undirected view of the graph is most common.
    undirected_graph = nx.Graph(graph) if graph.is_directed() else graph
    lcc_values = nx.clustering(undirected_graph)

    if per_node:
        return lcc_values
    else:
        if node is None:
            raise ValueError("Node must be specified.")
        return lcc_values.get(str(node))


def get_neighbor_non_connectivity(graph, per_node=False, node=None):
    """
    Calculates a measure of how disconnected a node's neighbors are from each other.
    Implemented as 1 - (density of the subgraph of neighbors).

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: Scores representing neighbor non-connectivity (higher is less connected).
    """
    non_conn_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    # This concept is clearer on an undirected graph.
    undirected_graph = nx.Graph(graph) if graph.is_directed() else graph

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            non_conn_scores[n_i_str] = 0.0
            continue

        neighbors_of_ni = list(undirected_graph.neighbors(n_i_str))

        if len(neighbors_of_ni) < 2:
            non_conn_scores[n_i_str] = 1.0  # Max non-connectivity
            continue

        neighbor_subgraph = undirected_graph.subgraph(neighbors_of_ni)

        density = nx.density(neighbor_subgraph)
        non_conn_scores[n_i_str] = 1.0 - density

    return non_conn_scores if per_node else non_conn_scores.get(str(node))


# --- Bridging Potential Index (BPI-DRI) ---


def get_bpi_dri(graph, per_node=False, node=None, sd_radius=1, normalize_components=True):
    """
    Calculates the Bridging Potential Index (BPI-DRI).
    Conceptual formula: BPI(u) = SD_ego(u) * (1 - LCC(u)) * NeighborNonConnectivity(u)

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        sd_radius (int): Radius for structural diversity calculation.
        normalize_components (bool): Whether to normalize components before multiplying.

    Returns:
        dict or float: BPI-DRI scores.
    """
    bpi_scores = {}
    all_nodes = list(graph.nodes())

    # 1. Calculate all components for all nodes
    sd_values_raw = get_structural_diversity_ego(graph, per_node=True, radius=sd_radius)
    lcc_values_raw = get_local_clustering_coefficient(graph, per_node=True)
    one_minus_lcc_values_raw = {str(n): (1.0 - lcc_values_raw.get(n, 0.0)) for n in all_nodes}
    neighbor_non_conn_raw = get_neighbor_non_connectivity(graph, per_node=True)

    if normalize_components:
        sd_values_norm = normalize_feature_dict(sd_values_raw)
        one_minus_lcc_norm = normalize_feature_dict(one_minus_lcc_values_raw)
        neighbor_non_conn_norm = normalize_feature_dict(neighbor_non_conn_raw)
    else:
        sd_values_norm = sd_values_raw
        one_minus_lcc_norm = one_minus_lcc_values_raw
        neighbor_non_conn_norm = neighbor_non_conn_raw

    # 2. Calculate BPI-DRI for each node
    for n_id in all_nodes:
        n_id_str = str(n_id)
        sd_val = sd_values_norm.get(n_id_str, 0.0)
        om_lcc_val = one_minus_lcc_norm.get(n_id_str, 0.0)
        nn_conn_val = neighbor_non_conn_norm.get(n_id_str, 0.0)

        # Multiplicative form
        bpi_scores[n_id_str] = sd_val * om_lcc_val * nn_conn_val

    if per_node:
        return bpi_scores
    else:
        if node is None:
            raise ValueError("Node must be specified.")
        return bpi_scores.get(str(node))


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing mechanism_focused_dri.py (BPI-DRI)...")

    # To run this standalone, the imported functions must be available.
    # We create temporary placeholder functions for testing purposes.
    # In the real project, these would be imported correctly.
    class MockGraphUtils:
        get_ego_network_minus_ego = staticmethod(_placeholder_get_ego_network_minus_ego)

    class MockLiteratureIndices:
        get_structural_diversity_ego = staticmethod(_placeholder_structural_diversity_ego)

    class MockCompositeDRI:
        normalize_feature_dict = staticmethod(normalize_feature_dict)

    # Re-assign globals for the test context
    globals()["get_ego_network_minus_ego"] = MockGraphUtils.get_ego_network_minus_ego
    globals()["get_structural_diversity_ego"] = MockLiteratureIndices.get_structural_diversity_ego
    globals()["normalize_feature_dict"] = MockCompositeDRI.normalize_feature_dict

    # Create a barbell graph, which is a good test case for bridging
    G_barbell = nx.barbell_graph(m1=5, m2=1)  # Two cliques of 5, connected by node 5
    # Nodes 0-4 are clique 1, node 5 is the bridge, nodes 6-10 are clique 2.

    bridge_node = 5
    clique_node = 0

    print(f"\n--- Testing BPI-DRI on Barbell Graph ---")

    bpi_all_nodes = get_bpi_dri(G_barbell, per_node=True, normalize_components=True)

    bpi_bridge_val = bpi_all_nodes.get(str(bridge_node))
    bpi_clique_val = bpi_all_nodes.get(str(clique_node))

    print(f"BPI-DRI (Normalized, bridge node {bridge_node}): {bpi_bridge_val:.3f} (expected to be high)")
    print(f"BPI-DRI (Normalized, clique node {clique_node}): {bpi_clique_val:.3f} (expected to be low)")

    # Let's inspect the components for the bridge node
    sd_bridge = get_structural_diversity_ego(
        G_barbell, node=bridge_node
    )  # Neighbors 4 and 6 are disconnected -> 2 components
    lcc_bridge = get_local_clustering_coefficient(
        G_barbell, node=bridge_node
    )  # No triangle involving node 5 -> LCC=0
    nnc_bridge = get_neighbor_non_connectivity(
        G_barbell, node=bridge_node
    )  # Neighbors 4,6 are not connected -> density=0, non-conn=1
    print(
        f"  Components for bridge node {bridge_node}: SD={sd_bridge}, LCC={lcc_bridge:.2f}, NNC={nnc_bridge:.2f}"
    )

    # Let's inspect the components for the clique node
    sd_clique = get_structural_diversity_ego(
        G_barbell, node=clique_node
    )  # Neighbors 1,2,3,4 are all connected -> 1 component
    lcc_clique = get_local_clustering_coefficient(
        G_barbell, node=clique_node
    )  # Neighbors form a clique -> LCC=1
    nnc_clique = get_neighbor_non_connectivity(
        G_barbell, node=clique_node
    )  # Neighbors are fully connected -> density=1, non-conn=0
    print(
        f"  Components for clique node {clique_node}: SD={sd_clique}, LCC={lcc_clique:.2f}, NNC={nnc_clique:.2f}"
    )

    print("\n--- Mechanism-Focused DRI Test Complete ---")


# # --- Main execution block (for testing this module independently) ---
# if __name__ == "__main__":
#     print("Testing mechanism_focused_dri.py (BPI-DRI)...")

#     # Create a sample graph for testing - A barbell graph is good for bridging
#     G_test = nx.barbell_graph(5, 1)  # Two cliques of 5 nodes, connected by a single path of 1 edge.
#     # Node 4 is one end of bridge, Node 5 is the bridge node, Node 6 other end.
#     test_bridge_node = 5  # The node on the bridge path
#     test_clique_node = 0  # A node within one of the cliques

#     print(
#         f"\n--- Testing BPI-DRI on Barbell Graph ({G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges) ---"
#     )
#     # Nodes in barbell_graph(m1, m2): 0 to m1-1 (first clique), m1 to m1+m2-1 (path), m1+m2 to 2*m1+m2-1 (second clique)
#     # For barbell_graph(5,1): Nodes 0-4 (clique1), 5 (bridge node, path of length 1 means nodes 4 and 5 are connected), 6-10 (clique2)
#     # The bridge edge is (4,5) and (5,6) if m2=1 means path length 1.
#     # nx.barbell_graph(m1,m2): m2 is number of nodes in path *between* cliques.
#     # If m2=0, cliques connected directly. If m2=1, one node connects them.
#     # Let's use m1=3, m2=1. Nodes 0,1,2 (clique1). Node 3 (bridge). Nodes 4,5,6 (clique2).
#     # Edges: (0,1),(0,2),(1,2), (2,3) <bridge edge1>, (3,4) <bridge edge2>, (4,5),(4,6),(5,6)
#     G_barbell = nx.Graph()
#     # Clique 1: 0,1,2
#     G_barbell.add_edges_from([(0, 1), (0, 2), (1, 2)])
#     # Bridge node: 3
#     G_barbell.add_edges_from(
#         [(2, 3), (3, 4)]
#     )  # Node 3 connects clique 1 (via node 2) to clique 2 (via node 4)
#     # Clique 2: 4,5,6
#     G_barbell.add_edges_from([(4, 5), (4, 6), (5, 6)])

#     bridge_node_actual = 3
#     clique_node_actual = 0
#     clique_edge_node_actual = 2  # Connects to bridge

#     print(f"Barbell-like graph created: Nodes: {G_barbell.nodes()}, Edges: {G_barbell.edges()}")

#     # Test LCC
#     lcc_all = get_local_clustering_coefficient(G_barbell, per_node=True)
#     print(f"\nLCC (all, sample): {{k: round(v,2) for k,v in list(lcc_all.items())}}")
#     # Node 3 (bridge) should have LCC=0. Nodes in clique (0,1,4,5,6) LCC=1. Node 2,4 LCC depends on connections.

#     # Test Avg Neighbor Non-Connectivity
#     annc_all = get_avg_neighbor_non_connectivity(G_barbell, per_node=True)
#     print(
#         f"Avg Neighbor Non-Connectivity (all, sample): {{k: round(v,2) for k,v in list(annc_all.items())}}"
#     )
#     # For node 3, neighbors are 2, 4. They are not connected. So non-conn should be high (1.0).

#     # Test Structural Diversity (using placeholder)
#     sd_all_test = _placeholder_structural_diversity_ego(G_barbell, per_node=True, radius=1)
#     print(f"Structural Diversity (placeholder, radius=1, all): {sd_all_test}")
#     # Node 3 (bridge) connects two components if its ego-net-minus-ego is considered. Neighbors are 2 and 4.
#     # Subgraph of {2,4} has 0 edges, so 2 components. SD(3) = 2.
#     # Node 0 (clique) neighbors 1,2. Subgraph {1,2} is connected by edge (1,2). SD(0) = 1.

#     # Test BPI-DRI
#     bpi_all_nodes = get_bpi_dri(G_barbell, per_node=True, sd_radius=1, normalize_components=True)
#     print(f"\nBPI-DRI (Normalized, all nodes, sample):")
#     for n, score in bpi_all_nodes.items():
#         print(
#             f"  Node {n}: SD={sd_all_test.get(n,0):.2f}, (1-LCC)={1-lcc_all.get(n,0):.2f}, NNC={annc_all.get(n,0):.2f} => BPI_norm={score:.3f}"
#         )

#     bpi_bridge = get_bpi_dri(G_barbell, node=bridge_node_actual, sd_radius=1, normalize_components=True)
#     bpi_clique = get_bpi_dri(G_barbell, node=clique_node_actual, sd_radius=1, normalize_components=True)

#     print(f"BPI-DRI (Normalized, bridge node {bridge_node_actual}): {bpi_bridge:.3f} (expected to be high)")
#     print(f"BPI-DRI (Normalized, clique node {clique_node_actual}): {bpi_clique:.3f} (expected to be lower)")

#     print("\n--- Mechanism-Focused DRI Test Complete ---")
