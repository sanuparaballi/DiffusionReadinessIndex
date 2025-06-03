# diffusion_readiness_project/structural_indices/mechanism_focused_tri.py
# Python 3.9

"""
Functions to calculate mechanism-focused Transmission Readiness Indices (TRIs).
Specifically, the Bridging Potential Index (BPI-TRI).
BPI-TRI aims to quantify a node's ability to connect otherwise
disparate parts of its local or semi-local environment.
"""

import networkx as nx
import numpy as np # For potential normalization if needed, or averaging

# For a full project, these would be imported:
# from .literature_indices import get_structural_diversity_ego
# from graph_utils.utils import get_ego_network_minus_ego
# For standalone testing:
def _placeholder_get_ego_network_minus_ego(graph, ego_node, radius=1):
    """Simplified k-hop subgraph extraction for local features, excluding ego."""
    if ego_node not in graph:
        return nx.Graph() if not graph.is_directed() else nx.DiGraph()
    
    nodes_in_k_hop = {ego_node}
    queue = deque([(ego_node, 0)]) # Using deque from collections
    visited = {ego_node}
    neighbor_nodes = set()

    while queue:
        current_node, depth = queue.popleft()
        if depth < radius: # Explore up to radius
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor != ego_node: # Add to neighbor_nodes if not ego
                        neighbor_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))
                elif neighbor != ego_node and neighbor not in neighbor_nodes: # If visited but not added (e.g. ego's direct neighbor)
                     neighbor_nodes.add(neighbor)


    # Subgraph of neighbors only
    return graph.subgraph(neighbor_nodes).copy()


def _placeholder_structural_diversity_ego(graph, per_node=False, node=None, radius=1):
    """Placeholder for structural diversity."""
    sd_scores = {}
    nodes_to_calc = [node] if (node is not None and not per_node) else list(graph.nodes())
    for n_val in nodes_to_calc:
        if n_val not in graph: sd_scores[n_val] = 0; continue
        ego_net_minus_ego = _placeholder_get_ego_network_minus_ego(graph, n_val, radius=radius)
        if not ego_net_minus_ego.nodes(): sd_scores[n_val] = 0
        else:
            if ego_net_minus_ego.is_directed():
                sd_scores[n_val] = nx.number_weakly_connected_components(ego_net_minus_ego)
            else:
                sd_scores[n_val] = nx.number_connected_components(ego_net_minus_ego)
    return sd_scores if per_node else sd_scores.get(n_val)

from collections import deque # For _placeholder_get_ego_network_minus_ego

# --- Bridging Potential Index (BPI-TRI) ---

def get_local_clustering_coefficient(graph, per_node=False, node=None):
    """
    Calculates the local clustering coefficient for nodes.
    Uses NetworkX's clustering function.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph. For DiGraph, uses the
                                       undirected version for LCC typically.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: LCC scores.
    """
    # nx.clustering works on an undirected view of the graph for DiGraphs by default
    # or considers triangles in all directions.
    # For a simple LCC, often the undirected version is implied.
    if graph.is_directed():
        # Consider the undirected version for standard LCC
        lcc_values = nx.clustering(nx.Graph(graph))
    else:
        lcc_values = nx.clustering(graph)
    
    if per_node:
        return lcc_values
    else:
        if node is None: raise ValueError("Node must be specified.")
        return lcc_values.get(node)

def get_avg_neighbor_non_connectivity(graph, per_node=False, node=None, use_resistance_distance=False):
    """
    Calculates a measure of how disconnected a node's neighbors are from each other
    (excluding paths through the node itself).
    A simple version: 1 - (density of the subgraph of neighbors).
    A more complex version could use average resistance distance between neighbors in G - {node}.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        use_resistance_distance (bool): If True, attempts to use resistance distance (computationally expensive).
                                        Currently, this is a placeholder for future implementation.

    Returns:
        dict or float: Scores representing neighbor non-connectivity. Higher means less connected.
    """
    # This is a conceptual placeholder for resistance distance.
    # For now, we implement the simpler 1 - density of neighbor subgraph.
    if use_resistance_distance:
        print("Warning: Resistance distance for neighbor non-connectivity is not fully implemented in this placeholder.")
        # Placeholder logic: would require removing 'node', then calculating pairwise resistance distances
        # between its neighbors in the remaining graph. This is very complex.
        # For now, will default to density-based if this path is taken.

    non_conn_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph: return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            non_conn_scores[n_i] = 0.0; continue

        neighbors_of_ni = list(graph.neighbors(n_i)) # For DiGraph, this is successors.
                                                    # For bridging, undirected neighborhood might be better.
                                                    # Let's assume undirected neighborhood for this concept.
        
        # Consider undirected neighborhood for calculating density among neighbors
        # If graph is directed, take neighbors from its undirected version for this specific calculation
        current_graph_view = nx.Graph(graph) if graph.is_directed() else graph
        neighbors_of_ni_undirected = list(current_graph_view.neighbors(n_i))

        if len(neighbors_of_ni_undirected) < 2: # Density is 0 or undefined if less than 2 neighbors
            non_conn_scores[n_i] = 1.0 # Max non-connectivity (as they can't be connected)
            continue

        neighbor_subgraph = current_graph_view.subgraph(neighbors_of_ni_undirected)
        
        # Density of the subgraph of neighbors
        # Density = 2*M / (N*(N-1)) for undirected graph
        # M = number of edges in neighbor_subgraph, N = number of nodes in neighbor_subgraph
        density = nx.density(neighbor_subgraph)
        
        # Non-connectivity score: 1 - density. Higher means neighbors are less connected.
        non_conn_scores[n_i] = 1.0 - density
        
    return non_conn_scores if per_node else non_conn_scores.get(node)


def get_bpi_tri(graph, per_node=False, node=None, sd_radius=1, normalize_components=True):
    """
    Calculates the Bridging Potential Index (BPI-TRI).
    Conceptual formula: BPI-TRI(u) = SD_ego(u) * (1 - LCC(u)) * NeighborNonConnectivity(u)
    All components should be normalized before multiplication if they are on different scales.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary for all nodes.
        node (any hashable, optional): The specific node.
        sd_radius (int): Radius for structural diversity calculation.
        normalize_components (bool): Whether to normalize SD, (1-LCC), and NonConnectivity
                                     before multiplying. Recommended.

    Returns:
        dict or float: BPI-TRI scores.
    """
    bpi_scores = {}
    
    # 1. Calculate all components for all nodes first if normalizing
    all_nodes = list(graph.nodes())
    
    # Component 1: Structural Diversity (SD_ego)
    # Using placeholder, replace with actual import from literature_indices
    sd_values_raw = _placeholder_structural_diversity_ego(graph, per_node=True, radius=sd_radius)

    # Component 2: (1 - Local Clustering Coefficient)
    lcc_values_raw = get_local_clustering_coefficient(graph, per_node=True)
    one_minus_lcc_values_raw = {n: (1.0 - lcc_values_raw.get(n, 0.0)) for n in all_nodes}

    # Component 3: Neighbor Non-Connectivity
    neighbor_non_conn_raw = get_avg_neighbor_non_connectivity(graph, per_node=True)

    if normalize_components:
        # Import normalize_feature_dict from composite_tri or define locally
        # from ..composite_tri import normalize_feature_dict # If in same package level
        # For standalone, define a simple min-max normalizer here
        def _temp_normalize(feature_dict):
            if not feature_dict: return {}
            vals = np.array(list(feature_dict.values()))
            if len(vals) == 0 or np.all(vals == vals[0]): return {k: 0.0 for k in feature_dict} # Or 0.5
            min_v, max_v = np.min(vals), np.max(vals)
            if max_v == min_v: return {k: 0.0 for k in feature_dict} # Or 0.5
            return {k: (v - min_v) / (max_v - min_v) for k, v in feature_dict.items()}

        sd_values_norm = _temp_normalize(sd_values_raw)
        one_minus_lcc_norm = _temp_normalize(one_minus_lcc_values_raw)
        neighbor_non_conn_norm = _temp_normalize(neighbor_non_conn_raw)
    else:
        sd_values_norm = sd_values_raw
        one_minus_lcc_norm = one_minus_lcc_values_raw
        neighbor_non_conn_norm = neighbor_non_conn_raw

    # 2. Calculate BPI-TRI for each node
    for n_id in all_nodes:
        sd_val = sd_values_norm.get(n_id, 0.0)
        om_lcc_val = one_minus_lcc_norm.get(n_id, 0.0)
        nn_conn_val = neighbor_non_conn_norm.get(n_id, 0.0)
        
        # Multiplicative form
        bpi_scores[n_id] = sd_val * om_lcc_val * nn_conn_val
        
    if per_node:
        return bpi_scores
    else:
        if node is None: raise ValueError("Node must be specified.")
        return bpi_scores.get(node)


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing mechanism_focused_tri.py (BPI-TRI)...")

    # Create a sample graph for testing - A barbell graph is good for bridging
    G_test = nx.barbell_graph(5, 1) # Two cliques of 5 nodes, connected by a single path of 1 edge.
                                   # Node 4 is one end of bridge, Node 5 is the bridge node, Node 6 other end.
    test_bridge_node = 5 # The node on the bridge path
    test_clique_node = 0 # A node within one of the cliques

    print(f"\n--- Testing BPI-TRI on Barbell Graph ({G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges) ---")
    # Nodes in barbell_graph(m1, m2): 0 to m1-1 (first clique), m1 to m1+m2-1 (path), m1+m2 to 2*m1+m2-1 (second clique)
    # For barbell_graph(5,1): Nodes 0-4 (clique1), 5 (bridge node, path of length 1 means nodes 4 and 5 are connected), 6-10 (clique2)
    # The bridge edge is (4,5) and (5,6) if m2=1 means path length 1.
    # nx.barbell_graph(m1,m2): m2 is number of nodes in path *between* cliques.
    # If m2=0, cliques connected directly. If m2=1, one node connects them.
    # Let's use m1=3, m2=1. Nodes 0,1,2 (clique1). Node 3 (bridge). Nodes 4,5,6 (clique2).
    # Edges: (0,1),(0,2),(1,2), (2,3) <bridge edge1>, (3,4) <bridge edge2>, (4,5),(4,6),(5,6)
    G_barbell = nx.Graph()
    # Clique 1: 0,1,2
    G_barbell.add_edges_from([(0,1), (0,2), (1,2)])
    # Bridge node: 3
    G_barbell.add_edges_from([(2,3), (3,4)]) # Node 3 connects clique 1 (via node 2) to clique 2 (via node 4)
    # Clique 2: 4,5,6
    G_barbell.add_edges_from([(4,5), (4,6), (5,6)])
    
    bridge_node_actual = 3
    clique_node_actual = 0
    clique_edge_node_actual = 2 # Connects to bridge

    print(f"Barbell-like graph created: Nodes: {G_barbell.nodes()}, Edges: {G_barbell.edges()}")

    # Test LCC
    lcc_all = get_local_clustering_coefficient(G_barbell, per_node=True)
    print(f"\nLCC (all, sample): {{k: round(v,2) for k,v in list(lcc_all.items())}}")
    # Node 3 (bridge) should have LCC=0. Nodes in clique (0,1,4,5,6) LCC=1. Node 2,4 LCC depends on connections.

    # Test Avg Neighbor Non-Connectivity
    annc_all = get_avg_neighbor_non_connectivity(G_barbell, per_node=True)
    print(f"Avg Neighbor Non-Connectivity (all, sample): {{k: round(v,2) for k,v in list(annc_all.items())}}")
    # For node 3, neighbors are 2, 4. They are not connected. So non-conn should be high (1.0).

    # Test Structural Diversity (using placeholder)
    sd_all_test = _placeholder_structural_diversity_ego(G_barbell, per_node=True, radius=1)
    print(f"Structural Diversity (placeholder, radius=1, all): {sd_all_test}")
    # Node 3 (bridge) connects two components if its ego-net-minus-ego is considered. Neighbors are 2 and 4.
    # Subgraph of {2,4} has 0 edges, so 2 components. SD(3) = 2.
    # Node 0 (clique) neighbors 1,2. Subgraph {1,2} is connected by edge (1,2). SD(0) = 1.


    # Test BPI-TRI
    bpi_all_nodes = get_bpi_tri(G_barbell, per_node=True, sd_radius=1, normalize_components=True)
    print(f"\nBPI-TRI (Normalized, all nodes, sample):")
    for n, score in bpi_all_nodes.items():
        print(f"  Node {n}: SD={sd_all_test.get(n,0):.2f}, (1-LCC)={1-lcc_all.get(n,0):.2f}, NNC={annc_all.get(n,0):.2f} => BPI_norm={score:.3f}")
        
    bpi_bridge = get_bpi_tri(G_barbell, node=bridge_node_actual, sd_radius=1, normalize_components=True)
    bpi_clique = get_bpi_tri(G_barbell, node=clique_node_actual, sd_radius=1, normalize_components=True)
    
    print(f"BPI-TRI (Normalized, bridge node {bridge_node_actual}): {bpi_bridge:.3f} (expected to be high)")
    print(f"BPI-TRI (Normalized, clique node {clique_node_actual}): {bpi_clique:.3f} (expected to be lower)")

    print("\n--- Mechanism-Focused TRI Test Complete ---")

