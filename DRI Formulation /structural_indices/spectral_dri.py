# diffusion_readiness_project/structural_indices/spectral_dri.py
# Python 3.9


"""
Functions to calculate Diffusion Readiness Indices (DRIs) based on
spectral graph theory concepts.
Includes:
- Localized Fiedler Value (Algebraic Connectivity of k-hop neighborhood)
- Node's component in the Global Fiedler Vector
- Localized Spectral Radius (of k-hop neighborhood adjacency matrix)
- API-DRI based on Personalized PageRank (PPR)
- API-DRI based on Heat Kernel
"""


import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla
import scipy.linalg
from collections import deque

# --- Import Project Modules ---
from graph_utils.utils import get_k_hop_neighborhood_subgraph

# --- Global Spectral Feature (calculated once for the graph) ---
_GLOBAL_FIEDLER_VECTOR = None
_GLOBAL_FIEDLER_VALUE = None


def precompute_global_fiedler_info(graph):
    """
    Precomputes the global Fiedler value and vector for the graph.
    This should be called once per graph if node_global_fiedler_component is used.
    """
    global _GLOBAL_FIEDLER_VECTOR, _GLOBAL_FIEDLER_VALUE

    # Initialize defaults for all nodes in the original graph
    _GLOBAL_FIEDLER_VALUE = 0.0
    _GLOBAL_FIEDLER_VECTOR = {str(n): 0.0 for n in graph.nodes()}

    if graph.number_of_nodes() < 2:
        return  # Defaults are already set

    graph_undirected = nx.Graph(graph) if graph.is_directed() else graph

    # Find the largest connected component (LCC) to work on
    if not nx.is_connected(graph_undirected):
        try:
            largest_cc_nodes = max(nx.connected_components(graph_undirected), key=len)
            if not largest_cc_nodes:
                return
            graph_undirected = graph_undirected.subgraph(largest_cc_nodes).copy()
        except ValueError:  # This can happen if the graph has no nodes/edges
            return  # Rely on the defaults set above

    num_nodes_in_component = graph_undirected.number_of_nodes()

    # Eigsh requires k < N. Here k=2, so we need N > 2.
    # We will handle N <= 2 cases manually and return.
    if num_nodes_in_component <= 2:
        print(
            f"  Info: Skipping global Fiedler value calculation for graph component of size {num_nodes_in_component}. Using default values."
        )
        if num_nodes_in_component == 2 and graph_undirected.number_of_edges() > 0:
            # For a connected 2-node graph, Fiedler value is 2.0
            _GLOBAL_FIEDLER_VALUE = 2.0
            node_list = list(graph_undirected.nodes())
            val = 1 / np.sqrt(2) if np.sqrt(2) > 0 else 0
            _GLOBAL_FIEDLER_VECTOR[str(node_list[0])] = val
            _GLOBAL_FIEDLER_VECTOR[str(node_list[1])] = -val
        # Otherwise, for N<2 or disconnected N=2, the default of 0.0 remains.
        return

    # For N > 2, proceed with the sparse solver
    try:
        L_norm = nx.normalized_laplacian_matrix(graph_undirected)
        # We need the 2nd smallest eigenvalue. k=2 to get the first two.
        eigenvalues, eigenvectors = sla.eigsh(
            L_norm, k=2, which="SM", tol=1e-3, maxiter=num_nodes_in_component * 20
        )

        if len(eigenvalues) > 1:
            _GLOBAL_FIEDLER_VALUE = float(eigenvalues[1])
            node_list_component = list(graph_undirected.nodes())
            fiedler_vec_component = eigenvectors[:, 1]
            for i, node_id in enumerate(node_list_component):
                _GLOBAL_FIEDLER_VECTOR[str(node_id)] = float(fiedler_vec_component[i])
        else:
            print(
                f"Warning: eigsh returned fewer than 2 eigenvalues for a component of size {num_nodes_in_component}. Using defaults."
            )

    except (sla.ArpackNoConvergence, ValueError, IndexError, TypeError) as e:
        print(f"Error precomputing global Fiedler info: {e}. Setting to defaults and continuing.")
        # Defaults are already set, so we can just pass.
        pass


# --- Node-Level Spectral DRI Functions ---


def get_localized_fiedler_value(graph, per_node=False, node=None, k_hop=1):
    """
    Calculates the Fiedler value (algebraic connectivity) of the k-hop
    neighborhood subgraph around each node.
    """
    lfv_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            lfv_scores[n_i_str] = 0.0
            continue

        subgraph = get_k_hop_neighborhood_subgraph(graph, n_i_str, k_hop)
        subgraph_undirected = nx.Graph(subgraph) if subgraph.is_directed() else subgraph

        if not nx.is_connected(subgraph_undirected):
            if subgraph_undirected.number_of_nodes() > 0:
                try:
                    largest_cc_nodes = max(nx.connected_components(subgraph_undirected), key=len)
                    if not largest_cc_nodes:
                        lfv_scores[n_i_str] = 0.0
                        continue
                    subgraph_undirected = subgraph_undirected.subgraph(largest_cc_nodes)
                except ValueError:
                    lfv_scores[n_i_str] = 0.0
                    continue
            else:
                lfv_scores[n_i_str] = 0.0
                continue

        N_sub = subgraph_undirected.number_of_nodes()

        if N_sub < 2:
            lfv_scores[n_i_str] = 0.0
            continue
        if N_sub == 2:
            lfv_scores[n_i_str] = 2.0 if subgraph_undirected.number_of_edges() > 0 else 0.0
            continue

        try:
            L_norm_sub = nx.normalized_laplacian_matrix(subgraph_undirected)
            eigenvalues_sub = sla.eigsh(
                L_norm_sub, k=2, which="SM", tol=1e-3, maxiter=N_sub * 20, return_eigenvectors=False
            )
            lfv_scores[n_i_str] = float(eigenvalues_sub[1]) if len(eigenvalues_sub) > 1 else 0.0
        except (
            sla.ArpackNoConvergence,
            ValueError,
            IndexError,
            TypeError,
        ) as e:  # Added TypeError to this handler
            lfv_scores[n_i_str] = 0.0

    return lfv_scores if per_node else lfv_scores.get(str(node))


def get_node_global_fiedler_component(graph, per_node=False, node=None):
    """
    Returns the component of the (precomputed) global Fiedler vector for a node.
    """
    global _GLOBAL_FIEDLER_VECTOR
    if _GLOBAL_FIEDLER_VECTOR is None:
        print(
            "Warning: Global Fiedler vector not precomputed. Call precompute_global_fiedler_info(graph) first."
        )
        default_vector = {n: 0.0 for n in graph.nodes()}
        return (
            {str(n): abs(default_vector.get(str(n), 0.0)) for n in graph.nodes()}
            if per_node
            else abs(default_vector.get(str(node), 0.0))
        )

    if per_node:
        return {str(n): abs(_GLOBAL_FIEDLER_VECTOR.get(str(n), 0.0)) for n in graph.nodes()}
    else:
        if node is None:
            raise ValueError("Node must be specified.")
        return abs(_GLOBAL_FIEDLER_VECTOR.get(str(node), 0.0))


def get_localized_spectral_radius(graph, per_node=False, node=None, k_hop=1):
    """
    Calculates the spectral radius of the k-hop neighborhood subgraph.
    """
    lsr_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            lsr_scores[n_i_str] = 0.0
            continue

        subgraph = get_k_hop_neighborhood_subgraph(graph, n_i_str, k_hop)
        N_sub = subgraph.number_of_nodes()

        if N_sub <= 1:
            lsr_scores[n_i_str] = 0.0
            continue

        A_sub = nx.adjacency_matrix(subgraph)

        if N_sub <= 2:
            try:
                dense_A_sub = A_sub.toarray()
                eigenvalues_dense = np.linalg.eigvals(dense_A_sub)
                lsr_scores[n_i_str] = (
                    float(np.max(np.abs(eigenvalues_dense))) if len(eigenvalues_dense) > 0 else 0.0
                )
            except Exception:
                lsr_scores[n_i_str] = 0.0
            continue

        try:
            eigenvalues_sub = sla.eigs(
                A_sub, k=1, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=N_sub * 20
            )
            lsr_scores[n_i_str] = float(np.abs(eigenvalues_sub[0]))
        except (sla.ArpackNoConvergence, ValueError, TypeError) as e:  # Added TypeError to this handler
            try:
                dense_A_sub = A_sub.toarray()
                eigenvalues_dense = np.linalg.eigvals(dense_A_sub)
                lsr_scores[n_i_str] = (
                    float(np.max(np.abs(eigenvalues_dense))) if len(eigenvalues_dense) > 0 else 0.0
                )
            except Exception:
                lsr_scores[n_i_str] = 0.0

    return lsr_scores if per_node else lsr_scores.get(str(node))


def get_ppr_api_dri(
    graph, per_node=False, node=None, k_sum_hops=1, ppr_alpha=0.85, ppr_tol=1e-4, ppr_max_iter=100
):
    """
    Calculates an API-DRI based on Personalized PageRank (PPR).
    """
    ppr_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    for n_i in nodes_to_compute:
        n_i_str = str(n_i)
        if n_i_str not in graph:
            ppr_api_scores[n_i_str] = 0.0
            continue

        personalization_vector = {n: 0.0 for n in graph.nodes()}
        personalization_vector[n_i_str] = 1.0

        try:
            ppr_values = nx.pagerank(
                graph,
                alpha=ppr_alpha,
                personalization=personalization_vector,
                tol=ppr_tol,
                max_iter=ppr_max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            ppr_api_scores[n_i_str] = 0.0
            continue

        score_sum = 0.0
        nodes_in_khop = set()
        q = deque([(n_i_str, 0)])
        visited_bfs = {n_i_str}

        while q:
            curr, depth = q.popleft()
            if depth < k_sum_hops:
                for succ_node in graph.successors(curr):
                    if succ_node not in visited_bfs:
                        visited_bfs.add(succ_node)
                        nodes_in_khop.add(succ_node)
                        q.append((succ_node, depth + 1))

        for neighbor in nodes_in_khop:
            score_sum += ppr_values.get(neighbor, 0.0)

        ppr_api_scores[n_i_str] = score_sum

    return ppr_api_scores if per_node else ppr_api_scores.get(str(node))


def get_heat_kernel_api_dri(graph, per_node=False, node=None, t_short=0.1, use_normalized_laplacian=True):
    """
    Calculates an API-DRI based on the Heat Kernel.
    NOTE: This is computationally very expensive for large graphs.
    """
    hk_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and str(node) not in graph:
        return None

    if graph.number_of_nodes() == 0:
        return {str(n): 0.0 for n in nodes_to_compute} if per_node else 0.0

    current_graph_view = nx.Graph(graph) if graph.is_directed() else graph

    if current_graph_view.number_of_nodes() > 1500:
        print(
            f"Warning: Heat Kernel API-DRI skipped for graph with {current_graph_view.number_of_nodes()} nodes due to cost."
        )
        return {str(n): 0.0 for n in nodes_to_compute} if per_node else 0.0

    try:
        if use_normalized_laplacian:
            L = nx.normalized_laplacian_matrix(current_graph_view)
        else:
            L = nx.laplacian_matrix(current_graph_view)

        H_matrix_dense = scipy.linalg.expm(-t_short * L.toarray())
        node_list = list(current_graph_view.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
    except Exception as e:
        print(f"Error computing Heat Kernel matrix: {e}. Returning zeros.")
        return {str(n): 0.0 for n in nodes_to_compute} if per_node else 0.0

    for n_i_orig in nodes_to_compute:
        n_i_str = str(n_i_orig)
        if n_i_str not in current_graph_view:
            hk_api_scores[n_i_orig] = 0.0
            continue

        idx_i = node_to_idx.get(n_i_str)
        if idx_i is None:
            hk_api_scores[n_i_orig] = 0.0
            continue

        current_hk_sum = 0.0
        for neighbor_orig in current_graph_view.neighbors(n_i_str):
            neighbor_str = str(neighbor_orig)
            idx_neighbor = node_to_idx.get(neighbor_str)
            if idx_neighbor is not None:
                current_hk_sum += H_matrix_dense[idx_i, idx_neighbor]
        hk_api_scores[n_i_orig] = current_hk_sum

    return hk_api_scores if per_node else hk_api_scores.get(str(node))


# --- Main execution block (for testing this module independently) ---
if __name__ == "__main__":
    print("Testing spectral_dri.py...")

    # Create a sample graph for testing
    G_test = nx.karate_club_graph()
    DG_test = nx.DiGraph(G_test.edges())  # Make a directed version for some tests
    test_node = 0  # A specific node in Karate club
    # Ensure test_node is string if node IDs are loaded as strings later
    # For Karate club, nodes are integers, so direct use is fine.

    print(f"\n--- Testing on Karate Club Graph ({G_test.number_of_nodes()} nodes) ---")

    # Precompute global Fiedler info (important for get_node_global_fiedler_component)
    print("\nPrecomputing global Fiedler information...")
    precompute_global_fiedler_info(G_test)  # Use undirected version for global
    print(f"Global Fiedler Value (lambda_2 of L_norm): {_GLOBAL_FIEDLER_VALUE:.4f}")

    # 1. Localized Fiedler Value
    lfv_all = get_localized_fiedler_value(G_test, per_node=True, k_hop=1)
    print(
        f"\nLocalized Fiedler Value (k=1, all, sample): {{k: round(v,3) for k,v in list(lfv_all.items())[:5]}}"
    )
    print(
        f"Localized Fiedler Value (k=1, node {test_node}): {get_localized_fiedler_value(G_test, node=test_node, k_hop=1):.3f}"
    )

    # 2. Node's Component in Global Fiedler Vector
    ngfc_all = get_node_global_fiedler_component(
        G_test, per_node=True
    )  # graph arg is just for node list here
    print(
        f"\nNode Global Fiedler Component (abs, all, sample): {{k: round(v,3) for k,v in list(ngfc_all.items())[:5]}}"
    )
    print(
        f"Node Global Fiedler Component (abs, node {test_node}): {get_node_global_fiedler_component(G_test, node=test_node):.3f}"
    )

    # 3. Localized Spectral Radius
    # Use DG_test for spectral radius of adjacency matrix if directionality matters for walks
    # Or G_test if using undirected definition
    lsr_all = get_localized_spectral_radius(DG_test, per_node=True, k_hop=1)
    print(
        f"\nLocalized Spectral Radius (k=1, DiGraph, all, sample): {{k: round(v,3) for k,v in list(lsr_all.items())[:5]}}"
    )
    print(
        f"Localized Spectral Radius (k=1, DiGraph, node {test_node}): {get_localized_spectral_radius(DG_test, node=test_node, k_hop=1):.3f}"
    )

    # 4. PPR-API-DRI
    ppr_api_all = get_ppr_api_dri(DG_test, per_node=True, k_sum_hops=1, ppr_alpha=0.85)
    print(
        f"\nPPR-API-DRI (k_sum=1, DiGraph, all, sample): {{k: round(v,4) for k,v in list(ppr_api_all.items())[:5]}}"
    )
    print(
        f"PPR-API-DRI (k_sum=1, DiGraph, node {test_node}): {get_ppr_api_dri(DG_test, node=test_node, k_sum_hops=1, ppr_max_iter=1000):.4f}"
    )  # Added max_iter

    # 5. Heat Kernel API-DRI
    # Using undirected G_test for Heat Kernel as standard Laplacian is on undirected.
    hk_api_all = get_heat_kernel_api_dri(G_test, per_node=True, t_short=0.1, use_normalized_laplacian=True)
    print(
        f"\nHeat Kernel API-DRI (t=0.1, Undirected Graph, all, sample): {{k: round(v,4) for k,v in list(hk_api_all.items())[:5]}}"
    )
    print(
        f"Heat Kernel API-DRI (t=0.1, Undirected Graph, node {test_node}): {get_heat_kernel_api_dri(G_test, node=test_node, t_short=0.1):.4f}"
    )

    # Test a very small graph for spectral radius edge cases
    G_tiny = nx.Graph()
    G_tiny.add_node(0)  # Single node
    print(
        f"\nLocalized Spectral Radius (k=1, node 0, tiny graph G_tiny): {get_localized_spectral_radius(G_tiny, node=0, k_hop=1):.3f}"
    )  # Expected 0.0
    G_tiny.add_node(1)
    G_tiny.add_edge(0, 1)  # Two nodes, one edge
    print(
        f"Localized Spectral Radius (k=1, node 0, 2-node graph G_tiny, k_hop=1): {get_localized_spectral_radius(G_tiny, node=0, k_hop=1):.3f}"
    )  # Subgraph for node 0 is (0,1). Adj = [[0,1],[1,0]]. Evals are 1, -1. Radius = 1.0
    print(
        f"Localized Spectral Radius (k=1, node 1, 2-node graph G_tiny, k_hop=1): {get_localized_spectral_radius(G_tiny, node=1, k_hop=1):.3f}"
    )  # Subgraph for node 1 is (0,1). Radius = 1.0

    print("\n--- Spectral DRI Test Complete ---")
