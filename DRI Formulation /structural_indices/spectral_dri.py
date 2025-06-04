# diffusion_readiness_project/structural_indices/spectral_dri.py
# Python 3.9

"""
Functions to calculate Transmission Readiness Indices (DRIs) based on
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
import scipy.sparse.linalg as sla  # For specific eigenvalues (e.g., Fiedler value)
import scipy.linalg  # For matrix exponential (expm for Heat Kernel)
from collections import deque  # For k-hop BFS if not using graph_utils

# It's assumed that graph_utils.utils might be in a different path.
# For the project structure, it would be:
from graph_utils.utils import get_k_hop_neighborhood_subgraph, get_graph_laplacian

# For now, let's define placeholders if used in testing block, or assume NetworkX for Laplacian.


# --- Placeholder for k-hop neighborhood (if not importing from graph_utils) ---
def _get_k_hop_subgraph_for_spectral(graph, center_node, k):
    """Simplified k-hop subgraph extraction for local spectral features."""
    if center_node not in graph:
        return nx.Graph() if not graph.is_directed() else nx.DiGraph()

    nodes_in_k_hop = {center_node}
    queue = deque([(center_node, 0)])
    visited = {center_node}

    while queue:
        current_node, depth = queue.popleft()
        if depth < k:
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_in_k_hop.add(neighbor)
                    queue.append((neighbor, depth + 1))

    return graph.subgraph(nodes_in_k_hop).copy()


# --- End Placeholder ---

# --- Global Spectral Feature (calculated once for the graph) ---
_GLOBAL_FIEDLER_VECTOR = None  # Cache for the global Fiedler vector
_GLOBAL_FIEDLER_VALUE = None  # Cache for the global Fiedler value


def precompute_global_fiedler_info(graph):
    """
    Precomputes the global Fiedler value and vector for the graph.
    This should be called once per graph if node_global_fiedler_component is used.
    """
    global _GLOBAL_FIEDLER_VECTOR, _GLOBAL_FIEDLER_VALUE
    if graph.number_of_nodes() < 2:
        _GLOBAL_FIEDLER_VALUE = 0.0
        _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}
        return

    # Ensure the graph is connected for a meaningful single Fiedler value,
    # or handle components separately if needed. NetworkX handles this by
    # usually operating on the largest connected component or raising errors.
    # For simplicity, let's assume we work on the graph as is or its largest component.

    # Use undirected version for Fiedler value/vector usually
    graph_undirected = nx.Graph(graph) if graph.is_directed() else graph

    # Get largest connected component to ensure meaningful Fiedler value
    if not nx.is_connected(graph_undirected):
        try:
            largest_cc_nodes = max(nx.connected_components(graph_undirected), key=len)
            if not largest_cc_nodes:  # Should not happen if graph has nodes
                _GLOBAL_FIEDLER_VALUE = 0.0
                _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}
                return
            graph_undirected = graph_undirected.subgraph(largest_cc_nodes).copy()  # Work on LCC
        except ValueError:  # No components found (empty graph)
            _GLOBAL_FIEDLER_VALUE = 0.0
            _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}
            return

    if graph_undirected.number_of_nodes() < 2:  # LCC too small
        _GLOBAL_FIEDLER_VALUE = 0.0
        # Ensure all original nodes get a default value
        _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}
        # Update for nodes in the small LCC (if any)
        for lcc_node in graph_undirected.nodes():
            _GLOBAL_FIEDLER_VECTOR[lcc_node] = 0.0
        return

    try:
        # Normalized Laplacian is often preferred for Fiedler vector interpretability
        L_norm = nx.normalized_laplacian_matrix(graph_undirected)

        num_nodes_in_component = graph_undirected.number_of_nodes()

        # For L_norm, the smallest eigenvalue is 0. We want the next one (k=2).
        # eigsh is for symmetric matrices; L_norm is symmetric.
        eigenvalues, eigenvectors = sla.eigsh(
            L_norm,
            k=2,  # Request the two smallest magnitude eigenvalues
            which="SM",
            tol=1e-3,  # Adjusted tolerance
            maxiter=num_nodes_in_component * 20,  # Adjusted maxiter
        )

        _GLOBAL_FIEDLER_VALUE = float(eigenvalues[1])  # Second smallest eigenvalue

        node_list_component = list(graph_undirected.nodes())
        fiedler_vec_component = eigenvectors[:, 1]

        _GLOBAL_FIEDLER_VECTOR = {
            node_id: 0.0 for node_id in graph.nodes()
        }  # Init for all original graph nodes
        for i, node_id in enumerate(node_list_component):
            _GLOBAL_FIEDLER_VECTOR[node_id] = float(fiedler_vec_component[i])

    except (sla.ArpackNoConvergence, ValueError, IndexError) as e:  # Added ValueError, IndexError
        print(
            f"Error precomputing global Fiedler info (LCC size {graph_undirected.number_of_nodes()}): {e}. Setting to defaults."
        )
        _GLOBAL_FIEDLER_VALUE = 0.0
        _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}


# --- Node-Level Spectral DRI Functions ---


def get_localized_fiedler_value(graph, per_node=False, node=None, k_hop=1):
    """
    Calculates the Fiedler value (algebraic connectivity) of the k-hop
    neighborhood subgraph around each node.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.
        k_hop (int): The number of hops for the neighborhood (e.g., 1 or 2).

    Returns:
        dict or float: Localized Fiedler values.
    """
    lfv_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph:
        return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            lfv_scores[n_i] = 0.0
            continue

        subgraph = _get_k_hop_subgraph_for_spectral(graph, n_i, k_hop)

        # K-core and other measures use undirected typically. For Fiedler on local patch:
        subgraph_undirected = nx.Graph(subgraph) if subgraph.is_directed() else subgraph

        # Ensure the subgraph is connected, or take largest component
        if not nx.is_connected(subgraph_undirected):
            if subgraph_undirected.number_of_nodes() > 0:
                try:
                    largest_cc_nodes = max(nx.connected_components(subgraph_undirected), key=len)
                    if not largest_cc_nodes:
                        lfv_scores[n_i] = 0.0
                        continue
                    subgraph_undirected = subgraph_undirected.subgraph(largest_cc_nodes)
                except ValueError:
                    lfv_scores[n_i] = 0.0
                    continue
            else:
                lfv_scores[n_i] = 0.0
                continue

        N_sub = subgraph_undirected.number_of_nodes()

        if N_sub < 2:
            lfv_scores[n_i] = 0.0
            continue
        if N_sub == 2:
            if subgraph_undirected.number_of_edges() > 0:  # Connected 2-node graph
                # For normalized Laplacian, eigenvalues are 0 and 2. Fiedler = 2.
                lfv_scores[n_i] = 2.0
            else:  # Two isolated nodes
                lfv_scores[n_i] = 0.0
            continue

        try:
            L_norm_sub = nx.normalized_laplacian_matrix(subgraph_undirected)
            # We need the 2nd smallest eigenvalue.
            # k=2 to get the first two (0 and Fiedler value for connected graph)
            eigenvalues_sub = sla.eigsh(
                L_norm_sub,
                k=2,
                which="SM",
                tol=1e-3,  # Adjusted tolerance
                maxiter=N_sub * 20,  # Adjusted maxiter
                return_eigenvectors=False,
            )
            lfv_scores[n_i] = float(eigenvalues_sub[1])

        except (
            nx.NetworkXError,
            nx.NetworkXAlgorithmError,
            sla.ArpackNoConvergence,
            ValueError,
            IndexError,
        ) as e:
            # print(f"Could not compute localized Fiedler for node {n_i} (N_sub={N_sub}): {e}")
            lfv_scores[n_i] = 0.0  # Default on error

    return lfv_scores if per_node else lfv_scores.get(node)


def get_node_global_fiedler_component(graph, per_node=False, node=None):
    """
    Returns the component of the (precomputed) global Fiedler vector for a node.
    Assumes precompute_global_fiedler_info(graph) has been called.

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph (used to check node existence).
        per_node (bool): If True, returns a dictionary.
        node (any hashable, optional): The specific node.

    Returns:
        dict or float: Node's value in the Fiedler vector (or its absolute value).
    """
    global _GLOBAL_FIEDLER_VECTOR
    if _GLOBAL_FIEDLER_VECTOR is None:
        print(
            "Warning: Global Fiedler vector not precomputed. Call precompute_global_fiedler_info(graph) first. Returning zeros."
        )
        default_vector = {n: 0.0 for n in graph.nodes()}
        return (
            {n: abs(default_vector.get(n, 0.0)) for n in graph.nodes()}
            if per_node
            else abs(default_vector.get(node, 0.0))
        )

    if per_node:
        # Return the absolute values as sign can be arbitrary for eigenvectors
        return {n: abs(_GLOBAL_FIEDLER_VECTOR.get(n, 0.0)) for n in graph.nodes()}
    else:
        if node is None:
            raise ValueError("Node must be specified.")
        return abs(_GLOBAL_FIEDLER_VECTOR.get(node, 0.0))


def get_localized_spectral_radius(graph, per_node=False, node=None, k_hop=1):
    """
    Calculates the spectral radius (largest eigenvalue of the adjacency matrix)
    of the k-hop neighborhood subgraph.

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph.
        per_node (bool): If True, returns dict.
        node (any hashable, optional): Specific node.
        k_hop (int): Hops for neighborhood.

    Returns:
        dict or float: Localized spectral radius.
    """
    lsr_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph:
        return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            lsr_scores[n_i] = 0.0
            continue

        subgraph = _get_k_hop_subgraph_for_spectral(graph, n_i, k_hop)
        N_sub = subgraph.number_of_nodes()

        if N_sub == 0:
            lsr_scores[n_i] = 0.0
            continue

        A_sub = nx.adjacency_matrix(subgraph)

        if N_sub <= 1:  # Handles N=0 (already caught) and N=1
            lsr_scores[n_i] = 0.0  # Spectral radius of a 1-node graph (no edges) is 0
            continue

        # For N_sub == 2, or other small N where sparse eigs might be problematic,
        # or if ArpackNoConvergence occurs, use dense solver.
        # The TypeError "k >= N - 1" for sparse A happens if N_sub=2 and k=1.
        if N_sub == 2:
            try:
                dense_A_sub = A_sub.toarray()
                eigenvalues_dense = np.linalg.eigvals(dense_A_sub)
                lsr_scores[n_i] = (
                    float(np.max(np.abs(eigenvalues_dense))) if len(eigenvalues_dense) > 0 else 0.0
                )
            except Exception as e_dense:
                # print(f"Dense eigvals failed for N_sub=2, node {n_i}: {e_dense}")
                lsr_scores[n_i] = 0.0
            continue  # Move to next node after handling N_sub=2

        # For N_sub > 2, attempt sparse solver first
        try:
            eigenvalues_sub = sla.eigs(
                A_sub, k=1, which="LM", return_eigenvectors=False, tol=1e-3, maxiter=N_sub * 20
            )
            lsr_scores[n_i] = float(np.abs(eigenvalues_sub[0]))
        except (sla.ArpackNoConvergence, ValueError) as e:
            # print(f"Sparse eigs failed for node {n_i} (subgraph size {N_sub}): {e}. Trying dense fallback.")
            try:
                dense_A_sub = A_sub.toarray()
                if dense_A_sub.size == 0:
                    lsr_scores[n_i] = 0.0
                else:
                    eigenvalues_dense = np.linalg.eigvals(dense_A_sub)
                    lsr_scores[n_i] = (
                        float(np.max(np.abs(eigenvalues_dense))) if len(eigenvalues_dense) > 0 else 0.0
                    )
            except Exception as e_dense:
                # print(f"Dense eigvals fallback also failed for node {n_i}: {e_dense}")
                lsr_scores[n_i] = 0.0

    return lsr_scores if per_node else lsr_scores.get(node)


def get_ppr_api_dri(
    graph, per_node=False, node=None, k_sum_hops=1, ppr_alpha=0.85, ppr_tol=1e-4, ppr_max_iter=100
):
    """
    Calculates an API-DRI based on Personalized PageRank (PPR).
    The DRI for node u is the sum of PPR scores of its k_sum_hops neighbors,
    where PPR is computed with u as the personalization/restart node.

    Args:
        graph (nx.DiGraph): Input directed graph.
        per_node (bool): If True, returns dict.
        node (any hashable, optional): Specific node.
        k_sum_hops (int): How many hops out to sum PPR scores of neighbors. (e.g., 1 or 2)
        ppr_alpha (float): Damping parameter for PageRank.
        ppr_tol (float): Tolerance for PageRank convergence.
        ppr_max_iter (int): Max iterations for PageRank.

    Returns:
        dict or float: PPR-API-DRI scores.
    """
    ppr_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph:
        return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            ppr_api_scores[n_i] = 0.0
            continue

        personalization_vector = {n: 0.0 for n in graph.nodes()}
        personalization_vector[n_i] = 1.0  # Restart at node n_i

        try:
            ppr_values_for_ni_restart = nx.pagerank(
                graph,
                alpha=ppr_alpha,
                personalization=personalization_vector,
                tol=ppr_tol,
                max_iter=ppr_max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            # print(f"PPR failed to converge for node {n_i}. Setting score to 0.")
            ppr_api_scores[n_i] = 0.0
            continue

        score_sum = 0.0
        nodes_in_khop_from_ni = set()
        q = deque([(n_i, 0)])  # (node, depth)
        visited_bfs = {n_i}  # Start with n_i so its own PPR score isn't summed if k_sum_hops=0

        # Collect nodes within k_sum_hops (successors) *excluding n_i itself*
        while q:
            curr, depth = q.popleft()
            if depth < k_sum_hops:
                for succ_node in graph.successors(curr):
                    if succ_node not in visited_bfs:
                        visited_bfs.add(succ_node)
                        nodes_in_khop_from_ni.add(succ_node)  # Add successor
                        q.append((succ_node, depth + 1))

        for neighbor_in_k_hop in nodes_in_khop_from_ni:
            score_sum += ppr_values_for_ni_restart.get(neighbor_in_k_hop, 0.0)

        ppr_api_scores[n_i] = score_sum

    return ppr_api_scores if per_node else ppr_api_scores.get(node)


def get_heat_kernel_api_dri(graph, per_node=False, node=None, t_short=0.1, use_normalized_laplacian=True):
    """
    Calculates an API-DRI based on the Heat Kernel.
    DRI_HeatKernel(u) = sum_{v in N1(u)} H_uv(t_short)
    where H(t) = exp(-tL), L is the graph Laplacian.
    NOTE: This is computationally very expensive for large graphs due to dense matrix exponential.

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph. For DiGraph, uses underlying undirected for Laplacian typically.
        per_node (bool): If True, returns dict.
        node (any hashable, optional): Specific node.
        t_short (float): A small diffusion time 't'.
        use_normalized_laplacian (bool): Whether to use normalized Laplacian.

    Returns:
        dict or float: Heat Kernel API-DRI scores.
    """
    hk_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph:
        return None

    if graph.number_of_nodes() == 0:
        return {n: 0.0 for n in nodes_to_compute} if per_node else 0.0

    # For DiGraph, use its undirected version for standard Laplacian
    current_graph_view = nx.Graph(graph) if graph.is_directed() else graph

    # Check graph size to decide if Heat Kernel is feasible
    # Set a threshold, e.g., 1500 nodes. Above this, matrix exponential is too slow.
    if current_graph_view.number_of_nodes() > 1500:  # Arbitrary threshold
        print(
            f"Warning: Heat Kernel API-DRI skipped for graph with {current_graph_view.number_of_nodes()} nodes due to computational cost. Returning zeros."
        )
        return {n: 0.0 for n in nodes_to_compute} if per_node else 0.0

    try:
        if use_normalized_laplacian:
            L = nx.normalized_laplacian_matrix(current_graph_view)
        else:
            L = nx.laplacian_matrix(current_graph_view)  # Combinatorial

        H_matrix_dense = scipy.linalg.expm(-t_short * L.toarray())  # This is the expensive step

        node_list = list(current_graph_view.nodes())  # Use nodes from the graph view L was based on
        node_to_idx = {n: i for i, n in enumerate(node_list)}

    except Exception as e:
        print(f"Error computing Heat Kernel matrix: {e}. Returning zeros.")
        return {n: 0.0 for n in nodes_to_compute} if per_node else 0.0

    for n_i_orig in nodes_to_compute:  # Iterate over original node IDs
        n_i = str(n_i_orig)  # Ensure string if node IDs are mixed types, matching node_list
        if n_i not in current_graph_view:  # Check if node is in the graph view used for L
            hk_api_scores[n_i_orig] = 0.0
            continue

        idx_i = node_to_idx.get(n_i)
        if idx_i is None:
            hk_api_scores[n_i_orig] = 0.0
            continue

        current_hk_sum = 0.0
        # For DiGraph, successors of original graph. For Graph, neighbors.
        # The heat flow is modeled on current_graph_view (undirected version for standard L).
        # So, neighbors should be from current_graph_view.
        for neighbor_orig in current_graph_view.neighbors(n_i):
            neighbor = str(neighbor_orig)
            idx_neighbor = node_to_idx.get(neighbor)
            if idx_neighbor is not None:
                current_hk_sum += H_matrix_dense[idx_i, idx_neighbor]

        hk_api_scores[n_i_orig] = current_hk_sum

    return hk_api_scores if per_node else hk_api_scores.get(node)


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
