# diffusion_readiness_project/structural_indices/spectral_tri.py
# Python 3.9

"""
Functions to calculate Transmission Readiness Indices (TRIs) based on
spectral graph theory concepts.
Includes:
- Localized Fiedler Value (Algebraic Connectivity of k-hop neighborhood)
- Node's component in the Global Fiedler Vector
- Localized Spectral Radius (of k-hop neighborhood adjacency matrix)
- API-TRI based on Personalized PageRank (PPR)
- API-TRI based on Heat Kernel
"""

import networkx as nx
import numpy as np
import scipy.sparse.linalg as sla # For specific eigenvalues (e.g., Fiedler value)
import scipy.linalg # For matrix exponential (expm for Heat Kernel)
from collections import deque # For k-hop BFS if not using graph_utils

# It's assumed that graph_utils.utils might be in a different path.
# For the project structure, it would be:
# from graph_utils.utils import get_k_hop_neighborhood_subgraph, get_graph_laplacian
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
_GLOBAL_FIEDLER_VECTOR = None # Cache for the global Fiedler vector
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
        largest_cc_nodes = max(nx.connected_components(graph_undirected), key=len)
        graph_undirected = graph_undirected.subgraph(largest_cc_nodes).copy() # Work on LCC
        if graph_undirected.number_of_nodes() < 2: # LCC too small
            _GLOBAL_FIEDLER_VALUE = 0.0
            _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()} # Original graph nodes
            return

    try:
        # Normalized Laplacian is often preferred for Fiedler vector interpretability
        L_norm = nx.normalized_laplacian_matrix(graph_undirected) 
        
        # Eigenvalues are sorted by magnitude by eigsh. Smallest non-zero is Fiedler.
        # We need the second smallest eigenvalue of L (or L_norm).
        # For L_norm, eigenvalues are between 0 and 2.
        # We need at least 2 eigenvalues if the graph is connected (smallest is 0).
        num_nodes_in_component = graph_undirected.number_of_nodes()
        if num_nodes_in_component < 2: # Should be caught above, but double check
             _GLOBAL_FIEDLER_VALUE = 0.0
             _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}
             return

        # For L_norm, the smallest eigenvalue is 0. We want the next one.
        # `k=2` for smallest two, `which='SM'` for smallest magnitude.
        # eigsh is for symmetric matrices; L_norm is symmetric.
        eigenvalues, eigenvectors = sla.eigsh(L_norm, k=2, which='SM', tol=1e-4) # tol for convergence
        
        _GLOBAL_FIEDLER_VALUE = float(eigenvalues[1]) # Second smallest eigenvalue
        
        # Map eigenvector components back to original node IDs
        # Need mapping from L_norm matrix indices to original node IDs of the component
        node_list_component = list(graph_undirected.nodes())
        fiedler_vec_component = eigenvectors[:, 1]
        
        _GLOBAL_FIEDLER_VECTOR = {node_id: 0.0 for node_id in graph.nodes()} # Init for all original nodes
        for i, node_id in enumerate(node_list_component):
            _GLOBAL_FIEDLER_VECTOR[node_id] = float(fiedler_vec_component[i])

    except Exception as e:
        print(f"Error precomputing global Fiedler info: {e}. Setting to defaults.")
        _GLOBAL_FIEDLER_VALUE = 0.0
        _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}


# --- Node-Level Spectral TRI Functions ---

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
    if node is not None and not per_node and node not in graph: return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            lfv_scores[n_i] = 0.0; continue
        
        subgraph = _get_k_hop_subgraph_for_spectral(graph, n_i, k_hop)
        
        if subgraph.number_of_nodes() < 2: # Fiedler value not well-defined or 0
            lfv_scores[n_i] = 0.0
            continue
        
        # K-core and other measures use undirected typically. For Fiedler on local patch:
        subgraph_undirected = nx.Graph(subgraph) if subgraph.is_directed() else subgraph

        # Ensure the subgraph is connected, or take largest component
        if not nx.is_connected(subgraph_undirected):
            if subgraph_undirected.number_of_nodes() > 0: # If it has nodes
                try:
                    largest_cc_nodes = max(nx.connected_components(subgraph_undirected), key=len)
                    subgraph_undirected = subgraph_undirected.subgraph(largest_cc_nodes)
                except ValueError: # No connected components (e.g. empty graph)
                    lfv_scores[n_i] = 0.0
                    continue
            else: # Empty subgraph
                lfv_scores[n_i] = 0.0
                continue
        
        if subgraph_undirected.number_of_nodes() < 2:
             lfv_scores[n_i] = 0.0
             continue
             
        try:
            # algebraic_connectivity (Fiedler value) can be slow for large subgraphs if not careful
            # For small k-hop subgraphs, it should be acceptable.
            # Uses Lanczos algorithm for eigendecomposition.
            # For normalized, use nx.normalized_laplacian_spectrum, then pick 2nd smallest.
            # nx.algebraic_connectivity is for the combinatorial Laplacian.
            # Let's use normalized for consistency with global Fiedler.
            L_norm_sub = nx.normalized_laplacian_matrix(subgraph_undirected)
            if L_norm_sub.shape[0] < 2: # too small
                lfv_scores[n_i] = 0.0
                continue

            # Get the second smallest eigenvalue
            eigenvalues_sub = sla.eigsh(L_norm_sub, k=min(2, L_norm_sub.shape[0]-1), which='SM', tol=1e-4, return_eigenvectors=False)
            lfv_scores[n_i] = float(eigenvalues_sub[1]) if len(eigenvalues_sub) > 1 else 0.0

        except (nx.NetworkXError, nx.NetworkXAlgorithmError, sla.ArpackNoConvergence) as e:
            # print(f"Could not compute localized Fiedler for node {n_i}: {e}")
            lfv_scores[n_i] = 0.0 # Default on error
            
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
        print("Warning: Global Fiedler vector not precomputed. Call precompute_global_fiedler_info(graph) first. Returning zeros.")
        _GLOBAL_FIEDLER_VECTOR = {n: 0.0 for n in graph.nodes()}

    if per_node:
        # Return the absolute values as sign can be arbitrary for eigenvectors
        return {n: abs(_GLOBAL_FIEDLER_VECTOR.get(n, 0.0)) for n in graph.nodes()}
    else:
        if node is None: raise ValueError("Node must be specified.")
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
    if node is not None and not per_node and node not in graph: return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            lsr_scores[n_i] = 0.0; continue

        subgraph = _get_k_hop_subgraph_for_spectral(graph, n_i, k_hop)
        if subgraph.number_of_nodes() == 0:
            lsr_scores[n_i] = 0.0
            continue
        
        # Adjacency matrix of the subgraph
        # For DiGraph, use as is. For Graph, also as is.
        A_sub = nx.adjacency_matrix(subgraph) # Returns scipy sparse matrix
        try:
            # Largest magnitude eigenvalue
            eigenvalues_sub = sla.eigs(A_sub, k=1, which='LM', return_eigenvectors=False, tol=1e-4) # LM for Largest Magnitude
            lsr_scores[n_i] = float(np.abs(eigenvalues_sub[0]))
        except (sla.ArpackNoConvergence, ValueError) as e: # ValueError if k >= N-1 for eigs
            # print(f"Could not compute localized spectral radius for node {n_i} (subgraph size {A_sub.shape[0]}): {e}")
            # Fallback for very small or problematic subgraphs
            if A_sub.shape[0] > 0:
                try:
                    dense_A_sub = A_sub.toarray()
                    eigenvalues_dense = np.linalg.eigvals(dense_A_sub)
                    lsr_scores[n_i] = float(np.max(np.abs(eigenvalues_dense))) if len(eigenvalues_dense) > 0 else 0.0
                except Exception:
                    lsr_scores[n_i] = 0.0
            else:
                lsr_scores[n_i] = 0.0
            
    return lsr_scores if per_node else lsr_scores.get(node)


def get_ppr_api_tri(graph, per_node=False, node=None, k_sum_hops=1, ppr_alpha=0.85, ppr_tol=1e-4):
    """
    Calculates an API-TRI based on Personalized PageRank (PPR).
    The TRI for node u is the sum of PPR scores of its k_sum_hops neighbors,
    where PPR is computed with u as the personalization/restart node.

    Args:
        graph (nx.DiGraph): Input directed graph.
        per_node (bool): If True, returns dict.
        node (any hashable, optional): Specific node.
        k_sum_hops (int): How many hops out to sum PPR scores of neighbors. (e.g., 1 or 2)
        ppr_alpha (float): Damping parameter for PageRank.
        ppr_tol (float): Tolerance for PageRank convergence.

    Returns:
        dict or float: PPR-API-TRI scores.
    """
    ppr_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph: return None

    for n_i in nodes_to_compute:
        if n_i not in graph:
            ppr_api_scores[n_i] = 0.0; continue
            
        personalization_vector = {n: 0.0 for n in graph.nodes()}
        personalization_vector[n_i] = 1.0 # Restart at node n_i
        
        try:
            ppr_values_for_ni_restart = nx.pagerank(
                graph, 
                alpha=ppr_alpha, 
                personalization=personalization_vector,
                tol=ppr_tol
            )
        except nx.PowerIterationFailedConvergence:
            # print(f"PPR failed to converge for node {n_i}. Setting score to 0.")
            ppr_api_scores[n_i] = 0.0
            continue
            
        # Sum PPR scores of k_sum_hops neighbors
        # Get k-hop neighborhood (successors for DiGraph)
        score_sum = 0.0
        # Using a simple BFS for k-hop successors from n_i
        # This should be nodes *reached by* n_i
        nodes_to_sum_ppr = set()
        q = deque([(n_i, 0)])
        visited_for_sum = {n_i} # To avoid cycles and re-adding

        # Collect nodes within k_sum_hops *excluding n_i itself*
        # These are nodes n_i potentially transmits its "rank" or "influence" to.
        temp_q = deque([(n_i,0)])
        visited_bfs = {n_i}
        nodes_in_khop_from_ni = set()

        while temp_q:
            curr, depth = temp_q.popleft()
            if depth < k_sum_hops:
                for succ_node in graph.successors(curr):
                    if succ_node not in visited_bfs:
                        visited_bfs.add(succ_node)
                        nodes_in_khop_from_ni.add(succ_node)
                        temp_q.append((succ_node, depth + 1))
        
        for neighbor_in_k_hop in nodes_in_khop_from_ni:
            score_sum += ppr_values_for_ni_restart.get(neighbor_in_k_hop, 0.0)
            
        ppr_api_scores[n_i] = score_sum
        
    return ppr_api_scores if per_node else ppr_api_scores.get(node)


def get_heat_kernel_api_tri(graph, per_node=False, node=None, t_short=0.1, use_normalized_laplacian=True):
    """
    Calculates an API-TRI based on the Heat Kernel.
    TRI_HeatKernel(u) = sum_{v in N1(u)} H_uv(t_short)
    where H(t) = exp(-tL), L is the graph Laplacian.

    Args:
        graph (nx.Graph or nx.DiGraph): Input graph.
        per_node (bool): If True, returns dict.
        node (any hashable, optional): Specific node.
        t_short (float): A small diffusion time 't'.
        use_normalized_laplacian (bool): Whether to use normalized Laplacian.

    Returns:
        dict or float: Heat Kernel API-TRI scores.
    """
    # This can be computationally very expensive as it involves matrix exponential
    # of the full Laplacian. For large graphs, approximations or different approaches
    # might be needed, or this might only be feasible for smaller test graphs.
    
    hk_api_scores = {}
    nodes_to_compute = [node] if (node is not None and not per_node) else list(graph.nodes())
    if node is not None and not per_node and node not in graph: return None

    if graph.number_of_nodes() == 0:
        if per_node: return {n:0.0 for n in nodes_to_compute} # should be empty
        else: return 0.0

    try:
        if use_normalized_laplacian:
            L = nx.normalized_laplacian_matrix(graph)
        else:
            L = nx.laplacian_matrix(graph) # Combinatorial
        
        # The matrix L is sparse. expm works with sparse but can be slow.
        # H = expm(-t * L)
        # scipy.linalg.expm converts to dense, which is problematic for large L.
        # For sparse L, sla.expm() is for matrix-vector products, not full matrix.
        # This is a known challenge. For smaller graphs, dense conversion is fine.
        
        if L.shape[0] > 1000 and not per_node: # Heuristic for "large"
             print(f"Warning: Heat Kernel on graph with {L.shape[0]} nodes can be very slow due to dense matrix exponential.")
        
        # If we compute H for all nodes, do it once.
        # Otherwise, if only for one node, this is still computing the full H.
        # A more optimized way for single node might be needed if full H is too big.
        
        H_matrix_dense = scipy.linalg.expm(-t_short * L.toarray()) # toarray() makes it dense!
        
        # Need a mapping from matrix row/col index to node ID
        node_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

    except Exception as e:
        print(f"Error computing Heat Kernel matrix: {e}. Returning zeros.")
        if per_node: return {n: 0.0 for n in nodes_to_compute}
        else: return 0.0


    for n_i in nodes_to_compute:
        if n_i not in graph:
            hk_api_scores[n_i] = 0.0; continue
        
        idx_i = node_to_idx.get(n_i)
        if idx_i is None: # Should not happen if node_list is from graph.nodes()
            hk_api_scores[n_i] = 0.0; continue

        current_hk_sum = 0.0
        # Sum heat diffused to immediate out-neighbors
        for neighbor in graph.successors(n_i): # For DiGraph, use successors
            idx_neighbor = node_to_idx.get(neighbor)
            if idx_neighbor is not None:
                current_hk_sum += H_matrix_dense[idx_i, idx_neighbor] # H_ui effectively, but matrix is H_ij(t) where u=i
                                                                  # Or H_neighbor,i if matrix represents flow to i
                                                                  # Standard H_uv(t) = e^(-tL)_uv, flow from u to v.
                                                                  # So, H_matrix_dense[idx_i, idx_neighbor] should be correct.
        
        hk_api_scores[n_i] = current_hk_sum
        
    return hk_api_scores if per_node else hk_api_scores.get(node)


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing spectral_tri.py...")

    # Create a sample graph for testing
    G_test = nx.karate_club_graph()
    DG_test = nx.DiGraph(G_test.edges()) # Make a directed version for some tests
    test_node = 0

    print(f"\n--- Testing on Karate Club Graph ({G_test.number_of_nodes()} nodes) ---")

    # Precompute global Fiedler info (important for get_node_global_fiedler_component)
    print("\nPrecomputing global Fiedler information...")
    precompute_global_fiedler_info(G_test) # Use undirected version for global
    print(f"Global Fiedler Value (lambda_2 of L_norm): {_GLOBAL_FIEDLER_VALUE:.4f}")


    # 1. Localized Fiedler Value
    lfv_all = get_localized_fiedler_value(G_test, per_node=True, k_hop=1)
    print(f"\nLocalized Fiedler Value (k=1, all, sample): {{k: round(v,3) for k,v in list(lfv_all.items())[:5]}}")
    print(f"Localized Fiedler Value (k=1, node {test_node}): {get_localized_fiedler_value(G_test, node=test_node, k_hop=1):.3f}")

    # 2. Node's Component in Global Fiedler Vector
    ngfc_all = get_node_global_fiedler_component(G_test, per_node=True)
    print(f"\nNode Global Fiedler Component (abs, all, sample): {{k: round(v,3) for k,v in list(ngfc_all.items())[:5]}}")
    print(f"Node Global Fiedler Component (abs, node {test_node}): {get_node_global_fiedler_component(G_test, node=test_node):.3f}")

    # 3. Localized Spectral Radius
    lsr_all = get_localized_spectral_radius(DG_test, per_node=True, k_hop=1) # Use DiGraph for Adjacency matrix interpretation
    print(f"\nLocalized Spectral Radius (k=1, DiGraph, all, sample): {{k: round(v,3) for k,v in list(lsr_all.items())[:5]}}")
    print(f"Localized Spectral Radius (k=1, DiGraph, node {test_node}): {get_localized_spectral_radius(DG_test, node=test_node, k_hop=1):.3f}")
    
    # 4. PPR-API-TRI
    ppr_api_all = get_ppr_api_tri(DG_test, per_node=True, k_sum_hops=1, ppr_alpha=0.85)
    print(f"\nPPR-API-TRI (k_sum=1, DiGraph, all, sample): {{k: round(v,4) for k,v in list(ppr_api_all.items())[:5]}}")
    print(f"PPR-API-TRI (k_sum=1, DiGraph, node {test_node}): {get_ppr_api_tri(DG_test, node=test_node, k_sum_hops=1):.4f}")

    # 5. Heat Kernel API-TRI
    # Note: Heat Kernel can be slow on larger graphs if not optimized. Karate Club is small enough.
    # Using undirected graph for Heat Kernel as Laplacian is typically for undirected.
    hk_api_all = get_heat_kernel_api_tri(G_test, per_node=True, t_short=0.1, use_normalized_laplacian=True)
    print(f"\nHeat Kernel API-TRI (t=0.1, Undirected Graph, all, sample): {{k: round(v,4) for k,v in list(hk_api_all.items())[:5]}}")
    print(f"Heat Kernel API-TRI (t=0.1, Undirected Graph, node {test_node}): {get_heat_kernel_api_tri(G_test, node=test_node, t_short=0.1):.4f}")

    print("\n--- Spectral TRI Test Complete ---")

