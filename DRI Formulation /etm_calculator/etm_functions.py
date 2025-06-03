# diffusion_readiness_project/etm_calculator/etm_functions.py
# Python 3.9

"""
Functions to calculate Effective Transmission Metrics (ETMs):
- Average Depth of Transmission (ADT)
- Maximum Cascade Subgraph Size (MCSS)
- High Neighborhood Reach (HNR_k)

These metrics are calculated based on observed or simulated cascade data.
The input `node_specific_cascade_data` is assumed to be a list of cascades
initiated by or significantly involving the `target_node`. Each cascade
in this list should ideally be represented as a list of propagation events
(e.g., (source, target, step/time_delay)) or a pre-built propagation tree/graph.
"""

import networkx as nx
from collections import Counter, defaultdict
import numpy as np # For averaging

# --- ETM Calculation Functions ---

def _build_cascade_graph(cascade_events):
    """
    Helper function to build a directed graph for a single cascade from its events.
    Events are expected to be (source, target, step/time) tuples.
    The graph represents the actual propagation path.
    """
    cascade_graph = nx.DiGraph()
    if not cascade_events:
        return cascade_graph
    
    # Assuming events are (u, v, step_or_time)
    # We only need the edges for structural analysis of the cascade
    edges = [(event[0], event[1]) for event in cascade_events]
    cascade_graph.add_edges_from(edges)
    
    # Add all involved nodes, even if some are only targets with no out-degree in this cascade
    all_nodes_in_cascade = set()
    for u, v, *_ in cascade_events: # *_ handles if step/time is not always there or needed here
        all_nodes_in_cascade.add(u)
        all_nodes_in_cascade.add(v)
    cascade_graph.add_nodes_from(list(all_nodes_in_cascade)) # Ensure all participating nodes are present
    
    return cascade_graph

def calculate_adt(target_node_cascades, target_node):
    """
    Calculates the Average Depth of Transmission (ADT) for a target_node.
    ADT is the average depth of the propagation trees/subgraphs rooted at (or significantly
    influenced by) the target_node across multiple cascades.
    Depth here means the longest shortest path from the target_node to any other
    node it (directly or indirectly) infected within that specific cascade.

    Args:
        target_node_cascades (list): A list of individual cascades. Each cascade is
                                     represented by a list of propagation events
                                     [(parent1, child1, step1), (parent2, child2, step2), ...].
                                     These should be cascades where `target_node` acted as a spreader.
        target_node (any hashable): The node for which ADT is being calculated.

    Returns:
        float: The Average Depth of Transmission. Returns 0.0 if no relevant cascades or
               if the target_node never spread to anyone.
    """
    all_cascade_depths = []

    if not target_node_cascades:
        return 0.0

    for cascade_events in target_node_cascades:
        if not cascade_events:
            continue

        # Build the specific graph for this cascade
        cascade_graph = _build_cascade_graph(cascade_events)

        if target_node not in cascade_graph:
            continue # Target node wasn't part of this cascade's graph representation

        # Find all nodes reachable from target_node *within this cascade_graph*
        # These are nodes that target_node (directly or indirectly) 'infected' in this cascade.
        # We only consider paths that start from target_node.
        
        # Filter cascade_graph to only include edges originating directly or indirectly from target_node
        # This means we are interested in the subgraph of influence of target_node *within* this cascade.
        
        # One way: find all descendants of target_node in this cascade_graph
        # However, target_node might not be the absolute root of the cascade_graph.
        # It's the depth of spread *from* target_node.
        
        # For each cascade, consider the subgraph where target_node is the source.
        # If target_node is not a source in any event, its depth of spread is 0 for that cascade.
        
        # Let's refine: we need to find the "downstream" part of the cascade from target_node.
        # Nodes reached from target_node in this cascade.
        
        if not any(event[0] == target_node for event in cascade_events): # target_node did not infect anyone
            # If target_node was infected but didn't spread, its depth contribution for this cascade is 0.
            # Or, if we only consider cascades *initiated* by target_node, this check might be different.
            # For now, let's assume target_node_cascades contains cascades where target_node *did* spread.
            # If it's just *part* of a cascade, we need to trace its specific influence.
            pass # This cascade might not be relevant if target_node didn't spread.

        max_depth_for_this_cascade = 0
        
        # Find all nodes reachable from target_node within this specific cascade
        # We need to ensure paths are within the cascade structure.
        # If target_node is in cascade_graph.nodes:
        try:
            # Get all nodes reachable from target_node
            descendants = nx.descendants(cascade_graph, target_node)
            # Add target_node itself to consider paths starting from it
            nodes_influenced_by_target = descendants.union({target_node})
            
            # Create the subgraph of influence for this cascade
            influence_subgraph = cascade_graph.subgraph(nodes_influenced_by_target)

            if not influence_subgraph.nodes() or target_node not in influence_subgraph:
                all_cascade_depths.append(0) # Or skip if node not in subgraph
                continue

            # Calculate shortest path lengths from target_node to all other nodes in its influence_subgraph
            # This gives the "depth" of each influenced node relative to target_node.
            path_lengths = nx.shortest_path_length(influence_subgraph, source=target_node)
            
            if path_lengths: # If target_node reached at least itself (or others)
                current_max_depth = max(path_lengths.values())
                max_depth_for_this_cascade = current_max_depth
            else: # target_node is isolated in its own influence subgraph (shouldn't happen if constructed correctly)
                max_depth_for_this_cascade = 0
                
        except nx.NetworkXError: # target_node might not be in cascade_graph, or no paths
             max_depth_for_this_cascade = 0 # Or handle as appropriate

        all_cascade_depths.append(max_depth_for_this_cascade)

    if not all_cascade_depths:
        return 0.0
    
    return np.mean(all_cascade_depths)


def calculate_mcss(target_node_cascades, target_node):
    """
    Calculates the Maximum Cascade Subgraph Size (MCSS) for a target_node.
    MCSS is the size (number of nodes) of the largest connected component
    of the propagation subgraph rooted at (or significantly influenced by) the target_node,
    averaged over multiple cascades. Or, it could be the max size observed.
    Let's define it as the average size of the subgraph of nodes influenced by target_node.

    Args:
        target_node_cascades (list): A list of individual cascades where target_node spread.
                                     Each cascade is a list of propagation events.
        target_node (any hashable): The node for which MCSS is being calculated.

    Returns:
        float: The average size of the influence subgraph caused by target_node.
               Returns 0.0 if no relevant cascades.
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
            # Nodes reached from target_node in this cascade
            descendants = nx.descendants(cascade_graph, target_node)
            # The size of the influence subgraph is target_node + its descendants in this cascade
            # If target_node has no descendants, size is 1 (itself).
            current_subgraph_size = 1 + len(descendants) 
        except nx.NetworkXError: # target_node might not be in cascade_graph
            current_subgraph_size = 0 # Or handle as appropriate

        all_cascade_subgraph_sizes.append(current_subgraph_size)

    if not all_cascade_subgraph_sizes:
        return 0.0
        
    return np.mean(all_cascade_subgraph_sizes)


def calculate_hnr_k(target_node_cascades, target_node, graph, k_hop=1):
    """
    Calculates the High Neighborhood Reach (HNR_k) for a target_node.
    HNR_k is the average fraction of a target_node's k-hop neighbors (in the static graph)
    that are also activated in cascades where the target_node participated as a spreader.

    Args:
        target_node_cascades (list): A list of individual cascades where target_node spread.
                                     Each cascade is a list of propagation events.
        target_node (any hashable): The node for which HNR_k is being calculated.
        graph (nx.DiGraph): The static underlying social/interaction graph.
        k_hop (int): The number of hops to define the neighborhood (e.g., 1 for direct neighbors).

    Returns:
        float: The average HNR_k value (between 0 and 1). Returns 0.0 if no relevant
               cascades or if the node has no k-hop neighbors.
    """
    all_hnr_k_values = []

    if target_node not in graph:
        # print(f"Warning: Target node {target_node} not in the main graph for HNR_k calculation.")
        return 0.0

    # Get k-hop neighbors of target_node from the static graph
    # For k=1, these are direct successors (out-neighbors)
    k_hop_neighbors = set()
    if k_hop == 1:
        k_hop_neighbors = set(graph.successors(target_node))
    else: # General k-hop (excluding self)
        # A bit more complex: all nodes reachable in k hops, excluding self.
        # Using BFS from graph_utils might be better here if that module is available.
        # For simplicity here, let's assume k=1 or a simplified k-hop.
        # nx.single_source_shortest_path_length can give distances
        try:
            path_lengths = nx.single_source_shortest_path_length(graph, target_node, cutoff=k_hop)
            for neighbor, dist in path_lengths.items():
                if 0 < dist <= k_hop: # Exclude self (dist=0)
                    k_hop_neighbors.add(neighbor)
        except nx.NetworkXError: # Node not in graph
             return 0.0


    if not k_hop_neighbors:
        # print(f"Warning: Target node {target_node} has no {k_hop}-hop neighbors in the static graph.")
        return 0.0 # No neighbors to reach

    if not target_node_cascades:
        return 0.0

    for cascade_events in target_node_cascades:
        if not cascade_events:
            continue

        # Identify all nodes activated in this specific cascade
        activated_in_this_cascade = set()
        # The cascade_events are (parent, child, step). We need all unique nodes involved.
        # Or, if we have the full list of activated nodes for the cascade, use that.
        # For now, let's assume cascade_events help reconstruct who got activated.
        # A simpler input for `target_node_cascades` might be:
        # list of sets, where each set is the activated nodes in a cascade involving target_node.
        
        # For now, let's build the cascade graph and see who got activated from target_node's influence.
        cascade_graph = _build_cascade_graph(cascade_events)
        if target_node not in cascade_graph:
            continue

        nodes_influenced_by_target_in_cascade = set()
        try:
            descendants = nx.descendants(cascade_graph, target_node)
            nodes_influenced_by_target_in_cascade = descendants.union({target_node}) # include self if activated
        except nx.NetworkXError:
            pass # target_node didn't spread or wasn't in this cascade representation
        
        # Find intersection: k-hop neighbors that were also activated in this cascade's influence spread
        reached_k_hop_neighbors = k_hop_neighbors.intersection(nodes_influenced_by_target_in_cascade)
        
        fraction_reached = len(reached_k_hop_neighbors) / len(k_hop_neighbors) if k_hop_neighbors else 0.0
        all_hnr_k_values.append(fraction_reached)

    if not all_hnr_k_values:
        return 0.0
        
    return np.mean(all_hnr_k_values)


# --- Pre-processing function for ETMs ---
def preprocess_cascade_data_for_node_etms(all_cascade_events_or_sims, all_nodes_in_graph):
    """
    Preprocesses raw cascade data (from CasFlow parser or IC simulations)
    to structure it per node for easier ETM calculation.

    Args:
        all_cascade_events_or_sims:
            - For CasFlow: A flat list of all infection events:
              `[{'cascade_id': cid, 'parent': p, 'target': t, 'infection_time_rel_to_parent': time}, ...]`
            - For IC Sims: A list of lists, where each inner list contains propagation events
              `[(source, target, step), ...]` for one simulation seeded by a particular node.
              Or, if run_multiple_ic_simulations returns a list of (activated_set, prop_events_list)
              per seed node, this needs to be adapted.

              Let's assume for IC sims, the input is a dictionary:
              `{seed_node1: [[(p,c,s),...]_sim1, [(p,c,s),...]_sim2, ...], seed_node2: ...}`

        all_nodes_in_graph (set or list): All unique node IDs in the static graph.

    Returns:
        dict: A dictionary where keys are node IDs and values are lists of
              cascades (each cascade being a list of (parent, child, step/time) tuples)
              in which that node acted as a spreader (parent).
              e.g., `{'nodeA': [[(A,B,1),(B,C,2)], [(A,D,1)]], 'nodeB': [...]}`
    """
    node_etm_input_data = defaultdict(list) # node_id -> list of its initiated/participated cascades

    # This logic needs to be robust based on the exact format of all_cascade_events_or_sims
    # For CasFlow (flat list of events):
    if isinstance(all_cascade_events_or_sims, list) and \
       all_cascade_events_or_sims and isinstance(all_cascade_events_or_sims[0], dict) and \
       'cascade_id' in all_cascade_events_or_sims[0]: # Heuristic for CasFlow event list
        
        cascades_by_id = defaultdict(list)
        for event in all_cascade_events_or_sims:
            # Ensure events have the expected structure
            if all(k in event for k in ('cascade_id', 'parent', 'target')):
                 # Using a simplified (parent, child) tuple for now for _build_cascade_graph
                 # The step/time might be event['infection_time_rel_to_parent'] or similar
                cascades_by_id[event['cascade_id']].append(
                    (str(event['parent']), str(event['target']), event.get('infection_time_rel_to_parent', 0))
                )

        # Now, for each node, find cascades where it acted as a spreader (parent)
        for node_id in all_nodes_in_graph:
            node_id_str = str(node_id)
            participated_cascades_for_node = []
            for cascade_id, events_in_cascade in cascades_by_id.items():
                # A node "participated as a spreader" if it's a 'parent' in any event of that cascade.
                # Or, more strictly, if it's the *source* of that cascade (needs source info).
                # For ETMs, we are interested in the spread *from* the target_node.
                # So, we need cascades where this node *initiated some part of the spread*.
                
                # We need to identify sub-cascades rooted at node_id_str.
                # This might be complex. A simpler approach for now:
                # Collect all cascades where node_id_str appears as a parent.
                # The ETM functions (ADT, MCSS) will then trace from node_id_str within those.
                
                is_spreader_in_cascade = any(event[0] == node_id_str for event in events_in_cascade)
                if is_spreader_in_cascade:
                    participated_cascades_for_node.append(events_in_cascade)
            
            if participated_cascades_for_node:
                node_etm_input_data[node_id_str] = participated_cascades_for_node

    # For IC Simulation output (dictionary of {seed_node: list_of_sim_event_lists})
    elif isinstance(all_cascade_events_or_sims, dict):
        for seed_node, list_of_simulations_for_seed in all_cascade_events_or_sims.items():
            seed_node_str = str(seed_node)
            # Each item in list_of_simulations_for_seed is a list of events for one simulation
            # These are directly cascades initiated by seed_node_str
            node_etm_input_data[seed_node_str].extend(list_of_simulations_for_seed)
            
            # If other nodes also spread within these simulations, and we want their ETMs,
            # we'd need to process `list_of_simulations_for_seed` similar to CasFlow above
            # to find all (parent, child) pairs. For now, this structure is primarily for
            # ETMs of the *seed nodes* of simulations.
            # For a general ETM for any node based on simulations where it *might* have spread (not just seed):
            # This would require pooling all simulation events and then processing like CasFlow.
            # Let's assume for now IC sim output is used to calculate ETMs for the SEED nodes.


    else:
        print("Warning: Unrecognized format for all_cascade_events_or_sims in preprocess_cascade_data_for_node_etms.")

    return dict(node_etm_input_data)


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing etm_functions.py...")

    # Sample static graph (used for HNR_k)
    G_static = nx.DiGraph()
    G_static.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'E'), ('B', 'F'),
        ('C', 'F'), ('C', 'G'),
        ('X', 'Y')
    ])
    print(f"Static Graph: Nodes: {G_static.nodes()}, Edges: {G_static.edges()}")


    # Sample cascade data for node 'A'
    # Cascade 1: A -> B -> E, A -> C -> F
    cascade1_events_A = [('A', 'B', 1), ('A', 'C', 1), ('B', 'E', 2), ('C', 'F', 2)]
    # Cascade 2: A -> D
    cascade2_events_A = [('A', 'D', 1)]
    # Cascade 3: A -> B (B doesn't spread further in this one from A's branch)
    cascade3_events_A = [('A', 'B', 1)]
    
    # Sample cascade data where 'A' is involved but not the primary spreader, or spreads less
    # Cascade 4: Z -> A -> B
    cascade4_events_Z = [('Z','A',1), ('A','B',2)]


    node_A_cascades = [cascade1_events_A, cascade2_events_A, cascade3_events_A, cascade4_events_Z]
    
    # Test ADT
    adt_A = calculate_adt(node_A_cascades, 'A')
    print(f"\nADT for node A: {adt_A:.2f}") # Expected: Cascade1 depth from A is 2 (A->B->E or A->C->F). C2 depth 1. C3 depth 1. C4 depth (A->B) is 1.
                                          # (2+1+1+1)/4 = 1.25

    # Test MCSS
    mcss_A = calculate_mcss(node_A_cascades, 'A')
    print(f"MCSS for node A (avg size of influence subgraph): {mcss_A:.2f}") # C1: A,B,C,E,F (5). C2: A,D (2). C3: A,B (2). C4: A,B (2)
                                                                        # (5+2+2+2)/4 = 2.75
    
    # Test HNR_k
    # k=1 neighbors of A in G_static are B, C, D
    hnr1_A = calculate_hnr_k(node_A_cascades, 'A', G_static, k_hop=1)
    print(f"HNR_1 for node A: {hnr1_A:.2f}")
    # C1: A influences B, C. (Reached B,C). Fraction = 2/3
    # C2: A influences D. (Reached D). Fraction = 1/3
    # C3: A influences B. (Reached B). Fraction = 1/3
    # C4: A influences B. (Reached B). Fraction = 1/3
    # Avg = (2/3 + 1/3 + 1/3 + 1/3) / 4 = (5/3) / 4 = 5/12 = 0.416

    # Test with a node that doesn't spread much or isn't a source often
    node_B_cascades = [cascade1_events_A] # B only spreads to E in C1 from A's cascade.
    adt_B = calculate_adt(node_B_cascades, 'B') # From B: B->E (depth 1)
    print(f"\nADT for node B (based on its role in cascade1 from A): {adt_B:.2f}") # Expected: 1.0
    mcss_B = calculate_mcss(node_B_cascades, 'B') # From B: B,E (size 2)
    print(f"MCSS for node B: {mcss_B:.2f}") # Expected: 2.0
    # k=1 neighbors of B in G_static are E, F
    hnr1_B = calculate_hnr_k(node_B_cascades, 'B', G_static, k_hop=1) # In C1, B reaches E. So 1/2.
    print(f"HNR_1 for node B: {hnr1_B:.2f}") # Expected: 0.5

    # Test preprocessing (conceptual)
    print("\nTesting preprocessing (conceptual example):")
    # Sample CasFlow-like flat event list
    all_events_sample = [
        {'cascade_id': 'c1', 'parent': 'A', 'target': 'B', 'infection_time_rel_to_parent': 1},
        {'cascade_id': 'c1', 'parent': 'A', 'target': 'C', 'infection_time_rel_to_parent': 1},
        {'cascade_id': 'c1', 'parent': 'B', 'target': 'D', 'infection_time_rel_to_parent': 1}, # B spreads in C1
        {'cascade_id': 'c2', 'parent': 'A', 'target': 'E', 'infection_time_rel_to_parent': 1},
        {'cascade_id': 'c3', 'parent': 'X', 'target': 'A', 'infection_time_rel_to_parent': 1}, # A is infected in C3
        {'cascade_id': 'c3', 'parent': 'A', 'target': 'F', 'infection_time_rel_to_parent': 1}, # A spreads in C3
    ]
    graph_nodes_sample = {'A', 'B', 'C', 'D', 'E', 'F', 'X'}
    processed_data = preprocess_cascade_data_for_node_etms(all_events_sample, graph_nodes_sample)
    
    if 'A' in processed_data:
        print(f"Cascades where A is a spreader: {len(processed_data['A'])} cascades")
        # print(processed_data['A']) 
        # Expected: A is spreader in c1 and c3.
        # processed_data['A'] should be a list containing two lists of events:
        # [ [('A','B',1), ('A','C',1), ('B','D',1)],  <-- events of c1
        #   [('X','A',1), ('A','F',1)] ]              <-- events of c3
        # The ETM functions will then trace from 'A' *within* these provided cascade contexts.
    if 'B' in processed_data:
         print(f"Cascades where B is a spreader: {len(processed_data['B'])} cascades")
         # Expected: B is spreader in c1.
         # processed_data['B'] should be: [ [('A','B',1), ('A','C',1), ('B','D',1)] ]

    print("\n--- ETM Functions Test Complete ---")

