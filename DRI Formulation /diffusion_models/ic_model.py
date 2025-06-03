# diffusion_readiness_project/diffusion_models/ic_model.py
# Python 3.9

"""
Implementation of the Independent Cascade (IC) model for simulating diffusion.
"""

import networkx as nx
import random # For probabilistic activation

# --- Independent Cascade (IC) Model ---

def run_ic_simulation(graph, seed_nodes, propagation_probability, max_iterations=1000, record_propagation_tree=False):
    """
    Simulates the Independent Cascade (IC) model on a given graph.

    Args:
        graph (nx.DiGraph): The input directed graph.
        seed_nodes (list or set): A collection of initially active nodes.
        propagation_probability (float): The uniform probability (p) with which an active
                                         node attempts to activate its inactive neighbors.
                                         Should be between 0 and 1.
        max_iterations (int): A safeguard to prevent excessively long simulations.
        record_propagation_tree (bool): If True, also returns the edges that
                                        successfully caused an activation.

    Returns:
        tuple: (final_activated_nodes, propagation_events)
            - final_activated_nodes (set): A set of all nodes activated by the end
                                           of the cascade.
            - propagation_events (list of tuples): If record_propagation_tree is True,
                                                   a list of (source, target, step) tuples
                                                   representing successful activations.
                                                   Otherwise, an empty list.
    """
    if not isinstance(graph, nx.DiGraph):
        # print("Warning: IC model typically expects a directed graph. Proceeding, but behavior might be unexpected if graph is undirected.")
        pass # Or raise TypeError if strict DiGraph is required

    if not 0 <= propagation_probability <= 1:
        raise ValueError("Propagation probability must be between 0 and 1.")

    if not seed_nodes:
        # print("Warning: No seed nodes provided for IC simulation. Returning empty sets.")
        return set(), []

    # Ensure all seed nodes are actually in the graph
    valid_seed_nodes = {node for node in seed_nodes if node in graph}
    if not valid_seed_nodes:
        # print("Warning: None of the provided seed nodes are in the graph. Returning empty sets.")
        return set(), []
    
    # `active_nodes_in_current_step`: nodes activated in the current iteration/step
    # `newly_activated_nodes`: nodes activated in the previous iteration, now trying to activate their neighbors
    # `final_activated_nodes`: cumulative set of all activated nodes
    
    final_activated_nodes = set(valid_seed_nodes)
    newly_activated_nodes = set(valid_seed_nodes) # Start with seeds as newly activated
    
    propagation_events = [] # To store (source, target, step)
    
    current_iteration = 0
    
    while newly_activated_nodes and current_iteration < max_iterations:
        current_iteration += 1
        
        # Nodes that get activated in *this* specific step, from `newly_activated_nodes` of *last* step
        activated_in_this_step = set() 
        
        for source_node in newly_activated_nodes:
            # For each neighbor of the source_node
            # In a DiGraph, graph.neighbors(u) or graph.successors(u) gives outgoing neighbors
            for target_node in graph.successors(source_node):
                # Check if the target is not already active
                if target_node not in final_activated_nodes:
                    # Attempt to activate the target_node
                    if random.random() < propagation_probability:
                        activated_in_this_step.add(target_node)
                        if record_propagation_tree:
                            propagation_events.append((str(source_node), str(target_node), current_iteration))
        
        # Update sets for the next iteration
        newly_activated_nodes = activated_in_this_step # These will attempt to activate in the next step
        final_activated_nodes.update(newly_activated_nodes)
        
    if current_iteration >= max_iterations:
        print(f"Warning: IC simulation reached max_iterations ({max_iterations}) before completion for seed(s): {list(valid_seed_nodes)[:3]}...")

    return final_activated_nodes, propagation_events

# --- Helper for running multiple simulations (e.g., for averaging) ---
def run_multiple_ic_simulations(graph, seed_node, propagation_probability, num_simulations, max_iterations=1000):
    """
    Runs multiple IC simulations starting from a single seed node and averages the cascade size.
    This is useful for getting a more stable measure of a node's spreading power.

    Args:
        graph (nx.DiGraph): The input directed graph.
        seed_node (any hashable): The single seed node for all simulations.
        propagation_probability (float): The uniform probability (p).
        num_simulations (int): The number of simulations to run.
        max_iterations (int): Max iterations for each simulation.

    Returns:
        float: The average cascade size (number of activated nodes, including the seed)
               over `num_simulations`.
        list: A list of all propagation event lists from each simulation (if needed for detailed ETMs).
    """
    total_activated_count = 0
    all_sim_propagation_events = [] # To store events from all simulations

    if seed_node not in graph:
        # print(f"Warning: Seed node {seed_node} not in graph for multiple simulations.")
        return 0.0, []

    for _ in range(num_simulations):
        activated_nodes, prop_events = run_ic_simulation(
            graph,
            seed_nodes=[seed_node], # Seed with a single node
            propagation_probability=propagation_probability,
            max_iterations=max_iterations,
            record_propagation_tree=True # Record events for potential ETM calculation
        )
        total_activated_count += len(activated_nodes)
        all_sim_propagation_events.append(prop_events)
    
    average_cascade_size = total_activated_count / num_simulations if num_simulations > 0 else 0.0
    return average_cascade_size, all_sim_propagation_events


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing ic_model.py...")

    # Create a sample directed graph for testing
    DG_test = nx.DiGraph()
    DG_test.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'),
        ('C', 'E'), ('C', 'F'), ('D', 'G'), ('E', 'G'), ('F', 'G'),
        ('G', 'H'), ('X', 'Y') # X, Y are disconnected
    ])
    print(f"Test DiGraph: Nodes: {DG_test.nodes()}, Edges: {DG_test.edges()}")

    # Test single IC simulation
    seed_nodes_test = ['A']
    prop_prob_test = 0.5
    print(f"\nRunning single IC simulation with seed(s) {seed_nodes_test}, p={prop_prob_test}")
    activated_set, events = run_ic_simulation(DG_test, seed_nodes_test, prop_prob_test, record_propagation_tree=True)
    print(f"Activated nodes: {activated_set}")
    print(f"Cascade size: {len(activated_set)}")
    print(f"Propagation events (source, target, step): {events[:5]} (up to 5)")

    seed_nodes_test_2 = ['X', 'A'] # Test with multiple seeds
    print(f"\nRunning single IC simulation with seed(s) {seed_nodes_test_2}, p={prop_prob_test}")
    activated_set_2, events_2 = run_ic_simulation(DG_test, seed_nodes_test_2, prop_prob_test, record_propagation_tree=True)
    print(f"Activated nodes: {activated_set_2}")
    print(f"Cascade size: {len(activated_set_2)}")


    # Test multiple IC simulations for averaging
    seed_node_for_avg = 'A'
    num_sims = 100
    print(f"\nRunning {num_sims} IC simulations for seed '{seed_node_for_avg}', p={prop_prob_test} to get average cascade size.")
    avg_size, all_events_lists = run_multiple_ic_simulations(DG_test, seed_node_for_avg, prop_prob_test, num_sims)
    print(f"Average cascade size for seed '{seed_node_for_avg}' over {num_sims} simulations: {avg_size:.2f}")
    # print(f"Number of event lists collected: {len(all_events_lists)}")

    seed_node_for_avg_isolated = 'X' # Test with a node that has limited spread
    avg_size_isolated, _ = run_multiple_ic_simulations(DG_test, seed_node_for_avg_isolated, prop_prob_test, num_sims)
    print(f"Average cascade size for seed '{seed_node_for_avg_isolated}' over {num_sims} simulations: {avg_size_isolated:.2f}")


    print("\n--- IC Model Test Complete ---")
