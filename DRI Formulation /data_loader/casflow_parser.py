# diffusion_readiness_project/data_loader/casflow_parser.py
# Python 3.9

"""
Parses the CasFlow meme dataset (dataset.txt).
Responsibilities:
1. Read cascade data line by line.
2. For each cascade:
    - Extract source node, start timestamp, and overall size.
    - Parse the sequence of infections, identifying parent-child relationships and infection timestamps/delays.
3. Reconstruct individual cascade structures (e.g., as a list of infection events or edges).
4. From all observed cascades, infer a static, directed user-to-user interaction graph.
   - An edge (A, B) exists if user A infected user B at least once.
   - Edge weights could represent frequency (optional, default to unweighted for structural TRIs).
"""

import networkx as nx
import pandas as pd # Optional: for easier handling of intermediate data structures

# --- Configuration ---
# Path to the CasFlow dataset file (can be passed as an argument or configured elsewhere)
# CASFLOW_DATA_PATH = "path/to/your/dataset.txt" # Example

# --- Main Parsing Function ---

def parse_casflow_data(file_path):
    """
    Parses the CasFlow dataset file to extract cascades and build an interaction graph.

    Args:
        file_path (str): The path to the CasFlow dataset.txt file.

    Returns:
        tuple: (all_cascades, interaction_graph)
            - all_cascades (list): A list of dictionaries, where each dictionary
                                   represents a single cascade's propagation events.
                                   Each event could be:
                                   {'cascade_id': str, 'source': str, 'target': str,
                                    'parent': str, 'infection_time_rel': float,
                                    'infection_time_abs': float (if calculable/needed)}
            - interaction_graph (nx.DiGraph): A NetworkX directed graph representing
                                              user interactions (A -> B if A infected B).
                                              Nodes are user IDs. Edges are unweighted by default.
    """
    print(f"Starting parsing of CasFlow data from: {file_path}")
    all_cascades_events = []  # To store individual infection events from all cascades
    interaction_edges = set() # To store unique (infector, infected) pairs for the graph

    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 4:
                    print(f"Warning: Skipping malformed line {line_number + 1}: {line} (not enough main parts)")
                    continue

                cascade_id_str, source_node_str, start_timestamp_str, num_infected_str = parts[0], parts[1], parts[2], parts[3]
                infection_sequence_str = parts[4:] # The rest are infection path segments

                # Basic type conversions and error handling
                try:
                    # cascade_id = int(cascade_id_str) # Or keep as string if preferred
                    cascade_id = cascade_id_str
                    source_node = str(source_node_str)
                    # start_timestamp = int(start_timestamp_str)
                    # num_infected = int(num_infected_str)
                except ValueError as e:
                    print(f"Warning: Skipping cascade {cascade_id_str} due to type conversion error in header: {e}")
                    continue

                # The first element in infection_sequence_str is the source itself with time 0
                # e.g., "1:0" or "source_node_id:0"
                # We process the actual infections which are subsequent elements
                # Format: "parent_id/infected_id:relative_time" or "infected_id:relative_time" (if parent is the main source)

                # Store cascade metadata if needed separately
                # current_cascade_metadata = {'id': cascade_id, 'source': source_node, 'start_time': start_timestamp, 'size': num_infected}

                # Iterate through the infection path
                for infection_event_str in infection_sequence_str:
                    try:
                        node_part, time_part = infection_event_str.rsplit(':', 1)
                        relative_infection_time = float(time_part)

                        parent_node = None
                        infected_node = None

                        if '/' in node_part: # Format: "parent/infected"
                            parent_str, infected_str = node_part.split('/', 1)
                            parent_node = str(parent_str)
                            infected_node = str(infected_str)
                        else: # Format: "infected" (parent is the main source of the cascade)
                            parent_node = source_node # Assume main source is the parent
                            infected_node = str(node_part)
                        
                        # Skip self-loops if any or if the "infected" is the source with time 0
                        if parent_node == infected_node: # Or (infected_node == source_node and relative_infection_time == 0)
                            if relative_infection_time == 0 and infected_node == source_node:
                                # This is the source node's own entry, not an "infection" from a parent
                                pass # Or handle as the start of the cascade
                            else:
                                # print(f"Note: Self-loop or redundant source entry skipped for {infected_node} in cascade {cascade_id}")
                                continue


                        # Add to cascade events
                        event_data = {
                            'cascade_id': cascade_id,
                            'parent': parent_node,
                            'target': infected_node,
                            'infection_time_rel_to_parent': relative_infection_time, # This interpretation might need refinement based on CasFlow exact definition
                            # 'infection_time_abs': start_timestamp + relative_infection_time # if relative_time is from cascade start
                        }
                        all_cascades_events.append(event_data)

                        # Add edge to interaction graph (parent -> infected)
                        if parent_node and infected_node and parent_node != infected_node : # Ensure valid edge
                           interaction_edges.add((parent_node, infected_node))

                    except ValueError as e:
                        print(f"Warning: Skipping malformed infection event '{infection_event_str}' in cascade {cascade_id}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Unexpected error parsing infection event '{infection_event_str}' in cascade {cascade_id}: {e}")
                        continue
        
        print(f"Parsed {line_number + 1} lines from the file.")
        print(f"Total individual infection events recorded: {len(all_cascades_events)}")
        print(f"Total unique interaction edges identified: {len(interaction_edges)}")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return [], nx.DiGraph()
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return [], nx.DiGraph()

    # Create the interaction graph
    interaction_graph = nx.DiGraph()
    interaction_graph.add_edges_from(list(interaction_edges))
    
    print(f"Interaction graph created with {interaction_graph.number_of_nodes()} nodes and {interaction_graph.number_of_edges()} edges.")

    # Post-processing: all_cascades_events might need further structuring
    # For ETMs, we might want to group events by cascade_id or by originating source node.
    # The current `all_cascades_events` is a flat list.
    # For now, this provides the raw infection links and the overall graph.
    # We will refine how `all_cascades` is structured based on ETM calculation needs.
    # For instance, ETMs might need a list of cascades, where each cascade is a list of (infector, infected, time) tuples.

    return all_cascades_events, interaction_graph


# --- Helper Functions (if any) ---
# e.g., function to calculate absolute infection times if needed,
# or to group events by cascade.


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    # This is for testing the parser directly.
    # Replace with the actual path to your dataset.txt file.
    # Make sure 'dataset.txt' is accessible from where you run this.
    test_file_path = 'dataset.txt' # Assumes dataset.txt is in the same directory or provide full path

    print(f"Testing CasFlow parser with file: {test_file_path}")
    
    # Check if networkx is available
    try:
        import networkx
        print(f"NetworkX version: {networkx.__version__}")
    except ImportError:
        print("Error: networkx library is not installed. Please install it via 'pip install networkx'")
        exit()

    cascades, graph = parse_casflow_data(test_file_path)

    if graph.number_of_nodes() > 0:
        print("\n--- Parser Test Summary ---")
        print(f"Number of cascades/events processed: {len(cascades)}")
        print(f"Interaction Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Example: Print some basic info about the graph
        if graph.number_of_edges() > 0:
            print("\nSample of graph edges (up to 5):")
            for i, edge in enumerate(graph.edges()):
                print(edge)
                if i >= 4:
                    break
            
            print("\nSample of cascade events (up to 5):")
            for i, event in enumerate(cascades):
                print(event)
                if i >= 4:
                    break
        else:
            print("No edges found in the interaction graph.")
            if not cascades:
                 print("No cascade events were successfully parsed.")
    else:
        print("Parsing resulted in an empty graph or no cascades. Check warnings above.")

    print("\n--- CasFlow Parser Test Complete ---")

