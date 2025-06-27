# diffusion_readiness_project/data_loader/snap_twitter_parser.py
# Python 3.9

"""
Parses the SNAP Twitter dataset (e.g., 78813.txt).
Responsibilities:
1. Read the edgelist file line by line.
2. Each line typically represents a directed edge (e.g., "follower followed" or "source target").
3. Construct a NetworkX DiGraph object from these edges.
"""

import networkx as nx

# --- Configuration ---
# Path to the SNAP Twitter dataset file (can be passed as an argument or configured elsewhere)
# SNAP_TWITTER_DATA_PATH = "path/to/your/78813.txt" # Example

# --- Main Parsing Function ---

def parse_snap_twitter_edgelist(file_path, comments_char='#', delimiter=None):
    """
    Parses a SNAP-style edgelist file (like the Twitter dataset 78813.txt)
    and creates a directed graph.

    The SNAP Twitter dataset (e.g., social circles from ego networks, 78813.txt)
    is typically an edgelist where each line "u v" means user u follows user v (a directed edge u -> v).
    Or, if it's an undirected dataset, it can be loaded as such.
    For diffusion, a directed graph is usually more appropriate.

    Args:
        file_path (str): The path to the edgelist file.
        comments_char (str): Character indicating comment lines to ignore (e.g., '#').
        delimiter (str, optional): Delimiter for splitting nodes on a line.
                                   Defaults to None (whitespace).

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the social network.
                    Returns an empty DiGraph if the file is not found or an error occurs.
    """
    print(f"Starting parsing of SNAP Twitter edgelist from: {file_path}")
    graph = nx.DiGraph()
    edges_added = 0
    lines_processed = 0
    
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f):
                lines_processed += 1
                line = line.strip()

                if not line or (comments_char and line.startswith(comments_char)):
                    continue  # Skip empty lines or comments

                try:
                    if delimiter:
                        parts = line.split(delimiter)
                    else: # Default: split by any whitespace
                        parts = line.split()
                    
                    if len(parts) >= 2:
                        u_node = str(parts[0]) # Ensure node IDs are strings
                        v_node = str(parts[1])
                        graph.add_edge(u_node, v_node)
                        edges_added += 1
                    else:
                        print(f"Warning: Skipping malformed line {line_number + 1} (not enough parts): {line}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_number + 1}: '{line}'. Error: {e}")
                    continue
        
        print(f"Parsed {lines_processed} lines from the file.")
        print(f"Added {edges_added} edges to the graph.")
        print(f"Twitter graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return nx.DiGraph() # Return an empty graph
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        return nx.DiGraph() # Return an empty graph

    return graph

# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    # This is for testing the parser directly.
    # Replace with the actual path to your 78813.txt file or similar edgelist.
    # Ensure the file is accessible.
    test_file_path = '78813.txt'  # Assumes 78813.txt is in the same directory or provide full path

    print(f"Testing SNAP Twitter edgelist parser with file: {test_file_path}")

    # Check if networkx is available
    try:
        import networkx
        print(f"NetworkX version: {networkx.__version__}")
    except ImportError:
        print("Error: networkx library is not installed. Please install it via 'pip install networkx'")
        exit()

    twitter_graph = parse_snap_twitter_edgelist(test_file_path)

    if twitter_graph.number_of_nodes() > 0:
        print("\n--- Parser Test Summary ---")
        print(f"SNAP Twitter Graph: {twitter_graph.number_of_nodes()} nodes, {twitter_graph.number_of_edges()} edges")
        
        # Example: Print some basic info about the graph
        if twitter_graph.number_of_edges() > 0:
            print("\nSample of graph edges (up to 5):")
            sample_edges = list(twitter_graph.edges())[:5]
            for edge in sample_edges:
                print(edge)
        else:
            print("No edges found in the graph. Check if the file was empty or all lines were comments/malformed.")
    else:
        print("Parsing resulted in an empty graph. Check file path and content, and any warnings above.")

    print("\n--- SNAP Twitter Parser Test Complete ---")
