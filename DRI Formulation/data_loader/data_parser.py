#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:39:08 2025

@author: sanup
"""


# diffusion_readiness_project/data_loader/data_parser.py
# Python 3.9

"""
Unified data parser for all datasets in the study.
- Handles parsing of CasFlow cascade data and inferring its graph.
- Handles parsing of SNAP edgelist datasets (Twitter, Enron, Digg, DBLP).
- Uses snap_downloader utility to fetch SNAP datasets if not present.
"""

import os
import networkx as nx
from collections import defaultdict

# Import the downloader utility from the same package
from . import snap_downloader

# --- Specific Parser Functions ---


def _parse_casflow_data(file_path):
    """
    Internal function to parse the CasFlow dataset file to extract cascades and
    build an interaction graph.

    Args:
        file_path (str): The path to the CasFlow dataset.txt file.

    Returns:
        tuple: (graph, all_cascades_events)
            - graph (nx.DiGraph): A NetworkX directed graph representing user interactions.
            - all_cascades_events (list): A list of dictionaries, each representing
                                          a single cascade's propagation events.
    """
    print(f"Parsing CasFlow data from: {file_path}")
    all_cascades_events = []
    interaction_edges = set()

    try:
        with open(file_path, "r") as f:
            for line_number, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                cascade_id, source_node, _, _ = parts[0], parts[1], parts[2], parts[3]
                infection_sequence_str = parts[4:]

                for infection_event_str in infection_sequence_str:
                    try:
                        node_part, time_part = infection_event_str.rsplit(":", 1)
                        parent_node, infected_node = source_node, str(node_part)
                        if "/" in node_part:
                            parent_str, infected_str = node_part.split("/", 1)
                            parent_node, infected_node = str(parent_str), str(infected_str)

                        if parent_node != infected_node:
                            event_data = {
                                "cascade_id": cascade_id,
                                "parent": parent_node,
                                "target": infected_node,
                                "infection_time_rel": float(time_part),
                            }
                            all_cascades_events.append(event_data)
                            interaction_edges.add((parent_node, infected_node))
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"Error: CasFlow file not found at {file_path}")
        return nx.DiGraph(), []

    interaction_graph = nx.DiGraph()
    interaction_graph.add_edges_from(list(interaction_edges))
    print(
        f"CasFlow graph inferred: {interaction_graph.number_of_nodes()} nodes, {interaction_graph.number_of_edges()} edges."
    )
    return interaction_graph, all_cascades_events


def _parse_edgelist_data(file_path, is_directed=True, comments_char="#", delimiter=None):
    """
    Internal function to parse a generic edgelist file into a NetworkX graph.

    Args:
        file_path (str): The path to the edgelist file.
        is_directed (bool): Whether to create a directed or undirected graph.
        comments_char (str): Character for comment lines.
        delimiter (str, optional): Delimiter between nodes. Defaults to whitespace.

    Returns:
        nx.Graph or nx.DiGraph: The loaded graph.
    """
    print(f"Parsing edgelist data from: {file_path} (Directed: {is_directed})")
    graph_type = nx.DiGraph if is_directed else nx.Graph

    try:
        # read_edgelist is highly efficient for this format
        graph = nx.read_edgelist(
            file_path,
            comments=comments_char,
            delimiter=delimiter,
            create_using=graph_type,
            nodetype=str,  # Ensure all node IDs are treated as strings
        )

        # For DBLP (undirected), we create a directed version with reciprocal edges
        # to model influence flowing both ways in a collaboration.
        if not is_directed:
            print(
                f"Converting undirected graph {os.path.basename(file_path)} to directed with reciprocal edges."
            )
            di_graph = nx.DiGraph()
            di_graph.add_edges_from(graph.edges())
            di_graph.add_edges_from([(v, u) for u, v in graph.edges()])
            graph = di_graph

    except FileNotFoundError:
        print(f"Error: Edgelist file not found at {file_path}")
        return graph_type()
    except Exception as e:
        print(f"An error occurred parsing edgelist {file_path}: {e}")
        return graph_type()

    print(f"Edgelist graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    return graph


# --- Main Unified Loader Function ---


def load_graph_and_cascades(dataset_name, config):
    """
    Main data loading router. Selects the correct parser based on dataset name.
    Downloads SNAP data if necessary.

    Args:
        dataset_name (str): The name of the dataset to load.
                            e.g., "casflow", "twitter", "enron", "digg", "dblp".
        config (dict): The main experiment configuration dictionary.

    Returns:
        tuple: (graph, cascade_data, graph_name_for_results)
            - graph (nx.DiGraph): The loaded static network.
            - cascade_data (any): The ground-truth cascade data. For edgelist datasets,
                                  this will be None, as cascades must be simulated.
                                  For CasFlow, it will be the list of parsed events.
            - graph_name_for_results (str): A clean name for saving results.
    """
    data_dir = config.get("data_dir", "datasets")  # Directory to store datasets

    if dataset_name == "casflow":
        file_path = config.get("casflow_data_path", "dataset.txt")
        graph, cascades = _parse_casflow_data(file_path)
        return graph, cascades, "CasFlow_Meme"

    elif dataset_name == "twitter":
        file_path = os.path.join(
            config.get("twitter_data_dir", "."), config.get("twitter_data_filename", "78813.txt")
        )
        graph = _parse_edgelist_data(file_path, is_directed=True)
        return graph, None, "SNAP_Twitter_78813"

    elif dataset_name == "enron":
        path = snap_downloader.download_and_extract_snap_dataset("email-Enron", data_dir=data_dir)
        if path is None:
            return nx.DiGraph(), None, "SNAP_Enron_Fail"
        graph = _parse_edgelist_data(path, is_directed=True)
        return graph, None, "SNAP_Enron"

    elif dataset_name == "digg":
        path = snap_downloader.download_and_extract_snap_dataset("soc-Digg-friends", data_dir=data_dir)
        if path is None:
            return nx.DiGraph(), None, "SNAP_Digg_Fail"
        graph = _parse_edgelist_data(path, is_directed=True)
        return graph, None, "SNAP_Digg"

    elif dataset_name == "dblp":
        path = snap_downloader.download_and_extract_snap_dataset("com-dblp", data_dir=data_dir)
        if path is None:
            return nx.DiGraph(), None, "SNAP_DBLP_Fail"
        # DBLP is an undirected co-authorship graph. We model influence as reciprocal.
        graph = _parse_edgelist_data(path, is_directed=False)
        return graph, None, "SNAP_DBLP"

    else:
        raise ValueError(f"Unknown dataset_name: '{dataset_name}'")


if __name__ == "__main__":
    print("Testing unified data_parser.py...")
    # This test requires the downloader and the actual data files.
    # It's better to test this module via the main.py script.

    # Example standalone test for edgelist parser
    # Create a dummy edgelist file for testing
    dummy_dir = "temp_parser_test"
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_edgelist_path = os.path.join(dummy_dir, "test_edgelist.txt")
    with open(dummy_edgelist_path, "w") as f:
        f.write("# This is a comment\n")
        f.write("1 2\n")
        f.write("1 3\n")
        f.write("2 3\n")

    print("\n--- Testing edgelist parser on dummy file ---")
    test_graph = _parse_edgelist_data(dummy_edgelist_path, is_directed=True)
    print(f"Dummy graph nodes: {test_graph.nodes()}")
    print(f"Dummy graph edges: {test_graph.edges()}")

    # To test the full loader, you would need a sample config dict
    sample_config = {
        "data_dir": dummy_dir,
        "twitter_data_dir": dummy_dir,  # Point to dummy dir
        "twitter_data_filename": "test_edgelist.txt",
    }

    print("\n--- Testing main loader function ---")
    g, c, name = load_graph_and_cascades("twitter", sample_config)
    print(f"Loaded graph for '{name}' with {g.number_of_nodes()} nodes.")
    print(f"Cascade data is None (as expected for edgelist): {c is None}")

    # Clean up
    import shutil

    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
        print(f"\nCleaned up temporary directory: '{dummy_dir}'")
