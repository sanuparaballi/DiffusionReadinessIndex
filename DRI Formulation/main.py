#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# diffusion_readiness_project/main.py
# Python 3.9
"""
Created on Mon Jun 16 20:39:08 2025

@author: sanup


Main script to orchestrate the Diffusion Readiness Index (DRI) research experiments.

This script serves as the central entry point for the entire project. It is designed
to be a configurable and automated pipeline that performs the following steps for
one or more datasets:

1.  Loads a JSON configuration file or uses a default configuration.
2.  Sets a random seed to ensure reproducibility of results that involve sampling.
3.  For each specified dataset:
    a. Loads the network graph and any associated ground-truth cascade data using the unified data_parser.
       This step includes automatically downloading SNAP datasets if they are not found locally.
    b. If no ground-truth cascades exist (e.g., for SNAP datasets), it runs IC model simulations
       to generate synthetic cascade data to serve as the ground truth for ETMs. This process is
       monitored with a progress bar.
    c. Calculates the three Effective Transmission Metrics (ETMs) for every node based on the cascade data.
       Each ETM calculation is monitored with a progress bar.
    d. Calculates a full suite of DRIs for every node, including baselines, literature benchmarks,
       and our proposed composite and spectral indices. This process is also monitored.
    e. Performs a comprehensive evaluation by calculating the correlation (Spearman's rho) and
       top-k performance (F1-Score) of each DRI against each ETM.
    f. Saves all raw scores and evaluation metrics to CSV files in a dataset-specific results directory.
    g. Generates and saves a series of plots (correlation heatmaps, performance bar charts) to
       visually summarize the findings.
"""

import os
import time
import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import random
from tqdm import tqdm  # Library for smart progress bars

# --- Import Project Modules ---
# These imports assume the project structure we've designed, where main.py
# is in the root and can access the other packages.
from data_loader import data_parser
from graph_utils import utils as graph_utils
from diffusion_models import ic_model
from etm_calculator import etm_functions
from structural_indices import (
    baselines,
    literature_indices,
    composite_dri,
    mechanism_focused_dri,
    spectral_dri,
)
from evaluation import metrics as eval_metrics
from evaluation import plotter

# --- Default Configuration ---
# This dictionary defines all parameters for the experiment. It can be
# overridden by providing a JSON file via the command line.
DEFAULT_CONFIG = {
    "datasets_to_run": ["casflow", "enron", "twitter", "digg", "dblp"],
    "random_seed": 42,  # For reproducibility of any random sampling (e.g., seed selection)
    "data_dir": "datasets",  # Master directory to store downloaded SNAP datasets
    "results_dir": "results",
    "casflow_data_path": "datasets/dataset.txt",
    "twitter_data_path": "datasets/78813.txt",
    "simulation_params": {
        "num_simulations_per_seed": 100,  # Number of IC simulations to average for each seed node
        "propagation_probabilities": {
            "default": 0.05,
            "twitter": 0.05,
            "enron": 0.01,
            "digg": 0.02,
            "dblp": 0.01,
        },
        "max_iterations_ic": 100,
        "seed_sampling_fraction": None,  # Set to a float (e.g., 0.1 for 10%) for faster testing, None for all nodes
    },
    "etm_params": {"hnr_k_hop": 1},
    "dri_selection": {
        "baselines": True,
        "literature": True,
        "composite_wmfs": True,
        "mechanism_bpi": True,
        "spectral": True,
    },
    "composite_wmfs_configs": [  # Define different combinations for the WMFS-DRI
        {
            "name": "WMFS_Simple",
            "features": [
                {"module": "baselines", "func": "get_degree_centrality", "weight": 0.5},
                {"module": "baselines", "func": "get_k_core_centrality", "weight": 0.5},
            ],
        },
        {
            "name": "WMFS_Advanced",
            "features": [
                {"module": "baselines", "func": "get_degree_centrality", "weight": 0.3},
                {
                    "module": "literature_indices",
                    "func": "get_structural_diversity_ego",
                    "weight": 0.4,
                    "params": {"radius": 1},
                },
                {"module": "baselines", "func": "get_eigenvector_centrality", "weight": 0.3},
            ],
        },
    ],
    "spectral_params": {"k_hop_local": 1, "ppr_k_sum_hops": 1, "ppr_alpha": 0.85, "heat_kernel_t": 0.1},
    "evaluation_params": {
        "top_k_values_perc": [0.01, 0.05, 0.10]  # Top-k percentages for F1-score calculation
    },
    "run_expensive_metrics": {  # Toggle for computationally intensive DRIs
        "betweenness": False,
        "gravity": False,
        "heat_kernel": False,
    },
    "save_plots": True,
    "save_results_csv": True,
}


def load_config(config_path=None):
    """Loads configuration from a JSON file, or returns the default."""
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)
    print("No config file provided. Using default configuration.")
    return DEFAULT_CONFIG


def safe_name(name_str):
    """Creates a filesystem-safe name from a given string."""
    return (
        name_str.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace("*", "")
        .replace(".", "p")
        .replace("%", "perc")
    )


def run_experiment(dataset_name, config):
    """
    Main function to run the full experimental pipeline for a single dataset.
    """
    start_time_exp = time.time()
    print(f"\n{'='*20} Starting Experiment for: {dataset_name.upper()} {'='*20}")

    results_subdir = os.path.join(config["results_dir"], dataset_name)
    os.makedirs(results_subdir, exist_ok=True)

    # --- 1. Load Data ---
    print("\n[1/5] Loading Data...")
    graph, cascade_data, graph_name_for_results = data_parser.load_graph_and_cascades(dataset_name, config)
    if graph.number_of_nodes() == 0:
        print(f"Error: Could not load graph for {dataset_name}. Aborting experiment for this dataset.")
        return

    # --- 2. Generate ETM Ground Truth ---
    print("\n[2/5] Generating ETM Ground Truth...")
    if cascade_data is None:  # For SNAP datasets, we need to simulate
        sim_params = config["simulation_params"]
        p = sim_params["propagation_probabilities"].get(
            dataset_name, sim_params["propagation_probabilities"]["default"]
        )
        print(f"No ground-truth cascades found. Running IC simulations with p={p}...")

        nodes_to_seed = list(graph.nodes())
        if sim_params["seed_sampling_fraction"] and sim_params["seed_sampling_fraction"] < 1.0:
            sample_k = int(len(nodes_to_seed) * sim_params["seed_sampling_fraction"])
            nodes_to_seed = random.sample(nodes_to_seed, k=max(1, sample_k))
            print(
                f"Using a random sample of {len(nodes_to_seed)} seed nodes ({sim_params['seed_sampling_fraction']*100}%)."
            )

        sim_cascades_by_seed = {}
        # TQDM provides a progress bar for this long-running loop
        for seed_node in tqdm(nodes_to_seed, desc="  Running IC Simulations", unit="seed"):
            _, sim_event_lists = ic_model.run_multiple_ic_simulations(
                graph, seed_node, p, sim_params["num_simulations_per_seed"], sim_params["max_iterations_ic"]
            )
            sim_cascades_by_seed[str(seed_node)] = sim_event_lists
        cascade_data = sim_cascades_by_seed

    node_centric_cascades = etm_functions.preprocess_cascade_data_for_node_etms(
        cascade_data, list(graph.nodes())
    )

    print("Calculating ETMs (ADT, MCSS, HNR_k)...")
    etm_scores = {}
    # TQDM provides progress bars for each ETM calculation
    etm_scores["ADT"] = {
        str(n): etm_functions.calculate_adt(node_centric_cascades.get(str(n), []), str(n))
        for n in tqdm(graph.nodes(), desc="    Calculating ADT ")
    }
    etm_scores["MCSS"] = {
        str(n): etm_functions.calculate_mcss(node_centric_cascades.get(str(n), []), str(n))
        for n in tqdm(graph.nodes(), desc="    Calculating MCSS")
    }
    etm_scores["HNR_k"] = {
        str(n): etm_functions.calculate_hnr_k(
            node_centric_cascades.get(str(n), []), str(n), graph, config["etm_params"]["hnr_k_hop"]
        )
        for n in tqdm(graph.nodes(), desc="    Calculating HNR_k")
    }

    # --- 3. Calculate DRIs ---
    print("\n[3/5] Calculating Diffusion Readiness Indices (DRIs)...")
    dri_results = {}

    if config["dri_selection"]["spectral"]:
        print("  Precomputing global Fiedler info...")
        spectral_dri.precompute_global_fiedler_info(graph)

    dri_modules = {
        "baselines": baselines,
        "literature_indices": literature_indices,
        "composite_dri": composite_dri,
        "mechanism_focused_dri": mechanism_focused_dri,
        "spectral_dri": spectral_dri,
    }

    DRI_FUNCTION_MAP = {
        "Degree (Out)": (baselines.get_degree_centrality, {}),
        "K-Core": (baselines.get_k_core_centrality, {}),
        "Eigenvector": (baselines.get_eigenvector_centrality, {}),
        "Closeness": (baselines.get_closeness_centrality, {}),
        "CI (l=2)": (literature_indices.get_collective_influence, {"l_dist": 2}),
        "SD_ego (r=1)": (literature_indices.get_structural_diversity_ego, {"radius": 1}),
        "FwLTR_struct": (literature_indices.get_fwltr_structural, {"uniform_threshold_fraction": 0.5}),
        "BPI-DRI": (mechanism_focused_dri.get_bpi_dri, {"normalize_components": True}),
        "LocFiedler (k=1)": (
            spectral_dri.get_localized_fiedler_value,
            {"k_hop": config["spectral_params"]["k_hop_local"]},
        ),
        "GlobFiedlerComp": (spectral_dri.get_node_global_fiedler_component, {}),
        "LocSpecRadius (k=1)": (
            spectral_dri.get_localized_spectral_radius,
            {"k_hop": config["spectral_params"]["k_hop_local"]},
        ),
        "PPR-API": (
            spectral_dri.get_ppr_api_dri,
            {
                "k_sum_hops": config["spectral_params"]["ppr_k_sum_hops"],
                "ppr_alpha": config["spectral_params"]["ppr_alpha"],
            },
        ),
    }

    if config["run_expensive_metrics"]["betweenness"]:
        DRI_FUNCTION_MAP["Betweenness"] = (
            baselines.get_betweenness_centrality,
            {"k_samples": 2000 if graph.number_of_nodes() > 10000 else None},
        )
    if config["run_expensive_metrics"]["gravity"]:
        DRI_FUNCTION_MAP["Gravity (r=3)"] = (literature_indices.get_gravity_centrality, {"r_radius": 3})
    if config["run_expensive_metrics"]["heat_kernel"]:
        DRI_FUNCTION_MAP["HeatKernel-API"] = (
            spectral_dri.get_heat_kernel_api_dri,
            {"t_short": config["spectral_params"]["heat_kernel_t"]},
        )

    # TQDM progress bar for main DRI calculation loop
    for dri_name, (func, params) in tqdm(DRI_FUNCTION_MAP.items(), desc="  Calculating all DRIs"):
        if graph.is_directed() or dri_name not in ["FwLTR_struct", "PPR-API"]:
            dri_results[dri_name] = func(graph, per_node=True, **params)

    if config["dri_selection"]["composite_wmfs"]:
        for wmfs_conf in config["composite_wmfs_configs"]:
            name = wmfs_conf["name"]
            print(f"  Calculating Composite DRI: {name}...")
            parsed_feature_configs = []
            for feat_conf in wmfs_conf["features"]:
                module = dri_modules[feat_conf["module"]]
                parsed_feature_configs.append(
                    {
                        "name": feat_conf["func"],
                        "func": getattr(module, feat_conf["func"]),
                        "weight": feat_conf["weight"],
                        "params": feat_conf.get("params", {}),
                    }
                )
            dri_results[name] = composite_dri.get_wmfs_dri(graph, parsed_feature_configs, per_node=True)

    # --- 4. Perform Evaluations ---
    print("\n[4/5] Performing Evaluations...")
    evaluation_summary = []

    for etm_name in etm_scores.keys():
        for dri_name in dri_results.keys():
            current_etm_scores = etm_scores[etm_name]
            current_dri_scores = dri_results[dri_name]

            common_nodes = list(set(current_etm_scores.keys()).intersection(set(current_dri_scores.keys())))
            common_nodes = [
                n
                for n in common_nodes
                if current_etm_scores.get(n) is not None and not np.isnan(current_etm_scores.get(n))
            ]
            if not common_nodes:
                continue

            ordered_etm = [current_etm_scores[n] for n in common_nodes]
            ordered_dri = [current_dri_scores.get(n, 0.0) for n in common_nodes]

            corr, p_val = eval_metrics.calculate_spearman_rank_correlation(ordered_dri, ordered_etm)
            row = {"ETM": etm_name, "DRI": dri_name, "Spearman_Correlation": corr, "P_value": p_val}

            for k_perc in config["evaluation_params"]["top_k_values_perc"]:
                true_top_k = eval_metrics.get_top_k_nodes(current_etm_scores, k_perc, is_percentage=True)
                pred_top_k = eval_metrics.get_top_k_nodes(current_dri_scores, k_perc, is_percentage=True)
                if not true_top_k:
                    continue
                prf_scores = eval_metrics.calculate_precision_recall_f1_at_k(true_top_k, pred_top_k)
                row[f"F1@{k_perc*100:.0f}%"] = prf_scores["f1_at_k"]
            evaluation_summary.append(row)

    results_df = pd.DataFrame(evaluation_summary)

    # --- 5. Save Results and Plots ---
    print("\n[5/5] Saving Results and Generating Plots...")
    if config["save_results_csv"]:
        results_df.to_csv(
            os.path.join(results_subdir, f"{graph_name_for_results}_evaluation_summary.csv"), index=False
        )
        print("  Evaluation summary saved.")

    if config["save_plots"]:
        plot_dir = os.path.join(results_subdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        try:
            # Correlation Heatmap
            corr_pivot_df = results_df.pivot(index="DRI", columns="ETM", values="Spearman_Correlation")
            plotter.plot_correlation_heatmap(
                corr_pivot_df,
                title=f"Spearman Correlation ({graph_name_for_results})",
                save_path=os.path.join(plot_dir, "correlation_heatmap.png"),
            )

            # Top-k F1 Bars for each ETM
            for etm_name in etm_scores.keys():
                f1_cols = [col for col in results_df.columns if "F1@" in col]
                if f1_cols:
                    f1_df = results_df[results_df["ETM"] == etm_name].set_index("DRI")[f1_cols]
                    f1_df.columns = [col.split("@")[1] for col in f1_cols]
                    if not f1_df.empty:
                        plotter.plot_top_k_performance_bars(
                            f1_df,
                            metric_name="F1-Score",
                            title_suffix=f"for {etm_name}",
                            save_path=os.path.join(plot_dir, f"top_k_f1_bars_{etm_name}.png"),
                        )
            print("  Plots generated.")
        except Exception as e:
            print(f"  An error occurred during plotting: {e}")

    exp_duration = time.time() - start_time_exp
    print(
        f"\n{'='*20} Experiment for {dataset_name.upper()} finished in {exp_duration:.2f} seconds. {'='*20}"
    )


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Diffusion Readiness Index Experiments.")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file.")
    parser.add_argument(
        "--datasets", nargs="+", help="List of datasets to run, e.g., --datasets enron twitter"
    )
    args = parser.parse_args()

    main_config = load_config(args.config)

    # Set the random seed for reproducibility of sampling and other random processes
    if "random_seed" in main_config and main_config["random_seed"] is not None:
        seed = main_config["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        print(f"Global random seed set to: {seed}")

    datasets_to_run = args.datasets if args.datasets else main_config["datasets_to_run"]

    print("\nStarting full experimental run...")
    for dataset in datasets_to_run:
        if dataset not in ["casflow", "twitter", "enron", "digg", "dblp"]:
            print(f"Warning: Dataset '{dataset}' is not a recognized option. Skipping.")
            continue
        run_experiment(dataset, main_config)

    print("\nAll specified experiments complete.")
