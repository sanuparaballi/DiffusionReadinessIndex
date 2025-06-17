#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:39:08 2025

@author: sanup


Workflow:
1. Load configuration/parameters (e.g., dataset paths, DRI selection, simulation params).
2. Load the specified dataset (CasFlow or SNAP Twitter).
3. If SNAP Twitter:
    a. Run IC model simulations to generate cascade data.
    b. Preprocess simulated cascade data for ETM calculation.
4. If CasFlow:
    a. CasFlow parser provides cascade events and an inferred interaction graph.
    b. Preprocess CasFlow events for ETM calculation.
5. Calculate Effective Transmission Metrics (ETMs: ADT, MCSS, HNR_k).
6. Calculate all selected Transmission Readiness Indices (DRIs) for all nodes.
   - Baselines (Degree, K-Core, Eigenvector, Betweenness, Closeness)
   - Literature Indices (CI, Gravity, FwLTR-structural, SD_ego)
   - Composite DRIs (WMFS-DRI with various feature sets)
   - Mechanism-Focused DRIs (BPI-DRI)
   - Spectral DRIs (Localized Fiedler, Global Fiedler Comp, Localized Spectral Radius, PPR-API, HeatKernel-API)
7. Perform evaluations:
   - Spearman rank correlations (DRIs vs. ETMs).
   - Top-k node identification performance (Precision@k, Recall@k, F1@k).
   - Imprecision metric.
8. Generate and save visualizations.
9. Save all quantitative results (scores, correlations, metrics) to files.

Main script to orchestrate the Diffusion Readiness Index (DRI) research experiments.
Handles the full pipeline for multiple datasets.
"""


import os
import time
import argparse
import json
import pandas as pd
import numpy as np
import networkx as nx
import random

# --- Import Project Modules ---
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
DEFAULT_CONFIG = {
    "datasets_to_run": ["casflow", "enron", "twitter", "digg", "dblp"],
    "random_seed": 42,  # For reproducibility of sampling
    "data_dir": "datasets",  # Master directory to store downloaded SNAP datasets
    "results_dir": "results",
    "casflow_data_path": "dataset.txt",
    "twitter_data_path": "78813.txt",
    "simulation_params": {
        "num_simulations_per_seed": 100,
        "propagation_probabilities": {
            "default": 0.05,
            "twitter": 0.05,
            "enron": 0.01,
            "digg": 0.02,
            "dblp": 0.01,
        },
        "max_iterations_ic": 100,
        "seed_sampling_fraction": None,  # e.g., 0.1 for 10% sample, None for all nodes
    },
    "etm_params": {"hnr_k_hop": 1},
    "dri_selection": {
        "baselines": True,
        "literature": True,
        "composite_wmfs": True,
        "mechanism_bpi": True,
        "spectral": True,
    },
    "composite_wmfs_configs": [
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
    "evaluation_params": {"top_k_values_perc": [0.01, 0.05, 0.10]},
    "run_expensive_metrics": {  # Toggle for slow metrics like Betweenness/Gravity
        "betweenness": False,
        "gravity": False,
        "heat_kernel": False,
    },
    "save_plots": True,
    "save_results_csv": True,
}


def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    print("Using default configuration.")
    return DEFAULT_CONFIG


def safe_name(name_str):
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
        total_seeds = len(nodes_to_seed)
        for i, seed_node in enumerate(nodes_to_seed):
            _, sim_event_lists = ic_model.run_multiple_ic_simulations(
                graph, seed_node, p, sim_params["num_simulations_per_seed"], sim_params["max_iterations_ic"]
            )
            sim_cascades_by_seed[str(seed_node)] = sim_event_lists
            if (i + 1) % 100 == 0:
                print(f"  Simulations complete for {i + 1}/{total_seeds} seeds...")
        cascade_data = sim_cascades_by_seed

    node_centric_cascades = etm_functions.preprocess_cascade_data_for_node_etms(
        cascade_data, list(graph.nodes())
    )

    print("Calculating ETMs (ADT, MCSS, HNR_k)...")
    etm_scores = {
        "ADT": {
            n: etm_functions.calculate_adt(node_centric_cascades.get(str(n), []), str(n))
            for n in graph.nodes()
        },
        "MCSS": {
            n: etm_functions.calculate_mcss(node_centric_cascades.get(str(n), []), str(n))
            for n in graph.nodes()
        },
        "HNR_k": {
            n: etm_functions.calculate_hnr_k(
                node_centric_cascades.get(str(n), []), str(n), graph, config["etm_params"]["hnr_k_hop"]
            )
            for n in graph.nodes()
        },
    }

    # --- 3. Calculate DRIs ---
    print("\n[3/5] Calculating Diffusion Readiness Indices (DRIs)...")
    dri_results = {}

    # Precompute global spectral info once if needed
    if config["dri_selection"]["spectral"]:
        print("  Precomputing global Fiedler info...")
        spectral_dri.precompute_global_fiedler_info(graph)

    # Dynamically call DRI functions based on config
    dri_modules = {
        "baselines": baselines,
        "literature_indices": literature_indices,
        "composite_dri": composite_dri,
        "mechanism_focused_dri": mechanism_focused_dri,
        "spectral_dri": spectral_dri,
    }

    # Add more DRI functions to this map as they are implemented
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
    # Add expensive ones conditionally
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

    for dri_name, (func, params) in DRI_FUNCTION_MAP.items():
        if graph.is_directed() or dri_name not in ["FwLTR_struct", "PPR-API"]:
            print(f"  Calculating {dri_name}...")
            dri_results[dri_name] = func(graph, per_node=True, **params)

    # Handle composite WMFS DRIs separately
    if config["dri_selection"]["composite_wmfs"]:
        for wmfs_conf in config["composite_wmfs_configs"]:
            name = wmfs_conf["name"]
            print(f"  Calculating {name}...")
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

    for etm_name, current_etm_scores in etm_scores.items():
        # Align nodes and scores for correlation
        common_nodes = list(
            set(current_etm_scores.keys()).intersection(*[set(s.keys()) for s in dri_results.values()])
        )
        common_nodes = [
            n
            for n in common_nodes
            if current_etm_scores.get(n) is not None and not np.isnan(current_etm_scores.get(n))
        ]
        if not common_nodes:
            continue

        ordered_etm = [current_etm_scores[n] for n in common_nodes]

        for dri_name, current_dri_scores in dri_results.items():
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

    # Set the random seed for reproducibility of sampling
    if "random_seed" in main_config and main_config["random_seed"] is not None:
        random.seed(main_config["random_seed"])
        np.random.seed(main_config["random_seed"])
        print(f"Random seed set to: {main_config['random_seed']}")

    datasets_to_run = args.datasets if args.datasets else main_config["datasets_to_run"]

    for dataset in datasets_to_run:
        if dataset not in ["casflow", "twitter", "enron", "digg", "dblp"]:
            print(f"Warning: Dataset '{dataset}' is not a recognized option. Skipping.")
            continue
        run_experiment(dataset, main_config)

    print("\nAll specified experiments complete.")
