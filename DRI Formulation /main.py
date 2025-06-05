# diffusion_readiness_project/main.py
# Python 3.9

# author - Sanup Araballi

"""
Main script to orchestrate the Diffusion Readiness Index research experiments.
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
"""

import os
import time
import argparse
import json  # For loading configurations
import pandas as pd
import numpy as np
import networkx as nx

# Import from project modules (assuming they are in PYTHONPATH or relative paths work)
from data_loader import casflow_parser, snap_twitter_parser
from graph_utils import utils as graph_utils_main  # Renamed to avoid conflict if utils is a common name
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

# --- Configuration Loading (Example) ---
DEFAULT_CONFIG = {
    "dataset_type": "casflow",  # "casflow" or "snap_twitter"
    "casflow_data_path": "data_loader/dataset.txt",  # Relative to project root or an absolute path
    "snap_twitter_data_path": "data_loader/78813.txt",
    "results_dir": "results",
    "simulation_params": {  # For SNAP Twitter
        "num_simulations_per_seed": 100,  # For averaging ETMs if seeded per node
        "propagation_probability": 0.05,
        "max_iterations_ic": 100,
    },
    "etm_calculation": {"hnr_k_hop": 1},
    "dri_selection": {  # Which DRIs to compute and evaluate
        "baselines": True,
        "literature": True,
        "composite_wmfs": True,
        "mechanism_bpi": True,
        "spectral": False,
    },
    "composite_wmfs_configs": [  # List of WMFS configurations to test
        {
            "name": "WMFS_Simple",
            "features": [
                {
                    "name": "degree_out",
                    "func_module": "baselines",
                    "func_name": "get_degree_centrality",
                    "weight": 0.4,
                    "params": {},
                },
                {
                    "name": "k_core",
                    "func_module": "baselines",
                    "func_name": "get_k_core_centrality",
                    "weight": 0.3,
                    "params": {},
                },
                {
                    "name": "sd_ego_1hop",
                    "func_module": "literature_indices",
                    "func_name": "get_structural_diversity_ego",
                    "weight": 0.3,
                    "params": {"radius": 1},
                },
            ],
        },
        # Add more WMFS configurations here, e.g., including spectral features
        {
            "name": "WMFS_Spectral_Basic",
            "features": [
                {
                    "name": "degree_out",
                    "func_module": "baselines",
                    "func_name": "get_degree_centrality",
                    "weight": 0.3,
                    "params": {},
                },
                {
                    "name": "k_core",
                    "func_module": "baselines",
                    "func_name": "get_k_core_centrality",
                    "weight": 0.2,
                    "params": {},
                },
                {
                    "name": "loc_fiedler_1hop",
                    "func_module": "spectral_dri",
                    "func_name": "get_localized_fiedler_value",
                    "weight": 0.3,
                    "params": {"k_hop": 1},
                },
                {
                    "name": "ppr_api_1hop",
                    "func_module": "spectral_dri",
                    "func_name": "get_ppr_api_dri",
                    "weight": 0.2,
                    "params": {"k_sum_hops": 1, "ppr_alpha": 0.85},
                },
            ],
        },
    ],
    "spectral_params": {
        "k_hop_local_spectral": 1,  # For localized fiedler/radius
        "ppr_k_sum_hops": 1,
        "ppr_alpha": 0.85,
        "heat_kernel_t": 0.1,
        "precompute_global_fiedler": True,  # If get_node_global_fiedler_component is used
    },
    "evaluation_params": {
        "top_k_values_abs": [10, 50, 100],  # Absolute top-k
        "top_k_values_perc": [0.01, 0.05, 0.10],  # Percentage top-k
    },
    "save_plots": True,
    "save_results_csv": True,
}


def load_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG


# --- Main Experiment Function ---
def run_experiment(config):
    """
    Main function to run the full experimental pipeline.
    """
    start_time_exp = time.time()
    print("Starting Diffusion Readiness Experiment...")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Create results directory if it doesn't exist
    os.makedirs(config["results_dir"], exist_ok=True)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    graph = None
    all_cascade_events_for_etm = None  # This will be list of lists of events, or dict for IC
    graph_name = ""

    if config["dataset_type"] == "casflow":
        graph_name = "CasFlow_Meme"
        raw_cascade_events, graph = casflow_parser.parse_casflow_data(config["casflow_data_path"])
        # `raw_cascade_events` is a flat list of dicts. Needs preprocessing for ETMs.
        # `graph` is the inferred interaction graph.
        if graph.number_of_nodes() == 0:
            print("Error: CasFlow graph is empty. Exiting.")
            return
        # Preprocess for ETMs: group events by cascade, then structure for node-centric ETMs
        all_cascade_events_for_etm = etm_functions.preprocess_cascade_data_for_node_etms(
            raw_cascade_events, list(graph.nodes())
        )
        print(
            f"CasFlow: Graph loaded with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
        )
        print(f"CasFlow: Preprocessed {len(raw_cascade_events)} raw events for ETM calculation.")

    elif config["dataset_type"] == "snap_twitter":
        graph_name = "SNAP_Twitter_78813"
        graph = snap_twitter_parser.parse_snap_twitter_edgelist(config["snap_twitter_data_path"])
        if graph.number_of_nodes() == 0:
            print("Error: SNAP Twitter graph is empty. Exiting.")
            return
        print(
            f"SNAP Twitter: Graph loaded with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
        )

        # Run IC model simulations
        print("\nRunning IC Model Simulations for SNAP Twitter data...")
        sim_params = config["simulation_params"]
        simulated_cascades_by_seed = {}  # {seed_node: [list_of_events_sim1, list_of_events_sim2, ...]}

        # Decide on seeds: all nodes, or a sample? For ETMs for all nodes, need to seed from all.
        # This can be very time-consuming.
        # Alternative: sample seeds, or use a fixed set of seeds.
        # For now, let's assume we want ETMs for all nodes, so we simulate from each as a seed.
        # This is for calculating ETMs *of the seed nodes*.

        # For a more general ETM calculation on simulated data (where any node can be a spreader, not just seed):
        # One might run simulations from random seeds, pool all propagation events,
        # and then use preprocess_cascade_data_for_node_etms similar to CasFlow.
        # For now, let's focus ETMs on the performance *as a seed*.

        nodes_to_seed = list(graph.nodes())  # Potentially very large! Consider sampling for testing.
        # nodes_to_seed = random.sample(list(graph.nodes()), k=min(1000, graph.number_of_nodes())) # Example sampling

        count_sim_done = 0
        for seed_node in nodes_to_seed:
            _, sim_event_lists_for_seed = ic_model.run_multiple_ic_simulations(
                graph,
                seed_node,
                sim_params["propagation_probability"],
                sim_params["num_simulations_per_seed"],
                sim_params["max_iterations_ic"],
            )
            simulated_cascades_by_seed[str(seed_node)] = sim_event_lists_for_seed  # Ensure string keys
            count_sim_done += 1
            if count_sim_done % 100 == 0:
                print(f"  Simulations done for {count_sim_done}/{len(nodes_to_seed)} seeds.")

        all_cascade_events_for_etm = (
            simulated_cascades_by_seed  # This is now {seed_node: list_of_cascade_event_lists}
        )
        print(f"IC Simulations: Completed for {len(nodes_to_seed)} seed nodes.")
    else:
        raise ValueError(f"Unsupported dataset_type: {config['dataset_type']}")

    all_graph_nodes = list(graph.nodes())  # Consistent list of nodes for iteration

    # --- 2. Calculate ETMs ---
    print("\n--- 2. Calculating ETMs ---")
    etm_scores = {}  # {'ADT': {node: score}, 'MCSS': ..., 'HNR_k': ...}

    # The `all_cascade_events_for_etm` should be structured by `preprocess_cascade_data_for_node_etms`
    # into a dict: {node_id: [list_of_cascades_where_node_spreads]}
    # For IC sims, if `all_cascade_events_for_etm` is {seed: [sim_events1, sim_events2,...]},
    # then ETMs are calculated for these seed nodes based on cascades they initiated.

    # If `all_cascade_events_for_etm` is already in the node-centric format from preprocessing:
    node_centric_cascades = all_cascade_events_for_etm

    etm_scores["ADT"] = {
        node: etm_functions.calculate_adt(node_centric_cascades.get(str(node), []), str(node))
        for node in all_graph_nodes
    }
    etm_scores["MCSS"] = {
        node: etm_functions.calculate_mcss(node_centric_cascades.get(str(node), []), str(node))
        for node in all_graph_nodes
    }
    etm_scores["HNR_k"] = {
        node: etm_functions.calculate_hnr_k(
            node_centric_cascades.get(str(node), []),
            str(node),
            graph,
            config["etm_calculation"]["hnr_k_hop"],
        )
        for node in all_graph_nodes
    }

    print("ETMs calculated.")
    if config["save_results_csv"]:
        for etm_name, scores_dict in etm_scores.items():
            pd.DataFrame(list(scores_dict.items()), columns=["node", etm_name]).to_csv(
                os.path.join(config["results_dir"], f"{graph_name}_{etm_name}_scores.csv"), index=False
            )

    # --- 3. Calculate DRIs ---
    print("\n--- 3. Calculating DRIs ---")
    dri_results = {}  # { 'dri_Name': {node: score}, ... }

    # Precompute global spectral info if needed
    if config["dri_selection"]["spectral"] and config["spectral_params"]["precompute_global_fiedler"]:
        print("Precomputing global Fiedler information for spectral DRIs...")
        spectral_dri.precompute_global_fiedler_info(graph)

    if config["dri_selection"]["baselines"]:
        print("Calculating Baseline DRIs...")
        dri_results["Degree (Out)"] = baselines.get_degree_centrality(graph, per_node=True)
        dri_results["K-Core"] = baselines.get_k_core_centrality(graph, per_node=True)
        dri_results["Eigenvector"] = baselines.get_eigenvector_centrality(graph, per_node=True)
        # Betweenness can be very slow, consider sampling for large graphs
        # dri_results['Betweenness'] = baselines.get_betweenness_centrality(graph, per_node=True, k_samples=1000 if graph.number_of_nodes() > 5000 else None)
        # dri_results['Closeness'] = baselines.get_closeness_centrality(graph, per_node=True)
        print("Baselines calculated.")

    if config["dri_selection"]["literature"]:
        print("Calculating Literature DRIs...")
        dri_results["CI (l=2)"] = literature_indices.get_collective_influence(graph, per_node=True, l_dist=2)
        # Gravity can be slow. Use radius or ensure graph is not too large.
        # dri_results['Gravity (r=3)'] = literature_indices.get_gravity_centrality(graph, per_node=True, r_radius=3)
        dri_results["SD_ego (r=1)"] = literature_indices.get_structural_diversity_ego(
            graph, per_node=True, radius=1
        )
        if graph.is_directed():
            dri_results["FwLTR_struct (t=0.5*indeg)"] = literature_indices.get_fwltr_structural(
                graph, per_node=True, uniform_threshold=0.5
            )
        print("Literature DRIs calculated.")

    if config["dri_selection"]["composite_wmfs"]:
        print("Calculating Composite WMFS-DRIs...")
        for wmfs_conf in config["composite_wmfs_configs"]:
            name = wmfs_conf["name"]
            # Dynamically get feature functions
            parsed_feature_configs = []
            for feat_conf in wmfs_conf["features"]:
                module_name = feat_conf["func_module"]
                func_name = feat_conf["func_name"]
                # This requires a way to map string names to actual function objects
                # Example: getattr(globals()[module_name], func_name)
                # This needs careful implementation for safety and correctness.
                # For now, assume functions are directly callable if modules are imported.
                # This part needs robust dynamic function loading.
                # Simplified:
                if module_name == "baselines":
                    module = baselines
                elif module_name == "literature_indices":
                    module = literature_indices
                elif module_name == "spectral_dri":
                    module = spectral_dri
                else:
                    raise ValueError(f"Unknown module: {module_name}")

                parsed_feature_configs.append(
                    {
                        "name": feat_conf["name"],  # For internal tracking if needed
                        "func": getattr(module, func_name),
                        "weight": feat_conf["weight"],
                        "params": feat_conf.get("params", {}),
                    }
                )
            dri_results[name] = composite_dri.get_wmfs_dri(graph, parsed_feature_configs, per_node=True)
            print(f"{name} calculated.")

    if config["dri_selection"]["mechanism_bpi"]:
        print("Calculating BPI-DRI...")
        dri_results["BPI-DRI (norm)"] = mechanism_focused_dri.get_bpi_dri(
            graph, per_node=True, normalize_components=True
        )
        print("BPI-DRI calculated.")

    if config["dri_selection"]["spectral"]:
        print("Calculating Spectral DRIs...")
        sp = config["spectral_params"]
        dri_results["LocFiedler (k=1)"] = spectral_dri.get_localized_fiedler_value(
            graph, per_node=True, k_hop=sp["k_hop_local_spectral"]
        )
        if sp["precompute_global_fiedler"]:
            dri_results["GlobFiedlerComp"] = spectral_dri.get_node_global_fiedler_component(
                graph, per_node=True
            )
        dri_results["LocSpecRadius (k=1)"] = spectral_dri.get_localized_spectral_radius(
            graph, per_node=True, k_hop=sp["k_hop_local_spectral"]
        )
        if graph.is_directed():  # PPR and Heat Kernel are often on DiGraphs for flow
            dri_results[f'PPR-API (k_sum={sp["ppr_k_sum_hops"]})'] = spectral_dri.get_ppr_api_dri(
                graph, per_node=True, k_sum_hops=sp["ppr_k_sum_hops"], ppr_alpha=sp["ppr_alpha"]
            )
            # Heat Kernel is very slow for large graphs due to dense matrix expm.
            # Only run on smaller graphs or if approximations are used.
            # if graph.number_of_nodes() < 1000: # Example threshold
            #    dri_results[f'HeatKernel-API (t={sp["heat_kernel_t"]})'] = spectral_dri.get_heat_kernel_api_dri(graph, per_node=True, t_short=sp["heat_kernel_t"])
            # else:
            #    print(f"Skipping Heat Kernel DRI for {graph_name} due to size ({graph.number_of_nodes()} nodes).")
        print("Spectral DRIs calculated.")

    if config["save_results_csv"]:
        for dri_name, scores_dict in dri_results.items():
            safe_dri_name = (
                dri_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "")
                .replace("*", "")
                .replace(".", "p")
            )
            pd.DataFrame(list(scores_dict.items()), columns=["node", dri_name]).to_csv(
                os.path.join(config["results_dir"], f"{graph_name}_{safe_dri_name}_scores.csv"), index=False
            )
    print("All DRIs calculated.")

    # --- 4. Perform Evaluations ---
    print("\n--- 4. Performing Evaluations ---")
    evaluation_results_summary = []  # List of dicts for overall results table

    for etm_name, current_etm_scores in etm_scores.items():
        print(f"\nEvaluating DRIs against ETM: {etm_name} on {graph_name}")

        # Ensure scores are aligned by node for correlation
        # And filter out nodes that might not have scores for both (e.g. isolated nodes for some ETMs)
        common_eval_nodes = list(
            set(current_etm_scores.keys()).intersection(
                *[set(scores.keys()) for scores in dri_results.values()]
            )
        )
        # Further filter for nodes where ETM score is not None or NaN
        common_eval_nodes = [
            n
            for n in common_eval_nodes
            if current_etm_scores.get(n) is not None and not np.isnan(current_etm_scores.get(n))
        ]

        if not common_eval_nodes:
            print(
                f"  No common nodes with valid ETM scores for {etm_name}. Skipping evaluation for this ETM."
            )
            continue

        ordered_etm_for_eval = [current_etm_scores[n] for n in common_eval_nodes]

        for dri_name, current_dri_scores in dri_results.items():
            ordered_dri_for_eval = [
                current_dri_scores.get(n, 0.0) for n in common_eval_nodes
            ]  # Default to 0 if node missing in DRI

            # a. Spearman Correlation
            corr, p_val = eval_metrics.calculate_spearman_rank_correlation(
                ordered_dri_for_eval, ordered_etm_for_eval
            )
            print(f"  Spearman Corr ({dri_name} vs {etm_name}): {corr:.3f} (p={p_val:.3g})")

            current_eval_row = {
                "ETM": etm_name,
                "DRI": dri_name,
                "Spearman_Correlation": corr,
                "P_value": p_val,
            }

            # b. Top-k Performance (Precision, Recall, F1)
            for k_perc in config["evaluation_params"]["top_k_values_perc"]:
                true_top_k = eval_metrics.get_top_k_nodes(current_etm_scores, k_perc, is_percentage=True)
                pred_top_k = eval_metrics.get_top_k_nodes(current_dri_scores, k_perc, is_percentage=True)

                # Filter true_top_k and pred_top_k to only include common_eval_nodes if necessary,
                # though get_top_k_nodes operates on the full dicts.
                # The metrics function should handle this alignment if sets are passed.

                if not true_top_k:  # Avoid issues if ETM scores are all same/zero
                    print(
                        f"    Skipping P/R/F1 for k={k_perc*100:.0f}% as true_top_k is empty for {etm_name}."
                    )
                    current_eval_row[f"F1@{k_perc*100:.0f}%"] = np.nan
                    continue

                prf_scores = eval_metrics.calculate_precision_recall_f1_at_k(true_top_k, pred_top_k)
                current_eval_row[f"Precision@{k_perc*100:.0f}%"] = prf_scores["precision_at_k"]
                current_eval_row[f"Recall@{k_perc*100:.0f}%"] = prf_scores["recall_at_k"]
                current_eval_row[f"F1@{k_perc*100:.0f}%"] = prf_scores["f1_at_k"]
                # print(f"    Top {k_perc*100:.0f}%: P={prf_scores['precision_at_k']:.3f}, R={prf_scores['recall_at_k']:.3f}, F1={prf_scores['f1_at_k']:.3f}")

            # c. Imprecision Metric (for one k value for brevity in summary)
            # k_for_imprecision = config["evaluation_params"]["top_k_values_perc"][1] # e.g., use middle k%
            # imprecision = eval_metrics.calculate_imprecision_metric(
            #     common_eval_nodes, # Pass only common nodes with valid scores
            #     {n: current_etm_scores[n] for n in common_eval_nodes},
            #     {n: current_dri_scores.get(n,0.0) for n in common_eval_nodes},
            #     k_for_imprecision, is_percentage=True
            # )
            # current_eval_row[f'Imprecision@{k_for_imprecision*100:.0f}%'] = imprecision
            # print(f"    Imprecision (Top {k_for_imprecision*100:.0f}%): {imprecision:.3f}")

            evaluation_results_summary.append(current_eval_row)

    # Save summary results
    results_df = pd.DataFrame(evaluation_results_summary)
    if config["save_results_csv"]:
        results_df.to_csv(
            os.path.join(config["results_dir"], f"{graph_name}_evaluation_summary.csv"), index=False
        )
    print("\nEvaluation summary saved.")
    print(results_df.to_string(max_rows=20))

    # --- 5. Generate Visualizations ---
    print("\n--- 5. Generating Visualizations ---")
    if config["save_plots"]:
        plot_dir = os.path.join(config["results_dir"], f"{graph_name}_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # a. Degree Distribution
        plotter.plot_degree_distribution(
            graph, graph_name, save_path=os.path.join(plot_dir, "degree_distribution.png")
        )

        # b. ETM Distributions
        for etm_name, scores in etm_scores.items():
            plotter.plot_etm_distribution(
                scores, etm_name, graph_name, save_path=os.path.join(plot_dir, f"dist_{etm_name}.png")
            )

        # c. Correlation Heatmap (requires results_df to be reshaped or a dedicated correlation matrix)
        # Reshape results_df for heatmap: index=DRI, columns=ETM, values=Spearman_Correlation
        try:
            corr_pivot_df = results_df.pivot(index="DRI", columns="ETM", values="Spearman_Correlation")
            plotter.plot_correlation_heatmap(
                corr_pivot_df,
                title=f"Spearman Correlation: DRIs vs. ETMs ({graph_name})",
                save_path=os.path.join(plot_dir, "correlation_heatmap.png"),
            )
        except Exception as e:
            print(f"Could not generate correlation heatmap: {e}")

        # d. Scatter/Rank Plots for a key ETM (e.g., MCSS) vs. a few key DRIs
        key_etm_for_scatter = "MCSS"  # Example
        if key_etm_for_scatter in etm_scores:
            key_dris_for_scatter = ["Degree (Out)", "CI (l=2)"]  # Add best proposed DRI name here once known
            if config["composite_wmfs_configs"]:
                key_dris_for_scatter.append(
                    config["composite_wmfs_configs"][0]["name"]
                )  # Add first WMFS-DRI

            for dri_name_sc in key_dris_for_scatter:
                if dri_name_sc in dri_results:
                    plotter.plot_scatter_rank_comparison(
                        dri_results[dri_name_sc],
                        etm_scores[key_etm_for_scatter],
                        tri_name=dri_name_sc,
                        etm_name=key_etm_for_scatter,
                        graph_name=graph_name,
                        save_path_prefix=os.path.join(
                            plot_dir, f"scatter_{safe_name(dri_name_sc)}_vs_{key_etm_for_scatter}"
                        ),
                    )

        # e. Top-k Performance Bar Charts (e.g., F1@k for each ETM)
        # Reshape results_df for F1 scores: index=DRI, columns=k_value, values=F1
        for etm_name_bc in etm_scores.keys():
            f1_cols = [
                col
                for col in results_df.columns
                if "F1@" in col and etm_name_bc in results_df[results_df["ETM"] == etm_name_bc]["ETM"].values
            ]
            if f1_cols:
                f1_df_for_etm = results_df[results_df["ETM"] == etm_name_bc].set_index("DRI")[f1_cols]
                f1_df_for_etm.columns = [
                    col.split("@")[1] for col in f1_cols
                ]  # Clean column names (e.g. "5%")
                if not f1_df_for_etm.empty:
                    plotter.plot_top_k_performance_bars(
                        f1_df_for_etm,
                        metric_name="F1-Score@Top-k%",
                        title_suffix=f"for {etm_name_bc} ({graph_name})",
                        save_path=os.path.join(plot_dir, f"top_k_f1_bars_{etm_name_bc}.png"),
                    )
        print("Visualizations generated and saved.")
    else:
        print("Plot saving is disabled in config.")

    end_time_exp = time.time()
    print(f"\nExperiment finished in {end_time_exp - start_time_exp:.2f} seconds.")
    print(f"All results and plots saved in: {config['results_dir']}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Diffusion Readiness Index Experiments.")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file.")
    # Add more command-line arguments to override specific config values if needed
    parser.add_argument("--dataset_type", type=str, choices=["casflow", "snap_twitter"])

    args = parser.parse_args()

    config_to_run = load_config(args.config)

    run_experiment(config_to_run)
