# diffusion_readiness_project/evaluation/plotter.py
# Python 3.9

"""
Functions to generate various plots for visualizing experimental results,
including dataset characteristics, ETM distributions, correlation heatmaps,
scatter plots, top-k performance plots, and network visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

# Ensure plots are displayed inline in some environments or saved to files.
# plt.rcParams['figure.figsize'] = (10, 6) # Default figure size

# --- Plotting Functions ---

def plot_degree_distribution(graph, graph_name="Graph", log_log_scale=True, save_path=None):
    """
    Plots the degree distribution of a graph.

    Args:
        graph (nx.Graph or nx.DiGraph): The input graph.
        graph_name (str): Name of the graph for the plot title.
        log_log_scale (bool): Whether to use log-log scale for axes.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    if graph.number_of_nodes() == 0:
        print(f"Graph '{graph_name}' is empty, cannot plot degree distribution.")
        return

    if graph.is_directed():
        # For directed graphs, one might plot in-degree, out-degree, or total degree.
        # Let's plot out-degree distribution as it's often related to influence spread.
        degrees = [d for n, d in graph.out_degree()]
        degree_type = "Out-Degree"
    else:
        degrees = [d for n, d in graph.degree()]
        degree_type = "Degree"
    
    if not degrees:
        print(f"No degrees to plot for graph '{graph_name}'.")
        return

    degree_counts = Counter(degrees)
    deg, cnt = zip(*sorted(degree_counts.items()))

    plt.figure(figsize=(10, 6))
    if log_log_scale:
        plt.loglog(deg, cnt, marker='o', linestyle='None')
    else:
        plt.plot(deg, cnt, marker='o', linestyle='-')
    
    plt.title(f"{degree_type} Distribution for {graph_name}")
    plt.xlabel(f"{degree_type}")
    plt.ylabel("Frequency (Count)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Degree distribution plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_etm_distribution(etm_scores_dict, etm_name="ETM", graph_name="Graph", bins=30, save_path=None):
    """
    Plots the distribution of ETM scores using a histogram and KDE.

    Args:
        etm_scores_dict (dict): Dictionary of {node_id: etm_score}.
        etm_name (str): Name of the ETM for labeling.
        graph_name (str): Name of the graph/dataset.
        bins (int): Number of bins for the histogram.
        save_path (str, optional): Path to save the figure.
    """
    if not etm_scores_dict:
        print(f"No ETM scores provided for {etm_name} on {graph_name} to plot distribution.")
        return

    scores = list(etm_scores_dict.values())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=bins, kde=True, stat="density", common_norm=False)
    plt.title(f"Distribution of {etm_name} Scores for {graph_name}")
    plt.xlabel(f"{etm_name} Score")
    plt.ylabel("Density")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"{etm_name} distribution plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(correlation_df, title="TRI vs. ETM Spearman Correlation", save_path=None):
    """
    Plots a heatmap of correlation coefficients.

    Args:
        correlation_df (pd.DataFrame): DataFrame where rows are TRIs, columns are ETMs,
                                       and values are correlation coefficients.
        title (str): Title for the heatmap.
        save_path (str, optional): Path to save the figure.
    """
    if correlation_df.empty:
        print("Correlation DataFrame is empty, cannot plot heatmap.")
        return

    plt.figure(figsize=(max(8, len(correlation_df.columns) * 1.2), max(6, len(correlation_df.index) * 0.8)))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_scatter_rank_comparison(tri_scores, etm_scores, tri_name="TRI", etm_name="ETM", graph_name="Graph", save_path_prefix=None):
    """
    Generates two plots:
    1. Scatter plot of TRI scores vs. ETM scores.
    2. Scatter plot of Rank by TRI vs. Rank by ETM.

    Args:
        tri_scores (dict): {node: tri_score}.
        etm_scores (dict): {node: etm_score}. Nodes should align.
        tri_name (str): Name of the TRI.
        etm_name (str): Name of the ETM.
        graph_name (str): Name of the graph/dataset.
        save_path_prefix (str, optional): Prefix for saving plot files.
    """
    common_nodes = list(set(tri_scores.keys()).intersection(set(etm_scores.keys())))
    if not common_nodes:
        print("No common nodes between TRI and ETM scores for scatter/rank plots.")
        return

    tri_vals = np.array([tri_scores[n] for n in common_nodes])
    etm_vals = np.array([etm_scores[n] for n in common_nodes])

    # 1. Score Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(tri_vals, etm_vals, alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.title(f"Scatter Plot: {tri_name} vs. {etm_name} ({graph_name})")
    plt.xlabel(f"{tri_name} Score")
    plt.ylabel(f"{etm_name} Score")
    plt.grid(True, ls="--", alpha=0.5)
    # Optional: Add a regression line
    # m, b = np.polyfit(tri_vals, etm_vals, 1)
    # plt.plot(tri_vals, m*tri_vals + b, color='red', linestyle='--')
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_score_scatter.png", bbox_inches='tight')
        print(f"Score scatter plot saved to {save_path_prefix}_score_scatter.png")
        plt.close()
    else:
        plt.show()

    # 2. Rank-Rank Plot
    # scipy.stats.rankdata can be used for ranking
    from scipy.stats import rankdata
    tri_ranks = rankdata([-x for x in tri_vals], method='ordinal') # Negative for descending, ordinal for unique ranks
    etm_ranks = rankdata([-x for x in etm_vals], method='ordinal')

    plt.figure(figsize=(8, 8))
    plt.scatter(tri_ranks, etm_ranks, alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.title(f"Rank-Rank Plot: {tri_name} vs. {etm_name} ({graph_name})")
    plt.xlabel(f"Rank by {tri_name} (Higher score = Lower rank number)")
    plt.ylabel(f"Rank by {etm_name} (Higher score = Lower rank number)")
    # Ideal: points along y=x line
    min_rank = 1
    max_rank = len(common_nodes)
    plt.plot([min_rank, max_rank], [min_rank, max_rank], color='red', linestyle='--')
    plt.xlim(min_rank -1 , max_rank + 1)
    plt.ylim(min_rank - 1, max_rank + 1)
    plt.gca().invert_xaxis() # Optional: if you want rank 1 at top-left
    plt.gca().invert_yaxis() # Optional: if you want rank 1 at top-left
    plt.grid(True, ls="--", alpha=0.5)
    
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_rank_rank_scatter.png", bbox_inches='tight')
        print(f"Rank-rank plot saved to {save_path_prefix}_rank_rank_scatter.png")
        plt.close()
    else:
        plt.show()


def plot_top_k_performance_bars(performance_data, metric_name="F1-Score@k", k_values=None, title_suffix="", save_path=None):
    """
    Plots bar charts comparing a performance metric (e.g., F1-score@k)
    for different TRIs across various k values or ETMs.

    Args:
        performance_data (pd.DataFrame): DataFrame where rows are TRIs,
                                         columns are k-values (or ETM types if k is fixed),
                                         and values are the performance metric.
                                         Example:
                                             k=5%  k=10% k=15%
                                     TRI_A  0.5   0.6   0.65
                                     TRI_B  0.4   0.55  0.62
        metric_name (str): Name of the metric being plotted.
        k_values (list, optional): List of k values (e.g., [0.01, 0.05, 0.10] for percentages).
                                   If None, uses DataFrame columns as x-axis labels.
        title_suffix (str): Suffix for the plot title (e.g., "for ETM_X").
        save_path (str, optional): Path to save the figure.
    """
    if performance_data.empty:
        print("Performance data is empty, cannot plot bar chart.")
        return

    performance_data.plot(kind='bar', figsize=(12, 7), width=0.8)
    plt.title(f"{metric_name} Comparison for Different TRIs {title_suffix}")
    plt.ylabel(metric_name)
    if k_values:
        plt.xlabel("Top-k Value (as fraction or count)")
        plt.xticks(ticks=range(len(k_values)), labels=[f"{k:.2%}" if isinstance(k, float) and k<=1 else str(k) for k in k_values], rotation=45, ha="right")
    else:
        plt.xlabel("Compared Categories (e.g., k-values or ETMs)")
        plt.xticks(rotation=45, ha="right")
        
    plt.legend(title="TRIs / Methods")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Top-k performance bar chart saved to {save_path}")
        plt.close()
    else:
        plt.show()

# Qualitative Network Visualization - This is more complex and often done with tools
# like Gephi or specialized Python libraries for interactive plots.
# A simple NetworkX draw can be a starting point for small subgraphs.
def plot_network_subgraph_highlight(graph, subgraph_nodes, node_colors=None, node_sizes=None,
                                    pos=None, title="Network Subgraph", save_path=None):
    """
    Draws a subgraph with nodes potentially colored or sized by scores.
    Very basic visualization; for publication quality, Gephi or other tools are better.

    Args:
        graph (nx.Graph or nx.DiGraph): The full graph.
        subgraph_nodes (list or set): Nodes to include in the visualization.
        node_colors (dict, optional): {node: color_value} for colormap.
        node_sizes (dict, optional): {node: size_value}.
        pos (dict, optional): Precomputed layout {node: (x,y)}. If None, computes one.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """
    subgraph = graph.subgraph(subgraph_nodes)
    if subgraph.number_of_nodes() == 0:
        print("Subgraph is empty, cannot plot.")
        return

    plt.figure(figsize=(12, 12))
    
    if pos is None:
        pos = nx.spring_layout(subgraph, k=0.5/np.sqrt(subgraph.number_of_nodes()), iterations=50) # k for spacing

    # Prepare colors and sizes
    colors_to_plot = [node_colors.get(n, 0) for n in subgraph.nodes()] if node_colors else 'skyblue'
    sizes_to_plot = [node_sizes.get(n, 300) for n in subgraph.nodes()] if node_sizes else 300
    
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
    nodes_drawn = nx.draw_networkx_nodes(subgraph, pos, node_color=colors_to_plot, node_size=sizes_to_plot, cmap=plt.cm.viridis, alpha=0.8)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    
    if node_colors:
        plt.colorbar(nodes_drawn, label="Node Scores (e.g., TRI or ETM)")

    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Network subgraph plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing evaluation/plotter.py...")

    # Sample data for plotting
    # 1. Degree Distribution
    G_sample = nx.barabasi_albert_graph(100, 2, seed=42)
    plot_degree_distribution(G_sample, graph_name="Sample BA Graph", save_path="test_degree_dist.png")

    # 2. ETM Distribution
    etm_sample_scores = {i: np.random.rand() * 10 for i in range(100)}
    plot_etm_distribution(etm_sample_scores, etm_name="Sample ETM", graph_name="Test Data", save_path="test_etm_dist.png")

    # 3. Correlation Heatmap
    corr_data = {
        'ETM1 (ADT)': {'TRI_A': 0.75, 'TRI_B': 0.60, 'Degree': 0.5},
        'ETM2 (MCSS)': {'TRI_A': 0.80, 'TRI_B': 0.65, 'Degree': 0.55},
        'ETM3 (HNR_k)': {'TRI_A': 0.70, 'TRI_B': 0.70, 'Degree': 0.45}
    }
    corr_df_sample = pd.DataFrame(corr_data).T # Transpose to have TRIs as rows, ETMs as columns
    plot_correlation_heatmap(corr_df_sample, title="Sample TRI vs. ETM Correlations", save_path="test_corr_heatmap.png")

    # 4. Scatter/Rank Plots
    tri_scores_sample = {i: np.random.normal(loc=5, scale=2) for i in range(50)}
    etm_scores_scatter_sample = {i: tri_scores_sample[i] * 0.8 + np.random.normal(loc=0, scale=1) for i in range(50)}
    plot_scatter_rank_comparison(tri_scores_sample, etm_scores_scatter_sample, 
                                 tri_name="SampleTRI", etm_name="SampleETM", graph_name="Test Scatter",
                                 save_path_prefix="test_scatter")
    
    # 5. Top-k Performance Bars
    perf_data = {
        'TRI_A': [0.5, 0.6, 0.65, 0.7],
        'TRI_B': [0.4, 0.55, 0.62, 0.68],
        'Degree': [0.3, 0.4, 0.45, 0.5]
    }
    k_vals_sample = [0.01, 0.05, 0.10, 0.15] # Example k values (as fractions for x-axis labels)
    perf_df_sample = pd.DataFrame(perf_data, index=[f"{k*100:.0f}% Top-k" for k in k_vals_sample])
    plot_top_k_performance_bars(perf_df_sample.T, metric_name="F1-Score@k", k_values=k_vals_sample, save_path="test_top_k_bars.png")


    # 6. Network Subgraph Highlight (Simple example)
    # G_karate = nx.karate_club_graph()
    # central_nodes = sorted(nx.degree_centrality(G_karate).items(), key=lambda x: x[1], reverse=True)[:5]
    # subgraph_to_plot_nodes = {n[0] for n in central_nodes}
    # for n, deg in G_karate.degree(list(subgraph_to_plot_nodes)): # Add neighbors of central nodes
    #     subgraph_to_plot_nodes.update(list(G_karate.neighbors(n)))
    # node_color_map = {n: G_karate.degree(n) for n in subgraph_to_plot_nodes}
    # plot_network_subgraph_highlight(G_karate, list(subgraph_to_plot_nodes), node_colors=node_color_map, 
    #                                 title="Karate Club Subgraph (Colored by Degree)", save_path="test_network_subgraph.png")
    
    print("\n--- Plotter Test Complete (Check for saved .png files) ---")

