#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:21:30 2025

@author: sanup
"""


import networkx as nx
import numpy as np
import random
import itertools

from scipy.stats import spearmanr, pearsonr, ttest_rel
from collections import defaultdict


# def load_graph(edge_path, weighted=False):
#     """
#     Load a directed graph from an edge list.
#     If weighted=True, assume the third column is a float weight.
#     """
#     G = nx.DiGraph()
#     with open(edge_path, "r") as fin:
#         for line in fin:
#             parts = line.strip().split()
#             if len(parts) < 2:
#                 continue
#             u, v = parts[0], parts[1]
#             w = float(parts[2]) if weighted and len(parts) > 2 else 1.0
#             G.add_edge(u, v, weight=w)
#     return G


def load_cascades_and_build_graph(path):
    """
    Parse cascades from `dataset.txt`, where each line is:
      cascade_id \t seed_id \t start_ts \t size \t activations...
    Activation tokens look like:
      user:0  user/other:306  user/other/more:49053 ...
    We infer time = int(text after last ':'), and build a global DiGraph
    with edges u->v (t_u < t_v), weight = count of cascades where this occurs.
    """
    cascades = []
    edge_counts = defaultdict(int)
    users = set()

    with open(path, "r") as fin:
        for line in fin:
            parts = line.strip().split("\t")
            # need at least 5 parts: id, seed, start_ts, size, activations
            if len(parts) < 5:
                continue
            # parse activation tokens
            act_tokens = parts[4].split()
            cascade = []
            for token in act_tokens:
                if ":" not in token:
                    continue
                user_part, time_part = token.rsplit(":", 1)
                user = user_part.split("/")[0]  # extract the user before any '/'
                try:
                    t = int(time_part)
                except ValueError:
                    continue
                cascade.append((user, t))
                users.add(user)
            if len(cascade) < 2:
                continue
            # sort by inferred time
            cascade.sort(key=lambda x: x[1])
            cascades.append(cascade)

            # count edges
            for i in range(len(cascade)):
                u, t_u = cascade[i]
                for j in range(i + 1, len(cascade)):
                    v, t_v = cascade[j]
                    if t_u < t_v:
                        edge_counts[(u, v)] += 1

    # build graph
    G = nx.DiGraph()
    G.add_nodes_from(users)
    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)

    print(f"Constructed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Loaded {len(cascades)} cascades.")
    return G, cascades


def compute_components(G):
    """
    Compute static DRI components for each node:
      - c_v: clustering coefficient
      - w_v: normalized avg incoming weight
      - r_v: reciprocity ratio
    Returns C, W, R dicts.
    """
    Gu = G.to_undirected()
    C = nx.clustering(Gu)
    raw_in = {v: sum(d["weight"] for _, _, d in G.in_edges(v, data=True)) for v in G.nodes()}
    indeg = {v: G.in_degree(v) for v in G.nodes()}
    avg_in = {v: (raw_in[v] / indeg[v] if indeg[v] > 0 else 0.0) for v in G.nodes()}
    vals = np.array(list(avg_in.values()), float)
    vmin, vmax = vals.min(), vals.max()
    W = {v: (avg_in[v] - vmin) / (vmax - vmin) if vmax > vmin else 0.0 for v in G.nodes()}
    rec = {}
    for v in G.nodes():
        deg = G.degree(v)
        rec[v] = sum(1 for u in G.predecessors(v) if G.has_edge(v, u)) / deg if deg > 0 else 0.0
    return C, W, rec


def compute_dri_scores(G, lambdas=(1 / 3, 1 / 3, 1 / 3), p=0.5):
    """
    Compute static DRI: (c_v^p + w_v^p + r_v^p)^(1/p).
    """
    lambda1, lambda2, lambda3 = lambdas
    C, W, R = compute_components(G)
    return {v: (C[v] ** p + W[v] ** p + R[v] ** p) ** (1 / p) for v in G.nodes()}


def compute_rich_dri(G, lambdas=(1 / 5,) * 5, p=0.5):
    """
    5-component DRI: [clustering, avg_in_weight, reciprocity, k-core, closeness].
    lambdas: tuple of 5 weights summing to 1.
    """

    lambda1, lambda2, lambda3, lambda4, lambda5 = lambdas

    # 1. Clustering
    C = nx.clustering(G.to_undirected())

    # 2. Avg incoming weight
    raw_in = {v: sum(d["weight"] for _, _, d in G.in_edges(v, data=True)) for v in G.nodes()}
    indeg = dict(G.in_degree())
    avg_in = {v: raw_in[v] / indeg[v] if indeg[v] > 0 else 0 for v in G.nodes()}
    w_vals = np.array(list(avg_in.values()), float)
    min_w, max_w = w_vals.min(), w_vals.max()
    W = {v: (avg_in[v] - min_w) / (max_w - min_w) if max_w > min_w else 0 for v in G.nodes()}

    # 3. Reciprocity
    R = {
        v: sum(1 for u in G.predecessors(v) if G.has_edge(v, u)) / G.degree(v) if G.degree(v) > 0 else 0
        for v in G.nodes()
    }

    # 4. K-core
    G_core = G.to_undirected().copy()
    G_core.remove_edges_from(nx.selfloop_edges(G_core))
    core = nx.core_number(G_core)
    # core = nx.core_number(G.to_undirected())
    max_core = max(core.values())
    K = {v: core[v] / max_core for v in G.nodes()}

    # 5. Closeness
    close = nx.closeness_centrality(G.to_undirected())
    # closeness already in [0,1]
    H = close

    features = [C, W, R, K, H]
    dri = {}
    for v in G.nodes():
        terms = [lambdas[i] * (features[i][v] ** p) for i in range(5)]
        dri[v] = (sum(terms)) ** (1 / p)
    return dri


def compute_baseline_metrics(G):
    """Compute standard centrality baselines."""
    return {
        "in_degree": dict(G.in_degree()),
        "out_degree": dict(G.out_degree()),
        "total_degree": dict(G.degree()),
        "pagerank": nx.pagerank(G, weight="weight"),
        "eigenvector": nx.eigenvector_centrality_numpy(G.to_undirected()),
        "katz": nx.katz_centrality_numpy(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
    }


def compute_correlations(dri_scores, baselines):
    """
    Compute Spearman and Pearson correlations between DRI and each baseline.
    Returns dict of metrics.
    """
    results = {}
    nodes = list(dri_scores.keys())
    dris = np.array([dri_scores[n] for n in nodes])
    for name, scores in baselines.items():
        vals = np.array([scores.get(n, 0) for n in nodes])
        rho, p_s = spearmanr(dris, vals)
        r, p_p = pearsonr(dris, vals)
        results[name] = {"spearman": (rho, p_s), "pearson": (r, p_p)}
    return results


def select_top_k(scores, k=50):
    """Return top-k nodes by value."""
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def run_ic(G, seeds, p=0.01, steps=0):
    """Independent Cascade model."""
    activated = set(seeds)
    new = set(seeds)
    t = 0
    while new and (steps == 0 or t < steps):
        t += 1
        nxt = set()
        for u in new:
            for v in G.successors(u):
                if v not in activated and random.random() <= p:
                    nxt.add(v)
        new = nxt
        activated |= new
    return len(activated)


def run_sir(G, seeds, beta=0.01, gamma=0.005, steps=0):
    """SIR model."""
    susceptible = set(G.nodes()) - set(seeds)
    infected = set(seeds)
    recovered = set()
    t = 0
    while infected and (steps == 0 or t < steps):
        t += 1
        new_inf = set()
        for u in infected:
            for v in G.successors(u):
                if v in susceptible and random.random() <= beta:
                    new_inf.add(v)
        for u in list(infected):
            if random.random() <= gamma:
                infected.remove(u)
                recovered.add(u)
        infected |= new_inf
        susceptible -= new_inf
    return len(recovered | infected)


def run_diffusion(G, seeds, model="IC", params=None, trials=100):
    """Run diffusion multiple times; return cascade sizes."""
    results = []
    for _ in range(trials):
        if model == "IC":
            res = run_ic(G, seeds, p=params.get("p", 0.01), steps=params.get("steps", 0))
        else:
            res = run_sir(
                G,
                seeds,
                beta=params.get("beta", 0.01),
                gamma=params.get("gamma", 0.005),
                steps=params.get("steps", 0),
            )
        results.append(res)
    return results


def load_cascade_data(cascade_path, G=None):
    """
    Load cascade logs from dataset.txt, filter seeds present in G if provided.
    Format: cascade_id\tseed_id\tstart_ts\tcascade_size\t...
    Returns list of ([seed_id], cascade_size) pairs.
    """
    cascades = []
    with open(cascade_path, "r") as fin:
        for line in fin:
            if "\t" not in line:
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            seed = parts[1]
            if G is not None and seed not in G:
                continue
            try:
                size = int(parts[3])
            except ValueError:
                continue
            cascades.append(([seed], size))
    print(f"Loaded {len(cascades)} cascades (filtered by graph nodes). ")
    return cascades


# def optimize_dri_weights(G, cascades, p_values=[0.2, 0.5, 1.0], grid_steps=11):
#     """
#     Grid-search optimize lambda weights and p to maximize Spearman correlation.
#     cascades: list of (seed_list, final_size)
#     Returns dict with best lam1/lam2/lam3, p, and score.
#     """
#     C, W, R = compute_components(G)
#     best = {'score': -np.inf}
#     # pre-filter cascades with valid seeds
#     filtered = [(s, size) for s, size in cascades if any(v in C for v in s)]
#     print(f"Optimizing over {len(filtered)} valid cascades.")
#     for lam1 in np.linspace(0, 1, grid_steps):
#         for lam2 in np.linspace(0, 1-lam1, grid_steps):
#             lam3 = 1 - lam1 - lam2
#             for p in p_values:
#                 preds, truths = [], []
#                 for seeds, size in filtered:
#                     vals = [(lam1*C[v]**p + lam2*W[v]**p + lam3*R[v]**p)**(1/p)
#                             for v in seeds if v in C]
#                     if not vals:
#                         continue
#                     preds.append(np.mean(vals))
#                     truths.append(size)
#                 if len(preds) < 3:
#                     continue
#                 rho, _ = spearmanr(preds, truths)
#                 if rho > best['score']:
#                     best = {'lam1': lam1, 'lam2': lam2, 'lam3': lam3, 'p': p, 'score': rho}
#     return best


def tune_dri_params(G, cascades):
    best_score = -np.inf
    best_params = None
    scores = []

    lambdas_list = [(l1, l2, 1 - l1 - l2) for l1 in np.linspace(0, 1, 5) for l2 in np.linspace(0, 1 - l1, 5)]
    p_list = [0.25, 0.5, 0.75, 1.0, 2.0]

    for lambdas, p in itertools.product(lambdas_list, p_list):
        if any(l < 0 or l > 1 for l in lambdas) or abs(sum(lambdas) - 1) > 1e-6:
            continue

        dri_scores = compute_dri_scores(G, lambdas=lambdas, p=p)

        seeds = []
        sizes = []
        for cascade in cascades:
            if not cascade:
                continue
            seed = str(cascade[0][0])  # force to string
            if seed in dri_scores:
                seeds.append(seed)
                sizes.append(len(cascade))

        if len(seeds) < 5:
            continue

        dri_vals = [dri_scores[s] for s in seeds]
        rho, _ = spearmanr(dri_vals, sizes)
        scores.append((rho, lambdas, p))
        if rho > best_score:
            best_score = rho
            best_params = (lambdas, p)

    if best_params is None:
        print("No valid cascades found after matching. Check ID format.")
    else:
        print(f"Best DRI params: {best_params} (Spearman: {best_score:.4f})")

    return best_params, scores


if __name__ == "__main__":
    # G = load_graph("./../Data/SNAPTwitterData/twitter/78813.edges", weighted=True)
    G, cascades = load_cascades_and_build_graph("./../Data/CasFlow Dataset/dataset.txt")
    # dri = compute_dri_scores(G, p=0.5)
    dri = compute_rich_dri(G)
    baselines = compute_baseline_metrics(G)

    # Evaluate static
    corr = compute_correlations(dri, baselines)
    print("=== Correlations DRI vs Baselines ===")
    for m, stats in corr.items():
        print(f"{m}: Spearman={stats['spearman'][0]:.4f} (p={stats['spearman'][1]:.2e})")

    # Diffusion performance
    strategies = {
        name: select_top_k(metric, 50) for name, metric in [("DRI", dri)] + list(baselines.items())
    }
    results = {}
    for name, seeds in strategies.items():
        sizes_ic = run_diffusion(G, seeds, "IC", {"p": 0.01, "steps": 0}, trials=50)
        sizes_sir = run_diffusion(G, seeds, "SIR", {"beta": 0.01, "gamma": 0.005, "steps": 0}, trials=50)
        results[name] = {"IC": sizes_ic, "SIR": sizes_sir}
    print("\n=== Seed-Set Diffusion Performance ===")
    for name, data in results.items():
        print(
            f"{name}: IC {np.mean(data['IC']):.2f}±{np.std(data['IC']):.2f}; "
            f"SIR {np.mean(data['SIR']):.2f}±{np.std(data['SIR']):.2f}"
        )

    # Paired t-tests
    print("\n=== Paired t-tests vs DRI ===")
    for model in ["IC", "SIR"]:
        base = np.array(results["DRI"][model])
        for name in baselines.keys():
            comp = np.array(results[name][model])
            t_stat, p_val = ttest_rel(base, comp)
            print(f"{model}: DRI vs {name} t={t_stat:.2f}, p={p_val:.3f}")

    # Optimize DRI weights
    # cascades = load_cascade_data("./../Data/CasFlow Dataset/dataset.txt")
    best_params, scores = tune_dri_params(G, cascades)
    print("Best DRI params:", best_params)

    # G = load_graph('./../Data/SNAPTwitterData/twitter/78813.edges', weighted=True)
    # cascades = load_cascade_data('./../Data/CasFlow Dataset/dataset.txt')
    # # 1. Original DRI
    # dri_orig = compute_dri_scores(G)

    # # 2. Ablations
    # #    a) Drop clustering: set all c_v to 0
    # C, W, R = compute_components(G)
    # dri_noC = {v: (0.0**0.5 + W[v]**0.5 + R[v]**0.5)**2 for v in G.nodes()}
    # #    b) Drop incoming weight
    # dri_noW = {v: (C[v]**0.5 + 0.0**0.5 + R[v]**0.5)**2 for v in G.nodes()}
    # #    c) Drop reciprocity
    # dri_noR = {v: (C[v]**0.5 + W[v]**0.5 + 0.0**0.5)**2 for v in G.nodes()}

    # # 3. Compute correlations for each
    # for label, scores in [('DRI', dri_orig), ('noC', dri_noC), ('noW', dri_noW), ('noR', dri_noR)]:
    #     corr = compute_correlations(scores, baselines)
    #     print(f"=== Correlations for {label} ===")
    #     for m, stats in corr.items():
    #         print(f"{m}: Spearman={stats['spearman'][0]:.4f}, p={stats['spearman'][1]:.2e}")

    # # 4. Seed-set diffusion (example for k=50)
    # k = 50
    # seed_sets = {
    #     'DRI': [n for n, _ in select_top_k(dri_orig, k)],
    #     'noC': [n for n, _ in select_top_k(dri_noC, k)],
    #     'noW': [n for n, _ in select_top_k(dri_noW, k)],
    #     'noR': [n for n, _ in select_top_k(dri_noR, k)],
    #     'Degree': [n for n, _ in select_top_k(baselines['total_degree'], k)],
    #     'PageRank': [n for n, _ in select_top_k(baselines['pagerank'], k)],
    # }

    # results = {}
    # for name, seeds in seed_sets.items():
    #     # replace run_diffusion with your simulator call
    #     cascade_sizes = run_diffusion(G, seeds, model='IC', params={'p': 0.01})
    #     # e.g. cascade_sizes could be a list from multiple runs
    #     results[name] = {
    #         'mean': np.mean(cascade_sizes),
    #         'std': np.std(cascade_sizes)
    #     }

    # print("=== Seed‐Set Diffusion Results ===")
    # for name, stats in results.items():
    #     print(f"{name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # # # Compute DRI scores
    # # dri_scores = compute_dri_scores(G, p=0.5)
    # # # Compute baseline metrics
    # # baselines = compute_baseline_metrics(G)

    # # # Print top 10 by DRI
    # # top10 = sorted(dri_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    # # print('Top 10 nodes by DRI:')
    # # for node, score in top10:
    # #     print(f"{node}\t{score:.4f}")

    # # # Example: correlation with PageRank
    # # import scipy.stats as stats
    # # dris = [dri_scores[n] for n in G.nodes()]
    # # pr = [baselines['pagerank'][n] for n in G.nodes()]
    # # rho, pval = stats.spearmanr(dris, pr)
    # # print(f"Spearman correlation DRI vs. PageRank: {rho:.4f} (p={pval:.2e})")
