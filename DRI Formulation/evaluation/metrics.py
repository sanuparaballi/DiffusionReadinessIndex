# diffusion_readiness_project/evaluation/metrics.py
# Python 3.9

"""
Functions to calculate evaluation metrics for comparing TRIs against ETMs:
- Spearman's Rank Correlation Coefficient
- Precision@k
- Recall@k
- F1-score@k
- Imprecision Metric (or similar rank-based error)
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score # For P@k, R@k, F1@k

# --- Evaluation Metric Functions ---

def calculate_spearman_rank_correlation(scores_array1, scores_array2):
    """
    Calculates Spearman's rank correlation coefficient between two arrays of scores.

    Args:
        scores_array1 (list or np.array): First list/array of scores (e.g., TRI scores).
        scores_array2 (list or np.array): Second list/array of scores (e.g., ETM scores).
                                          Must be of the same length as scores_array1.

    Returns:
        tuple: (correlation_coefficient, p_value) or (None, None) if an error occurs.
    """
    if len(scores_array1) != len(scores_array2):
        # print("Error: Input arrays for Spearman correlation must have the same length.")
        return None, None
    if len(scores_array1) < 2: # Spearman correlation needs at least 2 data points
        # print("Warning: Need at least 2 data points for Spearman correlation.")
        return None, None
        
    try:
        correlation, p_value = spearmanr(scores_array1, scores_array2)
        # Handle potential NaN from spearmanr if input arrays are constant
        if np.isnan(correlation):
            # If both arrays are constant, correlation is undefined by some definitions,
            # or could be considered 1 if they are perfectly "correlated" in their constancy.
            # scipy may return NaN. If one is constant and other is not, should be 0 or near 0.
            # For simplicity, if NaN, let's return 0 or a clear indicator.
            # print("Warning: Spearman correlation resulted in NaN, possibly due to constant input arrays.")
            # If std dev of either array is 0, spearmanr might return nan.
            if np.std(scores_array1) == 0 and np.std(scores_array2) == 0:
                return 1.0, 0.0 # Perfectly correlated in their constancy
            elif np.std(scores_array1) == 0 or np.std(scores_array2) == 0:
                 return 0.0, 1.0 # No rank correlation if one is constant and other varies
            return 0.0, 1.0 # Default for other NaN cases
        return correlation, p_value
    except Exception as e:
        # print(f"Error calculating Spearman correlation: {e}")
        return None, None


def get_top_k_nodes(node_scores_dict, k_value, is_percentage=False):
    """
    Identifies the top k (or top k%) nodes based on their scores.

    Args:
        node_scores_dict (dict): Dictionary of {node_id: score}.
        k_value (int or float): The number of top nodes (if int and not is_percentage)
                                or percentage of top nodes (if float and is_percentage).
        is_percentage (bool): If True, k_value is treated as a percentage (0.0 to 1.0).

    Returns:
        set: A set of node_ids representing the top k nodes.
    """
    if not node_scores_dict:
        return set()

    if is_percentage:
        if not (0.0 <= k_value <= 1.0):
            raise ValueError("k_value as percentage must be between 0.0 and 1.0")
        num_top_nodes = int(np.ceil(len(node_scores_dict) * k_value))
    else:
        num_top_nodes = int(k_value)

    if num_top_nodes <= 0:
        return set()
    
    # Sort nodes by score in descending order
    # Handles cases where scores might be NaN by placing them lower
    sorted_nodes = sorted(node_scores_dict.items(), key=lambda item: (item[1] is not None, item[1]), reverse=True)
    
    top_k_node_ids = {node_id for node_id, score in sorted_nodes[:num_top_nodes]}
    return top_k_node_ids


def calculate_precision_recall_f1_at_k(true_top_k_nodes, predicted_top_k_nodes, all_nodes_list=None):
    """
    Calculates Precision@k, Recall@k, and F1-score@k.

    Args:
        true_top_k_nodes (set): Set of node_ids that are truly in the top-k based on ETMs.
        predicted_top_k_nodes (set): Set of node_ids predicted to be in the top-k by a TRI.
        all_nodes_list (list, optional): A list of all unique node IDs in the graph.
                                         Required if using scikit-learn's precision/recall functions
                                         for binary classification style. If not provided, calculates manually.

    Returns:
        dict: {'precision_at_k': float, 'recall_at_k': float, 'f1_at_k': float}
    """
    if not true_top_k_nodes and not predicted_top_k_nodes:
        return {'precision_at_k': 1.0, 'recall_at_k': 1.0, 'f1_at_k': 1.0} # Both empty, perfect match
    if not true_top_k_nodes: # No true positives possible, recall is 0 or undefined
        return {'precision_at_k': 0.0 if predicted_top_k_nodes else 1.0, 'recall_at_k': 0.0, 'f1_at_k': 0.0}
    if not predicted_top_k_nodes: # No predictions made, precision is 0 or undefined
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0 if true_top_k_nodes else 1.0, 'f1_at_k': 0.0}

    true_positives = len(true_top_k_nodes.intersection(predicted_top_k_nodes))
    
    precision = true_positives / len(predicted_top_k_nodes) if len(predicted_top_k_nodes) > 0 else 0.0
    recall = true_positives / len(true_top_k_nodes) if len(true_top_k_nodes) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {'precision_at_k': precision, 'recall_at_k': recall, 'f1_at_k': f1}


def calculate_imprecision_metric(graph_nodes, etm_scores, tri_scores, k_value, is_percentage=False):
    """
    Calculates an imprecision metric similar to that used in some influence maximization papers.
    Imprecision = 1 - (sum of ETM scores of top-k nodes by TRI) / (sum of ETM scores of true top-k nodes by ETM)
    Or a rank-based variant focusing on rank difference.

    For now, let's implement the ETM score based version.
    Lower imprecision is better.

    Args:
        graph_nodes (list): List of all node IDs in the graph.
        etm_scores (dict): {node_id: etm_score}
        tri_scores (dict): {node_id: tri_score}
        k_value (int or float): The number or percentage for top-k.
        is_percentage (bool): If k_value is a percentage.

    Returns:
        float: Imprecision score. Returns None if denominators are zero or data is insufficient.
    """
    true_top_k_by_etm = get_top_k_nodes(etm_scores, k_value, is_percentage)
    predicted_top_k_by_tri = get_top_k_nodes(tri_scores, k_value, is_percentage)

    if not true_top_k_by_etm:
        # print("Warning: No true top-k nodes by ETM for imprecision calculation.")
        return None # Cannot calculate if true top-k is empty or ETM scores are all same/zero

    sum_etm_of_true_top_k = sum(etm_scores.get(node, 0.0) for node in true_top_k_by_etm)
    sum_etm_of_predicted_top_k = sum(etm_scores.get(node, 0.0) for node in predicted_top_k_by_tri)

    if sum_etm_of_true_top_k == 0:
        # print("Warning: Sum of ETM scores for true top-k is zero. Imprecision cannot be calculated.")
        # This could mean all top ETM scores are 0, or true_top_k_by_etm is empty
        if sum_etm_of_predicted_top_k == 0: # If both are zero, perfect match in terms of score sum
            return 0.0 
        return None # Or a very high value if predicted sum is non-zero

    imprecision = 1.0 - (sum_etm_of_predicted_top_k / sum_etm_of_true_top_k)
    return imprecision

# --- Main execution block (for testing this module independently) ---
if __name__ == '__main__':
    print("Testing evaluation/metrics.py...")

    # Sample scores for testing
    # Assume nodes are 0, 1, 2, 3, 4
    all_nodes = [0, 1, 2, 3, 4]
    etm_scores_sample = {0: 0.8, 1: 0.9, 2: 0.5, 3: 0.7, 4: 0.2} # True top-k should pick 1, 0, 3...
    tri1_scores_sample = {0: 0.7, 1: 0.85, 2: 0.4, 3: 0.75, 4: 0.1} # Good TRI
    tri2_scores_sample = {0: 0.1, 1: 0.2, 2: 0.8, 3: 0.7, 4: 0.9} # Bad TRI (inverse correlation)
    tri3_scores_sample = {0: 0.8, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.2} # Mix

    # For Spearman, convert dicts to lists in the same node order
    ordered_etm = [etm_scores_sample[n] for n in all_nodes]
    ordered_tri1 = [tri1_scores_sample[n] for n in all_nodes]
    ordered_tri2 = [tri2_scores_sample[n] for n in all_nodes]

    # 1. Test Spearman Rank Correlation
    corr1, p1 = calculate_spearman_rank_correlation(ordered_tri1, ordered_etm)
    print(f"\nSpearman Correlation (TRI1 vs ETM): Coeff={corr1:.3f}, P-value={p1:.3f} (Expected: High positive)")
    corr2, p2 = calculate_spearman_rank_correlation(ordered_tri2, ordered_etm)
    print(f"Spearman Correlation (TRI2 vs ETM): Coeff={corr2:.3f}, P-value={p2:.3f} (Expected: High negative or low positive)")

    # Test with constant array
    const_array = [0.5, 0.5, 0.5, 0.5, 0.5]
    corr_const, p_const = calculate_spearman_rank_correlation(ordered_tri1, const_array)
    print(f"Spearman Correlation (TRI1 vs Constant): Coeff={corr_const}, P-value={p_const} (Expected: 0.0 or NaN handled as 0.0)")


    # 2. Test Top-k Node Identification
    k_abs = 2 # Top 2 nodes
    k_perc = 0.4 # Top 40% (i.e., top 2 out of 5)

    true_top_2_nodes = get_top_k_nodes(etm_scores_sample, k_abs)
    print(f"\nTrue Top {k_abs} nodes by ETM: {true_top_2_nodes}") # Expected: {0, 1} (or {1,0}) -> sorted it's 1 then 0

    predicted_top_2_tri1 = get_top_k_nodes(tri1_scores_sample, k_abs)
    print(f"Predicted Top {k_abs} nodes by TRI1: {predicted_top_2_tri1}")

    predicted_top_2_tri2 = get_top_k_nodes(tri2_scores_sample, k_perc, is_percentage=True) # Test percentage
    print(f"Predicted Top {k_perc*100}% nodes by TRI2: {predicted_top_2_tri2}")


    # 3. Test Precision@k, Recall@k, F1-score@k
    print("\nTesting Precision, Recall, F1 @k=2:")
    # True top 2 by ETM are nodes 1 (0.9) and 0 (0.8) -> {0, 1}
    # TRI1 top 2 are nodes 1 (0.85) and 3 (0.75) -> {1, 3}
    # Intersection: {1}. TP = 1.
    # Precision = 1/2 = 0.5. Recall = 1/2 = 0.5. F1 = 0.5.
    metrics_tri1_k2 = calculate_precision_recall_f1_at_k(true_top_2_nodes, predicted_top_2_tri1)
    print(f"TRI1 vs ETM (k=2): Precision={metrics_tri1_k2['precision_at_k']:.2f}, Recall={metrics_tri1_k2['recall_at_k']:.2f}, F1={metrics_tri1_k2['f1_at_k']:.2f}")

    # True top 2 by ETM are {0,1}
    # TRI2 top 40% (k=2) are nodes 4 (0.9) and 2 (0.8) -> {2,4}
    # Intersection: {}. TP = 0.
    # Precision = 0/2 = 0. Recall = 0/2 = 0. F1 = 0.
    metrics_tri2_k2 = calculate_precision_recall_f1_at_k(true_top_2_nodes, predicted_top_2_tri2)
    print(f"TRI2 vs ETM (k=2 from 40%): Precision={metrics_tri2_k2['precision_at_k']:.2f}, Recall={metrics_tri2_k2['recall_at_k']:.2f}, F1={metrics_tri2_k2['f1_at_k']:.2f}")


    # 4. Test Imprecision Metric
    print("\nTesting Imprecision Metric @k=2:")
    imprecision1 = calculate_imprecision_metric(all_nodes, etm_scores_sample, tri1_scores_sample, k_abs)
    # True top 2 ETM: {1:0.9, 0:0.8}. Sum_ETM_true = 1.7
    # Predicted top 2 by TRI1: {1:0.85, 3:0.75}. ETMs for these are {1:0.9, 3:0.7}. Sum_ETM_pred = 1.6
    # Imprecision = 1 - (1.6 / 1.7) = 1 - 0.941 = 0.059
    print(f"Imprecision (TRI1 vs ETM, k=2): {imprecision1:.3f} (Expected low for good TRI)")

    imprecision2 = calculate_imprecision_metric(all_nodes, etm_scores_sample, tri2_scores_sample, k_abs)
    # Predicted top 2 by TRI2: {4:0.9, 2:0.8}. ETMs for these are {4:0.2, 2:0.5}. Sum_ETM_pred = 0.7
    # Imprecision = 1 - (0.7 / 1.7) = 1 - 0.412 = 0.588
    print(f"Imprecision (TRI2 vs ETM, k=2): {imprecision2:.3f} (Expected higher for bad TRI)")

    print("\n--- Evaluation Metrics Test Complete ---")

