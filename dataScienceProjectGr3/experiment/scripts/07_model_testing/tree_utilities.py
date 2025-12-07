"""
Utility functions for Decision Tree visualization and analysis.
"""

import matplotlib
# Ensure backend is set before importing pyplot; prefer TkAgg, fallback to Agg for headless environments
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd


def get_decision_path(clf, feature_names, node_id):
    """Get the decision path (rules) leading to a specific node."""
    path = []
    tree_ = clf.tree_

    def recurse(current_node=0, decisions=None):
        if decisions is None:
            decisions = []
        if current_node == node_id:
            path.extend(decisions)
            return True
        if tree_.children_left[current_node] != -1:
            # Left child path
            if recurse(
                tree_.children_left[current_node],
                decisions + [f"{feature_names[tree_.feature[current_node]]} <= {tree_.threshold[current_node]:.4f}"]
            ):
                return True
            # Right child path
            if recurse(
                tree_.children_right[current_node],
                decisions + [f"{feature_names[tree_.feature[current_node]]} > {tree_.threshold[current_node]:.4f}"]
            ):
                return True
        return False

    recurse()
    return path


def plot_tree(clf, feature_cols, path=None):
    """Plot the decision tree and save to file."""
    fig = plt.figure(figsize=(200, 160))
    _ = tree.plot_tree(clf, feature_names=feature_cols, class_names=['0', '1'], filled=True)
    fig.savefig(path)
    plt.close(fig)


def get_tree_stats(clf, feature_cols):
    """Extract per-node statistics from the decision tree."""
    tree_ = clf.tree_

    # Create DataFrame with node information
    tree_df = pd.DataFrame({
        "node_id": range(tree_.node_count),
        "left_child": tree_.children_left,
        "right_child": tree_.children_right,
        "feature": tree_.feature,
        "threshold": tree_.threshold,
        "impurity": tree_.impurity,
        "n_node_samples": tree_.n_node_samples,
        "weighted_n_node_samples": tree_.weighted_n_node_samples,
        "value": [tree_.value[i][0].tolist() for i in range(tree_.node_count)],
        "class": [clf.classes_[v.argmax()] for v in tree_.value]
    })

    return tree_df


def apply_decision_rules(df, rules):
    """Apply decision rules to filter a DataFrame."""
    filtered_df = df.copy()
    for rule in rules:
        # Parse rule: "feature_name <= threshold" or "feature_name > threshold"
        if " <= " in rule:
            feature, threshold = rule.split(" <= ")
            threshold = float(threshold)
            filtered_df = filtered_df[filtered_df[feature] <= threshold]
        elif " > " in rule:
            feature, threshold = rule.split(" > ")
            threshold = float(threshold)
            filtered_df = filtered_df[filtered_df[feature] > threshold]
    return filtered_df

