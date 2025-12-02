# full_pipeline_dt_onehot.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os

class TreeNode:
    def __init__(self, best_feature=None, best_threshold=None, is_leaf=False,
                 children=None, prediction=None, y_values=None):
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.is_leaf = is_leaf
        self.children = children if children is not None else {}
        self.prediction = prediction
        self.y_values = y_values

class DecisionTreeModel:
    def __init__(self, max_depth=3, categorical_cols=None, continuous_cols=None, using_gini_index=False):
        self.max_depth = max_depth
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.continuous_cols = continuous_cols if continuous_cols is not None else []
        self.tree = None
        self.using_gini_index = using_gini_index

    def fit(self, X, y):
        self.tree = self._build_tree(X.copy(), y.copy(), depth=0)

    def predict(self, X):
        return [self._predict_row(row, self.tree) for _, row in X.iterrows()]

    # --------------------------- TREE BUILDING ---------------------------
    def _build_tree(self, X, y, depth):
        prediction = int(np.bincount(y.astype(int)).argmax())

        # Base cases
        if len(set(y)) == 1 or depth == self.max_depth or X.shape[1] == 0 or len(y) == 0:
            return TreeNode(is_leaf=True, prediction=prediction, y_values=y)

        bestInfoGain = -float("inf")
        bestFeature = None
        bestThreshold = None
        bestSplits = None

        for column in X.columns:
            splits = {}
            y_splits = []
            threshold = None

            if column in (self.categorical_cols or []):
                # Categorical splitting (by value)
                for val in X[column].unique():
                    idx = X[column] == val
                    splits[val] = (X[idx].drop(columns=[column]), y[idx])
                    y_splits.append(y[idx])

            else:
                # treat as continuous (median split)
                # if column constant, skip
                col_vals = X[column]
                if col_vals.nunique() == 1:
                    continue
                threshold = col_vals.median()
                left_idx = col_vals <= threshold
                right_idx = col_vals > threshold
                splits["left"] = (X[left_idx], y[left_idx])
                splits["right"] = (X[right_idx], y[right_idx])
                y_splits.append(y[left_idx])
                y_splits.append(y[right_idx])

            # Avoid invalid splits (any empty split)
            if any(len(s) == 0 for s in y_splits):
                # still compute info gain, but skip if degenerate
                pass

            currGain = self.information_gain(y, y_splits)
            if currGain > bestInfoGain:
                bestInfoGain = currGain
                bestFeature = column
                bestThreshold = threshold
                bestSplits = splits

        if bestFeature is None or bestSplits is None:
            return TreeNode(is_leaf=True, prediction=prediction, y_values=y)

        children = {}
        for key, (X_child, y_child) in bestSplits.items():
            children[key] = self._build_tree(X_child.copy(), y_child.copy(), depth + 1)

        return TreeNode(best_feature=bestFeature,
                        best_threshold=bestThreshold,
                        is_leaf=False,
                        children=children,
                        prediction=prediction,
                        y_values=y)

    # --------------------------- PREDICTION ---------------------------
    def _predict_row(self, row, node):
        if node is None:
            return 0
        if node.is_leaf:
            return node.prediction

        feature = node.best_feature
        if feature in (self.categorical_cols or []):
            value = row.get(feature, None)
            if value not in node.children:
                return node.prediction
            return self._predict_row(row, node.children[value])
        else:
            value = row.get(feature, None)
            # if missing or children not as expected, return node prediction
            if value is None:
                return node.prediction
            # continuous branch expected "left"/"right"
            if "left" not in node.children or "right" not in node.children:
                return node.prediction
            if value <= node.best_threshold:
                return self._predict_row(row, node.children["left"])
            else:
                return self._predict_row(row, node.children["right"])

    # --------------------------- METRICS ---------------------------
    def entropy(self, y):
        if len(y) == 0:
            return 0.0
        values, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(np.where(probs > 0, probs * np.log2(probs), 0.0))

    def gini_index(self, y):
        if len(y) == 0:
            return 0.0
        values, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def information_gain(self, y, y_splits):
        if len(y) == 0 or len(y_splits) == 0:
            return 0.0
        if self.using_gini_index:
            H_before = self.gini_index(y)
            H_after = sum((len(split) / len(y)) * self.gini_index(split) for split in y_splits)
        else:
            H_before = self.entropy(y)
            H_after = sum((len(split) / len(y)) * self.entropy(split) for split in y_splits)
        return H_before - H_after

    # --------------------------- PRUNING ---------------------------
    def get_prunable_nodes(self, node=None):
        if node is None:
            node = self.tree
        nodes = []
        if node is None:
            return nodes
        if not node.is_leaf:
            nodes.append(node)
            for child in node.children.values():
                nodes.extend(self.get_prunable_nodes(child))
        return nodes

    def prune_node(self, node):
        node.children = {}
        node.is_leaf = True
        if node.y_values is not None and len(node.y_values) > 0:
            node.prediction = int(np.bincount(node.y_values.astype(int)).argmax())

    def count_nodes(self, node=None):
        if node is None:
            node = self.tree
        if node is None:
            return 0
        total = 1
        for child in node.children.values():
            total += self.count_nodes(child)
        return total

    # --------------------------- PRINT TREE ---------------------------
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node is None:
            print("Empty tree")
            return
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf → predicts {node.prediction} (n={len(node.y_values) if node.y_values is not None else 0})")
            return
        if node.best_feature in (self.categorical_cols or []):
            print(f"{indent}Node → {node.best_feature}")
        else:
            print(f"{indent}Node → {node.best_feature} <= {node.best_threshold}")
        for key, child in node.children.items():
            print(f"{indent}  [{key}]")
            self.print_tree(child, depth + 2)