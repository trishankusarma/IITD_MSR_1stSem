import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())


def post_prune_bottom_up(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Bottom-up post-pruning:
    - Recurse to children
    - After pruning children, consider pruning this node
    """

    history = []

    def compute_accuracy(model, X, y):
        return np.mean(np.array(model.predict(X)) == y)

    # initial values
    best_val_acc = compute_accuracy(model, X_val, y_val)
    best_train_acc = compute_accuracy(model, X_train, y_train)
    best_test_acc = compute_accuracy(model, X_test, y_test)
    num_nodes = count_nodes(model.tree)

    history.append((num_nodes, best_train_acc, best_val_acc, best_test_acc))
    logging.info(f"Start bottom-up pruning: nodes={num_nodes}, val_acc={best_val_acc:.4f}")

    # BOTTOM-UP PRUNING FUNCTION
    def prune_recursive(node):
        nonlocal best_val_acc, best_train_acc, best_test_acc

        # Leaf → nothing to prune
        if node.is_leaf:
            return

        # 1) FIRST prune all children
        for child in list(node.children.values()):
            prune_recursive(child)

        # 2) THEN evaluate pruning this node
        if node.is_leaf:
            return

        # Save original state
        orig_children = node.children
        orig_is_leaf = node.is_leaf
        orig_pred = node.prediction

        # Temporary prune (convert to leaf)
        node.children = {}
        node.is_leaf = True

        # Compute new validation accuracy
        val_acc_new = compute_accuracy(model, X_val, y_val)

        # If not improved → revert
        if val_acc_new <= best_val_acc + 1e-9:
            node.children = orig_children
            node.is_leaf = orig_is_leaf
            node.prediction = orig_pred
            return

        # Else: keep prune
        best_val_acc = val_acc_new
        best_train_acc = compute_accuracy(model, X_train, y_train)
        best_test_acc = compute_accuracy(model, X_test, y_test)
        num_nodes_after = count_nodes(model.tree)

        logging.info(f"Pruned node → nodes={num_nodes_after}, val_acc={best_val_acc:.4f}")

        history.append((num_nodes_after, best_train_acc, best_val_acc, best_test_acc))


    # Start bottom-up recursion
    prune_recursive(model.tree)

    logging.info(f"Bottom-up pruning complete. Final nodes={count_nodes(model.tree)}, best_val_acc={best_val_acc:.4f}")
    return history
