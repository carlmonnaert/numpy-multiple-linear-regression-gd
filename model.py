"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    if len(X)!=len(y):
        return None
        
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    N_train = int( len(X) * train_frac )
    N_val = int( len(X) * val_frac )
    N_test = len(X) - N_val - N_train

    train_idx = [ i for i in range(N_train) ]
    val_idx = [ i for i in range(N_train, N_train + N_val)]
    test_idx = [ i for i in range(N_train + N_val, N_train + N_val + N_test)]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    return [ X[:,i].mean() for i in range(X.shape[1])], [ X[:,i].std() if X[:,i].std() != 0 else 1.0 for i in range(X.shape[1])]

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    return (X - mean)/std

# Step 5 - add_bias_column
def add_bias_column(X):
    return np.hstack((np.ones((X.shape[0],1)),X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X = standardize_features(X, mean, std)
    return np.hstack( (np.ones((X.shape[0],1)) , X) )

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    return 1/len(y_true) * np.linalg.norm(y_pred - y_true,2)**2

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # mse = 1/n * ||Y - Xw||^2
    # Grad(mse) = 1/n * 2 * (Y - Xw).T * X
    return 2/len(y_true) * (y_pred - y_true).T @ X

# Step 10 - normal_equation
def normal_equation(X, y):
    # The normal equation is X.T X w = X.T y
    # This is because y = X w + noise
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(0,0.01,n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    y_pred = predict_linear(X,weights)
    grad = mse_gradient(X, y, y_pred)
    weights -= lr * grad
    return weights

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

