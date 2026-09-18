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
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
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
    mean , std = X.mean(axis = 0), np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    return (X - mean)/std

# Step 5 - add_bias_column
def add_bias_column(X):
    return np.hstack((np.ones((X.shape[0],1)),X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X = standardize_features(X, mean, std)
    return add_bias_column(X)

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
    return np.mean((y_pred - y_true)**2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # mse = 1/n * ||Y - Xw||^2
    # Grad(mse) = 1/n * 2 * (Y - Xw).T * X
    return 2/len(y_true) * (y_pred - y_true).T @ X

# Step 10 - normal_equation
def normal_equation(X, y):
    # The normal equation is X.T X w = X.T y
    A = X.T @ X
    
    # Form the right-hand side b = X.T @ y
    b = X.T @ y
    
    # Solve the linear system A w = b
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(X.T @ X) @ X.T @ y

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    np.random.seed(seed)
    return np.random.normal(0,0.01,n_features)

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

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    y_pred_train, y_pred_val = predict_linear(X_train, weights),  predict_linear(X_val, weights)
    return mse_loss(y_train, y_pred_train), mse_loss(y_val, y_pred_val)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = weights
        wait = 0
    
    else:
        wait += 1

    return (best_val_loss, wait, best_weights, (wait >= patience))

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features, seed)
    return { 'weights' : weights,
             'best_weights' : weights.copy(),
             'best_val_loss' : np.inf,
             'wait' : 0,
             'train_losses' : [],
             'val_losses' : [],
             'stopped' : False}

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    weights = state['weights']
    best_weights = state['best_weights']
    best_val_loss = state['best_val_loss']
    wait = state['wait']

    weights = gd_step(X_train, y_train, weights, lr)
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, weights)
    best_val_loss, wait, best_weights, stopped = update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience)
    
    state['weights'] = weights
    state['best_weights'] = best_weights
    state['best_val_loss'] = best_val_loss
    state['wait'] = wait
    state['stopped'] = stopped
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)
    
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    state = init_training_state(X_train.shape[1], seed)

    for epoch in range(epochs):
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state['stopped']:
            break

    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mse_loss(y_true,y_pred))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    SSres = sum( (y_true - y_pred) ** 2 )
    SStot = sum( (y_true - y_true.mean() ) ** 2 )
    return 1 - SSres/SStot if SStot != 0 else float('nan')

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    return { 'mae' : mean_absolute_error(y_true, y_pred),
             'rmse' : root_mean_squared_error(y_true, y_pred),
             'r2' : r_squared(y_true, y_pred) }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    epochs = list(range(1, len(train_losses)+1))
    train_list = np.array(train_losses).tolist()
    val_list = np.array(val_losses).tolist()
    return epochs, train_list, val_list

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    return np.linalg.norm(w_gd-w_closed)

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    return {
        'learning_rate' : learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    # Read Hyperparams
    lr, epochs, patience, seed = model['learning_rate'], model['epochs'], model['patience'], model['seed']

    # Compute stats of the train set
    mean, std = compute_feature_stats(X_train)

    # Normalize the train and val sets
    X_train_design, X_val_design = prepare_design_matrix(X_train, mean, std), prepare_design_matrix(X_val, mean, std)

    # Train the model with gd
    weights, train_losses, val_losses = train_batch_gd(X_train_design, y_train, X_val_design, y_val, lr, epochs, patience, seed)

    # Get the weights from normal equations
    normal_weights = normal_equation(X_train_design, y_train)

    # Write in the statistics
    model['mean'] = mean
    model['std'] = std
    model['weights'] = weights
    model['normal_weights'] = normal_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses

    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    
    mean, std = model['mean'],model['std']
    weights = model['weights']

    X_design = prepare_design_matrix(X, mean, std)
    y_pred = predict_linear(X_design, weights)

    return y_pred

# Step 27 - score_lr_model
import numpy as np
def score_lr_model(model, X, y):
    y_pred = predict_lr_model(model, X)
    return evaluate_regression(y, y_pred)

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

