"""
NumPy Multiple Linear Regression GD scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Demo: from-scratch multiple linear regression with batch GD in NumPy."""
import numpy as np


def main():
    np.random.seed(0)
    n_samples, n_features = 150, 3
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -2.0, 0.5])
    y = X @ true_weights + 0.3 + 0.1 * np.random.randn(n_samples)

    X, y = shuffle_xy(X, y, seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, train_frac=0.6, val_frac=0.2
    )
    print("Splits:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    model = create_lr_model(learning_rate=0.05, epochs=400, patience=25, seed=0)
    model = fit_lr_model(model, X_train, y_train, X_val, y_val)

    y_hat = predict_lr_model(model, X_test[:5])
    print("Sample preds:", np.round(y_hat, 4))
    print("Sample trues:", np.round(y_test[:5], 4))

    metrics = score_lr_model(model, X_test, y_test)
    print("Test MAE/RMSE/R2:", metrics)

    gap = compare_with_normal_equation(model)
    print("GD vs normal-eq L2 gap:", float(gap))

    train_losses = model.get("train_losses", [])
    val_losses = model.get("val_losses", [])
    if len(train_losses) > 0:
        epochs, tr, va = learning_curve_data(train_losses, val_losses)
        print("Final train/val MSE:", float(tr[-1]), float(va[-1]))
        print("Epochs run:", int(epochs[-1]) + 1 if len(epochs) else 0)


if __name__ == "__main__":
    main()
