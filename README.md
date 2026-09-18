# NumPy Multiple Linear Regression GD

Build a from-scratch multiple linear regression trainer in pure NumPy: standardize features, minimize MSE with batch gradient descent and early stopping, compare against the normal equation, and report MAE, RMSE, and R-squared via a reusable model API.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** shuffle_xy
- [x] **2.** split_train_val_test
- [x] **3.** compute_feature_stats
- [x] **4.** standardize_features
- [x] **5.** add_bias_column
- [x] **6.** prepare_design_matrix
- [x] **7.** predict_linear
- [x] **8.** mse_loss
- [x] **9.** mse_gradient
- [x] **10.** normal_equation
- [x] **11.** initialize_weights
- [x] **12.** gd_step
- [x] **13.** epoch_train_val_losses
- [x] **14.** update_early_stop_state
- [x] **15.** init_training_state
- [x] **16.** run_one_epoch
- [x] **17.** train_batch_gd
- [x] **18.** mean_absolute_error
- [x] **19.** root_mean_squared_error
- [x] **20.** r_squared
- [x] **21.** evaluate_regression
- [ ] **22.** learning_curve_data
- [ ] **23.** weights_l2_distance
- [ ] **24.** create_lr_model
- [ ] **25.** fit_lr_model
- [ ] **26.** predict_lr_model
- [ ] **27.** score_lr_model
- [ ] **28.** compare_with_normal_equation

---

Built on Deep-ML.
