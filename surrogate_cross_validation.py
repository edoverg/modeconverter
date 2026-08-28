'''Validates the surrogate model (Kriging) by evaluating it on an unseen test set.
    Computes R² score and creates diagnostic plots.
    
    Args:
        design_space: SMT DesignSpace object
        xdoe_train: training input samples (N_train, n_dims)
        ydoe_train: training output values (N_train, 1)
        xdoe_test: test input samples (N_test, n_dims)
        ydoe_test: test output values (N_test, 1)
        save_folder: folder to save plots
    Returns:
        Dictionary with metrics (r2_score, rmse, mae, sm)
    '''
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from smt.surrogate_models import KRG
from smt.design_space import DesignSpace

seed = 777
np.random.seed(seed)
xlimits = np.array([[100,700], [100,700]])
design_space = DesignSpace(xlimits, seed=seed)
results_folder = "results_symm_ego"
doe = np.loadtxt(f'{results_folder}/ego_doe.txt')
ndoe = doe.shape[0]
xdoe = doe[:, :2]
ydoe = doe[:, 2].reshape(-1, 1)
train_size = int(ndoe * 0.7)
xdoe_train = xdoe[:train_size].reshape(-1,2)
ydoe_train = ydoe[:train_size].reshape(-1,1)
xdoe_test = xdoe[train_size:].reshape(-1,2)
ydoe_test = ydoe[train_size:].reshape(-1,1)
doe_additional = np.loadtxt(f'{results_folder}/ego_additional.txt')
xdoe_additional = doe_additional[:, :2]
ydoe_additional = doe_additional[:, 2].reshape(-1, 1)

#ego optimal
optimum = np.loadtxt(f'{results_folder}/ego_opt_results.txt')
d0_opt = optimum[0]
d1_opt = optimum[1]
opt_val = optimum[2]
# Train the surrogate model
print("\n" + "="*60)
print("SURROGATE MODEL VALIDATION")
print("="*60)
print(f"Training set size: {xdoe_train.shape[0]} samples")
print(f"Test set size: {xdoe_test.shape[0]} samples")

xdoe_train_ref = np.vstack((xdoe_train, xdoe_additional))
ydoe_train_ref = np.vstack((ydoe_train, ydoe_additional))
sm = KRG(design_space=design_space, n_start=25, print_global=False)
sm.set_training_values(xdoe_train_ref, ydoe_train_ref)
sm.train()

# Predict on both sets
y_train_pred = sm.predict_values(xdoe_train)
y_test_pred = sm.predict_values(xdoe_test)
y_additional_pred = sm.predict_values(xdoe_additional)
y_train_ref_pred = sm.predict_values(xdoe_train_ref)
# Compute metrics
r2_train_ref = r2_score(ydoe_train_ref, y_train_ref_pred)
r2_test = r2_score(ydoe_test, y_test_pred)
rmse_train_ref = np.sqrt(mean_squared_error(ydoe_train_ref, y_train_ref_pred))
rmse_test = np.sqrt(mean_squared_error(ydoe_test, y_test_pred))
mae_train_ref = mean_absolute_error(ydoe_train_ref, y_train_ref_pred)
mae_test = mean_absolute_error(ydoe_test, y_test_pred)

print(f"\nTraining Set Metrics:")
print(f"  R² Score: {r2_train_ref:.4f}")
print(f"  RMSE:     {rmse_train_ref:.6e}")
print(f"  MAE:      {mae_train_ref:.6e}")

print(f"\nTest Set Metrics:")
print(f"  R² Score: {r2_test:.4f}")
print(f"  RMSE:     {rmse_test:.6e}")
print(f"  MAE:      {mae_test:.6e}")

if r2_test < 0.80:
    print("\n  WARNING: Test R² < 0.80. Consider adding more DOE samples.")
elif r2_test < 0.85:
    print("\n  CAUTION: Test R² < 0.85. Model quality is moderate.")
else:
    print("\n Model quality is good. Ready to proceed with EGO.")

print("="*60 + "\n")

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Predicted vs Actual (Training and Test)
ax = axes[0, 0]
ax.scatter(ydoe_train, y_train_pred, alpha=0.6, label='Training', s=50, color='C0')
ax.scatter(ydoe_test, y_test_pred, alpha=0.6, label='Test', s=50, marker='^', color='C1')
ax.scatter(ydoe_additional, y_additional_pred, alpha=0.6, label='Additional', s=50, marker='s', color='C2')
y_min, y_max = min(ydoe_train.min(), ydoe_test.min(), ydoe_additional.min()), max(ydoe_train.max(), ydoe_test.max(), ydoe_additional.max())
ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Output', fontsize=20)
ax.set_ylabel('Predicted Output', fontsize=20)
ax.set_title('Predicted vs Actual Values', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=16)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='top')

# Plot 2: Residuals (Test Set)
ax = axes[0, 1]
residuals = ydoe_test.flatten() - y_test_pred.flatten()
ax.scatter(y_test_pred, residuals, alpha=0.6, s=50, marker='^', color='C1')
ax.axhline(y=0, color='k', linestyle='--', lw=2)
ax.set_xlabel('Predicted Output', fontsize=20)
ax.set_ylabel('Residuals', fontsize=20)
ax.set_title('Residuals vs Predicted Values (Test Set)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='top')

# Plot 3: Design Space Coverage
ax = axes[1, 0]
ax.scatter(xdoe_train[:, 0], xdoe_train[:, 1], alpha=0.6, s=50, label='Training', color='C0')
ax.scatter(xdoe_test[:, 0], xdoe_test[:, 1], alpha=0.6, s=50, marker='^', label='Test', color='C1')
ax.scatter(xdoe_additional[:, 0], xdoe_additional[:, 1], alpha=0.6, s=50, marker='s', label='Additional', color='C2')
ax.scatter(d0_opt, d1_opt, color='red', s=100, marker='*', label='EGO Optimum')
ax.set_xlabel('d0 [nm]', fontsize=20)
ax.set_ylabel('d1 [nm]', fontsize=20)
ax.set_title('Design Space Coverage', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=16, framealpha=0.5, loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.98, '(c)', transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='top', horizontalalignment='right')

# Plot 4: R² Score Comparison
ax = axes[1, 1]
models = ['Training (Ref)', 'Test']
r2_scores = [r2_train_ref, r2_test]
colors = ['green' if r2 > 0.85 else 'orange' if r2 > 0.80 else 'red' for r2 in r2_scores]
bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2,)
#ax.axhline(y=0.80, color='orange', linestyle='--', linewidth=2, label='Acceptable (0.80)')
ax.set_ylabel('R² Score', fontsize=20)
ax.set_title('Model Performance', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.set_ylim([0, 1])
#ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=20, fontweight='bold', verticalalignment='top')
# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height-0.1,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95, hspace=0.35, wspace=0.3)
plt.savefig(f'{results_folder}/surrogate_model_validation.pdf', dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Create a grid for plotting the surrogate model surface
# predictions are very cheap thanks to the model)
x1 = np.linspace(xlimits[0, 0], xlimits[0, 1], 50)
x2 = np.linspace(xlimits[1, 0], xlimits[1, 1], 50)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack((X1.flatten(), X2.flatten()))
y_grid_pred = sm.predict_values(X_grid).reshape(X1.shape)
surf = ax.plot_surface(X1, X2, y_grid_pred, cmap='viridis', alpha=0.7, edgecolor='none')
# Plot training points
ax.scatter(xdoe_additional[:, 0], xdoe_additional[:, 1], ydoe_additional.flatten(), color='blue', marker='o', label='Additional', s=50)
# Plot only the original training points (without the EGO additional)
ax.scatter(xdoe_train[:, 0], xdoe_train[:, 1], ydoe_train.flatten(), color='green', label='Train', s=50, marker='d')
# Plot test points
ax.scatter(xdoe_test[:, 0], xdoe_test[:, 1], ydoe_test.flatten(), color='red', label='Test', s=50, marker='^')
# Highlight the optimal point found by EGO
ax.scatter(d0_opt, d1_opt, opt_val, color='gold', label='EGO Optimum', s=100, edgecolor='black', marker='*')
ax.set_xlabel('d0 [nm]', fontsize=20, labelpad=15)
ax.set_ylabel('d1 [nm]', fontsize=20, labelpad=15)
ax.set_zlabel('Cost Function Value', fontsize=20, labelpad=15)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=16)
plt.savefig(f'{results_folder}/surrogate_model_surface_final.pdf', dpi=300, bbox_inches='tight')
plt.close()

#add a level set curve to better visualize the surface of the surrogate model
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
# Create a grid for plotting the surrogate model surface
x1 = np.linspace(xlimits[0, 0], xlimits[0, 1], 100)
x2 = np.linspace(xlimits[1, 0], xlimits[1, 1], 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack((X1.flatten(), X2.flatten()))
y_grid_pred = sm.predict_values(X_grid).reshape(X1.shape)
# Plot the contour of the surrogate model
contour = ax.contourf(X1, X2, y_grid_pred, levels=50, cmap='viridis', alpha=0.8)
# Plot training points
ax.scatter(xdoe_additional[:, 0], xdoe_additional[:, 1], color='blue', marker='o', label='Additional', s=50)
# Plot only the original training points (without the EGO additional)
ax.scatter(xdoe_train[:, 0], xdoe_train[:, 1], color='green', label='Train', s=50, marker='d')
# Plot test points
ax.scatter(xdoe_test[:, 0], xdoe_test[:, 1], color='red', label='Test', s=50, marker='^')
# Highlight the optimal point found by EGO
ax.scatter(d0_opt, d1_opt, color='gold', label='EGO Optimum', s=100, edgecolor='black', marker='*')
ax.set_xlabel('d0 [nm]', fontsize=20)
ax.set_ylabel('d1 [nm]', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
cbar = plt.colorbar(contour, label='Cost Function Value')
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Cost Function Value', fontsize=20)
ax.legend(fontsize=16)
plt.savefig(f'{results_folder}/surrogate_model_contour_final.pdf', dpi=300, bbox_inches='tight')
plt.close() 


#let us now consider a different approach to cross validation and work on k-fold cross validation.
#Consider the whole DOE dataset (40 points). We split the initial dataset into 10 folds.
#Each fold has 4 points. We will train the surrogate model on 9 folds and test it on the remaining fold. We will repeat this process for all 10 folds and compute the average R² score, RMSE, and MAE across all folds.
#We make use of the KFold class from sklearn.model_selection to perform the k-fold cross validation.
#We select the best surrogate model based on the generalization error: the generalization error
#is computed in the following way: given a surrogate model trained with the 9 folds, we test it with the 
#remaining fold and compute the error as (predicted-true)^2. We then take the average error
#across the elements of the left-out fold.
from sklearn.model_selection import KFold

full_x_doe = doe[:, :2]
full_y_doe = doe[:, 2].reshape(-1, 1)
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

surrogates = []
folds_train_xdoe = []
folds_train_ydoe = []
avg_errs = []
for i, (train_index, test_index) in enumerate(kf.split(full_x_doe)):
    print(f"Fold {i}")
    print(f" Train: index={train_index}")
    print(f" Test: index={test_index}")
    train_values = full_x_doe[train_index]
    test_values = full_x_doe[test_index]
    train_outputs = full_y_doe[train_index]
    test_outputs = full_y_doe[test_index]
    sm_fold = KRG(design_space=design_space, n_start=25, print_global=False)
    sm_fold.set_training_values(train_values, train_outputs)
    sm_fold.train()
    test_pred_values = sm_fold.predict_values(test_values)
    avg_sq_err = np.mean((test_pred_values - test_outputs)**2)
    print(f" Average squared error for fold {i}: {avg_sq_err:.6e}")
    surrogates.append(sm_fold)
    avg_errs.append(avg_sq_err)
    #save the training data for this fold for future use
    folds_train_xdoe.append(train_values)
    folds_train_ydoe.append(train_outputs)
    #make a plot of the predicted vs actual values for the test set of this fold
    plt.figure(figsize=(10, 6))
    plt.scatter(test_outputs, test_pred_values, alpha=0.6, s=50, color='C0')
    y_min, y_max = min(test_outputs.min(), test_pred_values.min()), max(test_outputs.max(), test_pred_values.max())
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Output', fontsize=20)
    plt.ylabel('Predicted Output', fontsize=20)
    plt.title(f'Predicted vs Actual Values for Fold {i}', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{results_folder}/kfold_pred_vs_actual_fold_{i}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

#select the model with the lowest average squared error
best_model_index = np.argmin(avg_errs)
best_model = surrogates[best_model_index]
#make a plot of the average squared error for each fold
plt.figure(figsize=(10, 6))
plt.bar(range(n_splits), avg_errs, color='skyblue', edgecolor='black')
plt.xlabel('Fold Index', fontsize=20)
plt.ylabel('Average Squared Error', fontsize=20)
plt.title('Average Squared Error for Each Fold', fontsize=20)
plt.xticks(range(n_splits), [f'Fold {i}' for i in range(n_splits)], fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{results_folder}/kfold_avg_squared_error.pdf', dpi=300, bbox_inches='tight')
plt.close()

#save the model for future use and the corresponding training and result data for the best fold
import pickle
with open(f'{results_folder}/best_surrogate_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open(f'{results_folder}/best_surrogate_model_training_data.pkl', 'wb') as f:
    pickle.dump((folds_train_xdoe[best_model_index], folds_train_ydoe[best_model_index]), f)

#make a surface plot of the best surrogate model
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Create a grid for plotting the surrogate model surface
x1 = np.linspace(xlimits[0, 0], xlimits[0, 1], 50)
x2 = np.linspace(xlimits[1, 0], xlimits[1, 1], 50)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack((X1.flatten(), X2.flatten()))
y_grid_pred = best_model.predict_values(X_grid).reshape(X1.shape)
surf = ax.plot_surface(X1, X2, y_grid_pred, cmap='viridis', alpha=0.7, edgecolor='none')
# Plot the full doe training points
ax.scatter(full_x_doe[:, 0], full_x_doe[:, 1], full_y_doe.flatten(), color='blue', marker='o', label='DOE Points', s=50)
ax.set_xlabel('d0 [nm]', fontsize=20, labelpad=15)
ax.set_ylabel('d1 [nm]', fontsize=20, labelpad=15)
ax.set_zlabel('Cost Function Value', fontsize=20, labelpad=15)
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=16)
plt.savefig(f'{results_folder}/best_surrogate_model_surface.pdf', dpi=300, bbox_inches='tight')
plt.close()