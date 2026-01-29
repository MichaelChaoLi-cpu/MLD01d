# %% [markdown]
# # Check Explanation of the relationship

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# %%
import ExplainResult
import Modelling
import SettingForFeatures
import TestingTools

# %%
import importlib
importlib.reload(Modelling)


# %%

# %% [markdown]
# ## Functions

# %%
def json_serializable(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Type not serializable: {type(obj)}")


# %%
def compute_single_pdp_self_defined(
    var: str,
    X: pd.DataFrame,
    model_list: list,
    range_boundary=(0.05, 0.95),
    stripe: float = 0.2
) -> tuple:
    # Determine the grid boundaries based on the specified quantiles
    low_b = range_boundary[0]
    up_b = range_boundary[1]
    
    # Generate the discrete grid of feature values
    potenital_values = np.arange(low_b, up_b, stripe)
    X_adjust = X.copy()
    
    pdp_list = []
    
    # Iterate through each model in the ensemble/list
    for model in model_list:
        pdp = np.full_like(potenital_values, fill_value=np.nan)
        
        # Iterate through each grid point (potential value)
        for idx, potenital_value in enumerate(potenital_values):
            # 1. Substitute the feature column with the current fixed value
            X_adjust[var] = potenital_value.astype(float)
            
            # 2. Predict the outcome for the entire adjusted dataset
            y_pred = model.predict_proba(X_adjust)[:,1]
            
            # 3. Calculate the partial dependence (average prediction)
            pdp[idx] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    pdp_array = np.array(pdp_list)

    return potenital_values, pdp_array


# %%

# %% [markdown]
# ## Runs

# %%
if __name__ == '__main__':
    pass

# %%
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.chdir(os.getenv("PROJECT_ROOT"))

# %%
all_data = SettingForFeatures.data_load_combine_dataset()

# %%
always_inputs = SettingForFeatures.return_input_variables()

# %%
aim_variables = SettingForFeatures.return_output_variables()

# %%
n_splits = 10

# %%

# %% [markdown]
# ### Test

# %%
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    with open(f"./{aim_variable}_params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    print(params)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_list = []
    for train_idx, test_idx in kf.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
    
        # Predict
        y_pred = model.predict(X_test)
    
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
    
        print(accuracy, f1, precision, recall)
        r2_list.append(accuracy)

    r2_ols_list = []
    for train_idx, test_idx in kf.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = LogisticRegression()
        model.fit(X_train, y_train)
    
        # Predict
        y_pred = model.predict(X_test)
    
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
    
        print(accuracy, f1, precision, recall)
        r2_ols_list.append(accuracy)

    accuracy_comparison = params
    accuracy_comparison['XGBoost Mean Accuracy'] = np.mean(r2_list).astype(float)
    accuracy_comparison['XGBoost SD Accuracy'] = np.std(r2_list).astype(float)
    accuracy_comparison['XGBoost Min Accuracy'] = np.min(r2_list).astype(float)
    accuracy_comparison['XGBoost Max Accuracy'] = np.max(r2_list).astype(float)

    accuracy_comparison['Logistic Mean Accuracy'] = np.mean(r2_ols_list).astype(float)
    accuracy_comparison['Logistic SD Accuracy'] = np.std(r2_ols_list).astype(float)
    accuracy_comparison['Logistic Min Accuracy'] = np.min(r2_ols_list).astype(float)
    accuracy_comparison['Logistic Max Accuracy'] = np.max(r2_ols_list).astype(float)

    save_path = os.path.join('results', f'{aim_variable}_accuracy_comparison.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(
            accuracy_comparison,
            f,
            default=json_serializable,
            ensure_ascii=False,
            indent=4
        )

# %%

# %% [markdown]
# ### Check Importance

# %%
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    with open(f"./{aim_variable}_params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    print(params)

    feature_importance_df_list = []
    fold = 1
    
    for train_idx, test_idx in kf.split(X):
    
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
    
        # Metrics
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        print(f"Fold {fold}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}")
    
        # Importance
        booster = model.get_booster()
        score = booster.get_score(importance_type='gain')
    
        # Convert to aligned DF: feature as index, importance as column
        df_imp = pd.DataFrame(score, index=[0]).T
        df_imp.columns = [f"fold_{fold}"]
    
        feature_importance_df_list.append(df_imp)
        
        fold += 1
    
    # 🔥 Merge all folds by feature name
    feature_importance_full = pd.concat(feature_importance_df_list, axis=1).fillna(0)
    
    feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
    feature_importance_full = feature_importance_full.sort_values('mean_importance')
    
    # Plot
    plt.figure(figsize=(10, 15))
    plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain Importance")
    plt.title("XGBoost Feature Importance")
    plt.show()

    feature_importance_full.to_csv(os.path.join('results', f'{aim_variable}_importance.csv'))

# %%

# %% [markdown]
# ### Check PDP

# %% [markdown]
# #### Health indicator

# %%
for aim_variable in aim_variables:
    print(aim_variable)
    break

# %%
with open(f"./{aim_variable}_params.yaml", "r") as f:
    params = yaml.safe_load(f)

print(params)

# %%
aim_variable = aim_variables[0]

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%
model_list = Modelling.get_clsmodel_list(
    X, y,
    n_splits = n_splits,
    params = params,
)

# %%
X.columns

# %%
X['DisasterExpInd'].describe()

# %%
potenital_values, pdp_array = compute_single_pdp_self_defined(
    var = 'DisasterExpInd',
    X = X,
    model_list = model_list,
    range_boundary = (0.0, 11.0),
    stripe = 1.0
)

# %%
pdp_mean = np.mean(pdp_array, axis = 0)
pdp_std = np.std(pdp_array, axis = 0)

# 1. Plot the mean PDP line (same as before)
plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")

# 2. Add the Error/Confidence Band using fill_between
# The band represents [mean - std] to [mean + std]
plt.fill_between(
    potenital_values, 
    pdp_mean - pdp_std * 1.96,  # Lower bound
    pdp_mean + pdp_std * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
    label="$\pm 1.96 \sigma$" # Label for the legend
)

# Optional: Add labels and grid
plt.xlabel("Natural Disaster Count")
plt.ylabel("Predicted Disease Increase Probability")
plt.grid(True)
plt.legend()

#plt.savefig(os.path.join(FIGURES, f'fig06_{aim_variable}_PDP_{varname.upper()}_STD.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# #### Knowledge

# %% [markdown]
# #### Naive PDP

# %%
# Determine the grid boundaries based on the specified quantiles
low_b = 0.0
up_b = 11.0
stripe = 1.0
var = 'DisasterExpInd'

# Generate the discrete grid of feature values
potenital_values = np.arange(low_b, up_b, stripe)
X_adjust = X.copy()

pdp_list = []

# Iterate through each model in the ensemble/list
for model in model_list:
    pdp = np.full_like(potenital_values, fill_value=np.nan)
    
    # Iterate through each grid point (potential value)
    for idx, potenital_value in enumerate(potenital_values):
        # 1. Substitute the feature column with the current fixed value
        X_adjust[var] = potenital_value.astype(float)
        
        # 2. Predict the outcome for the entire adjusted dataset
        y_pred = model.predict_proba(X_adjust)[:,1]
        
        # 3. Calculate the partial dependence (average prediction)
        pdp[idx] = np.mean(y_pred)
    
    pdp_list.append(pdp)

pdp_array = np.array(pdp_list)

# %%
pdp_mean = np.mean(pdp_array, axis = 0)
pdp_std = np.std(pdp_array, axis = 0)

# 1. Plot the mean PDP line (same as before)
plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")

# 2. Add the Error/Confidence Band using fill_between
# The band represents [mean - std] to [mean + std]
plt.fill_between(
    potenital_values, 
    pdp_mean - pdp_std * 1.96,  # Lower bound
    pdp_mean + pdp_std * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
    label="$\pm 1.96 \sigma$" # Label for the legend
)

# Optional: Add labels and grid
plt.xlabel("Natural Disaster Count")
plt.ylabel("Predicted Disease Increase Probability")
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(FIGURES, f'fig05_naive_PDP.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# #### without know

# %%
# Determine the grid boundaries based on the specified quantiles
low_b = 0.0
up_b = 11.0
stripe = 1.0
var = 'DisasterExpInd'

# Generate the discrete grid of feature values
potenital_values = np.arange(low_b, up_b, stripe)
X_adjust = X.copy()

pdp_list = []

# Iterate through each model in the ensemble/list
for model in model_list:
    pdp = np.full_like(potenital_values, fill_value=np.nan)
    
    # Iterate through each grid point (potential value)
    for idx, potenital_value in enumerate(potenital_values):
        # 1. Substitute the feature column with the current fixed value
        X_adjust[var] = potenital_value.astype(float)
        X_adjust['HeardClimate_Dummy'] = 0.0
        
        # 2. Predict the outcome for the entire adjusted dataset
        y_pred = model.predict_proba(X_adjust)[:,1]
        
        # 3. Calculate the partial dependence (average prediction)
        pdp[idx] = np.mean(y_pred)
    
    pdp_list.append(pdp)

pdp_array = np.array(pdp_list)

# %%
pdp_array_without_knowledge = pdp_array.copy()

# %%
pdp_mean = np.mean(pdp_array, axis = 0)
pdp_std = np.std(pdp_array, axis = 0)

# 1. Plot the mean PDP line (same as before)
plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")

# 2. Add the Error/Confidence Band using fill_between
# The band represents [mean - std] to [mean + std]
plt.fill_between(
    potenital_values, 
    pdp_mean - pdp_std * 1.96,  # Lower bound
    pdp_mean + pdp_std * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
    label="$\pm 1.96 \sigma$" # Label for the legend
)

# Optional: Add labels and grid
plt.xlabel("Natural Disaster Count")
plt.ylabel("Predicted Disease Increase Probability")
plt.grid(True)
plt.legend()

#plt.savefig(os.path.join(FIGURES, f'fig06_{aim_variable}_PDP_{varname.upper()}_STD.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# #### knowing

# %%
# Determine the grid boundaries based on the specified quantiles
low_b = 0.0
up_b = 11.0
stripe = 1.0
var = 'DisasterExpInd'

# Generate the discrete grid of feature values
potenital_values = np.arange(low_b, up_b, stripe)
X_adjust = X.copy()

pdp_list = []

# Iterate through each model in the ensemble/list
for model in model_list:
    pdp = np.full_like(potenital_values, fill_value=np.nan)
    
    # Iterate through each grid point (potential value)
    for idx, potenital_value in enumerate(potenital_values):
        # 1. Substitute the feature column with the current fixed value
        X_adjust[var] = potenital_value.astype(float)
        X_adjust['HeardClimate_Dummy'] = 1.0
        
        # 2. Predict the outcome for the entire adjusted dataset
        y_pred = model.predict_proba(X_adjust)[:,1]
        
        # 3. Calculate the partial dependence (average prediction)
        pdp[idx] = np.mean(y_pred)
    
    pdp_list.append(pdp)

pdp_array = np.array(pdp_list)

# %%
pdp_array_with_knowledge = pdp_array.copy()

# %%
pdp_mean = np.mean(pdp_array, axis = 0)
pdp_std = np.std(pdp_array, axis = 0)

# 1. Plot the mean PDP line (same as before)
plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")

# 2. Add the Error/Confidence Band using fill_between
# The band represents [mean - std] to [mean + std]
plt.fill_between(
    potenital_values, 
    pdp_mean - pdp_std * 1.96,  # Lower bound
    pdp_mean + pdp_std * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
    label="$\pm 1.96 \sigma$" # Label for the legend
)

# Optional: Add labels and grid
plt.xlabel("Natural Disaster Count")
plt.ylabel("Predicted Disease Increase Probability")
plt.grid(True)
plt.legend()

#plt.savefig(os.path.join(FIGURES, f'fig06_{aim_variable}_PDP_{varname.upper()}_STD.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Plot togather

# %%
os.makedirs(FIGURES := 'figures', exist_ok = True)

# %%
pdp_mean_without_knowledge = np.mean(pdp_array_without_knowledge, axis = 0)
pdp_std_without_knowledge = np.std(pdp_array_without_knowledge, axis = 0)
pdp_mean_with_knowledge = np.mean(pdp_array_with_knowledge, axis = 0)
pdp_std_with_knowledge = np.std(pdp_array_with_knowledge, axis = 0)

# 1. Plot the mean PDP line (same as before)
plt.plot(potenital_values, pdp_mean_without_knowledge, linewidth=2, label="Mean Prediction Without Knowledge")
plt.plot(potenital_values, pdp_mean_with_knowledge, linewidth=2, label="Mean Prediction With Knowledge", color = 'red')


# 2. Add the Error/Confidence Band using fill_between
# The band represents [mean - std] to [mean + std]
plt.fill_between(
    potenital_values, 
    pdp_mean_without_knowledge - pdp_std_without_knowledge * 1.96,  # Lower bound
    pdp_mean_without_knowledge + pdp_std_without_knowledge * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
    label="$\pm 1.96 \sigma$" # Label for the legend
)
plt.fill_between(
    potenital_values, 
    pdp_mean_with_knowledge - pdp_std_with_knowledge * 1.96,  # Lower bound
    pdp_mean_with_knowledge + pdp_std_with_knowledge * 1.96,  # Upper bound
    color='gray',        # Color of the shaded area
    alpha=0.3,           # Transparency
)


# Optional: Add labels and grid
plt.xlabel("Natural Disaster Count")
plt.ylabel("Predicted Disease Increase Probability")
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(FIGURES, f'fig06_PDP.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%
np.save(os.path.join('results', 'pdp_array_without_knowledge.npy'), pdp_mean_without_knowledge)
np.save(os.path.join('results', 'pdp_array_with_knowledge.npy'), pdp_array_with_knowledge)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
