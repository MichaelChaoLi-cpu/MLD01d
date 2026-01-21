# %% [markdown]
# # Hyperparameter Search

# %% [markdown]
# These are experimental codes:      
# - Briefing the data
# - Tuning optimal hyperparameter
# - Visualize the performance of Models
# - Visualize the importance
# - Visualize and summerize impacts

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

# %%
import Modelling
import SettingForFeatures
import TestingTools

# %%
import importlib
importlib.reload(SettingForFeatures)


# %%

# %% [markdown]
# ## Functions

# %%
def random_search_aim(
    all_data : pd.DataFrame,
    aim_variable: str,
    always_inputs: list,
    reg_params : dict,
    n_splits : int = 10
): 
    all_data_use =  all_data[
        always_inputs + [aim_variable]
    ]
    X = all_data_use.drop(columns = aim_variable)
    y = all_data_use[aim_variable]

    _ = Modelling.xgb_cls_kfold_cv(
        X, y,
        n_splits=n_splits,
        params=reg_params,
        log_dir="logs",
        log_file="xgb_cls_booming.csv"
    )


# %%
cls_params = {
    "objective":"binary:logistic",  
    "eval_metric":"logloss",   
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda"
}

# %%

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
always_inputs = SettingForFeatures.return_input_variables()

# %%
potential_output = SettingForFeatures.return_output_variables()

# %%
df_all = SettingForFeatures.data_load_combine_dataset()

# %%
BEAUTIFUL_NAME = SettingForFeatures.return_beautiful_dict()

# %%

# %% [markdown]
# ### test

# %%
for aim_variable in potential_output:
    print(aim_variable)
    random_search_aim(
        all_data = df_all,
        aim_variable = aim_variable,
        always_inputs = always_inputs,
        reg_params = cls_params,
        n_splits = 10
    )

# %%

# %%

# %% [markdown]
# ### Basic Importance Check

# %%
n_splits = 10

# %%
importance_list = []

# %%
for aim_variable in potential_output:
    X, y = Modelling.prepare_data(
        all_data = df_all,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    feature_importance_df_list = []
    fold = 1
    
    for train_idx, test_idx in kf.split(X):
    
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = xgb.XGBClassifier(**cls_params)
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
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain Importance")
    plt.title("XGBoost Feature Importance")
    plt.show()

    importance_list.append(feature_importance_full)

# %%
os.makedirs(RESULTS := 'results', exist_ok=False)

# %%
for idx, aim_variable in enumerate(potential_output):
    importance_list[idx].index = importance_list[idx].index.map(
        lambda x: BEAUTIFUL_NAME.get(x, x)
    )
    importance_list[idx].to_excel(os.path.join(RESULTS, f'importnace_basic_{aim_variable}.xlsx'))

# %%
importance_list[idx]

# %%

# %%

# %%

# %%

# %%
