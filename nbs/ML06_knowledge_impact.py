# %% [markdown]
# # Check Explanation of the relationship

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import geopandas as gpd
import json
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from matplotlib import cm
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
importlib.reload(SettingForFeatures)


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
from matplotlib import patheffects as path_effects

def plot_spatial_difference(
    map_df: gpd.GeoDataFrame,
    save_address: str,
    col: str = "difference",
    vmin=None,
    vmax=None,
    title: str = "Difference",
    annotate: bool = True,
):
    """
    Plot a single spatial map for one column (default: 'difference').

    Parameters
    ----------
    map_df : geopandas.GeoDataFrame
        GeoDataFrame containing geometry and the target column.
    save_address : str
        Output path for saving the figure (e.g., 'out.png').
    col : str, default='difference'
        Column name to plot.
    vmin, vmax : float or None
        Color scale bounds. If None, use data min/max.
    title : str, default='Difference'
        Figure title.
    annotate : bool, default=True
        Whether to annotate each polygon with its index.
    """
    if col not in map_df.columns:
        raise ValueError(f"Column '{col}' not found in map_df.columns: {list(map_df.columns)}")

    # Infer vmin/vmax if not provided
    data = map_df[col]
    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    cmap = plt.cm.RdYlBu_r

    map_df.plot(
        column=col,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
    )

    # Optional annotation
    for i, row in map_df.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.text(cx, cy, f"{i[0]}\n{i[1]}", fontsize=7, ha='center', va='center').set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground='white'),
                path_effects.Normal()
            ])

    # Colorbar
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    # Style
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="-", alpha=0.4)
    ax.axis("on")

    plt.tight_layout()
    plt.savefig(save_address, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Map saved to: {save_address}")


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
map_df = SettingForFeatures.load_spatial_data()
map_df.columns = ['EcoBelt', "Province", 'geometry']

# Fix inconsistent province name
map_df.loc[map_df['Province'] == 'Sudur Pashchim', 'Province'] = 'Sudurpashchim'
# Set multi-index with Province and EcoBelt
map_df = map_df.set_index(['Province', 'EcoBelt'])

loc_df = all_data[['Prov', 'EcoBelt']]
loc_df.columns = ['Province', 'EcoBelt']

# %%
loc_df = loc_df.replace('Sudurpaschim', 'Sudurpashchim')

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
var = 'HeardClimate_Dummy'

# %%
# Generate the discrete grid of feature values
potenital_values = [0, 1]
X_adjust = X.copy()

pdp_list = []

y_df = np.full((X_adjust.shape[0], len(model_list), len(potenital_values)), fill_value = np.nan)

# Iterate through each model in the ensemble/list
for model_idx, model in enumerate(model_list):
    pdp = np.full_like(potenital_values, fill_value=np.nan)
    
    # Iterate through each grid point (potential value)
    for idx, potenital_value in enumerate(potenital_values):
        # 1. Substitute the feature column with the current fixed value
        X_adjust[var] =  potenital_value
        
        # 2. Predict the outcome for the entire adjusted dataset
        y_pred = model.predict_proba(X_adjust)[:,1]
        
        # 3. Calculate the partial dependence (average prediction)
        y_df[:, model_idx, idx] = y_pred

# %%
np.save(os.path.join("results", f"health_prediction_of_{var}.npy"), y_df)

# %%
y_df.shape

# %%
np.mean(np.mean(y_df, axis = 1)[:,1] - np.mean(y_df, axis = 1)[:,0] )

# %%
X[['negative_health_proba', 'positive_health_proba']] = np.mean(y_df, axis = 1)

# %%
X_output = X[['negative_health_proba', 'positive_health_proba']].copy()

# %%
X_output = X_output.merge(loc_df, left_index=True, right_index=True)

# %%
X_output['difference'] = X_output['positive_health_proba'] - X_output['negative_health_proba']

# %%
X_output.columns

# %%
X_output_region = X_output.groupby(['Province', 'EcoBelt']).mean().reset_index()

# %%
map_df = map_df.merge(X_output_region, on = ['Province', 'EcoBelt'])

# %%
map_df = map_df.set_index(['Province', 'EcoBelt'])

# %%
plot_spatial_difference(map_df, "figures/difference_map.png", col="difference", title="Spatial Difference")

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

# %%

# %%

# %%

# %%
