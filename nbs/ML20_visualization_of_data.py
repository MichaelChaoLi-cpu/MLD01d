# %% [markdown]
# # Visualization for Materials

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import patheffects as path_effects

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
def convert_count_data_into_map(
    count_df : pd.DataFrame
) -> np.ndarray:
    grid_array = np.full((180, 360), np.nan)
    count_df['lat_idx'] = (count_df['LATITUDE'] + 90).astype(int)
    count_df['lon_idx'] = (count_df['LONGITUDE'] + 180).astype(int)

    gridded_df = count_df.groupby(['lat_idx', 'lon_idx'])['Count'].sum().reset_index()

    lat_indices = gridded_df['lat_idx'].to_numpy()
    lon_indices = gridded_df['lon_idx'].to_numpy()
    values = gridded_df['Count'].to_numpy()

    grid_array[lat_indices, lon_indices] = values

    return grid_array[::-1,:]


# %%
def convert_data_into_map_by_mean(
    df : pd.DataFrame,
    var : str
) -> np.ndarray:
    grid_array = np.full((180, 360), np.nan)
    df['lat_idx'] = (df['LATITUDE'] + 90).astype(int)
    df['lon_idx'] = (df['LONGITUDE'] + 180).astype(int)

    gridded_df = df.groupby(['lat_idx', 'lon_idx'])[var].mean().reset_index()

    lat_indices = gridded_df['lat_idx'].to_numpy()
    lon_indices = gridded_df['lon_idx'].to_numpy()
    values = gridded_df[var].to_numpy()

    grid_array[lat_indices, lon_indices] = values

    return grid_array[::-1,:]


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
os.makedirs(FIGURES := "./figures", exist_ok = True)

# %%
always_inputs = SettingForFeatures.return_input_variables()

# %%
aim_variable = SettingForFeatures.return_output_variables()[0]

# %%
all_data = SettingForFeatures.data_load_combine_dataset()

# %%
map_df = SettingForFeatures.load_spatial_data()
map_df.columns = ['EcoBelt', "Province", 'geometry']

# Fix inconsistent province name
map_df.loc[map_df['Province'] == 'Sudur Pashchim', 'Province'] = 'Sudurpashchim'
# Set multi-index with Province and EcoBelt
map_df = map_df.set_index(['Province', 'EcoBelt'])

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%
# Define custom colormap
colors = ['blue', 'green', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# %%
# Define custom colormap
colors = ['blue', 'green', 'white', 'yellow', 'red']
custom_cmap_white = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# %%
os.makedirs(FIGURES := 'figures', exist_ok=True)

# %%
VARIABLE_MAP_RENAMED = SettingForFeatures.return_beautiful_dict()

# %%

# %%

# %% [markdown]
# ### Plot Map of Observation

# %%
loc_df = all_data[['Prov', 'EcoBelt', 'Year']].replace('Sudurpaschim', 'Sudurpashchim')
loc_df.columns = ['Province', 'EcoBelt', 'Year']
loc_df['count'] = 1
loc_df_count = loc_df.groupby(['Province', 'EcoBelt', 'Year']).sum().reset_index()

# %%
loc_df_wide = loc_df_count.pivot_table(
    index=['Province', 'EcoBelt'],
    columns='Year',
    values=loc_df_count.columns.difference(['Province', 'EcoBelt', 'Year']),
    aggfunc='sum'
).reset_index()

# %%
loc_df_wide.columns = ['Province', 'EcoBelt', '2016', '2022']

# %%
used_map_df = map_df.merge(loc_df_wide, on = ['Province', 'EcoBelt',], how = 'left').set_index(['Province', 'EcoBelt'])

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    #subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

vmin = 0
vmax = 1000

idxs = 'ab'

for ax, col, idx in zip(axes, ['2016', '2022'], idxs):

    im = used_map_df.plot(
        column=col,
        cmap=custom_cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
        missing_kwds={
        "color": "grey",
        "edgecolor": "black"
    }
    )

    # Optional annotation
    for i, row in used_map_df.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.text(
                cx, cy,
                f"{i[0]}\n{i[1]}",
                fontsize=7,
                ha="center",
                va="center"
            ).set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()
            ])

    ax.set_title(f"({idx}): Observation Spatial Distribution in {col}", fontsize=11, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("on")

# ---- Shared colorbar (one for both maps) ----
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 

sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=custom_cmap
)
sm._A = []

cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)
cbar.set_label('Total Respondents', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_address:=os.path.join(FIGURES, 'fig01_observation_distribution.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %% [markdown]
# ### Plot Map of Health

# %%
loc_df = all_data[['Prov', 'EcoBelt', 'Year', aim_variable]].replace('Sudurpaschim', 'Sudurpashchim')
loc_df.columns = ['Province', 'EcoBelt', 'Year', 'Health']
loc_df_count = loc_df.groupby(['Province', 'EcoBelt', 'Year']).mean().reset_index()

# %%
loc_df_count['Health'] = loc_df_count['Health'] *100

# %%
loc_df_wide = loc_df_count.pivot_table(
    index=['Province', 'EcoBelt'],
    columns='Year',
    values=loc_df_count.columns.difference(['Province', 'EcoBelt', 'Year']),
    aggfunc='mean'
).reset_index()

# %%
loc_df_wide.columns = ['Province', 'EcoBelt', '2016', '2022']

# %%
used_map_df = map_df.merge(loc_df_wide, on = ['Province', 'EcoBelt',], how = 'left').set_index(['Province', 'EcoBelt']) 

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    #subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

vmin = 0
vmax = 60

idxs = 'ab'

for ax, col, idx in zip(axes, ['2016', '2022'], idxs):

    im = used_map_df.plot(
        column=col,
        cmap=custom_cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
        missing_kwds={
            "color": "grey",
            "edgecolor": "black"
        }
    )

    # Optional annotation
    for i, row in used_map_df.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.text(
                cx, cy,
                f"{i[0]}\n{i[1]}",
                fontsize=7,
                ha="center",
                va="center"
            ).set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()
            ])

    ax.set_title(f"({idx}): Disease Probability Spatial Distribution in {col}", fontsize=11, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("on")

# ---- Shared colorbar (one for both maps) ----
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 

sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=custom_cmap
)
sm._A = []

cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)
cbar.set_label('Disease Probability (%)', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_address:=os.path.join(FIGURES, 'fig02_health.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %% [markdown]
# ### Plot Map of Natural Disater

# %%
this = 'DisasterExpInd'

# %%
loc_df = all_data[['Prov', 'EcoBelt', 'Year', this]].replace('Sudurpaschim', 'Sudurpashchim')
loc_df.columns = ['Province', 'EcoBelt', 'Year', this]
loc_df_count = loc_df.groupby(['Province', 'EcoBelt', 'Year']).mean().reset_index()

# %%
loc_df_wide = loc_df_count.pivot_table(
    index=['Province', 'EcoBelt'],
    columns='Year',
    values=loc_df_count.columns.difference(['Province', 'EcoBelt', 'Year']),
    aggfunc='mean'
).reset_index()

# %%
loc_df_wide.columns = ['Province', 'EcoBelt', '2016', '2022']

# %%
used_map_df = map_df.merge(loc_df_wide, on = ['Province', 'EcoBelt',], how = 'left').set_index(['Province', 'EcoBelt']) 

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    #subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

vmin = 0
vmax = 8

idxs = 'ab'

for ax, col, idx in zip(axes, ['2016', '2022'], idxs):

    im = used_map_df.plot(
        column=col,
        cmap=custom_cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
        missing_kwds={
            "color": "grey",
            "edgecolor": "black"
        }
    )

    # Optional annotation
    for i, row in used_map_df.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.text(
                cx, cy,
                f"{i[0]}\n{i[1]}",
                fontsize=7,
                ha="center",
                va="center"
            ).set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()
            ])

    ax.set_title(f"({idx}): Average Experienced Natural Disasters in {col}", fontsize=11, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("on")

# ---- Shared colorbar (one for both maps) ----
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 

sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=custom_cmap
)
sm._A = []

cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)
cbar.set_label('Types of Natural Disasters', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_address:=os.path.join(FIGURES, 'fig03_natural_disaster.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %%

# %% [markdown]
# ### Plot Map of HeardClimate_Dummy

# %%
this = 'HeardClimate_Dummy'

# %%
loc_df = all_data[['Prov', 'EcoBelt', 'Year', this]].replace('Sudurpaschim', 'Sudurpashchim')
loc_df.columns = ['Province', 'EcoBelt', 'Year', this]
loc_df_count = loc_df.groupby(['Province', 'EcoBelt', 'Year']).mean().reset_index()

# %%
loc_df_count[this] = loc_df_count[this] * 100

# %%
loc_df_wide = loc_df_count.pivot_table(
    index=['Province', 'EcoBelt'],
    columns='Year',
    values=loc_df_count.columns.difference(['Province', 'EcoBelt', 'Year']),
    aggfunc='mean'
).reset_index()

# %%
loc_df_wide.columns = ['Province', 'EcoBelt', '2016', '2022']

# %%
used_map_df = map_df.merge(loc_df_wide, on = ['Province', 'EcoBelt',], how = 'left').set_index(['Province', 'EcoBelt']) 

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    #subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

vmin = 0
vmax = 70

idxs = 'ab'

for ax, col, idx in zip(axes, ['2016', '2022'], idxs):

    im = used_map_df.plot(
        column=col,
        cmap=custom_cmap,
        edgecolor="black",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        alpha=0.8,
        legend=False,
        ax=ax,
        missing_kwds={
            "color": "grey",
            "edgecolor": "black"
        }
    )

    # Optional annotation
    for i, row in used_map_df.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.text(
                cx, cy,
                f"{i[0]}\n{i[1]}",
                fontsize=7,
                ha="center",
                va="center"
            ).set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()
            ])

    ax.set_title(f"({idx}): Percentage of Population with Climate Knowledge in {col}", fontsize=11, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("on")

# ---- Shared colorbar (one for both maps) ----
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 

sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=custom_cmap
)
sm._A = []

cbar = fig.colorbar(
    sm,
    cax=cbar_ax,
    orientation="horizontal"
)
cbar.set_label('Percentage of Population with Climate Knowledge (%)', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_address:=os.path.join(FIGURES, 'fig04_knowledge_perc.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %% [markdown]
# ### Plot Importance Variable

# %%
aim_variable

# %%
feature_importance_full = pd.read_csv(os.path.join('results', f'{aim_variable}_importance.csv'), index_col = 0).iloc[:,:10]

# %%
feature_importance_full_sum = feature_importance_full.sum(axis = 0)
feature_importance_full = feature_importance_full / feature_importance_full_sum * 100

feature_importance_full['mean'] = feature_importance_full.mean(axis = 1)
feature_importance_full['std'] = feature_importance_full.std(axis = 1)

# %%
feature_importance_full.index = feature_importance_full.index.map(VARIABLE_MAP_RENAMED)

# %%
feature_importance_full = feature_importance_full.sort_values('mean', ascending = False)

# %%
df_plot = feature_importance_full.copy()

# %%
fig, ax = plt.subplots(figsize=(10, 15))

ax.barh(
    df_plot.index,
    df_plot["mean"].astype(float),
    xerr=(df_plot["std"].astype(float) * 1.96),
    color="lightskyblue",
    alpha=0.9,
    capsize=4
)

ax.invert_yaxis()
ax.set_xlabel("Gain Importance (%)")

# Bold y-tick labels containing "LAI"
for label in ax.get_yticklabels():
    if "Climate Change Knowledge" in label.get_text():
        label.set_fontweight("bold")

ax.grid(linestyle="--", alpha=0.6)

fig.savefig(os.path.join(FIGURES, "fig05_importance.jpg"), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %% [markdown]
# ### Plot Geodifference

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
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%
var = 'HeardClimate_Dummy'

# %%
y_df = np.load(os.path.join("results", f"health_prediction_of_{var}.npy"))

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
X_output_region = X_output.groupby(['Province', 'EcoBelt']).mean().reset_index()

# %%
map_df_use = map_df.merge(X_output_region, on = ['Province', 'EcoBelt'], how = 'left')

# %%
map_df_use = map_df_use.set_index(['Province', 'EcoBelt'])

# %%
fig, ax = plt.subplots(
    figsize=(15, 10)
)

vmin = -0.07
vmax = -0.01

im = map_df_use.plot(
    column='difference',
    cmap=custom_cmap,
    edgecolor="black",
    linewidth=0.4,
    vmin=vmin,
    vmax=vmax,
    alpha=0.8,
    legend=False,
    ax=ax,
    missing_kwds={
        "color": "grey",
        "edgecolor": "black"
    }
)

# Optional annotation
for i, row in map_df_use.iterrows():
    geom = row.geometry
    if geom is not None and not geom.is_empty:
        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(
            cx, cy,
            f"{i[0]}\n{i[1]}",
            fontsize=7,
            ha="center",
            va="center"
        ).set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground="white"),
            path_effects.Normal()
        ])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.4)
ax.axis("on")

# Colorbar
sm = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
    cmap=custom_cmap
)
sm._A = []
cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.6)
cbar.ax.tick_params(labelsize=9)
cbar.set_label('Effects', fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(save_address:=os.path.join(FIGURES, 'fig06_spatial_effect.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %%

# %% [markdown]
# ### Plot Knowledge imapct

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%
var = 'HeardClimate_Dummy'

# %%
y_df = np.load(os.path.join("results", f"health_prediction_of_{var}.npy"))

# %%
np.mean(np.mean(y_df, axis = 1)[:,1] - np.mean(y_df, axis = 1)[:,0] )

# %%
X[['negative_health_proba', 'positive_health_proba']] = np.mean(y_df, axis = 1)

# %%
X['difference'] = X['positive_health_proba'] - X['negative_health_proba']

# %%
bins = [0.0, 0.1, 0.2,
       0.3, 0.4, 0.5,
       0.6, 1.0]

# %%
xtick_labels = ['0-10%', '10%-20%', '20%-30%', 
                '30%-40%', '40%-50%', '50%-60%', 
                '60%-100%']

# %%
variables = [
    'Literal_Ratio', 'Edu12_Ratio', 
    'Female_Ratio', 'A65_Ratio',
    'TotalIncome', 'DisasterExpInd'
]

# %%
len(bins)

# %%
n_panels = 6
vars_to_plot = variables[:n_panels]

fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharey=True)
axes = axes.flatten()

figure_index = 'abcdefg'

for i, variable in enumerate(vars_to_plot):
    ax = axes[i]

    x = X[variable].values
    y = X['difference'].values

    if variable == 'Female_Ratio':
        xtick_labels = ['10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-100%']
        bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    elif variable == 'TotalIncome':
        bins = list(range(0, 500_000 + 1, 50_000)) + [9.920000e+07 + 50_000]
        xtick_labels = ['0~50k', '~100k', '~150k', '~200k', '~250k', '~300k', '~350k', '~400k', '~450k', '~500k', '500k +',]
    elif variable == 'DisasterExpInd':
        bins = list(range(1, 11, 1)) + [15]
        xtick_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", ">= 10"]
    else:
        xtick_labels = ['0-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-100%']
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]

    df_temp = pd.DataFrame({'x': x, 'y': y})
    df_temp['x_bin'] = pd.cut(df_temp['x'], bins=bins)

    grouped = df_temp.groupby('x_bin')['y'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.dropna()

    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['x_idx'] = list(range(len(grouped)))
    
    ax.errorbar(
        grouped['x_idx'],
        grouped['mean'],
        yerr=grouped['se'] * 1.96,
        fmt='o',
        capsize=3
    )

    ax.axhline(0, linestyle='--', color='red', linewidth=1)

    ax.set_xticks(grouped['x_idx'])
    ax.set_xticklabels(xtick_labels, rotation=45, ha = 'center')

    ax.set_xlabel(VARIABLE_MAP_RENAMED.get(variable, variable))
    ax.set_ylabel('Mean of Effect')

    ax.grid(True)

    ax.text(
        0.02, 0.95,         
        figure_index[i],
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='top'
    )

for j in range(len(vars_to_plot), 4):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.savefig(save_address:=os.path.join(FIGURES, 'fig07_effect_with_ohters.jpg'), dpi=300, bbox_inches="tight")
plt.show()

# %%

# %% [markdown]
# ### PDP 

# %%
pdp_array = np.load(os.path.join('results', 'pdp_array_DisasterExpInd.npy'))

# %%
potenital_values = list(range(11))

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

plt.savefig(os.path.join(FIGURES, f'fig08_naive_PDP.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# ### PDP condisering knowledge

# %%
pdp_array_without_knowledge = np.load(os.path.join('results', 'pdp_array_without_knowledge.npy'))
pdp_array_with_knowledge = np.load(os.path.join('results', 'pdp_array_with_knowledge.npy') )

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

plt.savefig(os.path.join(FIGURES, f'fig09_PDP_with_knowledge.jpg'), dpi=300, bbox_inches='tight')
plt.show()

# %%
