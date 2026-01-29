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
