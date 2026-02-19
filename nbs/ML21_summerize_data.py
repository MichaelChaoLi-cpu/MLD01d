# %% [markdown]
# # Summerize Data

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import json
import numpy as np
import pandas as pd

# %%
import SettingForFeatures

# %%
import importlib
importlib.reload(SettingForFeatures)


# %% [markdown]
# ## Functions

# %%
def calculate_bootstrap_se(data_series, n_bootstraps=1000, statistic_func=np.mean):
    """
    Calculates the Bootstrap Standard Error (SE) for a statistic (default is mean).
    
    Parameters:
    data_series (pd.Series): The data series to calculate the statistic from.
    n_bootstraps (int): The number of resamples (bootstraps).
    statistic_func (function): The statistic function to apply (e.g., np.mean, np.median).
    
    Returns:
    float: The Bootstrap Standard Error of the statistic.
    """
    n_samples = len(data_series)
    bootstrap_statistics = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement, size equal to original sample
        resampled_data = np.random.choice(data_series, size=n_samples, replace=True)
        # Calculate the statistic (e.g., mean) on the resampled data
        stat = statistic_func(resampled_data)
        bootstrap_statistics.append(stat)
        
    # The Bootstrap SE is the standard deviation of the bootstrap distribution
    return np.std(bootstrap_statistics)


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
os.makedirs(TABLES := './tables', exist_ok = True)

# %%
all_data = SettingForFeatures.data_load_combine_dataset()

# %%
np.sum(all_data['Year'].notna())

# %%
wave_1_raw = all_data[all_data['Year']==2016][['Prov', 'EcoBelt']].value_counts(dropna=False).sort_index()

# %%
wave_2_raw = all_data[all_data['Year']==2022][['Prov', 'EcoBelt']].value_counts(dropna=False).sort_index()

# %%
merged_data = wave_1_raw.to_frame().reset_index().merge(wave_2_raw.to_frame().reset_index(), on = ['Prov', 'EcoBelt'], how = 'outer').replace('Sudurpaschim', 'Sudurpashchim').fillna(0)

# %%
merged_data.columns = ['Province', 'EcoBelt', 'Respondents in Wave 1', 'Respondents in Wave 2']

# %%
merged_data

# %%
merged_data.to_excel(os.path.join(TABLES, 'TableS1_respondentCount.xlsx'))

# %%

# %% [markdown]
# ### Data Summary

# %%
always_inputs = SettingForFeatures.return_input_variables()

# %%
aim_variable = SettingForFeatures.return_output_variables()[0]

# %%
data_summary = all_data[[aim_variable] + always_inputs].describe().T.reset_index()

# %%
data_summary

# %%
VARIABLE_MAP_RENAMED = SettingForFeatures.return_beautiful_dict()

# %%
data_summary['index'] = data_summary['index'].map(VARIABLE_MAP_RENAMED)

# %%
data_summary

# %%
data_summary.to_excel(os.path.join(TABLES, 'Table1_DataSummary.xlsx'))

# %%

# %% [markdown]
# ### Hyperparameter

# %%
with open(save_path := os.path.join('results', 'HumanDiseaseIncreasePast25_Dummy_accuracy_comparison.json'), 'r', encoding='utf-8') as f:
    accuracy_comparison = json.load(f)

# %%
df = pd.DataFrame({
    'model': accuracy_comparison
}).reset_index().rename(columns={'index': 'model'})

# %%
df

# %%
df.to_excel(os.path.join(TABLES, 'TableS2_HyperTable.xlsx'))

# %%

# %%

# %%
