# %% [markdown]
# # Data Cleansing

# %%
# %pwd

# %%
# %cd ..

# %%

# %% [markdown]
# ## Import Package

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# %%

# %% [markdown]
# ## Dataset 2016

# %%
# %ls PrivateData/Climate-2016/Data/Data/

# %% [markdown]
# ### Meta

# %%
df_HhLevel = pd.read_stata("PrivateData/Climate-2016/Data/Data/HhldLevel_KeyVariable.dta")

# %%
df_HhLevel

# %%
# Categorical summaries
for col in ["RespSex", "RespAge", "RespEdu", "Quintl", "CZONE"]:
    print(f"\n--- {col} ---")
    print(df_HhLevel[col].value_counts(dropna=False))

# %%
df_HhLevel.columns

# %%
df_kept = df_HhLevel[['PSU', 'HHLD', 'Quintl', 'CZONE']]

# %%
df_PSULevel = pd.read_stata("PrivateData/Climate-2016/Data/Data/PSULevel_KeyVariable.dta")

# %%
df_PSULevel

# %%
df_PSULevel.columns

# %%
# Categorical summaries
for col in ['Dist', 'UrbRur71', 'Strata', 'EcoBelt', 'Comb_Vuln', 'Comb_Risk', 'Comb_Adap']:
    print(f"\n--- {col} ---")
    print(df_PSULevel[col].value_counts(dropna=False))

# %%
print(df_PSULevel['Dist'].unique().to_list())

# %%
df_PSULevel.columns

# %%
# District → Province mapping (for the districts you listed)
district_to_province = {
    # Province 1 (Koshi)
    'Taplejung': 'Koshi',
    'Panchthar': 'Koshi',
    'Morang': 'Koshi',
    'Dhankuta': 'Koshi',

    # Province 2 (Madhesh)
    'Saptari': 'Madhesh',
    'Dhanusa': 'Madhesh',
    'Rautahat': 'Madhesh',

    # Province 3 (Bagmati)
    'Khotang': 'Bagmati',
    'Ramechhap': 'Bagmati',
    'Dolakha': 'Bagmati',
    'Kathmandu': 'Bagmati',
    'Dhading': 'Bagmati',

    # Province 4 (Gandaki)
    'Tanahu': 'Gandaki',
    'Kaski': 'Gandaki',
    'Mustang': 'Gandaki',
    'Baglung': 'Gandaki',

    # Province 5 (Lumbini)
    'Palpa': 'Lumbini',
    'Rupandehi': 'Lumbini',
    'Pyuthan': 'Lumbini',
    'Dang': 'Lumbini',

    # Province 6 (Karnali)
    'Salyan': 'Karnali',
    'Jumla': 'Karnali',
    'Kalikot': 'Karnali',

    # Province 7 (Sudurpaschim)
    'Bajura': 'Sudurpaschim',
    'Achham': 'Sudurpaschim',
    'Kailali': 'Sudurpaschim',
}

# Example usage in pandas:
df_PSULevel['Prov'] = df_PSULevel['Dist'].map(district_to_province)

# %%
df_select = df_PSULevel[['PSU', 'Dist', 'UrbRur71', 'Strata', 'EcoBelt', 'Prov']]

# %%
df_kept = df_kept.merge(df_select, on = 'PSU', how='left')

# %%
df_kept.head()

# %%
df_kept['Rural_Dummy'] = np.where(df_kept['UrbRur71'] == 'Rural', 1, 0)

# %%

# %% [markdown]
# ### Data Merging

# %% [markdown]
# #### Section 01

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S01.dta", convert_categoricals=False)

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
df_read = df_read[['PSU', 'HHLD', 'A07SEX', 'A07AGE', 'A08SEX', 'A08AGE', 'A11', 'A12']]

# %%
df_read.columns = ['PSU', 'HHLD', 'HeadHH_Female', 'HeadHH_Age',
                  'Respon_Female', 'Respon_Age', 'Edu', 'LivingYear']

# %%
df_read.loc[:, 'HeadHH_Female'] = df_read.loc[:, 'HeadHH_Female'] - 1
df_read.loc[:, 'Respon_Female'] = df_read.loc[:, 'Respon_Female'] - 1

# %%
df_read

# %%
plt.figure(figsize=(8, 5))
df_read["Respon_Age"].hist(bins=17, edgecolor="black")
plt.xlabel("age")
plt.ylabel("Frequency")
plt.show()

# %%
df_read = df_read.assign(
    Edu_UnderSLC     = np.where(df_read["Edu"] < 12, 1, 0),
    Edu_Certificate  = np.where(df_read["Edu"] == 12, 1, 0),
    Edu_Bachelor     = np.where(df_read["Edu"] == 13, 1, 0),
    Edu_Master       = np.where(df_read["Edu"] == 14, 1, 0),
    Edu_PhD          = np.where(df_read["Edu"] == 15, 1, 0),
    Edu_Literal      = np.where(df_read["Edu"] == 16, 1, 0),
    Edu_Illiterate   = np.where(df_read["Edu"] == 17, 1, 0)
) 

# %%
plt.figure(figsize=(8, 5))
df_read["Edu"].hist(bins=17, edgecolor="black")
plt.xlabel("Education Code")
plt.ylabel("Frequency")
plt.xticks(range(1, 18))
plt.show()

# %%
edu_year_map = {
    1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9, 10: 10, 11: 11,
    12: 12,      # Certificate level
    13: 16,      # Bachelor
    14: 18,      # Master
    15: 21,      # PhD
    16: 0,       # Literal
    17: 0        # Illiterate
}

df_read["Edu_year"] = df_read["Edu"].map(edu_year_map).fillna(0).astype(int)

# %%
df_01 = df_read.copy()

# %%
df_01

# %%
df_01.columns

# %%
df_select = df_01[['PSU', 'HHLD', 'HeadHH_Female', 'HeadHH_Age', 'Respon_Female',
       'Respon_Age', 'LivingYear', 'Edu_UnderSLC', 'Edu_Certificate',
       'Edu_Bachelor', 'Edu_Master', 'Edu_PhD', 'Edu_Literal',
       'Edu_Illiterate', "Edu_year"]]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.head()

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 02-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S02_1.dta")

# %%
# Categorical summaries
for col in ['B09OCC']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S02_1.dta", convert_categoricals=False)

# %%
# Categorical summaries
for col in ['B09OCC']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read.shape

# %%
df_read

# %%
df_read['Female_Ratio'] = df_read['B05SEX'] - 1
df_read['U18_Ratio'] = np.where(df_read["B06AGE"] < 18, 1, 0)
df_read['A65_Ratio'] = np.where(df_read["B06AGE"] >= 65, 1, 0)

df_read["B07EDU"] = df_read["B07EDU"].fillna(17)
df_read['Edu12_Ratio'] = np.where((df_read["B07EDU"] >= 12)&(df_read["B07EDU"] < 16), 1, 0)
df_read['Literal_Ratio'] = np.where((df_read["B07EDU"] >= 12)&(df_read["B07EDU"] < 17), 1, 0)

df_read["B09OCC"] = df_read["B09OCC"].fillna(7).astype(int)
df_occ_dummies = pd.get_dummies(df_read["B09OCC"], prefix="OCC").astype(int)
df_occ_dummies.columns = ['Occ_Agri', 'Occ_Wage', 'Occ_NonAgriBus', 
                          'Occ_Household', 'Occ_Stu', 'Occ_Hunting',  
                          'Occ_NoJob', 'Occ_Uable']
df_read = pd.concat([df_read, df_occ_dummies], axis=1)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'Female_Ratio', 'U18_Ratio', 'A65_Ratio',
       'Edu12_Ratio', 'Literal_Ratio', 'Occ_Agri', 'Occ_Wage',
       'Occ_NonAgriBus', 'Occ_Household', 'Occ_Stu', 'Occ_Hunting',
       'Occ_NoJob', 'Occ_Uable']]

# %%
df_count = df_read.groupby(['PSU', 'HHLD']).count().reset_index()['B02SN']

# %%
df_count

# %%
df_select_ratio = df_select.groupby(['PSU', 'HHLD']).mean().reset_index()

# %%
df_select_ratio['Household_memberNum'] = df_count

# %%
df_select_ratio

# %%
df_kept = df_kept.merge(df_select_ratio, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.head()

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 02-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S02_2.dta")

# %%
# Categorical summaries
for col in ['B12A', 'B12B', 'B12C', 'B13A', 'B13B', 'B13C']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
# Categorical summaries
for col in ['B17A', 'B17B', 'B17C', 'B17D', 'B17E', 'B17F', 'B17G', 'B17H', 'B17I', 'B17J', 'B17K', 'B17L']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S02_2.dta", convert_categoricals=False)

# %%
# Categorical summaries
for col in ['B17A', 'B17B', 'B17C', 'B17D', 'B17E', 'B17F', 'B17G', 'B17H', 'B17I', 'B17J', 'B17K', 'B17L']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read.shape

# %%
df_read

# %%
cols = ['B17A','B17B','B17C','B17D','B17E','B17F','B17G','B17H','B17I','B17J','B17K','B17L']

df_read["Radio_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["TV_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Cable_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["PC_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Net_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Phone_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Mobile_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Motorbike_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Car_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Bike_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["OtherVehi_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)
df_read["Refrige_dummy"] = (df_read[cols] == 1).any(axis=1).astype(int)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'B10', 'B11', 'B12A', 'B12B',
        'B12C', 'B13A', 'B13B', 'B13C', 'B14', 'B15', 'B16A', 'B16B', 'B16C',
         'B18', 'C01', 'Radio_dummy', 'TV_dummy',
       'Cable_dummy', 'PC_dummy', 'Net_dummy', 'Phone_dummy', 'Mobile_dummy',
       'Motorbike_dummy', 'Car_dummy', 'Bike_dummy', 'OtherVehi_dummy',
       'Refrige_dummy']].fillna(0)

# %%
df_select.columns = ['PSU', 'HHLD', 'Own_Resid', 'Resid_Type', 'WaterS1', 'WaterS2',
        'WaterS3', 'CookFuelS1', 'CookFuelS2', 'CookFuelS3', 'LightEnergy', 'Toilet', 'IncomeS1', 'IncomeS2', 'IncomeS3',
         'Remittance_dummy', 'Have_AgriLand', 'Radio_dummy', 'TV_dummy',
       'Cable_dummy', 'PC_dummy', 'Net_dummy', 'Phone_dummy', 'Mobile_dummy',
       'Motorbike_dummy', 'Car_dummy', 'Bike_dummy', 'OtherVehi_dummy',
       'Refrige_dummy']

# %%
df_select['Remittance_dummy'] = (df_select['Remittance_dummy'] - 2).abs()
df_select['Have_AgriLand'] = (df_select['Have_AgriLand'] - 2).abs()

# %%
df_select

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 03 - Complicated Skip

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S03.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%

# %%

# %% [markdown]
# #### Section 04

# %% [markdown]
# **Yes is Yes**

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S04.dta", convert_categoricals=False)

# %%
df_read.shape

# %%
df_read

# %%
df_read['HouseHead_AgriExpYear'] = df_read["D01"]
df_read['SavingMembership'] = np.where(df_read["D02"] == 1, 1, 0)
df_read['RegularSaving'] = np.where(df_read["D03"] == 1, 1, 0)
df_read['OrgMembership'] = np.where(df_read["D04"] == 1, 1, 0)
df_read['AgriSupport'] = np.where(df_read["D05"] == 1, 1, 0)
df_read['Dist_Road'] = df_read["D06"]
df_read['Dist_HealthCenter'] = df_read["D07"]
df_read['Dist_SecondarySchool'] = df_read["D08"]
df_read['Dist_Market'] = df_read["D09"]
df_read['Dist_AgriSupport'] = df_read["D10"]
df_read['FramMechan'] = np.where(df_read["D11"] == 1, 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'HouseHead_AgriExpYear', 'SavingMembership',
                        'RegularSaving', 'OrgMembership', 'AgriSupport', 'Dist_Road',
                        'Dist_HealthCenter', 'Dist_SecondarySchool', 'Dist_Market',
                        'Dist_AgriSupport', 'FramMechan']]

# %%
df_select = df_select.fillna(0)

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 05

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S05.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read['TotalIncome'] = df_read[['E01', 'E02', 'E03', 'E04', 'E05']].sum(axis=1)

# %%
df_read.columns = ['PSU', 'HHLD', 'CropIncome', 'LivestockIncome', 'OtherAgriIncome', 'NonAgriIncome', 'BusiIncome', 'TotalIncome']

# %%
df_read = df_read.fillna(0)

# %%
df_select = df_read

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 06

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S06.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['F01', 'F02', 'F03', 'F04A', 'F04B', 'F04C', 'F05A',
       'F05B', 'F05C', 'F05D', 'F05E', 'F06', 'F07', 'F09', 'F10']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['HeardClimate_Dummy'] = np.where(df_read['F01'] == 'Yes', 1, 0)
df_read['ClimateChanged_Dummy'] = np.where(df_read['F03'] == 'Yes', 1, 0)

# %%
f02_dummies = pd.get_dummies(df_read["F02"], prefix="ClimateInfo").astype(int)

# %%
f02_dummies.columns = [name[:17] for name in f02_dummies.columns]

# %%
df_read = pd.concat([df_read, f02_dummies], axis=1)

# %%
f04a_dummies = pd.get_dummies(df_read["F04A"], prefix="ClimateReasonA").astype(int)

# %%
f04a_dummies.columns = [name[:20] for name in f04a_dummies.columns]

# %%
df_read = pd.concat([df_read, f04a_dummies], axis=1)

# %%
f04b_dummies = pd.get_dummies(df_read["F04B"], prefix="ClimateReasonB").astype(int)

# %%
f04b_dummies.columns = [name[:20] for name in f04b_dummies.columns]

# %%
df_read = pd.concat([df_read, f04b_dummies], axis=1)

# %%
f04c_dummies = pd.get_dummies(df_read["F04C"], prefix="ClimateReasonC").astype(int)

# %%
f04c_dummies.columns = [name[:20] for name in f04c_dummies.columns]

# %%
df_read = pd.concat([df_read, f04c_dummies], axis=1)

# %%
df_read['SummerTemp_IncreaseDummy'] = np.where(df_read['F06'] == 'Increase', 1, 0)
df_read['SummerTemp_DecreaseDummy'] = np.where(df_read['F06'] == 'Decrease', 1, 0)
df_read['WinterTemp_IncreaseDummy'] = np.where(df_read['F07'] == 'Increase', 1, 0)
df_read['WinterTemp_DecreaseDummy'] = np.where(df_read['F07'] == 'Decrease', 1, 0)

# %%
df_read['MonsoonPreci_IncreaseDummy'] = np.where(df_read['F09'] == 'Increase', 1, 0)
df_read['MonsoonPreci_DecreaseDummy'] = np.where(df_read['F09'] == 'Decrease', 1, 0)
df_read['WinterPreci_IncreaseDummy'] = np.where(df_read['F10'] == 'Increase', 1, 0)
df_read['WinterPreci_DecreaseDummy'] = np.where(df_read['F10'] == 'Decrease', 1, 0)

# %%
df_read.columns = [name.replace( '\'' ,'x') for name in df_read.columns]

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'HeardClimate_Dummy', 'ClimateChanged_Dummy', 'ClimateInfo_Radio',
       'ClimateInfo_Telev', 'ClimateInfo_News ', 'ClimateInfo_Aware',
       'ClimateInfo_Local', 'ClimateInfo_Neigh', 'ClimateInfo_Famil',
       'ClimateInfo_Other', 'ClimateReasonA_Defor', 'ClimateReasonA_Natur',
       'ClimateReasonA_Indus', 'ClimateReasonA_Urban', 'ClimateReasonA_Human',
       'ClimateReasonA_Godxs', 'ClimateReasonA_Earth', 'ClimateReasonA_Other',
       'ClimateReasonA_Donxt', 'ClimateReasonB_Defor', 'ClimateReasonB_Natur',
       'ClimateReasonB_Indus', 'ClimateReasonB_Urban', 'ClimateReasonB_Human',
       'ClimateReasonB_Godxs', 'ClimateReasonB_Earth', 'ClimateReasonB_Other',
       'ClimateReasonC_Defor', 'ClimateReasonC_Natur', 'ClimateReasonC_Indus',
       'ClimateReasonC_Urban', 'ClimateReasonC_Human', 'ClimateReasonC_Godxs',
       'ClimateReasonC_Earth', 'ClimateReasonC_Other',
       'SummerTemp_IncreaseDummy', 'SummerTemp_DecreaseDummy',
       'WinterTemp_IncreaseDummy', 'WinterTemp_DecreaseDummy',
       'MonsoonPreci_IncreaseDummy', 'MonsoonPreci_DecreaseDummy',
       'WinterPreci_IncreaseDummy', 'WinterPreci_DecreaseDummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 06-4

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S06_4.dta",
                       convert_categoricals=False)

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['F14', 'F15', 'F16']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['ExpDummy'] = np.where(df_read['F14'] == 1, 1, 0)
df_read['ExpIncreaseDummy'] = np.where(df_read['F15'] == 1, 1, 0)
df_read['ExpDecreaseDummy'] = np.where(df_read['F15'] == 2, 1, 0)
df_read['Impact'] = df_read['F15'].fillna(0)

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="F12CODE", 
           values=["ExpDummy", 'ExpIncreaseDummy', 'ExpDecreaseDummy', 'Impact'])
    .reset_index()
)

# %%
df_wide

# %%
abbr_map = {
    1: "DR",
    2: "FF",
    3: "FS",
    4: "FL",
    5: "IN",
    6: "WS",
    7 :"TS",
    8: "HS",
    9: "HR",
    10: "SR",
    11: "SE",
    12: "LS",
    13: "SS",
    14: "AV",
    15: "GLOF",
    16: "HW",
    17: "CW",
    18: "DI",
    19: "OT"
}

# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 07-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S07_1.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['G03YR', 'G04', 'G05', 'G06',
       'G07', 'G08', 'G09', 'G10', 'G11', 'G12']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['DisasterFoodShortage_Dummy'] = np.where(df_read['G06'] == 'Yes', 1, 0)
df_read['DisasterDie_Dummy'] = np.where(df_read['G07'] == 'Yes', 1, 0)

# %%
df_read = df_read[['PSU', 'HHLD', 'G01CODE', 'DisasterFoodShortage_Dummy', 'DisasterDie_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="G01CODE", 
           values=["DisasterFoodShortage_Dummy", "DisasterDie_Dummy"])
    .reset_index()
)

# %%
abbr_map = {
    "Drought": "DR",
    "Fire (forest)": "FF",
    "Fire (settlement)": "FS",
    "Flood": "FL",
    "Inundation": "IN",
    "Windstorm": "WS",
    "Thunderstorm": "TS",
    "Hailstorm": "HS",
    "Heavy rain": "HR",
    "Sporadic rain": "SR",
    "Soil erosion": "SE",
    "Land slide": "LS",
    "Snowstorm": "SS",
    "Avalanche": "AV",
    "GLOF": "GLOF",
    "Heat wave": "HW",
    "Cold wave": "CW",
    "Diseases / insect": "DI",
    "Others": "OT"
}


# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 07-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S07_2.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['G15YN']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['DisasterMoneyLoss_Dummy'] = np.where(df_read['G15YN'] == 'Yes', 1, 0)

df_read['DisasterMoneyLoss_TotalNRs'] = np.nansum(df_read[['G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23']].fillna(0), axis = 1)

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="G13CODE", 
           values=["DisasterMoneyLoss_Dummy", "DisasterMoneyLoss_TotalNRs"])
    .reset_index()
)

# %%
abbr_map = {
    "Drought": "DR",
    "Fire (forest)": "FF",
    "Fire (settlement)": "FS",
    "Flood": "FL",
    "Inundation": "IN",
    "Windstorm": "WS",
    "Thunderstorm": "TS",
    "Hailstorm": "HS",
    "Heavy rain": "HR",
    "Sporadic rain": "SR",
    "Soil erosion": "SE",
    "Land slide": "LS",
    "Snowstorm": "SS",
    "Avalanche": "AV",
    "GLOF": "GLOF",
    "Heat wave": "HW",
    "Cold wave": "CW",
    "Diseases / insect": "DI",
    "Others": "OT"
}

# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 08

# %% [markdown]
# Yes is Yes

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S08.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read['NewDiseaseCropPast25_Dummy'] = np.where(df_read['H01'] == 'Yes', 1, 0)
df_read['NewInsectCropPast25_Dummy'] = np.where(df_read['H03'] == 'Yes', 1, 0)
df_read['NewDiseaseLivestockPast25_Dummy'] = np.where(df_read['H05'] == 'Yes', 1, 0)

df_read['HumanDiseaseIncreasePast25_Dummy'] = np.where(df_read['H07'] == 'Yes', 1, 0)
df_read['HumanVetorDisIncreasePast25_Dummy'] = np.where(df_read['H09'] == 'Yes', 1, 0)
df_read['HumanWaterDisIncreasePast25_Dummy'] = np.where(df_read['H10'] == 'Yes', 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'NewDiseaseCropPast25_Dummy', 'NewInsectCropPast25_Dummy',
       'NewDiseaseLivestockPast25_Dummy', 'HumanDiseaseIncreasePast25_Dummy',
       'HumanVetorDisIncreasePast25_Dummy',
       'HumanWaterDisIncreasePast25_Dummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 09

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S09.dta", convert_categoricals=False)

# %%
df_read.shape

# %%
df_read

# %%
df_read['WaterSourceRiver_IncreaseDummy'] = np.where(df_read['I01'] == 1, 1, 0)
df_read['WaterSourceRiver_DecreaseDummy'] = np.where(df_read['I01'] == 2, 1, 0)
df_read['WaterSourceRiver_DeteriorationDummy'] = np.where(df_read['I02'] == 1, 1, 0)

df_read['WaterSourceWell_IncreaseDummy'] = np.where(df_read['I03'] == 1, 1, 0)
df_read['WaterSourceWell_DecreaseDummy'] = np.where(df_read['I03'] == 2, 1, 0)
df_read['WaterSourceWell_DeteriorationDummy'] = np.where(df_read['I04'] == 1, 1, 0)

df_read['WaterSourceRiver_DriedDummy'] = np.where(df_read['I05'] == 1, 1, 0)
df_read['WaterSourceWell_DriedDummy'] = np.where(df_read['I07'] == 1, 1, 0)
df_read['WaterSourceSpout_DriedDummy'] = np.where(df_read['I09'] == 1, 1, 0)

df_read['WaterSourceSpout_IncreaseDummy'] = np.where(df_read['I08'] == 1, 1, 0)
df_read['WaterSourceSpout_DecreaseDummy'] = np.where(df_read['I08'] == 2, 1, 0)

df_read['WaterSourcePipe_IncreaseDummy'] = np.where(df_read['I10'] == 1, 1, 0)
df_read['WaterSourcePipe_DecreaseDummy'] = np.where(df_read['I10'] == 2, 1, 0)

df_read['WaterSourceChangePast25_Dummy'] = np.where(df_read['I11'] == 1, 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'WaterSourceRiver_IncreaseDummy', 'WaterSourceRiver_DecreaseDummy',
       'WaterSourceRiver_DeteriorationDummy', 'WaterSourceWell_IncreaseDummy',
       'WaterSourceWell_DecreaseDummy', 'WaterSourceWell_DeteriorationDummy',
       'WaterSourceRiver_DriedDummy', 'WaterSourceWell_DriedDummy',
       'WaterSourceSpout_DriedDummy', 'WaterSourceSpout_IncreaseDummy',
       'WaterSourceSpout_DecreaseDummy', 'WaterSourcePipe_IncreaseDummy',
       'WaterSourcePipe_DecreaseDummy', 'WaterSourceChangePast25_Dummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 10-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S10_1.dta")

# %%
df_read.shape

# %%
df_read

# %%
# Categorical summaries
for col in ['J03', 'J04', 'J05', 'J06', 'J07']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['Drop_Dummy'] = np.where(df_read['J03'] == 'Changed', 1, 0)

# %%
df_read.columns

# %%
df_read = df_read[['PSU', 'HHLD', 'J01CODE', 'Drop_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="J01CODE", 
           values="Drop_Dummy")
    .reset_index()
)

# %%
df_wide.columns

# %%
abbr_map = {
    "Tree": "Tree",
    "Shrub / bush": "Shrub",
    "Herbal plant / non-timber forest product": "Herbal",
    "Grass / Fodder": "Grass",
    "Aquatic animal": "AquAnimal",
    "Aquatic plant": "AquPlant",
    "Wild animal": "WildAnimal",
    "Birds": "Birds",
    "Insects": "Insects"
}
new_cols = []
for c in df_wide.columns:
    if c in ["PSU", "HHLD"]:
        new_cols.append(c)
    elif c in abbr_map:
        new_cols.append(f"{abbr_map[c]}_Drop_Dummy")
    else:
        new_cols.append(c)
df_wide.columns = new_cols

# %%
df_wide

# %%
df_kept = df_kept.merge(df_wide, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 10-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S10_2.dta")

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['J10YN', 'J11', 'J12', 'J13',
       'J14A', 'J14B', 'J14C']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['Invasion_Dummy'] = np.where(df_read['J10YN'] == 'Yes', 1, 0)

# %%
df_read = df_read[['PSU', 'HHLD', 'J09TYPE', 'Invasion_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="J09TYPE", 
           values="Invasion_Dummy")
    .reset_index()
)

# %%
df_wide.columns

# %%
abbr_map = {
    "1. Shrub / bush": "Shrub",
    "2. Climber": "Climber",
    "3. Creeper": "Creeper"
}
new_cols = []
for c in df_wide.columns:
    if c in ["PSU", "HHLD"]:
        new_cols.append(c)
    elif c in abbr_map:
        new_cols.append(f"{abbr_map[c]}_Invasion_Dummy")
    else:
        new_cols.append(c)
df_wide.columns = new_cols

# %%
df_kept = df_kept.merge(df_wide, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 10-3

# %% [markdown]
# Yes is Yes

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S10_3.dta")

# %%
df_read.shape

# %%
df_read

# %%
# Categorical summaries
for col in ['J23']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read.columns

# %%
name_list = ['EarlyTree_Dummy', 'LaterTree_Dummy', 'EarlyShrub_Dummy', 'LaterShrub_Dummy',
             'EarlyFruit_Dummy', 'LaterFruit_Dummy', 'EarlyHerb_Dummy', 'LaterHerb_Dummy']
code_list = ['J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22']
for name, code in zip(name_list, code_list):
    df_read[name] = np.where(df_read[code] == 'Yes', 1, 0)

# %%
df_read['FruitSizeIncrease_Dummy'] = np.where(df_read['J23'] == '', 1, 0)
df_read['FruitSizeDecrease_Dummy'] = np.where(df_read['J23'] == '', 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'EarlyTree_Dummy', 'LaterTree_Dummy', 'EarlyShrub_Dummy',
       'LaterShrub_Dummy', 'EarlyFruit_Dummy', 'LaterFruit_Dummy',
       'EarlyHerb_Dummy', 'LaterHerb_Dummy', 'FruitSizeIncrease_Dummy',
       'FruitSizeDecrease_Dummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 11

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S11.dta",
                       convert_categoricals=False)

# %%
df_read.shape

# %%
df_read.columns

# %%
df_read[['PSU', 'HHLD', 'K01']].describe()

# %%
df_read['TouristImportance'] = (df_read['K01'] - 2).abs()

# %%
df_select = df_read[['PSU', 'HHLD', 'TouristImportance']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 12

# %%
df_read = pd.read_stata("PrivateData/Climate-2016/Data/Data/S12.dta", convert_categoricals=False)

# %%
df_read.shape

# %%
for name in df_read.columns[2:]:
    df_read[name] = np.where(df_read[name] == 1, 1, 0)

# %%
df_read.columns = ['PSU', 'HHLD', 
                   'SkillTrainingPast25', 'ChangeCropPatternPast25', 'LeftLandFallow', 'RearedLivestockPAst25', 'SuppIrrigationPast25',
                   'InvestIrrigationPast25', 'ImprovedSeedPast25', 'ChangePlantingDatePast25', 'IncreaseInorganicFertilizersPAst25', 'IncreaseOrganicFertilizersPAst25',
                   'NewCropsPast25', 'NewLivestockPast25', 'InvestLivestockPestPast25', 'InsuranceLivestockPast25', 'InsuranceCropPast25',
                   'FarmingLivestockPast25', 'FarmingCropPast25', 'FarmingLivestockAndCropPast25', 'AgroForestPast25', 'CompatibleCropPast25',
                   'TunnelFramingPast25', 'ColdStoragePast25', 'SeedBankPast25', 'SoilWaterConservationPast25', 'VisitClimateOfficePast25',
                   'FoodConsumptionHabitPast25', 'OfffarmActiPast25', 'NonFarmEmployPast25', 'FamilyMigrationPast25', 'RiskReductionPast25',
                   'RoadImprovementPast25', 'CommunityPartipationPast25'
                  ]

# %%
df_read.describe()

# %%
df_select = df_read

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# ### Save Data

# %%
df_kept.to_parquet('Data/01_Napel2016.parquet')

# %%

# %%

# %% [markdown]
# ## DAtaset 2022

# %%
# %ls "PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/"

# %%

# %% [markdown]
# ### Data Merger

# %% [markdown]
# #### Section 01

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S01.dta", 
                        convert_categoricals=False)

# %%
df_read

# %%
df_read.columns

# %%
df_read = df_read[['psu', 'hhld', 'respsex', 'respage', 'a14', 'a15']]

# %%
df_read.columns = ['PSU', 'HHLD', 
                  'Respon_Female', 'Respon_Age', 'Edu', 'LivingYear']

# %%
df_read.loc[:, 'Respon_Female'] = df_read.loc[:, 'Respon_Female'] - 1

# %%
df_read = df_read.assign(
    Edu_UnderSLC     = np.where(df_read["Edu"] < 12, 1, 0),
    Edu_Certificate  = np.where(df_read["Edu"] == 12, 1, 0),
    Edu_Bachelor     = np.where(df_read["Edu"] == 13, 1, 0),
    Edu_Master       = np.where(df_read["Edu"] == 14, 1, 0),
    Edu_PhD          = np.where(df_read["Edu"] == 15, 1, 0),
    Edu_Literal      = np.where(df_read["Edu"] == 16, 1, 0),
    Edu_Illiterate   = np.where(df_read["Edu"] == 17, 1, 0)
) 

# %%
edu_year_map = {
    1: 1,  2: 2,  3: 3,  4: 4,  5: 5,  6: 6,  7: 7,  8: 8,  9: 9, 10: 10, 11: 11,
    12: 12,      # Certificate level
    13: 16,      # Bachelor
    14: 18,      # Master
    15: 21,      # PhD
    16: 0,       # Literal
    17: 0        # Illiterate
}

df_read["Edu_year"] = df_read["Edu"].map(edu_year_map).fillna(0).astype(int)

# %%
df_kept = df_read[['PSU', 'HHLD', 'Respon_Female',
       'Respon_Age', 'LivingYear', 'Edu_UnderSLC', 'Edu_Certificate',
       'Edu_Bachelor', 'Edu_Master', 'Edu_PhD', 'Edu_Literal',
       'Edu_Illiterate', "Edu_year"]]

# %%
df_kept

# %%

# %%

# %% [markdown]
# #### Section 02-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S02_1.dta", convert_categoricals=False)
df_read['PSU'] = df_read['psu']
df_read['HHLD'] = df_read['hhld']

# %%
df_read

# %%
df_read['Female_Ratio'] = df_read['b05sex'] - 1
df_read['U18_Ratio'] = np.where(df_read["b06age"] < 18, 1, 0)
df_read['A65_Ratio'] = np.where(df_read["b06age"] >= 65, 1, 0)

df_read["B07EDU"] = df_read["b07edu"].fillna(17)
df_read['Edu12_Ratio'] = np.where((df_read["b07edu"] >= 12)&(df_read["b07edu"] < 16), 1, 0)
df_read['Literal_Ratio'] = np.where((df_read["b07edu"] >= 12)&(df_read["b07edu"] < 17), 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'Female_Ratio', 'U18_Ratio', 'A65_Ratio',
       'Edu12_Ratio', 'Literal_Ratio']]

# %%
df_count = df_read.groupby(['PSU', 'HHLD']).count().reset_index()['b02sn']

# %%
df_count

# %%
df_select_ratio = df_select.groupby(['PSU', 'HHLD']).mean().reset_index()

# %%
df_select_ratio['Household_memberNum'] = df_count

# %%
df_select_ratio

# %%
df_kept = df_kept.merge(df_select_ratio, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.head()

# %%

# %%

# %% [markdown]
# #### Section 02-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S02_2.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['B12A', 'B12B', 'B12C', 'B13A', 'B13B', 'B13C']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
# Categorical summaries
for col in ['TV', 'LANDLINEPHONE',
       'ORDINRYMOBILE', 'SMARTMOBILE', 'COMPUTER', 'INTERNET', 'CARJEEP',
       'ELECTRCVEHICL', 'MOTORCYCLE', 'ELECTRCBIKE', 'BICYCLE', 'ELECTRCFAN',
       'REFRIGERATOR', 'WASHINGMACHINE', 'AIRCONDITNR', 'NOFACILITY']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S02_2.dta", convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
# Categorical summaries
for col in ['TV', 'LANDLINEPHONE',
       'ORDINRYMOBILE', 'SMARTMOBILE', 'COMPUTER', 'INTERNET', 'CARJEEP',
       'ELECTRCVEHICL', 'MOTORCYCLE', 'ELECTRCBIKE', 'BICYCLE', 'ELECTRCFAN',
       'REFRIGERATOR', 'WASHINGMACHINE', 'AIRCONDITNR', 'NOFACILITY']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
df_read["Radio_dummy"] = df_read["RADIO"]
df_read["TV_dummy"] = df_read["TV"]
df_read["PC_dummy"] = df_read["COMPUTER"]
df_read["Net_dummy"] = df_read["INTERNET"]
df_read["Phone_dummy"] = df_read["LANDLINEPHONE"]
df_read["Mobile_dummy"] = ((df_read["ORDINRYMOBILE"] == 1) | (df_read["SMARTMOBILE"] == 1)).astype(int)
df_read["Motorbike_dummy"] = df_read["MOTORCYCLE"]
df_read["Car_dummy"] = df_read["CARJEEP"]
df_read["Bike_dummy"] = df_read["BICYCLE"]
df_read["OtherVehi_dummy"] = df_read["ELECTRCBIKE"]
df_read["Refrige_dummy"] = df_read["REFRIGERATOR"]

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'B10', 'B11', 'B12A', 'B12B',
        'B12C', 'B13A', 'B13B', 'B13C', 'B14', 'B15', 'B16A', 'B16B', 'B16C',
         'B18', 'C01', 'C02', 'Radio_dummy', 'TV_dummy',
       'PC_dummy', 'Net_dummy', 'Phone_dummy', 'Mobile_dummy',
       'Motorbike_dummy', 'Car_dummy', 'Bike_dummy', 'OtherVehi_dummy',
       'Refrige_dummy']].fillna(0)

# %%
df_select.columns = ['PSU', 'HHLD', 'Own_Resid', 'Resid_Type', 'WaterS1', 'WaterS2',
        'WaterS3', 'CookFuelS1', 'CookFuelS2', 'CookFuelS3', 'LightEnergy', 'Toilet', 'IncomeS1', 'IncomeS2', 'IncomeS3',
         'Remittance_dummy', 'Have_AgriLand', 'HouseHead_AgriExpYear', 'Radio_dummy',
       'TV_dummy', 'PC_dummy', 'Net_dummy', 'Phone_dummy', 'Mobile_dummy',
       'Motorbike_dummy', 'Car_dummy', 'Bike_dummy', 'OtherVehi_dummy',
       'Refrige_dummy']

# %%
df_select['Remittance_dummy'] = (df_select['Remittance_dummy'] - 2).abs()
df_select['Have_AgriLand'] = (df_select['Have_AgriLand'] - 2).abs()

# %%
df_select

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 04

# %% [markdown]
# **Yes is Yes**

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S04.dta", convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read['SavingMembership'] = np.where(df_read["D01"] == 1, 1, 0)
df_read['RegularSaving'] = np.where(df_read["D02"] == 1, 1, 0)
df_read['OrgMembership'] = np.where(df_read["D07"] == 1, 1, 0)
df_read['AgriSupport'] = np.where(df_read["D10"] == 1, 1, 0)
df_read['Dist_Road'] = df_read["D12"]
df_read['Dist_HealthCenter'] = df_read["D13"]
df_read['Dist_SecondarySchool'] = df_read["D14"]
df_read['Dist_Market'] = df_read["D15"]
df_read['Dist_AgriSupport'] = df_read["D16"]
df_read['FramMechan'] = np.where(df_read["D17"] == 1, 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'SavingMembership',
       'RegularSaving', 'OrgMembership', 'AgriSupport', 'Dist_Road',
       'Dist_HealthCenter', 'Dist_SecondarySchool', 'Dist_Market',
       'Dist_AgriSupport', 'FramMechan']]

# %%
df_select = df_select.fillna(0)

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 05

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S05.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read['TotalIncome'] = df_read[['E01', 'E02', 'E03', 'E04']].sum(axis=1)

# %%
df_read.columns = ['PSU', 'HHLD', 'CropIncome', 'LivestockIncome', 'NonAgriIncome', 'BusiIncome', 'TotalIncome']

# %%
df_read = df_read.fillna(0)

# %%
df_select = df_read

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 06

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S06.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
df_read['HeardClimate_Dummy'] = np.where(df_read['F01'] == 'Yes', 1, 0)
df_read['ClimateChanged_Dummy'] = np.where(df_read['F03'] == 'Yes', 1, 0)

# %%
f02_dummies = pd.get_dummies(df_read["F02"], prefix="ClimateInfo").astype(int)

# %%
f02_dummies.columns = [name[:17] for name in f02_dummies.columns]

# %%
f02_dummies = f02_dummies[['ClimateInfo_Radio',
       'ClimateInfo_Telev', 'ClimateInfo_News ', 'ClimateInfo_Aware',
       'ClimateInfo_Local', 'ClimateInfo_Neigh', 'ClimateInfo_Famil',
       'ClimateInfo_Other']]

# %%
df_read = pd.concat([df_read, f02_dummies], axis=1)

# %%
f04a_dummies = pd.get_dummies(df_read["F04A"], prefix="ClimateReasonA").astype(int)

# %%
f04a_dummies.columns = [name[:20] for name in f04a_dummies.columns]

# %%
f04a_dummies = f04a_dummies[['ClimateReasonA_Defor', 'ClimateReasonA_Natur',
       'ClimateReasonA_Indus', 'ClimateReasonA_Urban', 'ClimateReasonA_Human',
       'ClimateReasonA_God\'s', 'ClimateReasonA_Earth', 'ClimateReasonA_Other',
       'ClimateReasonA_Don\'t']]

# %%
df_read = pd.concat([df_read, f04a_dummies], axis=1)

# %%
f04b_dummies = pd.get_dummies(df_read["F04B"], prefix="ClimateReasonB").astype(int)

# %%
f04b_dummies.columns = [name[:20] for name in f04b_dummies.columns]

# %%
f04b_dummies = f04b_dummies[['ClimateReasonB_Defor', 'ClimateReasonB_Natur',
       'ClimateReasonB_Indus', 'ClimateReasonB_Urban', 'ClimateReasonB_Human',
       'ClimateReasonB_God\'s', 'ClimateReasonB_Earth', 'ClimateReasonB_Other']]

# %%
df_read = pd.concat([df_read, f04b_dummies], axis=1)

# %%
f04c_dummies = pd.get_dummies(df_read["F04C"], prefix="ClimateReasonC").astype(int)

# %%
f04c_dummies.columns = [name[:20] for name in f04c_dummies.columns]

# %%
f04c_dummies = f04c_dummies[['ClimateReasonC_Defor', 'ClimateReasonC_Natur', 'ClimateReasonC_Indus',
       'ClimateReasonC_Urban', 'ClimateReasonC_Human', 'ClimateReasonC_God\'s',
       'ClimateReasonC_Earth', 'ClimateReasonC_Other']]

# %%
df_read = pd.concat([df_read, f04c_dummies], axis=1)

# %%
df_read.columns = [name.replace( '\'' ,'x') for name in df_read.columns]

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'HeardClimate_Dummy', 'ClimateChanged_Dummy', 'ClimateInfo_Radio',
       'ClimateInfo_Telev', 'ClimateInfo_News ', 'ClimateInfo_Aware',
       'ClimateInfo_Local', 'ClimateInfo_Neigh', 'ClimateInfo_Famil',
       'ClimateInfo_Other', 'ClimateReasonA_Defor', 'ClimateReasonA_Natur',
       'ClimateReasonA_Indus', 'ClimateReasonA_Urban', 'ClimateReasonA_Human',
       'ClimateReasonA_Godxs', 'ClimateReasonA_Earth', 'ClimateReasonA_Other',
       'ClimateReasonA_Donxt', 'ClimateReasonB_Defor', 'ClimateReasonB_Natur',
       'ClimateReasonB_Indus', 'ClimateReasonB_Urban', 'ClimateReasonB_Human',
       'ClimateReasonB_Godxs', 'ClimateReasonB_Earth', 'ClimateReasonB_Other',
       'ClimateReasonC_Defor', 'ClimateReasonC_Natur', 'ClimateReasonC_Indus',
       'ClimateReasonC_Urban', 'ClimateReasonC_Human', 'ClimateReasonC_Godxs',
       'ClimateReasonC_Earth', 'ClimateReasonC_Other']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 06-3

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S06_3.dta",
                       convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read

# %%
df_read['ExpDummy'] = np.where(df_read['F17'] == 1, 1, 0)
df_read['Impact'] = df_read['F18'].fillna(0)

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="F15SN", 
           values=["ExpDummy", 'Impact'])
    .reset_index()
)

# %%
df_wide

# %%
abbr_map = {
    1: "DR",
    2: "FF",
    3: "FS",
    4: "FL",
    5: "IN",
    6: "WS",
    7 :"TS",
    8: "HS",
    9: "HR",
    10: "SR",
    11: "SE",
    12: "LS",
    13: "SS",
    14: "AV",
    15: "GLOF",
    16: "HW",
    17: "CW",
    18: "DI",
    19: "OT"
}

# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 07-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S07_1.dta",
                       convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['G03YR', 'G04', 'G05', 'G06',
       'G07', 'G08', 'G09', 'G10', 'G11', 'G12']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['DisasterFoodShortage_Dummy'] = np.where(df_read['G06'] == 1, 1, 0)
df_read['DisasterDie_Dummy'] = np.where(df_read['G07'] == 1, 1, 0)

# %%
df_read = df_read[['PSU', 'HHLD', 'G01SN', 'DisasterFoodShortage_Dummy', 'DisasterDie_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="G01SN", 
           values=["DisasterFoodShortage_Dummy", "DisasterDie_Dummy"])
    .reset_index()
)

# %%
df_wide

# %%
abbr_map = {
    1: "DR",
    2: "FF",
    3: "FS",
    4: "FL",
    5: "IN",
    6: "WS",
    7 :"TS",
    8: "HS",
    9: "HR",
    10: "SR",
    11: "SE",
    12: "LS",
    13: "SS",
    14: "AV",
    15: "GLOF",
    16: "HW",
    17: "CW",
    18: "DI",
    19: "OT"
}

# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 07-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S07_2.dta",
                       convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['G15SN']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['DisasterMoneyLoss_Dummy'] = np.where(df_read['G17YN'] == 'Yes', 1, 0)

df_read['DisasterMoneyLoss_TotalNRs'] = df_read['G18'].fillna(0)

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="G15SN", 
           values=["DisasterMoneyLoss_Dummy", "DisasterMoneyLoss_TotalNRs"])
    .reset_index()
)

# %%
abbr_map = {
    1: "DR",
    2: "FF",
    3: "FS",
    4: "FL",
    5: "IN",
    6: "WS",
    7 :"TS",
    8: "HS",
    9: "HR",
    10: "SR",
    11: "SE",
    12: "LS",
    13: "SS",
    14: "AV",
    15: "GLOF",
    16: "HW",
    17: "CW",
    18: "DI",
    19: "OT"
}

# %%
df_wide.columns = [
    col if isinstance(col, str) else f"{col[0]}{abbr_map.get(col[1], col[1])}"
    for col in df_wide.columns.to_flat_index()
]

# %%
df_wide

# %%
df_select = df_wide

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %% [markdown]
# #### Section 08

# %% [markdown]
# Yes is Yes

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S08.dta",
                       convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read['NewDiseaseCropPast25_Dummy'] = np.where(df_read['H01'] == 'Yes', 1, 0)
df_read['NewInsectCropPast25_Dummy'] = np.where(df_read['H03'] == 'Yes', 1, 0)
df_read['NewDiseaseLivestockPast25_Dummy'] = np.where(df_read['H05'] == 'Yes', 1, 0)

df_read['HumanDiseaseIncreasePast25_Dummy'] = np.where(df_read['H07'] == 'Yes', 1, 0)
df_read['HumanVetorDisIncreasePast25_Dummy'] = np.where(df_read['H09'] == 'Yes', 1, 0)
df_read['HumanWaterDisIncreasePast25_Dummy'] = np.where(df_read['H10'] == 'Yes', 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'NewDiseaseCropPast25_Dummy', 'NewInsectCropPast25_Dummy',
       'NewDiseaseLivestockPast25_Dummy', 'HumanDiseaseIncreasePast25_Dummy',
       'HumanVetorDisIncreasePast25_Dummy',
       'HumanWaterDisIncreasePast25_Dummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 09

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S09.dta",
                       convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read['WaterSourceRiver_IncreaseDummy'] = np.where(df_read['I05'] == 1, 1, 0)
df_read['WaterSourceRiver_DecreaseDummy'] = np.where(df_read['I05'] == 2, 1, 0)


df_read['WaterSourceRiver_DriedDummy'] = np.where(df_read['I06'] == 1, 1, 0)

df_read['WaterSourceSpout_IncreaseDummy'] = np.where(df_read['I03'] == 1, 1, 0)
df_read['WaterSourceSpout_DecreaseDummy'] = np.where(df_read['I03'] == 2, 1, 0)

df_read['WaterSourcePipe_IncreaseDummy'] = np.where(df_read['I07'] == 1, 1, 0)
df_read['WaterSourcePipe_DecreaseDummy'] = np.where(df_read['I07'] == 2, 1, 0)

# %%
df_read.columns

# %%
df_select = df_read[['PSU', 'HHLD', 'WaterSourceRiver_IncreaseDummy',
       'WaterSourceRiver_DecreaseDummy', 'WaterSourceRiver_DriedDummy',
       'WaterSourceSpout_IncreaseDummy', 'WaterSourceSpout_DecreaseDummy',
       'WaterSourcePipe_IncreaseDummy', 'WaterSourcePipe_DecreaseDummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 10-1

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S10_1.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
# Categorical summaries
for col in ['J03']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['Drop_Dummy'] = np.where(df_read['J03'] == 'Changed', 1, 0)

# %%
df_read.columns

# %%
df_read = df_read[['PSU', 'HHLD', 'J01SN', 'Drop_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="J01SN", 
           values="Drop_Dummy")
    .reset_index()
)

# %%
df_wide.columns

# %%
abbr_map = {
    "Tree": "Tree",
    "Shrub": "Shrub",
    "Herbal plant / non-timber forest product": "Herbal",
    "Grass / fodder": "Grass",
    "Aquatic animal": "AquAnimal",
    "Aquatic plant": "AquPlant",
    "Wild animal": "WildAnimal",
    "Birds": "Birds",
    "Insects": "Insects"
}
new_cols = []
for c in df_wide.columns:
    if c in ["PSU", "HHLD"]:
        new_cols.append(c)
    elif c in abbr_map:
        new_cols.append(f"{abbr_map[c]}_Drop_Dummy")
    else:
        new_cols.append(c)
df_wide.columns = new_cols

# %%
df_wide

# %%
df_kept = df_kept.merge(df_wide, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 10-2

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S10_2.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
df_read.columns

# %%
# Categorical summaries
for col in ['J10SN', 'J12YN']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read['Invasion_Dummy'] = np.where(df_read['J12YN'] == 'Yes', 1, 0)

# %%
df_read = df_read[['PSU', 'HHLD', 'J10SN', 'Invasion_Dummy']]

# %%
df_wide = (
    df_read
    .pivot(index=["PSU", "HHLD"], 
           columns="J10SN", 
           values="Invasion_Dummy")
    .reset_index()
)

# %%
df_wide.columns

# %%
abbr_map = {
    "Shrub": "Shrub",
    "Climber plant climb on Tree": "Climber",
    "Creeper plant on land": "Creeper"
}
new_cols = []
for c in df_wide.columns:
    if c in ["PSU", "HHLD"]:
        new_cols.append(c)
    elif c in abbr_map:
        new_cols.append(f"{abbr_map[c]}_Invasion_Dummy")
    else:
        new_cols.append(c)
df_wide.columns = new_cols

# %%
df_wide.columns

# %%
df_wide = df_wide[['PSU', 'HHLD', 'Shrub_Invasion_Dummy', 'Climber_Invasion_Dummy',
       'Creeper_Invasion_Dummy']]

# %%
df_kept = df_kept.merge(df_wide, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Section 10-3

# %% [markdown]
# Yes is Yes

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S10_3.dta")

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
df_read

# %%
# Categorical summaries
for col in ['J23']:
    print(f"\n--- {col} ---")
    print(df_read[col].value_counts(dropna=False))

# %%
df_read.columns

# %%
name_list = ['EarlyTree_Dummy', 'LaterTree_Dummy', 'EarlyShrub_Dummy', 'LaterShrub_Dummy',
             'EarlyFruit_Dummy', 'LaterFruit_Dummy', 'EarlyHerb_Dummy', 'LaterHerb_Dummy']
code_list = ['J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25']
for name, code in zip(name_list, code_list):
    df_read[name] = np.where(df_read[code] == 'Yes', 1, 0)

# %%
df_select = df_read[['PSU', 'HHLD', 'EarlyTree_Dummy', 'LaterTree_Dummy', 'EarlyShrub_Dummy',
       'LaterShrub_Dummy', 'EarlyFruit_Dummy', 'LaterFruit_Dummy',
       'EarlyHerb_Dummy', 'LaterHerb_Dummy']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %% [markdown]
# #### Section 11

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/S11.dta", convert_categoricals=False)

# %%
df_read.columns = [name.upper() for name in df_read.columns]

# %%
df_read.shape

# %%
for name in df_read.columns[2:]:
    df_read[name] = np.where(df_read[name] == 1, 1, 0)

# %%
df_read

# %%
df_read.columns

# %%
df_read = df_read[['PSU', 'HHLD', 'K01', 'K02', 'K03', 'K04', 'K05', 
                   'K06', 'K07', 'K08', 'K09', 'K10',
                  'K12', 'K13', 'K14', 'K16', 'K19',
                  'K22', 'K23', 'K24', 'K25', 'K26',
                  'K28', 'K30', 'K31', 'K32',
                  'K38', 'K39', 'K40', 'K41', 'K42',
                  'K43', 'K44']]

# %%
df_read.columns = ['PSU', 'HHLD', 
                   'SkillTrainingPast25', 'ChangeCropPatternPast25', 'LeftLandFallow', 'RearedLivestockPAst25', 'SuppIrrigationPast25',
                   'InvestIrrigationPast25', 'ImprovedSeedPast25', 'ChangePlantingDatePast25', 'IncreaseInorganicFertilizersPAst25', 'IncreaseOrganicFertilizersPAst25',
                   'NewCropsPast25', 'NewLivestockPast25', 'InvestLivestockPestPast25', 'InsuranceLivestockPast25', 'InsuranceCropPast25',
                   'FarmingLivestockPast25', 'FarmingCropPast25', 'FarmingLivestockAndCropPast25', 'AgroForestPast25', 'CompatibleCropPast25',
                   'TunnelFramingPast25', 'ColdStoragePast25', 'SoilWaterConservationPast25', 'VisitClimateOfficePast25',
                   'FoodConsumptionHabitPast25', 'OfffarmActiPast25', 'NonFarmEmployPast25', 'FamilyMigrationPast25', 'RiskReductionPast25',
                   'RoadImprovementPast25', 'CommunityPartipationPast25'
                  ]

# %%
df_read.describe()

# %%
df_select = df_read

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %% [markdown]
# #### Weights

# %%
df_read = pd.read_stata("PrivateData/Climate-2022/Data 2022/NCCS 2022/Data/Weight.dta")

# %%
df_read['Rural_Dummy'] = np.where(df_read['UrbRur'] == 'Rural', 1, 0)

# %%
df_read.columns

# %%
df_read.head()

# %%
df_read['Prov'].unique()

# %%
df_select = df_read[['PSU', 'HHLD', 'Rural_Dummy', 'EcoBelt', 'Prov']]

# %%
df_kept = df_kept.merge(df_select, on = ['PSU', 'HHLD'], how='left')

# %%
df_kept.shape

# %%

# %%

# %%

# %% [markdown]
# ### Save Data

# %%
df_kept.to_parquet('Data/01_Napel2022.parquet')

# %%
