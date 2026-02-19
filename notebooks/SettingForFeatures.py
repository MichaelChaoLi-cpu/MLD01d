import geopandas as gpd
import os
import pandas as pd

def data_load_combine_dataset():
    df_2016 = pd.read_parquet('data/processed/01_Napel2016.parquet')
    df_2016['Year'] = 2016
    df_2022 = pd.read_parquet('data/processed/01_Napel2022.parquet')
    df_2022['Year'] = 2022
    df_all = pd.concat([df_2016, df_2022], axis = 0)
    df_all = df_all.dropna(axis=1, how="any")
    df_all = df_all.set_index(['PSU', 'HHLD'])

    df_all['DisasterExpInd'] = df_all[[
        'ExpDummyDR', 'ExpDummyFF', 'ExpDummyFS', 'ExpDummyFL', 'ExpDummyIN', 'ExpDummyWS', 
        'ExpDummyTS', 'ExpDummyHS', 'ExpDummyHR', 'ExpDummySR', 'ExpDummySE', 'ExpDummyLS', 
        'ExpDummySS', 'ExpDummyAV', 'ExpDummyGLOF', 'ExpDummyHW', 'ExpDummyCW', 'ExpDummyDI', 
        'ExpDummyOT',
    ]].sum(axis = 1)
    df_all['DisasterFoodShortageInd'] = df_all[[
        'DisasterFoodShortage_DummyDR', 'DisasterFoodShortage_DummyFF', 'DisasterFoodShortage_DummyFS', 
        'DisasterFoodShortage_DummyFL', 'DisasterFoodShortage_DummyIN', 'DisasterFoodShortage_DummyWS', 
        'DisasterFoodShortage_DummyTS', 'DisasterFoodShortage_DummyHS', 'DisasterFoodShortage_DummyHR', 
        'DisasterFoodShortage_DummySR', 'DisasterFoodShortage_DummySE', 'DisasterFoodShortage_DummyLS', 
        'DisasterFoodShortage_DummySS', 'DisasterFoodShortage_DummyAV', 'DisasterFoodShortage_DummyGLOF', 
        'DisasterFoodShortage_DummyHW', 'DisasterFoodShortage_DummyCW', 'DisasterFoodShortage_DummyDI', 
        'DisasterFoodShortage_DummyOT', 'DisasterDie_DummyDR', 
    ]].sum(axis = 1)
    df_all['DisasterDieInd'] = df_all[[                   
        'DisasterDie_DummyFF', 'DisasterDie_DummyFS', 'DisasterDie_DummyFL', 'DisasterDie_DummyIN', 
        'DisasterDie_DummyWS', 'DisasterDie_DummyTS', 'DisasterDie_DummyHS', 'DisasterDie_DummyHR', 
        'DisasterDie_DummySR', 'DisasterDie_DummySE', 'DisasterDie_DummyLS', 'DisasterDie_DummySS', 
        'DisasterDie_DummyAV', 'DisasterDie_DummyGLOF', 'DisasterDie_DummyHW', 'DisasterDie_DummyCW', 
        'DisasterDie_DummyDI', 'DisasterDie_DummyOT',  
    ]].sum(axis = 1)
    df_all['DisasterMoneyLoss'] = df_all[[                   
       'DisasterMoneyLoss_DummyDR', 'DisasterMoneyLoss_DummyFF', 'DisasterMoneyLoss_DummyFS', 
       'DisasterMoneyLoss_DummyFL', 'DisasterMoneyLoss_DummyIN', 'DisasterMoneyLoss_DummyWS', 
       'DisasterMoneyLoss_DummyTS', 'DisasterMoneyLoss_DummyHS', 'DisasterMoneyLoss_DummyHR', 
       'DisasterMoneyLoss_DummySR', 'DisasterMoneyLoss_DummySE', 'DisasterMoneyLoss_DummyLS', 
       'DisasterMoneyLoss_DummySS', 'DisasterMoneyLoss_DummyAV', 'DisasterMoneyLoss_DummyGLOF',
       'DisasterMoneyLoss_DummyHW', 'DisasterMoneyLoss_DummyCW', 'DisasterMoneyLoss_DummyDI', 
       'DisasterMoneyLoss_DummyOT', 
    ]].sum(axis = 1)
    
    df_all["EcoBelt"] = df_all["EcoBelt"].str.replace('Tarai', 'Terai')
    
    eco_dummies = pd.get_dummies(df_all["EcoBelt"], prefix="EcoBelt").astype(int)
    df_all = pd.concat([df_all, eco_dummies], axis=1)
    prov_dummies = pd.get_dummies(df_all["Prov"], prefix="Prov").astype(int)
    df_all = pd.concat([df_all, prov_dummies], axis=1)

    df_all['IncomeResAgri_dummy'] = (df_all[['IncomeS1', 'IncomeS2', 'IncomeS3']] == 1).any(axis=1).astype(int)
    df_all['IncomeResWage_dummy'] = (df_all[['IncomeS1', 'IncomeS2', 'IncomeS3']] == 2).any(axis=1).astype(int)
    df_all['IncomeResNonAgriBusi_dummy'] = (df_all[['IncomeS1', 'IncomeS2', 'IncomeS3']] == 3).any(axis=1).astype(int)
    df_all['IncomeResRemit_dummy'] = (df_all[['IncomeS1', 'IncomeS2', 'IncomeS3']] == 4).any(axis=1).astype(int)
    df_all['IncomeResOthers_dummy'] = (df_all[['IncomeS1', 'IncomeS2', 'IncomeS3']] == 5).any(axis=1).astype(int)

    df_all['ResidenceOwn_dummy'] = (df_all['Own_Resid'] == 1).astype(int)
    df_all['ResidenceRent_dummy'] = (df_all['Own_Resid'] == 2).astype(int)
    df_all['ResidenceInstitu_dummy'] = (df_all['Own_Resid'] == 3).astype(int)
    df_all['ResidenceOthers_dummy'] = (df_all['Own_Resid'] == 4).astype(int)

    df_all['ResidInfraPerman_dummy'] = (df_all['Resid_Type'] == 1).astype(int)
    df_all['ResidInfraSemi_dummy'] = (df_all['Resid_Type'] == 2).astype(int)
    df_all['ResidInfraKachchi_dummy'] = (df_all['Resid_Type'] == 3).astype(int)
    df_all['ResidInfraOthers_dummy'] = (df_all['Resid_Type'] == 4).astype(int)

    df_all['Ind_income'] = df_all['TotalIncome'] / df_all['Household_memberNum']
    return df_all

def return_input_variables():
    return [
        'HeardClimate_Dummy', 'ClimateChanged_Dummy',
        'Respon_Female', 'Respon_Age', 'LivingYear', 'Edu_Literal', 'Edu_Illiterate', 'Edu_year', # S01
        'Female_Ratio', 'U18_Ratio', 'A65_Ratio', 'Edu12_Ratio', 'Literal_Ratio', # S02-1
        'EcoBelt_Hill', 'EcoBelt_Mountain', 'EcoBelt_Terai', 
        'Prov_Bagmati', 'Prov_Koshi', 'Prov_Lumbini', 'Prov_Madhesh', 'Prov_Sudurpaschim',
        'Prov_Gandaki', 'Prov_Karnali', # location     
        'ResidenceOwn_dummy', 'ResidenceRent_dummy', 'ResidenceInstitu_dummy', 'ResidenceOthers_dummy',
        'ResidInfraPerman_dummy', 'ResidInfraSemi_dummy', 'ResidInfraKachchi_dummy', 'ResidInfraOthers_dummy', # house
        'Remittance_dummy', 
        'Have_AgriLand', 'HouseHead_AgriExpYear',
        'Radio_dummy', 'TV_dummy', 'PC_dummy', 'Net_dummy', 'Phone_dummy',
        'Mobile_dummy', 'Motorbike_dummy', 'Car_dummy', 'Bike_dummy', 'OtherVehi_dummy', 'Refrige_dummy',
        'SavingMembership', 'RegularSaving', 'OrgMembership', 'AgriSupport', 
        'Dist_Road', 'Dist_HealthCenter', 'Dist_SecondarySchool', 'Dist_Market', 'Dist_AgriSupport', 
        'FramMechan',
        'IncomeResAgri_dummy', 'IncomeResWage_dummy', 'IncomeResNonAgriBusi_dummy', 'IncomeResRemit_dummy',
        'IncomeResOthers_dummy', 'TotalIncome',
        'Year',
        'DisasterExpInd', 'DisasterFoodShortageInd', 'DisasterDieInd', 'DisasterMoneyLoss',
    ]


def return_output_variables():
    return [
        "HumanDiseaseIncreasePast25_Dummy",     # main
        #"HumanWaterDisIncreasePast25_Dummy",    # robustness
        #"HumanVetorDisIncreasePast25_Dummy"     # robustness
    ]

def return_beautiful_dict():
    variname_readable = {
        'HumanDiseaseIncreasePast25_Dummy':'Increase in Household Disease Incidence Dummy',
        'HumanWaterDisIncreasePast25_Dummy':'Increase in Water-related Diseases Dummy',
        'HumanVetorDisIncreasePast25_Dummy':'Increase in Vector-borne Diseases Dummy',
        'ClimateChanged_Dummy':'Climate Change Awareness',
        'HeardClimate_Dummy':'Climate Change Knowledge', 'Respon_Female':'Respondent Female Dummy', 
        'Respon_Age':'Repsondent Age', 'LivingYear':'Years Living in Community', 'Edu_UnderSLC':'Education under Secondary Certificate Dummy', 
        'Edu_Certificate':'Education with Secondary Certificate Dummy', 'Edu_Bachelor':'Education with Bachelor Dummy', 
        'Edu_Master':'Education with Master Dummy',  'Edu_PhD':'Education with PhD Dummy', 
        'Edu_Literal':'Literate Education Dummy',  'Edu_Illiterate':'Illiterate Dummy', 'Edu_year':'Education Year',
        'Female_Ratio':'Female Ratio in Household', 'U18_Ratio':'Member Under 18 Ratio', 'A65_Ratio':'Seniors Ratio',
        'Edu12_Ratio':"Member with 12-Year Education or above Ratio", "Literal_Ratio": "Literate Member Ratio",
        'EcoBelt_Hill': "EcoBelt Hill Dummy", 'EcoBelt_Mountain': "EcoBelt Mountain Dummy", 'EcoBelt_Terai': "EcoBelt Terai Dummy",
        'Prov_Bagmati': "Province Bagmati Dummy", 'Prov_Koshi': "Province Koshi Dummy", 'Prov_Lumbini': "Province Lumbibi Dummy",
        'Prov_Madhesh': "Province Madhesh Dummy", 'Prov_Sudurpaschim': "Province Sudurpaschim Dummy", 'Prov_Gandaki': "Province Gandaki Dummy",
        'Prov_Karnali': "Province Karnali Dummy",
        'ResidenceOwn_dummy': "Owned Residence Ownership Dummy", 'ResidenceRent_dummy': 'Rented Residence Ownership Dummy', 
        'ResidenceInstitu_dummy': 'Institutional Residence Ownership Dummy', 'ResidenceOthers_dummy': 'Other-type Residence Ownership Dummy',
        'ResidInfraPerman_dummy': 'Permanent Residence Dummy', 'ResidInfraSemi_dummy': 'Semi-Permanent Residence Dummy', 
        'ResidInfraKachchi_dummy': "Kachchi Residence Dummy", 'ResidInfraOthers_dummy': 'Other Residence Infrastructure Dummy', # house
        'Remittance_dummy' : "Have Remittance",
        'Have_AgriLand': "Having Agricultural Land Dummy", 'HouseHead_AgriExpYear': "Household Head Agricultural Experience",
        'Radio_dummy': "Having Radio Dummy", 'TV_dummy': "Having TV Dummy", 'PC_dummy': "Having Computer Dummy", 'Net_dummy': "Having Internet Dummy",
        'Phone_dummy': "Having Telephone Dummy", 'Mobile_dummy': "Having Mobile Dummy", 'Motorbike_dummy': "Having Motorbike Dummy", 
        'Car_dummy': "Having Car Dummy", 'Bike_dummy': "Having Bike Dummy", 'OtherVehi_dummy': "Having Other Vehicle Dummy", 
        'Refrige_dummy':"Having Refrigator Dummy",
        'SavingMembership': "Saving Membership Dummy", 'RegularSaving': 'Having Regular Saving Dummy', 
        'OrgMembership': 'Having Organization Membership Dummy', 'AgriSupport':'Agricultural Supporting Dummy', 
        'Dist_Road': 'Distance to Motorable Road', 'Dist_HealthCenter': "Distance to Health Center", 
        'Dist_SecondarySchool': "Distance to Secondary School", 'Dist_Market':"Distance to Market", 'Dist_AgriSupport': 'Distance to Agricultural Center', 
        'FramMechan':'Farm Mechanization Dummy',
        'IncomeResAgri_dummy': "Agricultural Income Source Dummy", 'IncomeResWage_dummy': "Wage Income Source Dummy",
        'IncomeResNonAgriBusi_dummy': "Non-Agricultural Business Income Source Dummy", 'IncomeResRemit_dummy': "Remittance Income Dummy",
        'IncomeResOthers_dummy': "Others Income Source Dummy", 
        'CropIncome': "Crop Income", 'LivestockIncome': "Livestock Income", 'NonAgriIncome': "Non-agricultural Income", 
        'BusiIncome': "Business Income", 'TotalIncome': "Total Income",
        'Year': "Survey Year",
        'DisasterExpInd': 'Natural Disaster Experience Indicator',
        'DisasterFoodShortageInd': 'Natural Disaster-related Food Shortage Indicator',
        'DisasterDieInd': 'Natural Disaster-related Death Indicator',
        'DisasterMoneyLoss': 'Natural Disaster-related Loss',
        'Household_memberNum': 'Household Member',
    }
    return variname_readable

def load_spatial_data() -> gpd.GeoDataFrame:
    """
    Load and combine Nepal EcoBelt and Province spatial data.

    This function:
    1. Loads the EcoBelt shapefile (3-class version: Mountain, Hill, Terai).
    2. Loads the Province shapefile.
    3. Converts both to the same CRS (EPSG:4326, WGS84).
    4. Computes the spatial intersection between EcoBelts and Provinces.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing intersected polygons 
                          with both EcoBelt and Province attributes.
    """
    # --- Load EcoBelt shapefile ---
    gdf_eco = gpd.read_file('data/raw/SpatialMaps/nepal_ecobelt_data/3_class_shape/Ecobelts_3Class.shp')
    gdf_eco = gdf_eco.to_crs(epsg=4326)
    
    # --- Load Province shapefile ---
    gdf_nepal = gpd.read_file('data/raw/SpatialMaps/02_PROVINCE/PROVINCE.shp')
    gdf_nepal = gdf_nepal[['geometry', 'Province']].to_crs(epsg=4326)
    
    # --- Intersection ---
    try:
        gdf_intersect = gpd.overlay(gdf_eco, gdf_nepal, how='intersection')
    except Exception as e:
        raise RuntimeError(f"❌ Failed to overlay EcoBelt and Province shapefiles: {e}")

    # --- Optional cleanup ---
    gdf_intersect = gdf_intersect.rename(columns={'EcoBelt': 'Eco_Belt'}) if 'EcoBelt' in gdf_intersect.columns else gdf_intersect
    
    print(f"✅ Loaded spatial data with {len(gdf_intersect)} intersected polygons.")
    return gdf_intersect    