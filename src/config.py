import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/data/housing/data/"
OMI_TRANSACTIONS_PATH = DATA_PATH+"input/other/aggregate_transactions/"
BI_SURVEY_PATH = DATA_PATH+"input/other/sondaggio-bi/"
MEF_INCOMES_PATH = DATA_PATH+"input/other/incomes/"
OMI_PRICES_PATH = DATA_PATH+"input/other/aggregate_prices/"
ISTAT_LOCATIONS_PATH = DATA_PATH+"input/other/locations/"
CENSUS_PATH = DATA_PATH+"input/census/"
INTERMEDIATE_DATA_PATH = DATA_PATH+"intermediate/"
SHAPEFILES_CENSUS_2011_PATH = CENSUS_PATH+"shapefiles_2011/"
RISK_PATH = DATA_PATH+"input/risk/"
ISP_PATH = DATA_PATH+"input/isp/"
FLOOD_EVENTS_PATH = RISK_PATH + "Flood_events/"
INTERMEDIATE_FLOOD_EVENTS_PATH = INTERMEDIATE_DATA_PATH + 'Flood_Events/'
hydro_risk_path = INTERMEDIATE_DATA_PATH+"cut_risk/All_CutRisk_Italy.shp"
census_hydro_risk_path = INTERMEDIATE_DATA_PATH + "All_Italy_census/Italy_census_shapefile.shp"
omi_shapefile_path = INTERMEDIATE_DATA_PATH+'All_OMI_geometry/OMI_geometry_shapefile.shp'
cap_shapefile_path = DATA_PATH + "input/other/cap/CAP.shp"
input_Copernicus_flood_metadata_path = FLOOD_EVENTS_PATH+'flood_events_metadata.csv'
flood_events_shape_path = INTERMEDIATE_FLOOD_EVENTS_PATH + "Flood_Events.shp"
master_table_locations_path = INTERMEDIATE_DATA_PATH + 'master_table_locations.csv'

df_contratti_2024_01_path = ISP_PATH+"FACT_TABLE_CONTRATTI_masked.csv"
df_catasto_2024_01_path = ISP_PATH+"TAB_REL_GAR_BENE_CATASTO_masked.csv"
df_condizioni_2024_01_path = ISP_PATH+"TAB_REL_CONDIZIONI_CONTRATTO_masked.csv"
isp_data_cleaned_path = INTERMEDIATE_DATA_PATH+"isp_cleaned.csv"
isp_distance_and_elevation_floods_path = INTERMEDIATE_DATA_PATH+'distance_and_elevation_flood_events.csv'

df_contratti_2024_04_path = ISP_PATH+"data_isp_2024_05_22/FACT_TABLE_CONTRATTI_masked.csv"
df_catasto_2024_04_path = ISP_PATH+"data_isp_2024_05_22/TAB_REL_GAR_BENE_CATASTO_masked.csv"
df_condizioni_2024_04_path = ISP_PATH+"data_isp_2024_05_22/TAB_REL_CONDIZIONI_CONTRATTO_masked.csv"
isp_data_cleaned_only_new_data_2024_04_path = INTERMEDIATE_DATA_PATH+"isp_cleaned_only_new_data_2024_04.csv"
isp_data_cleaned_2024_04_path = INTERMEDIATE_DATA_PATH+"isp_cleaned_up_to_2024_04.csv"
isp_distance_and_elevation_floods_only_new_data_2024_04_path = INTERMEDIATE_DATA_PATH+'distance_and_elevation_flood_events_only_new_data_2024_04.csv'
isp_distance_and_elevation_floods_2024_04_path = INTERMEDIATE_DATA_PATH+'distance_and_elevation_flood_events_up_to_2024_04.csv'

ISPRA_eventsList_path = DATA_PATH+'input/ISPRA_historical_events.csv'

isp_cleaned_latest = INTERMEDIATE_DATA_PATH+"data_processed_isp_2024_09_30/isp_cleaned_up_to_2024_08_masked_unique.csv"
isp_empirical = INTERMEDIATE_DATA_PATH+'data_processed_isp_2024_09_30/isp_cleaned_up_to_2024_08_masked.csv'
flood_empirical = INTERMEDIATE_DATA_PATH+'data_processed_isp_2024_09_30/distance_and_elevation_flood_events_up_to_2024_08.csv'

hedonic_regression_ABM_path = '/data/housing/data/intermediate/HedonicRegressionABM/'

default_missing = pd._libs.parsers.STR_NA_VALUES
default_missing.remove('NA')
default_missing.remove('None')

