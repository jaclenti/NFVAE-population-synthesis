import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

import requests
import time
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points


import utils
import config


# Params for the call to opentopodata
call_params = {
        'samples_per_request' :100, #max 100
        'delay_between_request' : 1 , #minimum 1
        'wait_api_limit' : 60, #60, 
        'wait_between_floods' : 120, 
        'url' : "https://api.opentopodata.org/v1/eudem25m?",  #using eudem25m api
        'max_distance2event':100000 # distance in meters: 100000 = 100 km
        }


# Select variables to keep for each DataFrame

#TODO: include SUM_of_VAL_VTR and SUM_of_IMP_BMTP_VALORE_BENE if we end up using it. 
var_keep_contratti = ["COD_TIPO_SOGGETTO","COD_TIPOLOGIA_CONTRATTO_DMC","COD_CONTRATTO","DES_PRD_PADRE",\
                     "DES_SEGMENTO","DES_DESTINAZIONE_FINANZIAMENTO","DAT_EROGAZIONE","IMP_MUTUO",
                     "IMP_SALDO","NUM_DURATA_AMMORT_IN_MESI","NUM_BENE","ANNO_COSTRUZIONE","VAL_BENE",
                     "COD_CLASSE_ENERG_ABS","NUM_MQ_SUPERF_COMM","NUM_MQ_SUPERF_EQUIV",\
                      "PRC_INDAF_PERC_LTV","DES_FMP_DES_FINAL_SING",
                     "IMP_FMP_PREZZO_FIN","SUM_of_IMP_ANPA_IMP_REDD_MESE","FLAG_GIOVANI","DES_INDIRIZZO_BENE",
                     "COD_CAP_BENE","COD_CAT_COMUNE_BENE","DES_COMUNE_BENE","DES_CICLO_VITA_BENE","BENE_OGGETTO_ASTA",
                     "GEO_LATITUDINE_BENE_ROUNDED","GEO_LONGITUDINE_BENE_ROUNDED","COD_SUPERNSG_HASH","COD_NDG_HASH",
                     "DES_REGIONE_BENE","COD_PROVINCIA_BENE"]
var_keep_catasto_not_eu = ["COD_CONTRATTO","NUM_BENE","NUM_MQ_SUPERF_COMM","NUM_MQ_SUPERF_EQUIV","VAL_BENE","COD_ID_DATI_CAT_SEQ",
                   "COD_CAT_CATASTALE"]

# Create a new column 'flag_acquisto' based on conditions on DES_DESTINAZIONE_FINANZIAMENTO
flag_acquisto = pd.Series(["ACQUISTO IMMOBILE AD USO ABITATIVO      1A CASA",
                           "COSTRUZIONE ABITAZIONI 1A CASA",
                                                   "ACQUISTO IMMOBILE AD USO ABITATIVO      2A CASA",
                                                   "BANDI 2010-2011 SARDEGNA ACQUISTO       1^CASA",
                                                   "ACQUISTO IMM USO ABITAT                 1A CASA",
                                                   "COSTRUZIONE ABITAZIONI 2A CASA",
                                                   "ACQ ABITAZIONE PRINCIPALE DA ALTRI      SOGGETTI (NO FAM CONSUMATRICI)",
                                                   "BANDI 2010/011 SARDEGNA COSTRUZ. 1^CASA",
                                                   "ACQUISTO IMM USO ABITAT                 2A CASA",
                          "FONDIARIO/EDILIZIO IMMOB.USO ABITATIVO",
                          "ACQUISTO IMM. USO ABITATIVO",
                          "ACQUISTO DI ABITAZIONE PRINCIPALE       SENZA INTERVENTI DI RISTRUT."])

# Define a map from all the values in the ISP data to a set of floor keys. We do this for the most common entries.
floor_map = {"T":"0", "T-1":"0-1", "T-1-2":"0-1-2", '1\\u00B0':"1","Terra":"0", "1-2":"1-2", '2\\u00B0':"2", \
"Primo": "1", "R":"0", "TERRA":"0", "PRIMO":"1", "S1-T-1":"0-1", '3\\u00B0':"3", "terra":"0", \
"primo":"1", "Secondo":"2", "S1-T": "0", "T-S1": "0", "SECONDO":"2", 'T-1\\u00B0':"0", "T 1": "0-1", \
"-":np.nan, "T/1":"0-1", "Terzo":"3", "t-1":"0-1", "t":"0","TERZO":"3", "T-1-S1": "0-1", \
'4\\u00B0':"4", "PT":"0", "P1":"1", "S":"0", "S1":"0", "Quarto":"4", "P2":"2", "secondo":"2", \
 "terzo":"3", "T 1 2":"0-1-2", "Rialzato":"0", "S1-T-1-2":"0-1-2", "T - 1": "0-1", "QUARTO":"4", \
"RIALZATO":"0", "1 2":"1-2", '5\\u00B0':"5", "t-1-2":"0-1-2", "S-T-1":"0-1", "TERRA-PRIMO":"0-1",\
"PT-P1":"0-1", "Quinto":"5", "Rialza":"0", "Second":"2", "T-1-2-S1":"0-1-2", "T/1/2":"0-1-2",\
"1/2":"1-2", "P,1":"1", "T 1": "0-1", "T 1\\u00B0":"0-1", "QUINTO":"5", "second":"2", "T-2": "0-1-2",\
"1\\u00B0-2\\u00B0":"1-2","T+1":"0-1", '6\\u00B0':"6", "quarto":"4", "2-S1":"2", "1-S1": "1", \
"/":np.nan, "rialzato":"0", "T/1\\u00B0":"0-1", "SECOND":"2", "S1-1":"1", "P,T,":"0", "P,2":"2", \
"S1 T 1":"0-1", "2\\u00B0-3\\u00B0":"2-3", "-1":"0", "T,1":"0-1", "SESTO":"6", "T-1-2-":"0-1-2", \
"P3":"3", "RIALZA":"0","P,3":"3", "T  1":"0-1", "TERRA PRIMO":"0-1", "2 3":"2-3", "rialza":"0",\
"S-T":"0", '7\\u00B0':"7", "Terra - Primo":"0-1", "PIANO PRIMO":"1", "-1 T 1":"0-1", "Terra-Primo":"0-1",\
"t 1":"0-1", "T1":"0-1", "PT/P1":"0-1", "Terra e Primo":"0-1", "Rialz,":"0", "S1/T/1":"0-1",\
"PT-1":"0-1", "PIANO TERRA":"0", "Terra e primo":"0-1", "Sesto":"6", "quinto":"5", "2/3":"2-3", "0 1":"0-1",\
"PIANO SECONDO":"2", "I":np.nan, "T-1\\u00B0-2":"0-1-2", "s1-t-1":"0-1", "P,T":"0", "P":np.nan, "P4":"4"}

# It may be useful to also have a numeric column for floors. In this case, choose the lowest floor in case of multi-floor apartments
# Mapping of combined floor ranges to single numeric values
floor_map_numeric = {"0-1":"0", "0-1-2":"0", "1-2":"1", "2-3":"2", "3-4":"3", "4-5":"4","5-6":"5", "1-2-3":"1", "6-7":"6"}

acceptable_floors = pd.Series(list(set(floor_map.values()))+["8","9","10","11","12","13","1-2-3", "3-4", "4-5", "5-6", "6-7"])

def select_relevant_data(df_contratti, flag_acquisto):
    """
    Filters the DataFrame `df_contratti` based on specified conditions and prepares it for further analysis.
    This is mostly to exclude mortgages that are likely to lead to spurious results.

    Parameters:
    df_contratti (pd.DataFrame): The input DataFrame containing contract data.
    flag_acquisto (pd.Series): A list of relevant mortgages that are associated to a purchase.

    Returns:
    pd.DataFrame: The filtered and modified DataFrame.
    """

    # Define values to exclude from the "DES_SEGMENTO" column
    exclude_DES_SEGMENTO = pd.Series(["Aziende", "Imprese", "Agribusiness"])

    # Filter the DataFrame based on multiple conditions:
    # 1. Exclude rows where "COD_TIPO_SOGGETTO" is "G"
    # 2. Exclude rows where "COD_TIPOLOGIA_CONTRATTO_DMC" is "D"
    # 3. Exclude rows where "DES_PRD_PADRE" is "Aedifica e Altro Fondiario"
    # 4. Exclude rows where "DES_SEGMENTO" is in the exclude_DES_SEGMENTO list
    # 5. Include rows where "BENE_OGGETTO_ASTA" is "NO"
    df_contratti_selected = df_contratti[
        (df_contratti["COD_TIPO_SOGGETTO"] != "G") &
        (df_contratti["COD_TIPOLOGIA_CONTRATTO_DMC"] != "D") &
        (df_contratti["DES_PRD_PADRE"] != "Aedifica e Altro Fondiario") &
        ~(df_contratti["DES_SEGMENTO"].isin(exclude_DES_SEGMENTO)) &
        (df_contratti["BENE_OGGETTO_ASTA"] == "NO")
    ]

    # Remove unnecessary columns from the filtered DataFrame
    df_contratti_selected.drop(columns=["COD_TIPOLOGIA_CONTRATTO_DMC", "DES_PRD_PADRE", "DES_SEGMENTO", "BENE_OGGETTO_ASTA"], inplace=True)

    # Initialize the 'flag_acquisto' column to 0
    df_contratti_selected["flag_acquisto"] = 0

    # Set 'flag_acquisto' to 1 for rows where "DES_DESTINAZIONE_FINANZIAMENTO" is in the flag_acquisto list
    df_contratti_selected.loc[df_contratti["DES_DESTINAZIONE_FINANZIAMENTO"].isin(flag_acquisto), "flag_acquisto"] = 1

    return df_contratti_selected



def define_na_and_adjust_types(df):
    """
    Adjusts the types of the variables from reading a csv (e.g. decimal separator in ISP data is "," but pandas expects ".") 
    and sets as NA the values that are not acceptable (e.g. a price or a square footage equal to zero). 

    Parameters:
    df (pd.DataFrame): The input DataFrame containing data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    #TODO: uncomment SUM_of_VAL_VTR and SUM_of_IMP_BMTP_VALORE_BENE if we end up using it. 
    
    # Convert the 'DAT_EROGAZIONE' column to datetime format, handling errors by coercing them to NaT (Not a Time)
    df.loc[:, 'DAT_EROGAZIONE'] = pd.to_datetime(df['DAT_EROGAZIONE'], format='%d/%m/%Y', errors='coerce')

    # Convert the 'IMP_MUTUO' column to numeric format, replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'IMP_MUTUO'] = pd.to_numeric(df['IMP_MUTUO'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

    # Convert the 'IMP_SALDO' column to numeric format, replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'IMP_SALDO'] = pd.to_numeric(df['IMP_SALDO'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

    # Convert the 'NUM_DURATA_AMMORT_IN_MESI' column to numeric format, replacing thousands separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'NUM_DURATA_AMMORT_IN_MESI'] = pd.to_numeric(df['NUM_DURATA_AMMORT_IN_MESI'].str.replace(',', '.'), errors='coerce')

    # Convert the 'ANNO_COSTRUZIONE' column to numeric format, handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'ANNO_COSTRUZIONE'] = pd.to_numeric(df['ANNO_COSTRUZIONE'], errors='coerce')

    # Set NaN for values in 'ANNO_COSTRUZIONE' greater than 2024, as they are likely erroneous data
    df.loc[df['ANNO_COSTRUZIONE'] > 2024, 'ANNO_COSTRUZIONE'] = np.nan

    # Convert the 'VAL_BENE' column to numeric format, replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'VAL_BENE'] = pd.to_numeric(df['VAL_BENE'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

    # Set NaN for values in 'COD_CLASSE_ENERG_ABS' equal to "ND" or "NC", as they indicate missing or unknown data
    df.loc[df['COD_CLASSE_ENERG_ABS'].isin(["ND", "NC"]), 'COD_CLASSE_ENERG_ABS'] = np.nan

    # Convert the 'NUM_MQ_SUPERF_COMM' column to numeric format, replacing thousands separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df.loc[:, 'NUM_MQ_SUPERF_COMM'] = pd.to_numeric(df['NUM_MQ_SUPERF_COMM'].str.replace(',', '.'), errors='coerce')

    # Set NaN for values in 'NUM_MQ_SUPERF_COMM' equal to 0, as they likely represent missing or erroneous data
    df.loc[df['NUM_MQ_SUPERF_COMM'] == 0, 'NUM_MQ_SUPERF_COMM'] = np.nan

    # Convert the 'PRC_INDAF_PERC_LTV' column to numeric format, replacing decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df['PRC_INDAF_PERC_LTV'] = pd.to_numeric(df['PRC_INDAF_PERC_LTV'].str.replace(',', '.'), errors='coerce')

    # Set NaN for values in 'PRC_INDAF_PERC_LTV' equal to 0, as they likely represent missing or erroneous data
    df.loc[df['PRC_INDAF_PERC_LTV'] == 0, 'PRC_INDAF_PERC_LTV'] = np.nan

    # Convert the 'IMP_FMP_PREZZO_FIN' column to numeric format, handling errors by coercing them to NaN (Not a Number)
    # Because in some cases the source data come in the format with comma used as a decimal separator and in other cases
    # the source data come in the format where the dot is used as decimal separator, distinguish the two cases.
    if df["IMP_FMP_PREZZO_FIN"].str.contains(',').any():
        df.loc[:,'IMP_FMP_PREZZO_FIN'] = pd.to_numeric(df['IMP_FMP_PREZZO_FIN'].str.replace('.', '').str.replace(',', '.'), errors='coerce')
    else:
        df.loc[:,'IMP_FMP_PREZZO_FIN'] = pd.to_numeric(df['IMP_FMP_PREZZO_FIN'], errors='coerce')

    # Set NaN for values in 'IMP_FMP_PREZZO_FIN' equal to 0, as they likely represent missing or erroneous data
    df.loc[df['IMP_FMP_PREZZO_FIN'] == 0, 'IMP_FMP_PREZZO_FIN'] = np.nan

    # Convert the 'SUM_of_IMP_ANPA_IMP_REDD_MESE' column to numeric format, handling errors by coercing them to NaN (Not a Number)
    df['SUM_of_IMP_ANPA_IMP_REDD_MESE'] = pd.to_numeric(df['SUM_of_IMP_ANPA_IMP_REDD_MESE'], errors='coerce')

    # Assign a default value of 0 to the new column 'flag_giovani'
    df["flag_giovani"] = 0

    # Set 'flag_giovani' to 1 where 'FLAG_GIOVANI' equals "S"
    df.loc[df["FLAG_GIOVANI"] == "S", "flag_giovani"] = 1

    # Delete the column 'FLAG_GIOVANI' from the DataFrame
    del df["FLAG_GIOVANI"]

#     df.loc[:,'SUM_of_VAL_VTR'] = pd.to_numeric(df['SUM_of_VAL_VTR'].str.replace('.', '').str.replace(',', '.'), errors='coerce')
#     df.loc[df['SUM_of_VAL_VTR']==0,'SUM_of_VAL_VTR']=np.nan
#     df.loc[:,'SUM_of_IMP_BMTP_VALORE_BENE'] = pd.to_numeric(df['SUM_of_IMP_BMTP_VALORE_BENE'].str.replace('.', '').str.replace(',', '.'), errors='coerce')
    return df


def check_values_same_and_select_one(group, variable):
    """
    Check the values of a specified variable within a group and return a single value.

    This function is designed to handle groups of data in a DataFrame and ensures that 
    the values of a specified variable within each group are consistent. It is particularly 
    useful for situations where duplicate rows might exist and you need to verify the 
    consistency of a specific column's values.

    Parameters:
    group (pd.DataFrame): A group of rows from a DataFrame. This is typically created 
                          using the groupby method.
    variable (str): The name of the variable (column) to check for consistency within 
                    the group.

    Returns:
    The value of the specified variable if it is consistent within the group. If the group 
    has only one row, the function returns the value of that row. If the group has multiple 
    rows but all values of the specified variable are the same, it returns one of those values.
    
    Raises:
    ValueError: If the specified variable has multiple different values within the group, 
                indicating inconsistency.
    """
    # If the group has only one row, return the sum of the values in the specified variable
    if len(group) == 1:
        return group[variable].sum()
    # If all values in the specified variable within the group are the same, return one of the values
    elif group[variable].nunique() == 1:
        return group[variable].iloc[0]
    else:
        # If there are multiple different values in the specified variable within the group, raise a ValueError
        if group[variable].nunique() > 1:
            raise ValueError("Multiple different " + variable + " values for duplicates")
        # If values in the specified variable are the same for duplicates, return one of the values
        return group[variable].iloc[0]

def aggregate_val_bene(df_catasto, df):
    """
    Always SUM in CATASTO if NUM_BENE is different (otherwise check if the VAL_BENE is the same and if yes consider only one,
    if no raise exception), and then replace in CONTRATTI

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    
    # Convert the 'VAL_BENE' column to numeric format, replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number) 
    df_catasto.loc[:,'VAL_BENE'] = pd.to_numeric(df_catasto['VAL_BENE'].str.replace('.', '').str.replace(',', '.'), \
                                                  errors='coerce')

    # Apply the 'check_values_same_and_select_one' function to each group defined by unique combinations of 'COD_CONTRATTO' and 'NUM_BENE'
    result_df = df_catasto.groupby(['COD_CONTRATTO', 'NUM_BENE']).apply(check_values_same_and_select_one, 'VAL_BENE').reset_index()

    # Sum up the 'VAL_BENE' values within each 'COD_CONTRATTO' group after aggregation
    result_df = result_df.groupby('COD_CONTRATTO')[0].sum().reset_index()

    # Rename the columns of the 'result_df' DataFrame
    result_df.columns = ['COD_CONTRATTO', 'VAL_BENE']

    # Rename the 'VAL_BENE' column in the original DataFrame 'df' to 'VAL_BENE_OLD'
    df = df.rename(columns={'VAL_BENE': 'VAL_BENE_OLD'})

    # Merge the 'result_df' DataFrame with the original DataFrame 'df' based on the 'COD_CONTRATTO' column
    df = pd.merge(df, result_df, on="COD_CONTRATTO", how="left")
    
    return df

def aggregate_superf(df_catasto, df):
    """
    Always SUM in CATASTO if NUM_BENE is different (otherwise check if the VAL_BENE is the same and if yes consider only one,
    if no raise exception), and then replace in CONTRATTI

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    
    # Convert the 'NUM_MQ_SUPERF_COMM' column in the 'df_catasto' DataFrame to numeric format,
    # replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df_catasto.loc[:, 'NUM_MQ_SUPERF_COMM'] = pd.to_numeric(df_catasto['NUM_MQ_SUPERF_COMM'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

    # Convert the 'NUM_MQ_SUPERF_EQUIV' column in the 'df_catasto' DataFrame to numeric format,
    # replacing thousands separator '.' and decimal separator ',' with '', handling errors by coercing them to NaN (Not a Number)
    df_catasto.loc[:, 'NUM_MQ_SUPERF_EQUIV'] = pd.to_numeric(df_catasto['NUM_MQ_SUPERF_EQUIV'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

    # Apply the 'check_values_same_and_select_one' function to each group defined by unique combinations of 'COD_CONTRATTO' and 'NUM_BENE'
    ix = df_catasto["COD_CAT_CATASTALE"].str.slice(0,1) == "A"
    result_df_comm = df_catasto[ix].groupby(['COD_CONTRATTO', 'NUM_BENE']).apply(check_values_same_and_select_one, 'NUM_MQ_SUPERF_COMM').reset_index()
    result_df_equiv = df_catasto[ix].groupby(['COD_CONTRATTO', 'NUM_BENE']).apply(check_values_same_and_select_one, 'NUM_MQ_SUPERF_EQUIV').reset_index()
    #The exception is never raised, so indeed same NUM_MQ_SUPERF_COMM and NUM_MQ_SUPERF_EQUIV for each duplicate of NUM_BENE

    # Sum up the 'NUM_MQ_SUPERF_COMM' values within each 'COD_CONTRATTO' group after aggregation
    result_df_comm = result_df_comm.groupby('COD_CONTRATTO')[0].sum().reset_index()
    result_df_comm.columns = ['COD_CONTRATTO', 'NUM_MQ_SUPERF_COMM']

    # Sum up the 'NUM_MQ_SUPERF_EQUIV' values within each 'COD_CONTRATTO' group after aggregation
    result_df_equiv = result_df_equiv.groupby('COD_CONTRATTO')[0].sum().reset_index()
    result_df_equiv.columns = ['COD_CONTRATTO', 'NUM_MQ_SUPERF_EQUIV']

    # Rename the 'NUM_MQ_SUPERF_COMM' and 'NUM_MQ_SUPERF_EQUIV' columns in the original DataFrame 'df' to 'NUM_MQ_SUPERF_COMM_OLD' and 'NUM_MQ_SUPERF_EQUIV_OLD', respectively
    df = df.rename(columns={'NUM_MQ_SUPERF_COMM': 'NUM_MQ_SUPERF_COMM_OLD', 'NUM_MQ_SUPERF_EQUIV': 'NUM_MQ_SUPERF_EQUIV_OLD'})

    # Merge the 'result_df_comm' DataFrame with the original DataFrame 'df' based on the 'COD_CONTRATTO' column
    df = pd.merge(df, result_df_comm, on="COD_CONTRATTO", how="left")

    # Merge the 'result_df_equiv' DataFrame with the original DataFrame 'df' based on the 'COD_CONTRATTO' column
    df = pd.merge(df, result_df_equiv, on="COD_CONTRATTO", how="left")

    # Replace 0 values in 'NUM_MQ_SUPERF_COMM' and 'NUM_MQ_SUPERF_EQUIV' columns with NaN (Not a Number)
    df.loc[df['NUM_MQ_SUPERF_COMM'] == 0, 'NUM_MQ_SUPERF_COMM'] = np.nan
    df.loc[df['NUM_MQ_SUPERF_EQUIV'] == 0, 'NUM_MQ_SUPERF_EQUIV'] = np.nan

    #Because the Data Owner told us that NUM_MQ_SUPERF_COMM is more reliable (although difference in only 4400 cases),
    #we rely on NUM_MQ_SUPERF_COMM and only use NUM_MQ_SUPERF_EQUIV to fill in missing values.
    #We also keep the name NUM_MQ_SUPERF_COMM for compatibility with downstream notebooks
    df.loc[df['NUM_MQ_SUPERF_COMM'].isna(), 'NUM_MQ_SUPERF_COMM'] = df.loc[df['NUM_MQ_SUPERF_COMM'].isna(), 'NUM_MQ_SUPERF_EQUIV']
    
    return df


def annexes_and_cadastrial_code(df_catasto,df):
    """
    Include the annexes (garage and basement/attic) and select the cadastrial category of the housing unit with highest value
    among the residential options (A02, A03, etc.)

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    # Find unique 'COD_CONTRATTO' values associated with garages ('COD_CAT_CATASTALE' == "C06")
    cod_contratto_with_garage = df_catasto.loc[df_catasto["COD_CAT_CATASTALE"] == "C06", "COD_CONTRATTO"].unique()

    # Initialize a new column 'flag_garage' in the DataFrame 'df' with default value 0
    df["flag_garage"] = 0

    # Set 'flag_garage' to 1 for rows where 'COD_CONTRATTO' is in the list 'cod_contratto_with_garage'
    df.loc[df["COD_CONTRATTO"].isin(cod_contratto_with_garage), "flag_garage"] = 1

    
    # Find unique 'COD_CONTRATTO' values associated with pertinenze ('COD_CAT_CATASTALE' == "C02")
    cod_contratto_with_pertinenza = df_catasto.loc[df_catasto["COD_CAT_CATASTALE"] == "C02", "COD_CONTRATTO"].unique()

    # Initialize a new column 'flag_pertinenza' in the DataFrame 'df' with default value 0
    df["flag_pertinenza"] = 0

    # Set 'flag_pertinenza' to 1 for rows where 'COD_CONTRATTO' is in the list 'cod_contratto_with_pertinenza'
    df.loc[df["COD_CONTRATTO"].isin(cod_contratto_with_pertinenza), "flag_pertinenza"] = 1

    
    # Filter rows in 'df_catasto' where 'COD_CAT_CATASTALE' is not NaN and starts with 'A' (residenziale)
    df_filtered = df_catasto[(~df_catasto['COD_CAT_CATASTALE'].isna()) & df_catasto['COD_CAT_CATASTALE'].str.startswith('A')]

    # Sort each group by 'VAL_BENE' in descending order
    df_sorted = df_filtered.sort_values(by='VAL_BENE', ascending=False)

    # Group the sorted DataFrame by 'COD_CONTRATTO' and select the first non-NaN 'COD_CAT_CATASTALE' for each group
    result_df = df_sorted.groupby('COD_CONTRATTO')['COD_CAT_CATASTALE'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()

    # Merge the resulting DataFrame with the original DataFrame 'df' based on the 'COD_CONTRATTO' column
    df = pd.merge(df, result_df, on="COD_CONTRATTO", how="left")
    
    return df



def energy_class(df_catasto,df):
    """
    Select the energy class of the housing unit with highest value when using the variable COD_EU_CL_ENERGETICA.
    If that variable is not available, use instead COD_CLASSE_ENERG_ABS

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    # Replace values "AG" and "NC" in the 'COD_EU_CL_ENERGETICA' column of 'df_catasto' with NaN
    df_catasto.loc[df_catasto["COD_EU_CL_ENERGETICA"].isin(["AG", "NC"]), "COD_EU_CL_ENERGETICA"] = np.nan

    # Filter rows in 'df_catasto' where 'COD_CAT_CATASTALE' is not NaN and 'COD_CAT_CATASTALE' starts with 'A' (residenziale)
    df_filtered = df_catasto[(~df_catasto['COD_CAT_CATASTALE'].isna()) & df_catasto['COD_CAT_CATASTALE'].str.startswith('A')]

    # Sort each group by 'VAL_BENE' in descending order
    df_sorted = df_filtered.sort_values(by='VAL_BENE', ascending=False)

    # Group by 'COD_CONTRATTO' and select the first non-NaN 'COD_EU_CL_ENERGETICA' for each group
    result_df = df_sorted.groupby('COD_CONTRATTO')['COD_EU_CL_ENERGETICA'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()

    # Merge with main dataframe
    df = pd.merge(df,result_df,on="COD_CONTRATTO",how="left")

    # Normally, consider as energy class 'COD_CLASSE_ENERG_ABS'
    df["energy_class"] = df["COD_CLASSE_ENERG_ABS"]

    # When 'COD_EU_CL_ENERGETICA' is not NA, consider that instead of 'COD_CLASSE_ENERG_ABS'
    ix=df["COD_EU_CL_ENERGETICA"].isnotna()
    df.loc[ix,"energy_class"] = df.loc[ix,"COD_EU_CL_ENERGETICA"]

    return df



def renovation(df_catasto,df):
    """
    Select the most recent renovation date of the housing unit.

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    # Extract the substring from index 5 to 8 from the 'DAT_EU_RISTRUTTURAZIONE' column,
    # convert it to float, and assign it back to the 'DAT_EU_RISTRUTTURAZIONE' column
    df_catasto["DAT_EU_RISTRUTTURAZIONE"] = df_catasto["DAT_EU_RISTRUTTURAZIONE"].str.slice(5,9).astype(float)

    # Filter rows in 'df_catasto' where 'COD_CAT_CATASTALE' is not NaN,
    # 'DAT_EU_RISTRUTTURAZIONE' is not NaN, and 'COD_CAT_CATASTALE' starts with 'A' (residenziale)
    df_filtered = df_catasto[(~df_catasto['COD_CAT_CATASTALE'].isna()) & \
                             (df_catasto['DAT_EU_RISTRUTTURAZIONE'].isnotna()) & \
                             df_catasto['COD_CAT_CATASTALE'].str.startswith('A')]

    # Sort the filtered DataFrame by 'DAT_EU_RISTRUTTURAZIONE' in descending order
    df_sorted = df_filtered.sort_values(by='DAT_EU_RISTRUTTURAZIONE', ascending=False)

    # Group the sorted DataFrame by 'COD_CONTRATTO' and select the most recent non-NaN 'DAT_EU_RISTRUTTURAZIONE' for each group
    result_df = df_sorted.groupby('COD_CONTRATTO')['DAT_EU_RISTRUTTURAZIONE'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()

    # Merge the resulting DataFrame with the original DataFrame 'df' based on the 'COD_CONTRATTO' column
    df = pd.merge(df,result_df,on="COD_CONTRATTO",how="left")
    
    return df



def air_conditioning(df_catasto,df):
    """
    Process the air conditioning information on the housing unit.

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    # Convert the 'NUM_EU_SUP_RAFFRESCATA' column values to numeric after removing dots and commas,
    # with errors coerced (invalid parsing will be set as NaN)
    df_catasto.loc[:,'NUM_EU_SUP_RAFFRESCATA'] = pd.to_numeric(df_catasto['NUM_EU_SUP_RAFFRESCATA'].str.replace('.', '').str.replace(',', '.'), \
                                                  errors='coerce')
    # Define a dataframe corresponding to the rows in 'df_catasto' where 'NUM_EU_SUP_RAFFRESCATA' is greater than 0
    df_positive = df_catasto[(df_catasto['NUM_EU_SUP_RAFFRESCATA']>0)]
    # Define a dataframe corresponding to the rows in 'df_catasto' where 'NUM_EU_SUP_RAFFRESCATA' is equal to 0
    df_zero = df_catasto[(df_catasto['NUM_EU_SUP_RAFFRESCATA']==0)]

    # Get unique 'COD_CONTRATTO' values where 'NUM_EU_SUP_RAFFRESCATA' is greater than 0
    cod_contratto_with_ac_positive = df_positive["COD_CONTRATTO"].unique()
    # Get unique 'COD_CONTRATTO' values where 'NUM_EU_SUP_RAFFRESCATA' is equal to 0
    cod_contratto_with_ac_zero = df_zero["COD_CONTRATTO"].unique()

    # Initialize a new column 'flag_air_conditioning' in DataFrame 'df' with NaN values
    df["flag_air_conditioning"] = np.nan
    # Assign 0 to 'flag_air_conditioning' where 'COD_CONTRATTO' is in 'cod_contratto_with_ac_zero'
    df.loc[df["COD_CONTRATTO"].isin(cod_contratto_with_ac_zero),"flag_air_conditioning"] = 0
    # Assign 1 to 'flag_air_conditioning' where 'COD_CONTRATTO' is in 'cod_contratto_with_ac_positive'
    df.loc[df["COD_CONTRATTO"].isin(cod_contratto_with_ac_positive),"flag_air_conditioning"] = 1
    # This way we make sure to distinguish between the cases in which there is no air conditioning  
    # and the cases for which we do not know.

    return df







def floor(df_catasto, df, floor_map, floor_map_numeric, acceptable_floors):
    """
    Choose the floor of the housing unit with highest value. 
    Because it is possible that a housing unit is on multiple floors, also define:
    -a numeric variable keeping the lower floor.
    -an indicator variable indicating if the housing unit is on multiple floors.

    Parameters:
    df_catasto (pd.DataFrame): The catasto DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.
    floor_map (dictionary): A dictionary associating unique floor values to various codes.
    floor_map_numeric (dictionary): A dictionary associating numeric values to mutlifloor housing units.
    acceptable_floors (list): A list with the acceptable values

    Returns:
    pd.DataFrame: The modified DataFrame.
    """

    # Filter rows in 'df_catasto' where 'COD_CAT_CATASTALE' is not NaN and 'COD_CAT_CATASTALE' starts with 'A' (residenziale)
    df_filtered = df_catasto[(~df_catasto['COD_CAT_CATASTALE'].isna()) & df_catasto['COD_CAT_CATASTALE'].str.startswith('A')]

    # Sort each group by 'VAL_BENE' in descending order
    df_sorted = df_filtered.sort_values(by='VAL_BENE', ascending=False)

    # Group by 'COD_CONTRATTO' and select the first non-NaN 'COD_EU_NUM_PIANO_IMM' for each group
    result_df = df_sorted.groupby('COD_CONTRATTO')['COD_EU_NUM_PIANO_IMM'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()

    for key in list(floor_map.keys()):
        # Filter rows in result_df where 'COD_EU_NUM_PIANO_IMM' matches the current key
        ix = result_df["COD_EU_NUM_PIANO_IMM"] == key
        # Replace matching keys with their corresponding values from floor_map
        result_df.loc[ix,"COD_EU_NUM_PIANO_IMM"] = floor_map[key]

    # Replace values in 'COD_EU_NUM_PIANO_IMM' not found in acceptable_floors with NaN
    result_df.loc[(~result_df["COD_EU_NUM_PIANO_IMM"].isin(acceptable_floors)),"COD_EU_NUM_PIANO_IMM"] = np.nan
    # Merge result_df into df on the 'COD_CONTRATTO' column
    df = pd.merge(df,result_df,on="COD_CONTRATTO",how="left")


    # Initialize a new column 'flag_multi_floor' in DataFrame 'df' with NaN values
    df["flag_multi_floor"] = np.nan
    # Assign 1 to 'flag_multi_floor' where 'COD_EU_NUM_PIANO_IMM' is in floor_map_numeric keys
    df.loc[df["COD_EU_NUM_PIANO_IMM"].isin(pd.Series(floor_map_numeric.keys())),"flag_multi_floor"] = 1
    # Assign 0 to 'flag_multi_floor' where 'COD_EU_NUM_PIANO_IMM' is not NaN and not in floor_map_numeric keys
    df.loc[(df["COD_EU_NUM_PIANO_IMM"].notna()) & \
           (~df["COD_EU_NUM_PIANO_IMM"].isin(pd.Series(floor_map_numeric.keys()))),"flag_multi_floor"] = 0

    # Copy 'COD_EU_NUM_PIANO_IMM' values to a new column 'floor_numeric' in DataFrame 'df'
    df["floor_numeric"] = df["COD_EU_NUM_PIANO_IMM"].copy()
    # Iterate through each key in floor_map_numeric dictionary
    for key in list(floor_map_numeric.keys()):
        # Replace matching keys in 'floor_numeric' with their corresponding values from floor_map_numeric
        ix = df["floor_numeric"] == key
        df.loc[ix,"floor_numeric"] = floor_map_numeric[key]
    # Convert 'floor_numeric' column values to numeric
    df["floor_numeric"] = pd.to_numeric(df["floor_numeric"])

    return df


def condizioni(df_condizioni, df):
    """
    Get information on the interest rate type (fixed, variable, mixed) and the interest rate from the mortgage
    with highest value of residual amortization.
    
    Parameters:
    df_condizioni (pd.DataFrame): The condizioni DataFrame.
    df (pd.DataFrame): The DataFrame that is being created to harmonize the data.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    
    # Convert values in 'PRC_TASSO_CLI_APPLCT' column to numeric, replacing commas with dots
    # and handling errors by coercing them to NaN
    df_condizioni.loc[:,'PRC_TASSO_CLI_APPLCT'] = pd.to_numeric(df_condizioni['PRC_TASSO_CLI_APPLCT'].str.replace('.', '').str.replace(',', '.'), \
                                                  errors='coerce')

    # Sort DataFrame 'df_condizioni' by the column 'IMP_DEBT_RESD_NETTO' in descending order
    df_sorted = df_condizioni.sort_values(by='IMP_DEBT_RESD_NETTO', ascending=False)

    # Group by 'COD_CONTRATTO' and select the first non-NaN 'COD_TIP_TASSO_KTO' and 'PRC_TASSO_CLI_APPLCT' for each group
    result_df = df_sorted.groupby('COD_CONTRATTO')['COD_TIP_TASSO_KTO'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()
    result_df_2 = df_sorted.groupby('COD_CONTRATTO')['PRC_TASSO_CLI_APPLCT'].apply(lambda x: next((i for i in x if pd.notna(i)), pd.NA)).reset_index()

    df = pd.merge(df,result_df,on="COD_CONTRATTO",how="left")
    df = pd.merge(df,result_df_2,on="COD_CONTRATTO",how="left")

    return df


def OMI_preference(group_contratto):
    """
    Resolve overlaps between OMI regions by selecting the OMI region with lowest letter and number in OMI_categ

    Parameters:
    group_contratto (pd.DataFrame): A DataFrame representing a group of contracts that 
                                    potentially overlap in OMI regions.

    Returns:
    pd.Series: A single row (contract) from the group that has the highest priority 
               based on the 'OMI_categ' sorting.
    """

    # Reset the index of the group to ensure it's sequential and starts from 0
    duplicates = group_contratto.reset_index(drop=True)
    
    # Sort the group by 'OMI_categ' to determine the preference
    sort_dup = duplicates.sort_values(by='OMI_categ')
    
    # Select the first row (highest priority) from the sorted DataFrame
    dup_keep = sort_dup.iloc[0, :]
    
    return dup_keep

def spatial_matching(df, census, hydro_risk, omi_og, cap):

    # Transform PRO_COM codes to 6 digit string format
    census['PRO_COM_formated'] = census['PRO_COM'].apply(lambda x: '{:06d}'.format(int(x)))
    census=census.drop(columns = 'PRO_COM').rename(columns={'PRO_COM_formated':'PRO_COM'})
    census = census.loc[:,['SEZ2011','PRO_COM','geometry']] 
    census['SEZ2011'] =census['SEZ2011'].astype(str)

    # Transform missing coordinates (ND) to NaN
    df.loc[df['GEO_LATITUDINE_BENE_ROUNDED'] =='ND','GEO_LATITUDINE_BENE_ROUNDED'] = np.nan
    df.loc[df['GEO_LONGITUDINE_BENE_ROUNDED'] =='ND','GEO_LONGITUDINE_BENE_ROUNDED'] = np.nan

    # Create Shapely points from the coordinates 
    df_geo = gpd.points_from_xy(df['GEO_LONGITUDINE_BENE_ROUNDED'], df['GEO_LATITUDINE_BENE_ROUNDED'], z=None, crs="EPSG:4326")
    df_geo = df_geo.to_crs('EPSG:3035')
    df_gpd = gpd.GeoDataFrame(df, geometry= df_geo).reset_index(drop=True)

    # Find subset with valid coordinates (not NA)
    df_valid= df_gpd.loc[(~df_gpd.GEO_LATITUDINE_BENE_ROUNDED.isna()) & (~df_gpd.GEO_LONGITUDINE_BENE_ROUNDED.isna()),: ].reset_index(drop=True)
    df_na= df_gpd.loc[(df_gpd.GEO_LATITUDINE_BENE_ROUNDED.isna()) | (df_gpd.GEO_LONGITUDINE_BENE_ROUNDED.isna()),: ].reset_index(drop=True)

    # Assign PRO_COM from census tracts with valid coordinates
    df_valid = gpd.tools.sjoin(df_valid, census, predicate="within", how='left')
    df_valid.drop(columns = 'index_right',inplace=True)

    # Add flag for valid coordinates
    df_valid.loc[:,'flag_geo_valid'] = 1

    # Add OMI_ids 
    omi = omi_og.loc[omi_og.OMI_categ.notna(),['OMI_id','OMI_categ','geometry']]

    # Find CAP code
    df_valid_cap = gpd.tools.sjoin(df_valid, cap[['CAP','geometry']], predicate="within", how='left')
    df_valid_cap=df_valid_cap.drop(columns='index_right')

    # Find position in OMI areas
    df_valid_omi= gpd.tools.sjoin(df_valid_cap, omi, predicate="within", how='left')
    df_valid_omi=df_valid_omi.drop(columns='index_right')

    # Find duplicates
    dups = df_valid_omi.loc[df_valid_omi.COD_CONTRATTO.duplicated(keep=False),:].reset_index(drop=True)
    no_dups = df_valid_omi.loc[~df_valid_omi.COD_CONTRATTO.duplicated(keep=False),:].reset_index(drop=True)

    # Keep duplicate with lowest letter and number in OMI_categ 
    dups_to_keep= dups.groupby(by='COD_CONTRATTO').apply(lambda group_contract: OMI_preference(group_contract))

    # Merge together the selected rows
    data_omi = pd.concat([no_dups,dups_to_keep.reset_index(drop=True)]).reset_index(drop=True)

    # Spatial matching of houses and Risk areas: 
    df_risk = gpd.tools.sjoin(data_omi, hydro_risk, predicate="within", how='left')
    df_risk.drop(columns = 'index_right',inplace=True)
    df_risk.scenario.fillna('No Risk',inplace = True)

    # Rename risk levels 
    risk_levels = {"Pericolosita' idraulica bassa - LowProbabilityHazard":'Low Risk',\
                    "Pericolosita' idraulica media - MediumProbabilityHazard":'Medium Risk',\
                    "Pericolosita' idraulica elevata - HighProbabilityHazard": 'High Risk'}

    df_risk['scenario'].replace(list(risk_levels.keys()), list(risk_levels.values()) ,inplace=True)

    # Add samples with NA geometry
    df_na.loc[:,['PRO_COM','SEZ2011','scenario','OMI_categ','OMI_id']] = np.nan
    df_na.loc[:,'flag_geo_valid'] = 0

    # Re-combine df_na and df_valid
    df_merge = pd.concat([df_risk, df_na]).reset_index(drop=True)

    # Add flag if point is inside italy
    df_merge.loc[:,'flag_italy'] = 0
    df_merge.loc[:,'flag_italy'] = 1*df_merge.PRO_COM.notna()

    # Correct risk in samples with coordinates flag_italy =0
    df_merge.loc[df_merge['flag_italy']== 0,'scenario']= np.nan

    # Correct dtype of some variables: 
    int_col = list(df.loc[:,df.dtypes == 'int64'].columns) + ['flag_italy', 'flag_geo_valid']
    df_merge[int_col] = df_merge[int_col].astype('int64')

    # Drop geometry: 
    df_final = pd.DataFrame(df_merge.drop(columns='geometry'))
    
    return df_final

def Elevation_difference(df):
    """
    Function that computes the difference in elevation between the house and the nearest point to each 
    flood event
    """
    col_el = [c for c in df.columns if 'elevation' in c and 'house' not in c]
    for c in col_el:
        df.loc[:,c[0:8]+'elevation_diff'] = df['house_elevation']-df[c]

    return df


def Call_opentopodata(df, COLUMN_POINTS, KEY_ELE, call_params):
    """
    This function calls the OpenTopoData API to get elevation data for a set of coordinates.
    The function splits the DataFrame into batches and sends a request for each batch.
    If the daily limit of API calls is reached, the function sleeps for 24 hours.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the coordinates.
    COLUMN_POINTS (str): the column name in df containing the shapley point to retrieve the coordinates in crs=WGS-86.
    KEY_ELE (str): Column name to be created to save the elevation data. 
    call_params (dict): A dictionary containing parameters for the API call. 
                        It should have the following keys:\\
                        - 'samples_per_request': The number of samples in each call (max. 100).\\
                        - 'delay_between_request': The delay between requests in seconds.\\
                        - 'wait_api_limit': Time to wait after reaching the maximum API calls per day  (min 8400s = 1 day).\\
                        - 'url': The API url, corresponds to the EUDEM 25m dataset.

    Returns:
    df (pandas.DataFrame): The DataFrame with the elevation data added.
    """
    # converting point from EPSG:3035 to EPSG:4326
    df_copy = df.copy(deep=True)
    df_copy = df_copy.reset_index(drop = True)
    df_copy.set_geometry(col=COLUMN_POINTS,crs = "EPSG:3035", inplace=True)
    point_lat_long=df_copy.to_crs("EPSG:4326")
    df_copy['LAT']=point_lat_long.geometry.y
    df_copy['LNG']=point_lat_long.geometry.x

    if(COLUMN_POINTS!='geometry'):
        df_copy = df_copy.loc[(df_copy[COLUMN_POINTS[0:8]+'distance']<= call_params['max_distance2event']) & (df_copy['flag_'+COLUMN_POINTS[0:7]]==0),:]
    
    df_copy = df_copy.reset_index(drop = True)
    #Split df into batches
    df_batch= [d for _, d in df_copy.groupby(df_copy.index//call_params['samples_per_request'])]

    df_copy[KEY_ELE+'elevation'] = np.nan

    counter_calls_day = 0

    for i in tqdm(range(len(df_batch))):
        # Check if we reached the counter limit
        if counter_calls_day >= 999:
            
            print('Num calls done today: ', counter_calls_day+1)
            print("API limit reached, sleeping for "+ str(call_params['wait_api_limit']/60)+' min')
            time.sleep(call_params['wait_api_limit'])  # sleep for 1 day
            counter_calls_day = 0
        
        batch=df_batch[i]
        batch_ids = batch.index
        
        location = "|".join([f"{row['LAT']},{row['LNG']}" for _, row in batch.iterrows()])
        url_spec = call_params['url'] + "locations=" + location 

        result = requests.get((url_spec))
        
        if result.status_code == 200:
            results = pd.json_normalize(result.json()["results"])
            df_copy.loc[batch_ids,KEY_ELE+'elevation'] = results["elevation"].values
            
        else:
            print(result, result.json())   

        counter_calls_day += 1
        time.sleep(call_params['delay_between_request'])  # free api
    
    # setting elevation equal to zero for Nan values in the elevation columns
    df_copy = df_copy.fillna(0)

    # adding the added column in df_copy to df
    df_copy = df_copy[['COD_CONTRATTO',KEY_ELE+'elevation']]
    df= pd.merge(df,df_copy,on='COD_CONTRATTO', how = 'left')

    return df


def Match_Houses2FloodEvents(df_isp,df_flood):
    '''
    This function adds to df_isp three columns for each flood events reported in df_flood.
    Each of these columns is named as: flood_code + '_near_point' or '_distance'. flood_code reports the alpha-numeric code identifying
    the flood event as reported in the Coopernicus dataset. The column flood_code + '_near_point' contains the closest point in the flooded
    area to the house at hand. The flood_code + '_distance' reports the horizzontal distance (in meters) between the house and the closest
    point of the flood. Finally, the columns 'flag' + flood_code is 1 when the house is located in the flooded area or 0 otherwise.
    '''

    # loop across the flood events (column Event_code in df_flood)
    for code in tqdm(df_flood.Event_code):
        event = df_flood.loc[df_flood.Event_code == code,"geometry"].reset_index(drop=True)
        # find the flooded points closest to the ISP houses 
        p1=[ nearest_points(event ,point)[0][0] for point in tqdm(df_isp.geometry)]
        df_isp[code+'_near_point'] = p1

        df_isp[code+'_distance'] = df_isp.apply(lambda row: row['geometry'].distance(row[code+'_near_point']), axis=1)
        df_isp['flag_'+code]=(df_isp[code+'_distance']==0)*1

    return df_isp
