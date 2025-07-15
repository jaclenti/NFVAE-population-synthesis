import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
import geopandas as gpd
from _10_functions_and_lists import OMI_preference
from sklearn.preprocessing import OneHotEncoder
import pickle
from pyfixest.estimation import feols
import sys


def remove_outliers(series, lower_quantile=0.001, upper_quantile=0.999):
    series_copy = series.copy()
    lower_bound = series_copy.quantile(lower_quantile)
    upper_bound = series_copy.quantile(upper_quantile)
    series_copy.loc[(series_copy < lower_bound) | (series_copy >= upper_bound)] = np.nan
    return series_copy
pd.Series.remove_outliers = remove_outliers

def isnotna(series):
    return ~series.isna()
pd.Series.isnotna = isnotna

def isnotin(series):
    return ~series.isin()
pd.Series.isnotin = isnotin

def CorrectTypes(isp):
    numeric_columns = ["IMP_MUTUO","IMP_SALDO","NUM_DURATA_AMMORT_IN_MESI", "ANNO_COSTRUZIONE", "VAL_BENE_OLD",
                               "NUM_MQ_SUPERF_COMM", "PRC_INDAF_PERC_LTV", "IMP_FMP_PREZZO_FIN", "floor_numeric",
                               "SUM_of_IMP_ANPA_IMP_REDD_MESE", "GEO_LATITUDINE_BENE_ROUNDED", "VAL_BENE",
                               "GEO_LONGITUDINE_BENE_ROUNDED","PRC_TASSO_CLI_APPLCT","flag_giovani","flag_italy", "flag_acquisto",
                               "flag_geo_valid","flag_garage", "flag_pertinenza","NUM_MQ_SUPERF_COMM_OLD","flag_air_conditioning",
                               "flag_multi_floor","floor_numeric",'DAT_EU_RISTRUTTURAZIONE',"NUM_MQ_SUPERF_EQUIV_OLD","NUM_MQ_SUPERF_EQUIV"]
    # I added DAT_EU_RISTRUTTURAZIONE to numeric columns because I noted that only the year was reported

    date_columns = ["DAT_EROGAZIONE"]

    for c in numeric_columns:
        isp[c] = pd.to_numeric(isp[c],errors='coerce')

    for c in date_columns:
        isp[c] = pd.to_datetime(isp[c], yearfirst=True)
    
    # Fix 'NA' code for Napoli province in ISP:
    master = pd.read_csv('/data/housing/data/intermediate/master_table_locations.csv',dtype = str,keep_default_na=False, encoding='latin1')
    comuni_in_napoli = master.loc[master.prov_abbrv =='NA','comune_cod_istat']
    isp.loc[(isp.PRO_COM.isin(comuni_in_napoli)) & (isp.DES_REGIONE_BENE == 'CAMPANIA'),'COD_PROVINCIA_BENE'] = 'NA'

    # Only changing for transaction we know for sure are in Napoli and in campania (there were some strange cases that did not meet it, need to double check).
    return isp


def CorrectTypes_ABM(isp):
    numeric_columns = ["IMP_MUTUO","IMP_SALDO","NUM_DURATA_AMMORT_IN_MESI", "ANNO_COSTRUZIONE", "VAL_BENE_OLD",
                               "NUM_MQ_SUPERF_COMM", "PRC_INDAF_PERC_LTV", "IMP_FMP_PREZZO_FIN", #"floor_numeric",
                               "SUM_of_IMP_ANPA_IMP_REDD_MESE", "GEO_LATITUDINE_BENE_ROUNDED", "VAL_BENE",
                               "GEO_LONGITUDINE_BENE_ROUNDED","PRC_TASSO_CLI_APPLCT","flag_giovani","flag_italy", "flag_acquisto",
                               "flag_geo_valid","flag_garage", "flag_pertinenza","NUM_MQ_SUPERF_COMM_OLD",
                               'DAT_EU_RISTRUTTURAZIONE',"NUM_MQ_SUPERF_EQUIV_OLD","NUM_MQ_SUPERF_EQUIV",
                               'log_mq','log_price','scenario_Risk','scenario_HighRisk','scenario_LowRisk','scenario_MediumRisk','scenario_NoRisk']
    
    numeric_columns = numeric_columns + list(isp.columns[isp.columns.str.contains('floor_')]) + list(isp.columns[isp.columns.str.contains('COD_CAT')])+ list(isp.columns[isp.columns.str.contains('_energy_class')]) + list(isp.columns[isp.columns.str.contains('year_erogaz_20')])

    # I added DAT_EU_RISTRUTTURAZIONE to numeric columns because I noted that only the year was reported

    date_columns = ["DAT_EROGAZIONE"]

    for c in numeric_columns:
        if c in isp.columns:
            isp[c] = pd.to_numeric(isp[c],errors='coerce')

    for c in date_columns:
        if c in isp.columns:
            isp[c] = pd.to_datetime(isp[c], yearfirst=True)
    
    return isp


def AddMasterTableInfo(data, col2add, left_merge, right_merge):
    path_inter = "/data/housing/data/intermediate/"
    master = pd.read_csv(path_inter+'master_table_locations.csv',dtype = str,encoding='utf-8',keep_default_na=False)
    master['P1'] = master["P1"].astype(int)
    data_master = pd.merge(data,master[col2add], how = 'left', left_on = left_merge, right_on = right_merge)
    data_master.drop(columns=right_merge,inplace = True)
    return data_master

def AddMasterRegrTableInfo(data, col2add, left_merge, right_merge):
    path_inter = "/data/housing/data/intermediate/"
    master = pd.read_csv(path_inter+'master_table_regressions.csv',dtype = str,encoding='utf-8',keep_default_na=False)
    data_master = pd.merge(data,master[col2add], how = 'left', left_on = left_merge, right_on = right_merge)
    return data_master


def save_dataset_with_only_new_data(main_dataset_path, new_dataset_path):
    """
    Saves a new dataset which only contains the new data
    Parameters:
    main_dataset_path (str): The path to the main dataset.
    new_dataset_path (str): The path to the new dataset.

    Returns:
    """
    df_main = pd.read_csv(main_dataset_path, sep=",", dtype=str,encoding="latin1")
    df_new = pd.read_csv(new_dataset_path, sep=",", dtype=str,encoding="latin1")
    df_new = df_new[~df_new["COD_CONTRATTO"].isin(df_main["COD_CONTRATTO"])]
    df_new.to_csv(new_dataset_path.replace(".csv", "")+"_only_new_data.csv", index=False)    

def join_datasets_old_new_data(old_data_path, new_data_path, output_path):
    """
    Saves a new dataset which contains both old and new data
    Parameters:
    new_data_path (str): The path to the new dataset.

    Returns:
    """
    df_old = pd.read_csv(old_data_path, sep=",", dtype=str, encoding="latin1")
    df_new = pd.read_csv(new_data_path, sep=",", dtype=str, encoding="latin1")
    df_main = pd.concat([df_old, df_new])
    df_main.to_csv(output_path,index=False)


def ImportantCoefficients(regression,imp_var):
    """
    Summarizes the regression parameters including coefficients, p-values, standard errors, t-values, and confidence intervals.
    Additionally, filters and sorts the parameters for a subset of important variables based on p-values.
    Parameters:
    - regression: A regression model object that has methods coef(), pvalue(), se(), tstat(), and confint() for extracting regression parameters.
    - imp_var (list of str): A list of variable names considered important for which the summary statistics are specifically extracted and sorted.
    Returns:
    - df (DataFrame): A pandas DataFrame containing the regression parameters for all variables, sorted by p-value.
    - df_important (DataFrame): A pandas DataFrame containing the regression parameters for the important variables specified in imp_var, sorted by p-value.
    The DataFrame 'df' includes the following columns:
    - 'Coefficient': The coefficients of the regression model.
    - 'p-value': The p-values associated with each model coefficient, indicating the probability of observing the given result by chance.
    - 'Std. Error': The standard errors of the coefficients.
    - 't value': The t-statistics for the coefficients.
    - '2.5%': The lower bound of the 95% confidence interval for the coefficient.
    - '97.5%': The upper bound of the 95% confidence interval for the coefficient.
    The DataFrame 'df_important' is a subset of 'df' that includes only the rows for the variables specified in 'imp_var'.
    """
    # df is a dataframe with the regression parameters for all variables
    df = pd.DataFrame({
    'Coefficient':regression.coef(),
    'p-value':regression.pvalue(),
    'Std. Error':regression.se(),
    't value':regression.tstat(),
    '2.5%':regression.confint()[regression.confint().columns[0]],
    '97.5%':regression.confint()[regression.confint().columns[1]]
    })

    imp_var = '|'.join(imp_var)
    # dataframe with variables you want to see
    df_important = df.loc[df.index.str.contains(imp_var),:].sort_values(by = 'p-value')

    # sort df according to p-value
    df = df.sort_values(by = 'p-value')

    return df_important, df



def spatial_matching_ABM(df,hydro_risk,census,omi_og,cap):
    """
    Function to do the spatial matching getting the census_tract, PRO_COM, risk, CAP, province and region
    Inputs:
        - df =  dataframe of the synthetic population
        - hydro_risk = hydrological risk from ISPRA
        - census = census shapefile
        - omi_og = OMI shapefiles
        - CAP =  CAP shapefiles

    Output: a dataframe called df_final that has the same columns as df, plus the following:
        - SEZ2011 = census tract
        - PRO_COM = municipality
        - flag_geo_valid = flag to find wether a coordinate is valid (1) or not (0)
        - OMI_id = OMI microzone (composition of cod_catastale comune and OMI_categ)
        - OMI_categ = OMI classification (for example, codes for central or peripherial areas)
        - CAP = CAP (ZIP code)
        - COD_CONTRATTO = a synthetic code for each line that works as COD_CONTRATTO in the ISP data. (It is usefull for the function)
        - scenario_HighRisk = flag that is 1 when the synthetic house is in a High Risk area
        - scenario_MediumRisk = flag that is 1 when the synthetic house is in a Medium Risk area
        - scenario_LowRisk = flag that is 1 when the synthetic house is in a Low Risk area
        - scenario_NoRisk = flag that is 1 when the synthetic house is outside a risk area 
        - scenario_Risk = flag that is 1 when the synthetic house is in a risk area (Low, Medium or High) 
        - flag_italy = flag that is 1 when the synthetic house is inside Italy boundaries
        - COD_REG = couple of number that identify each Italian region
        - regione_nome = name of the Italian region inj which the considered house is
        - COD_PROV = three digit identyfing the Italian province in which the considered house is in
        - prov_nome = name of the province
        - prov_abbrv = abbreviation of the province (e.g., TO for Turin)

    """
    # full column names
    df = df.rename(columns = {"x": "GEO_LONGITUDINE_BENE_ROUNDED", "y": "GEO_LATITUDINE_BENE_ROUNDED"})

    # Create COD_CONTRATTO
    df['COD_CONTRATTO']='syn_'+ df.index.astype(str)

    # Transform PRO_COM codes to 6 digit string format
    census['PRO_COM_formated'] = census['PRO_COM'].apply(lambda x: '{:06d}'.format(int(x)))
    census=census.drop(columns = 'PRO_COM').rename(columns={'PRO_COM_formated':'PRO_COM'})
    census = census.loc[:,['SEZ2011','PRO_COM','geometry']] 
    census['SEZ2011'] =census['SEZ2011'].astype(str)

    # Create Shapely points from the coordinates 
    df_geo = gpd.points_from_xy(df['GEO_LONGITUDINE_BENE_ROUNDED'], df['GEO_LATITUDINE_BENE_ROUNDED'], z=None, crs="EPSG:4326")
    df_geo = df_geo.to_crs('EPSG:3035')
    df_gpd = gpd.GeoDataFrame(df, geometry= df_geo).reset_index(drop=True)

    # Find subset with valid coordinates (not NA)
    df_valid= df_gpd.loc[(~df_gpd.GEO_LATITUDINE_BENE_ROUNDED.isna()) & (~df_gpd.GEO_LONGITUDINE_BENE_ROUNDED.isna()),: ].reset_index(drop=True)

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
    df_risk.drop_duplicates("COD_CONTRATTO", inplace = True) # rare duplicates when houses are in the edge between two polygons
    df_risk.drop(columns = 'index_right',inplace=True)
    df_risk.scenario.fillna('No Risk',inplace = True)

    # Rename risk levels 
    risk_levels = {"Pericolosita' idraulica bassa - LowProbabilityHazard":'Low Risk',\
                    "Pericolosita' idraulica media - MediumProbabilityHazard":'Medium Risk',\
                    "Pericolosita' idraulica elevata - HighProbabilityHazard": 'High Risk'}

    df_risk['scenario'].replace(list(risk_levels.keys()), list(risk_levels.values()) ,inplace=True)
    
    # Scenario dummies
    encoder = OneHotEncoder()
    # encoder = OneHotEncoder(sparse=True)
    encoded = encoder.fit_transform(df_risk[['scenario']])
    column_names = encoder.get_feature_names_out(['scenario'])
    dummy_risk = pd.DataFrame.sparse.from_spmatrix(encoded, columns=column_names) #, columns=encoder_OMI.get_feature_names_out(['categoria'])

    df_risk = pd.concat([df_risk, dummy_risk], axis=1)
    df_risk = df_risk.rename(columns={'scenario_High Risk': 'scenario_HighRisk','scenario_Low Risk': 'scenario_LowRisk', \
        'scenario_Medium Risk': 'scenario_MediumRisk', 'scenario_No Risk': 'scenario_NoRisk'})
    
    # need all the columns, also for the provinces without risk areas
    for r in ["High", "Medium", "Low", "No"]:
        df_risk[f"scenario_{r}Risk"] = df_risk.get(f"scenario_{r}Risk", default = pd.Series(np.zeros(len(df_risk))))
    
    # dummy unique risk
    df_risk['scenario_Risk'] = 1*(df_risk.loc[:,['scenario_HighRisk','scenario_MediumRisk','scenario_LowRisk']].sum(axis=1)>0)

    # Add flag if point is inside italy
    df_risk.loc[:,'flag_italy'] = 0
    df_risk.loc[:,'flag_italy'] = 1*df_risk.PRO_COM.notna()

    # Correct risk in samples with coordinates flag_italy =0
    df_risk.loc[df_risk['flag_italy']== 0,'scenario']= np.nan

    # Drop geometry: 
    df_final = pd.DataFrame(df_risk.drop(columns='geometry'))

    # add region info
    df_final = AddMasterRegrTableInfo(df_final, ['PRO_COM','COD_PROV','COD_REG'],left_merge='PRO_COM',right_merge='PRO_COM')
    df_final['COD_REG'] = df_final['COD_REG'].str.zfill(2)
    map_region_name = AddMasterTableInfo(df_final, ['comune_cod_istat','regione_cod_istat','regione_nome'],left_merge='PRO_COM',right_merge='comune_cod_istat')
    map_region_name = map_region_name[['COD_REG','regione_nome']].drop_duplicates().dropna().sort_values('COD_REG').reset_index(drop=True)
    df_final =pd.merge(df_final,map_region_name,how='left',on='COD_REG')

    # add province info
    map_prov_name = AddMasterTableInfo(df_final, ['comune_cod_istat','prov_code_istat','prov_nome','prov_abbrv'],left_merge='PRO_COM',right_merge='comune_cod_istat')
    map_prov_name = map_prov_name[['COD_PROV','prov_nome','prov_abbrv']].drop_duplicates().dropna().sort_values('COD_PROV').reset_index(drop=True)
    df_final =pd.merge(df_final,map_prov_name,how='left',on='COD_PROV')

    # define year_erogaz_prov
    df_final['year_erogaz_prov'] = '2024_' + df_final.COD_PROV
    
    return df_final



def Train_RegressionHedonicPrice(train_data,sp_fix_eff,region, save):
    """
    Function to compute the hedonic price of the houses in the synthetic population
    Inputs:
     - train_data: ISP data
     - sp_fix_eff: spatial fixed effects (we start using PRO_COM)
     - save: flag that is True if you want to save the regression parameters in this file f'/data/housing/data/intermediate/HedonicRegressionABM/reg_risk_{sp_fix_eff}.pkl'. 

    Outputs:
     - reg_risk: regression objetc
    """
    if region != None:
        train_data= train_data.loc[train_data.COD_REG == region,:]
    # Drop colinearity variables
    drop_coliniarity= ['ANNO_COSTRUZIONE_1500_1965','Medium_energy_class',
                        'COD_CAT_A_04_05','floor_0.0','year_erogaz_2016']
    train_data = train_data.drop(columns=drop_coliniarity)

    # Add dummies to formula: 
    floor_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('floor_')]))
    year_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('year_erogaz_20')]))
    energy_class_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('_energy_class')]))
    ANNO_COSTRUZIONE_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('ANNO_COSTRUZIONE')]))
    COD_CAT_dummies = ' + '.join(list(train_data.columns[train_data.columns.str.contains('COD_CAT_')]))

    dummies_vec = COD_CAT_dummies + '+' + ANNO_COSTRUZIONE_dummies + '+' + energy_class_dummies + ' + ' + floor_dummies + '+' + year_dummies

    # Regression specification
    base_features = 'flag_garage + flag_pertinenza + flag_air_conditioning + flag_multi_floor +' + dummies_vec
    mq_feature = 'log_mq' 

    eq_regr_risk = 'log_price ~ scenario_Risk +' + mq_feature + '+' + base_features + '|' +  sp_fix_eff  # + C(month_erogaz) year_erogaz_prov +
    
    # Train regression
    reg_risk = feols(fml=eq_regr_risk, data=train_data, drop_intercept=False ,vcov = {"CRV1":sp_fix_eff})
    if save==True:
        model_state = {"_beta_hat": reg_risk._beta_hat,
                    "_coefnames": reg_risk._coefnames,
                    "_se": reg_risk._se,
                    "_tstat": reg_risk._tstat,
                    "_pvalue": reg_risk._pvalue,
                    "_conf_int": reg_risk._conf_int,
                    "_vcov": reg_risk._vcov,
                    "_fml": reg_risk._fml,
                    "_sumFE": reg_risk._sumFE,
                    "_fixef": reg_risk._fixef,
                    "_fixef_dict": reg_risk._fixef_dict,
                    "_data": reg_risk._data}
        with open(f'/data/housing/data/intermediate/HedonicRegressionABM/reg_risk_{sp_fix_eff}_{region}.pkl', 'wb') as handle:
            pickle.dump(model_state, handle)

    return reg_risk

def Predict_HedonicPrice_SyntheticHomes(isp,train_path,test_data):
    """
    Function to estimate the hedonic price of a house. 

    Input variables:
        -isp: data from ISP data (used to initialize the regression model)
        - train_path: path to parameters of the trained model
        - test_data: data for which you want to estimate the home hedonic price

    Output:
        - test_data: same Dataframe as the input with the estimated hedonic price corrected by inflation
    and flood_risk effects (``price_corrected``). Moreover, the price is not in log form.
    """

  
    # Update reg_risk with good params
    master_loc = pd.read_csv('/data/housing/data/intermediate/master_table_regressions2locations.csv',dtype = str,encoding='utf-8',keep_default_na=False)

    list_provinces =pd.Series(test_data.COD_PROV.unique()).apply(lambda x: str('{:03d}'.format(int(x)))).values
    
    region_code = master_loc.loc[master_loc.COD_PROV.isin(list_provinces),'COD_REG'].unique()
    region_code= pd.Series(region_code).apply(lambda x: str('{:02d}'.format(int(x)))).values    

    #Initialise dummy regresion
    reg_risk = Train_RegressionHedonicPrice(isp.sample(1000),'PRO_COM',None,save=False)

    if len(region_code)==1:
        # Load regression trained at region scale
        with open(train_path+ f'reg_risk_OMI_id_{region_code[0]}.pkl', 'rb') as handle:
            model_state = pickle.load(handle)

        reg_risk._beta_hat = model_state["_beta_hat"]
        reg_risk._coefnames = model_state["_coefnames"]
        reg_risk._se = model_state["_se"]
        reg_risk._tstat = model_state["_tstat"]
        reg_risk._pvalue = model_state["_pvalue"]
        reg_risk._conf_int = model_state["_conf_int"]
        reg_risk._vcov = model_state["_vcov"]
        reg_risk._fml = model_state["_fml"]
        reg_risk._sumFE = model_state["_sumFE"]
        reg_risk._fixef = model_state["_fixef"]
        reg_risk._fixef_dict = model_state["_fixef_dict"]
        reg_risk._data = model_state["_data"]

        test_data['log_price_estimation'] = reg_risk.predict(test_data)
        # Correct price estimations with Risk coefficient and average Inflation
        _, df_coeff = ImportantCoefficients(reg_risk,['Risk'])

        # Risk correction
        test_data['price_corrected'] = test_data['log_price_estimation'] - (df_coeff.loc['scenario_Risk','Coefficient']*test_data['scenario_Risk'])

        # Correct inflation
        df_year,_ = ImportantCoefficients(reg_risk,['year']) # av_inflation
        avg_inflation = df_year.Coefficient.mean()
        
        for i in df_year.index:
            test_data['price_corrected'] = test_data['price_corrected'] - (df_year.loc[i,'Coefficient']*test_data[i])
        test_data['price_corrected'] = test_data['price_corrected'] + avg_inflation
        
        # Remove log and drop unnecessary columns
        test_data['price_corrected'] = np.exp(test_data['price_corrected'])
        test_data.drop(columns='log_price_estimation',inplace=True)
    else:
        print('ERROR, data belong to two different regions.')
        sys.exit()
    
    return test_data





# defining colormap for macroareas
areas = ['North-West','North-East','Center','South','Islands']

cmap_macroareas = {}
for i in range(len(areas)):
    cmap_macroareas[areas[i]]=colors.rgb2hex(plt.cm.get_cmap('Paired', len(areas))(i)) #plt.cm.get_cmap('Paired', len(unique_categories))plt.cm.Paired(i)

# defining the colormap to compare the ISP and OMI data
cmap_dataset= {'ISP':'#35618F', 'OMI':'#E590C7', 'BankItaly':'#8DE4D3', 'MEF':'#5BCF29'}

# defining colormap for house sizes (m^2)
cmap_mq = {r"$0<m^2<50$":"#6450A8", r"$50<m^2<85$":"#EBAA8C", r"$85<m^2<115$":"#972554", r"$115<m^2<145$":"#F79302", r"$m^2>145$":"#724933"}

# defining colormap for years
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
colormap = plt.cm.get_cmap('Blues', len(years))
cmap_year = {category: colormap(i) for i, category in enumerate(years)}


