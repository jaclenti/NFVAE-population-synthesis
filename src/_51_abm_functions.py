""""
The script contains all the functions used to run the ABM, and to create the dataframes used to analyse the output
- prepare_data_for_abm_region is the main function used to prepare the ABM input in a consistent format
- run_abm_reg initialize the ABM and run the simulation
- create_df_db_transactions_model, create_df_db_buyers_model, create_df_db_sellers_model create dataframe to make the output of the ABM ready for the analyses
"""


import numpy as np
import pandas as pd
import sys
from glob import glob
sys.path += ["../src"]
import config
import utils
from _52_abm_auxiliary_functions import *
from _50_abm import ABM



def load_data():
    master_locations = pd.read_csv("/data/housing/data/intermediate/master_table_regressions.csv", #locations.csv", \
                                dtype={'comune_cod_istat': str},na_values=config.default_missing)
    master_locations = master_locations.assign(COD_REG = lambda x: [f"{u:03d}" for u in x["COD_REG"]],
                                                PRO_COM = lambda x: [f"{u:06d}" for u in x["PRO_COM"]],
                                                COD_PROV = lambda x: [f"{u:03d}" for u in x["COD_PROV"]])
    data_path = '/data/housing/data/intermediate/ISP_data_for_ABM/ISP_ABM_up_to_2024_08.csv'
    isp = pd.read_csv(data_path,dtype=str,encoding='latin1',na_values=config.default_missing)

    drop_col = ['floor_numeric','NUM_MQ_SUPERF_COMM','COD_CAT_CATASTALE','year_erogaz']
    isp = isp.drop(columns=drop_col)
    isp = utils.CorrectTypes_ABM(isp)

    return master_locations, isp

master_locations, isp = load_data()
isp = isp.rename(columns = {"IMP_FMP_PREZZO_FIN": "price"})

# dataframe with province names and cod provinces
cod_prov_abbrv_df = pd.read_csv("/data/housing/data/intermediate/master_table_locations.csv",
                                dtype={'prov_code_istat': str},na_values=config.default_missing)[["prov_abbrv", "prov_code_istat"]].drop_duplicates().rename(columns = {"prov_code_istat": "COD_PROV"})

# dataframe with region names and cod regions
cod_reg_abbrv_df = pd.read_csv("/data/housing/data/intermediate/master_table_locations.csv",
                                dtype={'prov_code_istat': str},na_values=config.default_missing)[["regione_abbrv", "regione_cod_istat", "regione_nome"]].assign(COD_REG = lambda x: [f"{u:02d}" for u in x["regione_cod_istat"]]).drop(columns = ["regione_cod_istat"]).drop_duplicates()

# columns needed in the regression
spatial_columns = ['COD_PROV', "COD_REG", "PRO_COM", "CAP", "OMI_id"]
home_columns = ['flag_garage', 'flag_pertinenza', 'flag_air_conditioning',
       'flag_multi_floor', 
       'log_mq', 'ANNO_COSTRUZIONE_1500_1965',
       'ANNO_COSTRUZIONE_1965_1985', 'ANNO_COSTRUZIONE_1985_2005',
       'ANNO_COSTRUZIONE_2005_2025', 'ANNO_COSTRUZIONE_Missing',
       'High_energy_class', 'Low_energy_class', 'Medium_energy_class',
       'Missing_energy_class', 'COD_CAT_A02', 'COD_CAT_A03',
       'COD_CAT_A_01_07_08', 'COD_CAT_A_04_05', 'floor_0.0', 'floor_1.0',
       'floor_2.0', 'floor_3.0', 'floor_Missing', 'floor_plus_4',
       'flag_air_conditioning_Missing', 
       'scenario_HighRisk', 'scenario_LowRisk',
       'scenario_MediumRisk', 'scenario_NoRisk', 'scenario_Risk', 'flag_italy',
       'year_erogaz_prov', 'year_erogaz_2016', 'year_erogaz_2017',
       'year_erogaz_2018', 'year_erogaz_2019', 'year_erogaz_2020',
       'year_erogaz_2021', 'year_erogaz_2022', 'year_erogaz_2023',
       'year_erogaz_2024']


"""
split the categorical features as one-hot encoding
a categorical variable with k classes is split into k binary variables
"""
def add_cat_features(df):
    df = (df.assign(more_energy = lambda x: x[[u for u in df.columns if "_energy" in u]].sum(axis = 1),
              more_COD = lambda x: x[[u for u in df.columns if "COD_CAT_" in u]].sum(axis = 1),
              more_ANNO = lambda x: x[[u for u in df.columns if "ANNO_COSTRUZIONE" in u]].sum(axis = 1))
              .query("(more_ANNO == 1)&(more_COD == 1)&(more_energy == 1)")
              .drop(columns = ["more_ANNO","more_COD","more_energy"]))    
    
    df["energy_class"] = df[[u for u in df.columns if "_energy" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]
    df["COD_CAT"] = [u[8:] for u in df[[u for u in df.columns if "COD_CAT_" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]]
    df["anno_costruzione"] = [u[17:] for u in df[[u for u in df.columns if "ANNO_COSTRUZIONE" in u]].stack().rename("col").reset_index().query("col == 1")["level_1"]]
    return df

"""
it transform the number of transactions per month and the population in each region
i) maintaining the same ratio number of transactions / population in each region
ii) keeping the same number of transactions for the smallest region
iii) setting the number of transactions equal to corrected_population for the largest region
iv) applying only a linear transformation
"""
def correct_pop_and_contratti(time0 = pd.to_datetime("2017-09-01"), corrected_population = 200, 
                              isp = isp, master_locations = master_locations):
    pop_per_prov, pop_per_reg = population_per_region_province(isp, master_locations)
    # dataframe with the number of transactions / month (isp data) and the population (census) per region
    contratti_pop_regions = (pd.concat([isp.query("DAT_EROGAZIONE > @time0").assign(year_month = lambda x: pd.PeriodIndex(x["DAT_EROGAZIONE"], freq = "M").to_timestamp())
                                        .groupby(["COD_REG", "year_month"]).count()["COD_CONTRATTO"].reset_index()
                                        .groupby("COD_REG").mean()["COD_CONTRATTO"], 
                                        pd.Series(pop_per_reg)], axis = 1).rename(columns = {"COD_CONTRATTO": "contratti_isp", 0: "population"})
                                        .assign(houses = lambda x: x["population"] / 2)
                                        .assign(ratio_houses_contratti = lambda x: x["houses"] / x["contratti_isp"]))
    u = contratti_pop_regions["contratti_isp"]
    contratti_pop_regions["corrected_contracts"] = (u - u.min()) * (corrected_population - u.min()) / (u.max() - u.min()) + u.min()
    contratti_pop_regions = contratti_pop_regions.assign(corrected_houses = lambda x: (x["houses"] * x["corrected_contracts"] / x["contratti_isp"]).astype(int))
    return contratti_pop_regions

"""
compute beta with the trained hedonic regression
beta is the discount given by the risk area
"""
def observed_beta(cod_reg, with_conf_int = False):
    train_path = f'/data/housing/data/intermediate/HedonicRegressionABM/'
    # model_state stores the coefficients of the hedonic regression
    with open(train_path+ f'reg_risk_OMI_id_{cod_reg}.pkl', 'rb') as handle:
        model_state = utils.pickle.load(handle)


    beta_risk = model_state["_beta_hat"][0]
    pvalue = model_state["_pvalue"][0]
    conf_int = model_state["_conf_int"][:,0]

    return beta_risk, pvalue, conf_int

"""
extract the summary statistics from the observed data
in the ABM calibration we compare the summary statistics of the observed and simulated data
in this case, the summary statistics are the (corrected) number of transactions per month, the average price and beta (with confidence interval)
"""
def get_observed_stats(cod_reg, isp = isp, time0 = pd.to_datetime("2017-09-01"), 
                       time1 = pd.to_datetime("2024-09-01"), corrected_population = None):
    isp_reg = isp.query("(COD_REG == @cod_reg)&(DAT_EROGAZIONE > @time0)")
    delta_years = int(pd.Timedelta(time1 - time0).days / 365)
    n_transactions_per_month = len(isp_reg) / (12 * delta_years)
    n_transactions_per_month_ts = isp_reg.assign(year_month = lambda x: pd.PeriodIndex(x["DAT_EROGAZIONE"], freq = "M").to_timestamp()).groupby("year_month").count()
    price_ts = isp_reg.assign(year_month = lambda x: pd.PeriodIndex(x["DAT_EROGAZIONE"], freq = "M").to_timestamp())[["year_month", "price"]].groupby("year_month").mean()
    avg_price = price_ts["price"].mean()
    std_price = price_ts["price"].std()
    beta, pvalue, conf_int = observed_beta(cod_reg)
    if corrected_population:
        contratti_pop_regions = correct_pop_and_contratti(time0 = pd.to_datetime("2017-09-01"), 
                                                          corrected_population = corrected_population, isp = isp, master_locations = master_locations)
        avg_transactions_corrected = contratti_pop_regions.loc[cod_reg, "corrected_contracts"]
        n_transactions_per_month_ts = n_transactions_per_month_ts * avg_transactions_corrected / n_transactions_per_month_ts.mean()
    avg_transactions, std_transactions = n_transactions_per_month_ts.mean().iloc[0], n_transactions_per_month_ts.std().iloc[0]
    obs_stats_reg = pd.Series([avg_transactions, std_transactions, avg_price, std_price, beta, conf_int[0], conf_int[1]], 
                              index = ["avg_transactions", "std_transactions", "avg_price", "std_price", "beta_risk", "beta_conf_left", "beta_conf_right"])
    return obs_stats_reg

"""
return the census population in each province and each region
"""
def population_per_region_province(isp, master_locations):
    pop_per_prov_ = (pd.concat([pd.read_csv(file,encoding = 'latin1', sep = ";")
                           .assign(CODPRO = lambda x: [f"{u:03d}" for u in x["CODPRO"]])
                           .groupby(["CODPRO", "PROVINCIA"]).sum()["P1"].reset_index() 
                           for file in sorted(glob("/data/housing/data/input/census/Dati_regionali_2021/*_indicatori_2021_sezioni.csv"))])
                .rename(columns = {"P1": "pop", "CODPRO": "COD_PROV", "PROVINCIA": "prov_nome"}))
    
    pop_per_prov = (pop_per_prov_.merge(isp[["COD_PROV", "COD_REG"]].drop_duplicates(), how = "left")
                        .merge(master_locations.drop(columns = ["PRO_COM"]).assign(COD_REG = lambda x: [f"{int(u):02d}" for u in x["COD_REG"]]))
                        .drop_duplicates().fillna("Sardegna"))
    pop_per_reg = dict(pop_per_prov.groupby("COD_REG").sum(numeric_only = True)["pop"])
    return pop_per_prov, pop_per_reg


""" 
dataset that is given as input to the ABM
 - the number of houses can be (i) the minimum between the real number of houses (from census) and max_agents, OR (ii) the population given by the corrected populaiton
 - it sample from the synthetic population the selected number of houses 
 - the number of agents in the ABM is equal to the number of houses 
 - df_params assign to each spatial area an income class
 - in the initialization of the ABM, an income is assigned to each agent, and this is coherent with the real spatial area
 """
def prepare_data_for_abm_region(cod_reg, seed = None, 
                                master_locations = master_locations, 
                                isp = isp,
                                max_agents = 1000000, 
                                n_agents = None, 
                                corrected_population = None):
    
    pop_per_prov, pop_per_reg = population_per_region_province(isp, master_locations)
    
    if n_agents is None:
        n_agents = min([int(pop_per_reg[cod_reg] / 2), max_agents])
    if corrected_population:
        contratti_pop_regions = correct_pop_and_contratti(time0 = pd.to_datetime("2017-09-01"), corrected_population = corrected_population, 
                                                          isp = isp, master_locations = master_locations)
        n_agents = contratti_pop_regions["corrected_houses"][cod_reg]

    df_params = pd.read_csv(f"/data/housing/data/intermediate/df_params.csv", index_col = 0,dtype={"spatial_area_income": str}).reset_index()
    df_params_agg = pd.read_csv(f"/data/housing/data/intermediate/incomes_lognorm_params_higher_aggregation.csv", index_col = 0)
    df_params_agg = df_params_agg.query(f"location_type == 'region' and location == '{cod_reg}'")

    syn_pop = pd.read_csv(f"/data/housing/data/intermediate/jl_pop_synth/abm_region/{cod_reg}_syn_pop.csv", index_col = 0)
    
    syn_pop["PRO_COM"] = syn_pop["PRO_COM"].astype(str).str.zfill(6)
    syn_pop["CAP"] = syn_pop["CAP"].astype(str).str.replace(r"\.0$", "", regex=True)
    syn_pop["spatial_area_income"] = syn_pop["PRO_COM"]
    sp = df_params["spatial_area_income"]
    caps_with_data = sp[sp.str.len()==5]
    ix = syn_pop["CAP"].isin(caps_with_data)
    syn_pop.loc[ix,"spatial_area_income"] = syn_pop.loc[ix,"CAP"]    
    
    syn_pop = syn_pop.dropna(subset = ["spatial_area_income"]).sample(n_agents, random_state = seed)
    syn_pop["price_hedonic"] = syn_pop["price_corrected"]

    
    exogenous_cross_sections = {}
    exogenous_cross_sections["homes_synth_pop"] = syn_pop
    exogenous_cross_sections["df_params"] = df_params
    exogenous_cross_sections["df_params_agg"] = df_params_agg

    return exogenous_cross_sections


"""
a single run of the ABM
1) prepare the synthetic data
2) define the ABM
3) initialize the ABM
4) run a full simulation

parameters encode all the parameters of the simulations, the length of the simulation (number of timesteps), 
the length of the burn-in (number of timesteps that are discarded before the beginning of the simulation),
the awareness scenario

it is possible to show/hide the progress of the simulation with hide_progress and hide_progress_transient (burn_in)
"""
def run_abm_reg(cod_reg, parameters, awareness_scenario = None, 
            seed = None, master_locations = master_locations, isp = isp, 
            hide_progress = False, hide_progress_transient = True, 
            max_agents = 1000000, corrected_population = None):
    if master_locations is None:
        master_locations, isp = load_data()
    
    if awareness_scenario is None:
        # Description for awarenesses
        awareness_scenario = "timeseries"
        awarenesses_input = np.exp(np.linspace(0,1,parameters["T"]+1))
        parameters["awarenesses_input"] = awarenesses_input
        
    
    exogenous_cross_sections = prepare_data_for_abm_region(cod_reg,
                                                           seed = seed, 
                                                           master_locations = master_locations, 
                                                           isp = isp, max_agents = max_agents,
                                                           corrected_population = corrected_population)
    parameters["n_homes"] = len(exogenous_cross_sections["homes_synth_pop"])
    # Dictionary with parameters
    abm = ABM(exogenous_cross_sections,awareness_scenario,parameters)
    
    vars_init = abm.initialize_abm(seed_init = seed)
    variables = abm.setup_variables(seed_run = seed, vars_init = vars_init)
    variables = abm.run(seed_run = seed, vars_init = vars_init, 
                        hide_progress = hide_progress, 
                        hide_progress_transient = hide_progress_transient)

    return abm, variables, vars_init


"""
transform the timesteps into dates, starting from t_start
each timestep is one month
"""
def obtain_dates(t_start, t_end, abm, datetimes):
    if abm.Tb > t_start and abm.Tb >= t_end:
        dates = np.repeat(datetimes[0],t_end-t_start)
    if abm.Tb > t_start and t_end > abm.Tb:
        dates = np.concatenate([np.repeat(datetimes[0],abm.Tb-t_start),\
                                np.array([datetimes[i] for i in range(0,t_end-abm.Tb)])])
    if t_start >= abm.Tb:
        dates = np.array([datetimes[i] for i in range(t_start - abm.Tb,t_end - abm.Tb)])
    return dates
 
"""
given the output of a set of simulations, it build the dataframe of all the transactions with the following columns
- id_buyer
- id_seller
- home_id
- price
- initial_asking_price
- timestep
- buyer features
- seller features
- house features
"""
def create_df_db_transactions_model(run_ids, t_start, t_end, variables_all_runs, vars_init_all_runs, abm, datetimes):
    
    dates = obtain_dates(t_start, t_end, abm, datetimes)
    df_transaction_list = []
    for run_id in range(len(run_ids)):
        variables = variables_all_runs[run_id]
        vars_init = vars_init_all_runs[run_id]
        
        for t in range(t_start,t_end):
            
            initial_asking_prices = np.array([variables["price_required_list_copy"][int(t-variables["time_on_market_copy"][t][i]+0.5)][i] \
                                              for i in variables["which_homes_transacted"][t]])
            
            transaction_dict = {
                "id_seller": variables["successful_sellers"][t],
                "id_buyer": variables["successful_buyers"][t],
                "home_id": variables["which_homes_transacted"][t],
                "price": variables["prices_transactions_this_step"][t],
                "timestep": t,
                "awareness_buyer":variables["awarenesses_successful_buyers"][t],
                "awareness_seller":variables["awarenesses_successful_sellers"][t],
                "initial_asking_price": initial_asking_prices,
                "res_price_buyer": variables["reservation_price_buyers"][t],
                "res_price_seller": variables["reservation_price_sellers"][t],
                "transaction_price_to_initial_asking_price":  variables["prices_transactions_this_step"][t] / initial_asking_prices,
                "income_buyer": list(vars_init["ag"]["income_cross_section"][variables["successful_buyers"][t]]),
                "income_seller": list(vars_init["ag"]["income_cross_section"][variables["successful_sellers"][t]]),
                "monthly_payment_to_income_buyer": list(variables["prices_transactions_this_step"][t] / variables["reservation_price_buyers"][t] * abm.parameters["monthly_payment_to_income"]),
                "run_id": run_id,
                "interest_rate": abm.parameters["interest_rates"],
                "loan_to_value": abm.parameters["loan_to_value"],
                "date": dates[t - t_start]
                }
                
            df_transaction_list.append(pd.DataFrame(transaction_dict))
            
    df_db_transactions_model = pd.concat(df_transaction_list)
    df_db_transactions_model = df_db_transactions_model.merge(vars_init["da"][["price_hedonic", "risk_score"] + spatial_columns + home_columns], 
                                                              left_on="home_id", right_index=True )


    return df_db_transactions_model.reset_index()

"""
given the output of a set of simulations, it build the dataframe of buyers information
a buyer is an agent who sold the house and is in the market
here we store the timestep she become buyer, the time she remains a buyer, her reservation price, her income
"""
def create_df_db_buyers_model(run_ids, t_start, t_end, variables_all_runs, vars_init_all_runs, abm, datetimes):
    dates = obtain_dates(t_start, t_end, abm, datetimes)
    
    df_buyers_list = []

    for run_id in range(len(run_ids)):
        variables = variables_all_runs[run_id]
        vars_init = vars_init_all_runs[run_id]
        for t in range(t_start,t_end):
            buyers_ids = np.concatenate([variables["successful_buyers"][t], variables["unsuccessful_buyers"][t]])
            buyers_dict = {
                "id_buyer": buyers_ids,
                "successful_buyer": np.concatenate([np.repeat(True, len(variables["successful_buyers"][t])), 
                                                    np.repeat(False, len(variables["unsuccessful_buyers"][t]))]),
                "income_buyer": list(vars_init["ag"]["income_cross_section"][buyers_ids]),
                "res_price_buyer": np.concatenate([variables["reservation_price_buyers"][t], variables["reservation_price_unsuccessful_buyers"][t]]),
                "awareness_buyer": np.concatenate([variables["awarenesses_successful_buyers"][t], variables["awarenesses_unsuccessful_buyers"][t]]),
                "time_unsuccessful_buying": np.concatenate([np.repeat(None, len(variables["successful_buyers"][t])), variables["time_unsuccessful_buying"][t]]).astype(np.float32),
                "home_id": np.concatenate([variables["which_homes_transacted"][t], np.repeat(np.nan, len(variables["unsuccessful_buyers"][t]))]),
                "run_id": run_id,
                "interest_rate": abm.parameters["interest_rates"],
                "loan_to_value": abm.parameters["loan_to_value"],
                "timestep": t,
                "date": dates[t - t_start]
            }

            for col in ["price_hedonic", "risk_score"] + spatial_columns:
                buyers_dict[col] = list(vars_init["da"].loc[variables["which_homes_transacted"][t], col]) + list(np.repeat(None, len(variables["unsuccessful_buyers"][t])))
            
            df_buyers_list.append(pd.DataFrame(buyers_dict))

    df_db_buyers_model = pd.concat(df_buyers_list)
    return df_db_buyers_model.reset_index()#.reset_index()


"""
given the output of a set of simulations, it builds the dataframe of sellers information
a seller is an agent who decide to sell the house
once she sells the house, she becomes a buyer
here we store the timestep she become seller, the time she remains a seller, her asking price at each timestep
"""   
def create_df_db_sellers_model(run_ids, t_start, t_end, variables_all_runs, vars_init_all_runs, abm, datetimes):
    
    dates = obtain_dates(t_start, t_end, abm, datetimes)
    
    df_sellers_list = []

    for run_id in range(len(run_ids)):
        variables = variables_all_runs[run_id]
        vars_init = vars_init_all_runs[run_id]
        for t in range(t_start,t_end):
            sellers_ids = np.concatenate([variables["successful_sellers"][t], variables["unsuccessful_sellers"][t]])
            home_ids = np.concatenate([variables["which_homes_transacted"][t], variables["which_homes_not_sold"][t]])
            initial_asking_prices = np.array([variables["price_required_list_copy"][int(t-variables["time_on_market_copy"][t][i]+0.5)][i] \
                                              for i in home_ids])
            
            sellers_dict = {
                "id_seller": sellers_ids,
                "successful_seller": np.concatenate([np.repeat(True, len(variables["successful_sellers"][t])), 
                                                    np.repeat(False, len(variables["unsuccessful_sellers"][t]))]),
                "income_seller": list(vars_init["ag"]["income_cross_section"][sellers_ids]),
                "res_price_seller": np.concatenate([variables["reservation_price_sellers"][t], variables["reservation_price_unsuccessful_sellers"][t]]),
                "awareness_seller": np.concatenate([variables["awarenesses_successful_sellers"][t], variables["awarenesses_unsuccessful_sellers"][t]]),
                "time_on_market": variables["time_on_market_copy"][t][home_ids],
                "home_id": home_ids,
                "asking_price": variables["price_required_list"][t][home_ids],
                "initial_asking_prices": initial_asking_prices,
                "run_id": run_id,
                "interest_rate": abm.parameters["interest_rates"],
                "loan_to_value": abm.parameters["loan_to_value"],
                "timestep": t,
                "date": dates[t - t_start]                
            }

            for col in ["price_hedonic", "risk_score"] + spatial_columns:
                sellers_dict[col] = list(vars_init["da"].loc[home_ids, col])
            
            df_sellers_list.append(pd.DataFrame(sellers_dict))

    df_db_sellers_model = pd.concat(df_sellers_list)
    return df_db_sellers_model.reset_index()#.reset_index()


"""
run several time the ABM

input
- cod_reg: select the region with the relative code (the dataframe cod_reg_abbrv_df contains all the codes)
- parameters: all the parameters of the simulations, including agents climate awareness (awareness_scenario and awareness input), 
    length of the simulation (T), length of the transient (Tb), and the parameters that are estimated from data 
    during the calibration phase (probability_sell, fraction_buyers, markup, price_reduction, awarenesses_input)
- n_runs: number of runs, that is the number of different simulations we want to run with the same set of parameters
- awareness_scenario: we can model the ABM with different awareness scenario. In the future simulations, this depend on the number of floods
- seeds: list of seeds for initializing the ABM
- master_locations: census dataset 
- isp: isp transactions dataset
- hide_progress: if it is False, it shows the progress bar (this can slow the simulations)
- hide_progress_transient: if it is False, it shows the progress bar during the transient (this can slow the simulations)
- max_agents: limit the synthetic population to avoid memory overload
- corrected_population: maximum number of transactions per month. The number of agents (population) will be linearly transformed maintain the ratio numer of transactions / population

output
- df_db_transactions_model: dataframe with all the transactions in each simulation run 
- df_db_buyers_model: dataframe with all the buyers states in each simulation run
- df_db_sellers_model: dataframe with all the sellers states in each simulation run
"""
def run_abms_reg(cod_reg, parameters, n_runs, awareness_scenario = None, seeds = [], master_locations = master_locations, 
             isp = isp, hide_progress = False, hide_progress_transient = True,
             max_agents = 1000000, corrected_population = None):
    if master_locations is None:
        master_locations, isp = load_data()
    if len(seeds) == 0:
        seeds = [None for _ in range(n_runs)]
    
    abm_all_runs, variables_all_runs, vars_init_all_runs = [], [], []
    run_ids = range(n_runs)
    for run_id in run_ids:
        vars = run_abm_reg(cod_reg, parameters, awareness_scenario = awareness_scenario, seed = seeds[run_id], 
                            master_locations = master_locations, isp = isp, hide_progress = hide_progress, 
                            hide_progress_transient = hide_progress_transient, max_agents = max_agents, corrected_population = corrected_population)
        abm_all_runs.append(vars[0])
        variables_all_runs.append(vars[1])
        vars_init_all_runs.append(vars[2])

    t_start = 1
    t_end = abm_all_runs[0].T+abm_all_runs[0].Tb
    timestamps = pd.date_range(start = '2024-01-01', 
                               end = '2100-12-01', freq='MS') + pd.offsets.Day(14)
    df_db_transactions_model = create_df_db_transactions_model(run_ids, t_start, t_end,
                                                               variables_all_runs, vars_init_all_runs, 
                                                               abm_all_runs[0], timestamps)

    df_db_buyers_model = create_df_db_buyers_model(run_ids, t_start, t_end,
                                                      variables_all_runs, vars_init_all_runs, 
                                                      abm_all_runs[0], timestamps)

    df_db_sellers_model = create_df_db_sellers_model(run_ids, t_start, t_end,
                                                        variables_all_runs, vars_init_all_runs, 
                                                        abm_all_runs[0], timestamps)
    return df_db_transactions_model, df_db_buyers_model, df_db_sellers_model










