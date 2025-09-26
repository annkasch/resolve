
import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["PYTENSOR_FLAGS"] = "compiledir=./pytensor_cache,mode=FAST_COMPILE,optimizer=None"
import argparse
import numpy as np
import pandas as pd
import yaml
np.random.seed(42)
import os
from resum.polynomial_chaos_expansion import PCEMultiFidelityModel
import matplotlib.pyplot as plt

def get_combined_variance(mu, sigma, nsamples):
    mu_err=np.sum(nsamples*(mu-np.ones(mu.shape[0])*mu.mean())**2)
    var_comb = (np.sum((nsamples-1.)*sigma**2)+mu_err)/(nsamples**2-1.)
    return var_comb



def main(path_to_settings):
    with open(path_to_settings, "r") as f:
        config_file = yaml.safe_load(f)

    version       = config_file["path_settings"]["version"]
    path_out_cnp  = config_file["path_settings"]["path_out_cnp"]
    path_out_pce  = config_file["path_settings"]["path_out_pce"]
    file_in       = f'{path_out_cnp}/cnp_{version}_output.csv'
    path_out = f'{path_out_pce}/{version}/deg{config_file["regression_settings"]["polynomial_order"]}'

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Set parameter name/x_labels -> needs to be consistent with data input file
    x_labels        = config_file["simulation_settings"]["theta_headers"]
    y_label_cnp     = 'y_cnp'
    y_err_label_cnp = 'y_cnp_err'
    y_label_sim     = 'y_raw'

    # Set parameter boundaries
    xmin    = config_file["simulation_settings"]["theta_min"]
    xmax    = config_file["simulation_settings"]["theta_max"]

    parameters={}
    for i,x in enumerate(x_labels):
        parameters[x]=[xmin[i],xmax[i]]

    data=pd.read_csv(file_in)
    data = data.drop_duplicates()

    #LF_cnp_noise=np.mean(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_err_label_cnp].to_numpy())
    #LF_cnp_noise=np.var(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_cnp].to_numpy())**2
    #LF_sim_noise=np.var(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_sim].to_numpy())
    HF_sim_noise = np.var(data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy())
    LF_cnp_noise = get_combined_variance(data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_label_cnp].to_numpy(),
                                         data.loc[(data['fidelity']==0.) & (data['iteration']==0)][y_err_label_cnp].to_numpy(),
                                         config_file["regression_settings"]["nsamples_per_lf"])

    x_train_hf = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][x_labels].to_numpy()
    y_train_hf = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][y_label_sim].to_numpy()

    #x_train_mf = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][x_labels].to_numpy()
    #y_train_mf = data.loc[(data['fidelity']==1.) & (data['iteration']==0)][ y_label_cnp].to_numpy()

    #x_train_lf_sim = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy()
    #y_train_lf_sim = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][ y_label_sim].to_numpy()

    x_train_lf = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][x_labels].to_numpy()
    y_train_lf = data.loc[(data['fidelity']==0.) & (data['iteration']==0)][ y_label_cnp].to_numpy()
    

    # Initialize the model
    trainings_data = {
        "lf": [x_train_lf, y_train_lf], 
    #    "mf": [x_train_mf, y_train_mf], 
        "hf": [x_train_hf, y_train_hf]
    }

    # Prior configurations
    # sigma_coeffs_prior_types: "default", "auto", "cauchy", "lasso"
    priors = {
        "lf": {"sigma_coeffs_prior_type": config_file["regression_settings"]["priors"]["lf"]["sigma_coeffs_prior_type"], "sigma_coeffs": config_file["regression_settings"]["priors"]["lf"]["sigma_coeffs"], "sigma_y": LF_cnp_noise},
    #    "mf": {"mu_rho": 1., "sigma_rho": 0.5, "sigma_coeffs_prior_type": "default","sigma_coeffs_delta": 0.05, "sigma": HF_cnp_noise},
        "hf": {"mu_rho": config_file["regression_settings"]["priors"]["hf"]["mu_rho"], "sigma_rho": config_file["regression_settings"]["priors"]["hf"]["sigma_rho"], "sigma_coeffs_prior_type": config_file["regression_settings"]["priors"]["hf"]["sigma_coeffs_prior_type"], "sigma_coeffs_delta": config_file["regression_settings"]["priors"]["hf"]["sigma_coeffs"], "sigma_y": HF_sim_noise}
    }

    polynomial_order = config_file["regression_settings"]["polynomial_order"]

    # Initialize the multi-fidelity model
    multi_fidelity_model = None
    multi_fidelity_model = PCEMultiFidelityModel(trainings_data, priors, parameters,degree=polynomial_order)

    multi_fidelity_model.build_model()
    #multi_fidelity_model.sanity_check_of_basis()

    draws = config_file["regression_settings"]["n_draws"]
    tune = config_file["regression_settings"]["tune"]
    chains = config_file["regression_settings"]["n_chains"]
    cores = config_file["regression_settings"]["n_cores"]
    target_accept = config_file["regression_settings"]["target_accept"]
    #trace = multi_fidelity_model.run_inference(method="advi", n_steps=10000, n_samples=20000)
    trace = multi_fidelity_model.run_inference(method="nuts", n_samples=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept)

    #deg_str = f"deg{multi_fidelity_model.degree[0]}" + "".join(f"_and_{n}" for n in multi_fidelity_model.degree[1:])
    prefix = path_out
    os.makedirs(prefix, exist_ok=True)


    multi_fidelity_model.save_trace(f"{prefix}/trace.nc")
    #multi_fidelity_model.readin_trace(f"{prefix}/trace.nc")

    # some diagnostics
    multi_fidelity_model.summarize_sampler()
    fig = multi_fidelity_model.pareto_k(f"{prefix}/pareto_k.png")
    fig = multi_fidelity_model.plot_diagnostics(f"{prefix}/bfmi.png")
    multi_fidelity_model.print_table(f"{prefix}/table.csv")
    fig = multi_fidelity_model.plot_pair(f"{prefix}/pair.png")
    fig = multi_fidelity_model.plot_trace(f"{prefix}/trace.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_settings', type=str, default="../binary-black-hole/settings.yaml")
    args = parser.parse_args()

    main(args.path_to_settings)
