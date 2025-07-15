#!/bin/bash

cliopts="$@"	# optional parameters
max_procs=4	# parallelism
timeout=30000	# timeout in seconds


latent_dims_vae="12 8"
flows_conf_vae="0 1 2 3"
num_epochs_vae="4000"
lr_vae_="10 100 1000 10000"
opt_vae="Adam"
date="vae_241031"


parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python jl_vae.py {1} {2} {3} {4} {5} {6}" ::: $latent_dims_vae ::: $flows_conf_vae ::: $num_epochs_vae ::: $lr_vae_ ::: $opt_vae ::: $date