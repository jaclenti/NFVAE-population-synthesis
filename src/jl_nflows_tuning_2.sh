#!/bin/bash

cliopts="$@"	# optional parameters
max_procs=1	# parallelism
timeout=60000	# timeout in seconds


num_layers_nf="48"
hidden_features_nf="48"
num_epochs_nf="20000"
lr_nf_="10"
opt_nf="Adam"
flow="NSF_CL"
date="nf_241128"

parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python jl_nflows_geo_coordinates_2.py {1} {2} {3} {4} {5} {6} {7}" ::: $num_layers_nf ::: $hidden_features_nf ::: $num_epochs_nf ::: $lr_nf_ ::: $opt_nf ::: $flow ::: $date
