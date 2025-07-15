#!/bin/bash

cliopts="$@"	# optional parameters
max_procs=4	# parallelism
timeout=30000	# timeout in seconds


num_layers_nf="16 32 64 128"
hidden_features_nf="8 16 32 64 128"
num_epochs_nf="20000"
lr_nf_="1 10 100 1000"
opt_nf="Adam RMSprop"
date="nf_241115"

parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python jl_nflows_geo_coordinates.py {1} {2} {3} {4} {5} {6}" ::: $num_layers_nf ::: $hidden_features_nf ::: $num_epochs_nf ::: $lr_nf_ ::: $opt_nf ::: $date