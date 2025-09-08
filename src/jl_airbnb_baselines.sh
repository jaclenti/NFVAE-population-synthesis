#!/bin/bash

cliopts="$@"	# optional parameters
max_procs=4	# parallelism
timeout=30000	# timeout in seconds


citynum="0 1 2 3 4 5 6 7 8 9 10 11 12"


parallel --timeout ${timeout} --ungroup --max-procs ${max_procs} "python src/airbnb_all_baselines.py {1}" ::: $citynum