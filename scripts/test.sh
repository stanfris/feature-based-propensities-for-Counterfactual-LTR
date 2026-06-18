#!/bin/bash

# python run.py -m \
#   experiment=test \
#   data=mslr30k \
#   random_state=43 \
#   policy_temperature=0.5 \
#   propensity_model=frequency-based \
#   ips.model=ips \
#   ips.n_sessions=10000 \
#   "$@"


# propensity_model=MLPclassifier,kmeans,cosine,knn,frequency-based,true_propensity \
  # ips.n_sessions=1000,5000,10000,50000,100000,500000 \
  # ips.n_sessions=100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000,500000, \


python run.py -m \
  experiment=est_pos_bias \
  data=mslr30k,yahoo,istella \
  random_state=40,41,42,43,44,45,46,47,48,49 \
  policy_temperature=0.5 \
  ips.model=ips,dm,dr \
  propensity_model=true_propensity,frequency-based,MLPregression \
  ips.n_sessions=10000 \
  ips.position_bias.source=estimate \
  ips.position_bias.estimator=global_all_pairs \
  "$@"

