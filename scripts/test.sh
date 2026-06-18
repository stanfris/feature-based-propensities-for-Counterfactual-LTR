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
  experiment=test_pbm \
  data=mslr30k \
  random_state=45 \
  policy_temperature=0.5 \
  propensity_model=MLPregression \
  ips.model=ips \
  ips.n_sessions=10000 \
  ips.position_bias.source=estimate \
  ips.position_bias.estimator=global_all_pairs \
  "$@"

