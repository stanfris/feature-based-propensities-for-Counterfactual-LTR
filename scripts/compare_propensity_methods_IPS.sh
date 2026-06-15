#!/bin/bash

python run_IPS.py -m \
  experiment=compare_propensity_methods_IPS_policy_0.5 \
  data=mslr30k \
  random_state=42,43,44 \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=-0.5,0.5 \
  policy_temperature=0.5 \
  propensity_model=frequency-based \
  ips.n_sessions=100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000,500000,1000000,5000000 \
  $@


# propensity_model=MLPclassifier,kmeans,cosine,knn,frequency-based,true_propensity \
  # ips.n_sessions=1000,5000,10000,50000,100000,500000 \
  # ips.n_sessions=100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000,500000, \
