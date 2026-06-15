#!/bin/bash

python run.py -m \
  experiment=real_targets \
  data=istella,mslr30k,yahoo \
  random_state=40,41,42 \
  policy_temperature=0.0,0.25,0.50,0.75,1.0 \
  propensity_model=frequency-based,MLPregression,true_propensity \
  ips.model=ips,dr,dm \
  ips.n_sessions=10000 \
  $@


# propensity_model=MLPclassifier,kmeans,cosine,knn,frequency-based,true_propensity \
  # ips.n_sessions=1000,5000,10000,50000,100000,500000 \
  # ips.n_sessions=100,500,1000,2500,5000,7500,10000,25000,50000,75000,100000,500000, \
