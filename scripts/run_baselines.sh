#!/bin/bash

python run.py -m \
  experiment=true_baselines \
  data=yahoo \
  random_state=40,41,42 \
  policy_temperature=0.5 \
  ips.model=logging_policy \
  propensity_model=frequency-based \
  ips.n_sessions=100 \
  data.preprocessor.top_x=25 \
  $@


# propensity_model=MLPregression,frequency-based,true_propensity \
# ips.n_sessions=1000,5000,10000,50000,100000,500000 \
# ips.n_sessions=500,1000,5000,10000,50000,100000,500000, \
# random_state=40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
# propensity_model=MLPclassifier,kmeans,cosine,knn,frequency-based,true_propensity \
