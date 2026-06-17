#!/bin/bash

python run.py -m \
  experiment=real_targets \
  data=yahoo,mslr30k,istella \
  random_state=50,51,52,53,54,55,56,57,58,59 \
  policy_temperature=0.5 \
  ips.model=ips,dm,dr \
  propensity_model=MLPregression,frequency-based,true_propensity  \
  ips.n_sessions=500,1000,5000,10000,50000,100000,500000,1000000 \
  data.preprocessor.top_x=25 \
  $@


# propensity_model=MLPregression,frequency-based,true_propensity \
# ips.n_sessions=1000,5000,10000,50000,100000,500000 \
# ips.n_sessions=500,1000,5000,10000,50000,100000,500000, \
# random_state=40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
# propensity_model=MLPclassifier,kmeans,cosine,knn,frequency-based,true_propensity \
