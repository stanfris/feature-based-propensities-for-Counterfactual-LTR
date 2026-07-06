#!/bin/bash

python run.py -m \
  experiment=real_targets \
  data=mslr30k,yahoo,istella \
  random_state=40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59\
  policy_temperature=0.0,0,25,0.75,1.0 \
  propensity_model=frequency-based,MLPregression,true_propensity \
  ips.model=ips,dr,dm \
  ips.n_sessions=10000 \
  $@


# add 0.5 if you don't run the policy models