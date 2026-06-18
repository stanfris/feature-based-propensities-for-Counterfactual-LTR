#!/bin/bash


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