#!/bin/bash

python train_lp_predictor.py -m \
  experiment=2-example \
  data=mslr30k \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1.0 \
  policy_temperature=0.0,0.5,1.0 \
  random_state=2023 \
  use_propensity_weighting=False \
  $@
