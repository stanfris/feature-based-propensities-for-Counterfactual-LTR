#!/bin/bash

python run.py -m \
  experiment=1-example \
  ips.model=add-two-tower \
  test_set_mode=clicks \
  data=mslr30k \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1.0 \
  policy_temperature=0.0 \
  random_state=2023 \
  use_propensity_weighting=False \
  $@
