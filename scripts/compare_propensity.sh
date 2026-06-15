#!/bin/bash

python compare_propensity.py -m \
  experiment=3-example \
  data=mslr30k \
  relevance=deep \
  logging_policy_ranker=deep \
  relevance_tower=deep \
  policy_strength=1.0 \
  policy_temperature=0.5\
  random_state=2023 \
  $@
