#!/bin/bash

python run.py -m \
  experiment=1_example \
  data=mslr30k \
  policy_temperature=0.5 \
  ips.model=ips \
  propensity_model=MLPregression  \
  ips.n_sessions=10000 \
  data.preprocessor.top_x=25 \
  "$@"
