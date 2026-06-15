#!/bin/bash

python compare_propensity_param.py -m \
  experiment=compare_propensity_estimation \
  data=mslr30k \
  policy_temperature=0.5 \
  data.preprocessor.max_documents_per_query=25 \
  random_state=2023 \
  ips.n_sessions=75000 \
  test_clicks=30000 \
  propensity_model=euclidean \
  propensity_model.euclidean.euclidean_threshold=3.5 \
  $@
