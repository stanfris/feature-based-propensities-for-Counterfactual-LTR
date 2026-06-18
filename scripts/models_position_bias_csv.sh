#!/bin/bash

CSV_PATH="results/position_bias_csv_smoke.csv"

# python estimate_position_bias.py -m \
#   experiment=position_bias_csv_smoke \
#   data=yahoo,istella \
#   random_state=41 \
#   policy_temperature=0.5 \
#   propensity_model=frequency-based \
#   ips.n_sessions=1000,10000,100000,1000000 \
#   ips.position_bias.export_path="${CSV_PATH}" \
#   $@

python run.py -m \
  experiment=position_bias_models \
  data=istella,yahoo \
  random_state=40,41,42 \
  policy_temperature=0.5 \
  propensity_model=frequency-based,MLPregression,true_propensity \
  ips.model=ips \
  ips.n_sessions=10000 \
  ips.position_bias.source=estimate \
  ips.position_bias.estimator=global_all_pairs \
  "$@"

