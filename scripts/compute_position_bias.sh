#!/bin/bash

CSV_PATH="results/position_bias_csv_smoke.csv"

python estimate_position_bias.py -m \
  experiment=position_bias_csv_smoke \
  data=yahoo \
  random_state=41 \
  policy_temperature=0.5 \
  propensity_model=frequency-based \
  ips.n_sessions=10000,100000 \
  ips.position_bias.export_path="${CSV_PATH}" \
  "$@"