#!/bin/bash

python run.py -m \
  experiment=test \
  data=mslr30k \
  random_state=42 \
  propensity_model=frequency-based \
  ips.n_session_percentage=0.1 \
  ips.model=max-score \
  ips.trainer.max_epochs=20 \
  data.preprocessor.top_x=20 \
  ips.debug=true \
  "$@"
