#!/bin/bash

pvapy-hpc-consumer \
  --input-channel pvapy:image \
  --control-channel processor:*:control \
  --status-channel processor:*:status \
  --output-channel processor:*:output \
  --processor-file inferPtychoNNImageProcessor.py \
  --processor-class InferPtychoNNImageProcessor \
  --processor-args '{"net": "eno1"}' \
  --report-period 5 \
  --n-consumers 16 \
  --server-queue-size 100 \
  --monitor-queue-size 1000 \
  --distributor-updates 8 \
  --disable-curses