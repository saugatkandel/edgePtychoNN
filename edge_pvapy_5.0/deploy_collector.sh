#!/bin/bash

echo "deploying inference collector"
pvapy-hpc-collector \
  --collector-id 1 \
  --producer-id-list "range(1,27,1)" \
  --input-channel processor:*:output \
  --control-channel collector:*:control \
  --status-channel collector:*:status \
  --output-channel collector:*:output \
  --processor-class pvapy.hpc.userDataProcessor.UserDataProcessor \
  --report-period 5 \
  --server-queue-size 10000 \
  --collector-cache-size 1000 \
  --monitor-queue-size 2000 \
  --disable-curses