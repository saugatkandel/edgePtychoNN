#!/bin/bash

rm -rf data/collected &&  mkdir -p data/collected
pvapy-hpc-consumer \
    --input-channel collector:1:output \
    --output-channel file:*:output \
    --control-channel file:*:control \
    --status-channel file:*:status \
    --processor-class pvapy.hpc.adOutputFileProcessor.AdOutputFileProcessor \
    --processor-args '{"outputDirectory" : "data/collected/", "outputFileNameFormat" : "bdp_{uniqueId:06d}.{processorId}.tiff"}' \
    --n-consumers 4 \
    --report-period 10 \
    --server-queue-size 1000 \
    --monitor-queue-size 10000 \
    --distributor-updates 1 \
    --disable-curses