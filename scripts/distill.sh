#!/bin/bash

# Loop over 10
for i in $(seq 1 10)
do
  echo "Run number: $i"
  # Kill any process using port 8080 after 4 hours
  ( sleep 14400 && fuser -k 8080/tcp ) & \
  bash scripts/run_distillation.sh "$@"
done
