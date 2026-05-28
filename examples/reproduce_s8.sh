#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs
python3 generate_wilf_from_shape_d8.py \
  --input data/reverse_k8.json \
  --output-json outputs/generated_wilf_k8_from_shape_d8.json \
  --output-txt outputs/generated_wilf_k8_from_shape_d8.txt
