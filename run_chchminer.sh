#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Processing ChChMiner CSV files ==="
python data/chchminer/processed/process_chchminer.py

echo "=== Preprocessing ChChMiner molecular graphs ==="
python graphpreprocess.py \
  --config_file data/chchminer/processed/config_chchminer.yml

echo "=== Running ChChMiner smoke test ==="
python smoke_test.py

echo "=== Training ChChMiner on the full processed dataset ==="
python train_task.py \
  --config_file data/chchminer/processed/config_chchminer.yml
