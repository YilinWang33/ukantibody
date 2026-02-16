#!/bin/bash

# 设置遇到错误即停止 (可选但推荐)
set -e

echo "开始预处理抗体数据..."

# 定义变量 (这样修改路径更方便)
JSON_PATH="/DATA/disk3/yilin/dyMEAN/all_data/test.json"
PDB_DIR="/DATA/disk3/yilin/dyMEAN/all_data/pdb"
OUT_DIR="./data/antibody_processed_2"

# 执行 Python 脚本
python script_utils/preprocess_antibody_data.py \
  --json_path "$JSON_PATH" \
  --pdb_dir "$PDB_DIR" \
  --out_dir "$OUT_DIR"

echo "预处理流程结束。"