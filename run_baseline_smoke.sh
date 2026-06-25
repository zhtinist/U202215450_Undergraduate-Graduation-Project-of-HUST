#!/usr/bin/env bash
set -euo pipefail

# 用法:
# bash run_baseline_smoke.sh <model_name_or_path>
# 示例:
# bash run_baseline_smoke.sh /path/to/local_t5_or_udop_checkpoint

MODEL_PATH="${1:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "请传入本地模型路径，例如: /path/to/local_checkpoint"
  exit 1
fi

source /home/zht/miniconda3/etc/profile.d/conda.sh
conda activate zht

export PYTHONPATH="/home/zht/HaotianZhu/src/LecSlides_370K:${PYTHONPATH:-}"
export LECSLIDES_T5_TOKENIZER_PATH="${MODEL_PATH}"

cd /home/zht/HaotianZhu

python src/LecSlides_370K/train/train.py \
  --model_map_name udop_t5 \
  --task my_summary \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path /home/zht/HaotianZhu/data/slideshare_data_merging_v2_4_train_10pct_local.json \
  --output_dir /home/zht/HaotianZhu/log/repro/smoke_udop_t5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --save_steps 1000 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --report_to none
