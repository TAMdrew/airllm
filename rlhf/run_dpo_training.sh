#!/usr/bin/env bash
# =============================================================================
# DPO (Direct Preference Optimization) training script.
#
# This is a research/training utility, separate from the core AirLLM inference
# library. It trains a 33B model using QLoRA + DPO on a single GPU.
# =============================================================================

set -x -e

run_id=$(date +%s)
echo "RUN ID: ${run_id}"

echo "START TIME: $(date)"

ROOT_DIR_BASE=./Anima_run
OUTPUT_PATH="${ROOT_DIR_BASE}/output_${run_id}"

mkdir -p "${OUTPUT_PATH}"

python qlora_dpo.py --dataset="lyogavin/Anima33B_rlhf_belle_eval_1k" `# rlhf dataset` \
    --dataset_format="hh-rlhf" `# follow hh-rlhf format` \
    --learning_rate 0.0001 `# QLoRA paper appendix B Table 9 `\
    --per_device_train_batch_size 1 `# fix for fitting mem `\
    --gradient_accumulation_steps 16 `# QLoRA paper appendix B Table 9  `\
    --max_steps 100 `# run 100 steps`\
    --model_name_or_path "lyogavin/Anima33B-merged" `# the base model to train on` \
    --reference_model "lyogavin/Anima33B-merged" `# the reference model the training should reference` \
    --source_max_len 600  `# 600 roughly covers 90th percentile of lengths`\
    --target_max_len 600 `# 600 roughly covers 90th percentile of lengths`\
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 10 `# eval every 10 steps to monitor training`  \
    --output_dir "${OUTPUT_PATH}" \
    --report_to 'wandb' \
    --sample_generate `# test sample generation every once a while`  \
    --save_steps 10 `# save every 10 steps for reproducibility` \
    --train_on_source true \
    --lora_r 256 \
    --beta 0.1 `# Temperature parameter for the DPO loss, typically 0.1 to 0.5.`
    #--debug_mode `# only set when it's debug mode` \
