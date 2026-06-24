#!/bin/bash

# GPU
export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python Environment
RUN_PYTHON_PATH="${RUN_PYTHON_PATH:-$(command -v python)}"
if [ -z "$RUN_PYTHON_PATH" ] || [ ! -x "$RUN_PYTHON_PATH" ]; then
    echo "Error: cannot find Python executable. Activate your env or set RUN_PYTHON_PATH=/path/to/python"
    exit 1
fi

require_user_path() {
    local name="$1"
    local value="$2"
    if [[ "$value" == REPLACE_WITH_* ]]; then
        echo "Error: please edit $name in $0 and set it to your local path."
        exit 1
    fi
}

# Dataset
TRAIN_SET_PATH="REPLACE_WITH_PREPROCESSED_TRAIN_SET_DIR"
TRAIN_SET_LIST_PATH="REPLACE_WITH_TRAIN_SET_LIST_JSON"

# Pretrained RiskDiffuser
RISKDIFFUSER_CHECKPOINT="$PROJECT_ROOT/checkpoints/"

# Training Config
EXP_NAME="risk_net_head_only"
BATCH_SIZE=64 #128
EPOCHS=50  #30临近稳定
LR=2e-4

# Run Training

echo "Starting RiskNet Training..."
echo "Experiment: $EXP_NAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "LR: $LR"
echo "----------------------------------"

require_user_path "TRAIN_SET_PATH" "$TRAIN_SET_PATH"
require_user_path "TRAIN_SET_LIST_PATH" "$TRAIN_SET_LIST_PATH"

"$RUN_PYTHON_PATH" train_risk_net.py \
--name $EXP_NAME \
--train_set $TRAIN_SET_PATH \
--train_set_list $TRAIN_SET_LIST_PATH \
--pretrained_riskdiffuser_ckpt $RISKDIFFUSER_CHECKPOINT \
--freeze_encoder true \
--batch_size $BATCH_SIZE \
--train_epochs $EPOCHS \
--learning_rate $LR \
--device cuda

echo "Training Finished"
