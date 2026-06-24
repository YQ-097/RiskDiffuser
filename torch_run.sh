export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="${RUN_PYTHON_PATH:-$(command -v python)}" # set RUN_PYTHON_PATH to override the Python executable
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

# Set training data path
TRAIN_SET_PATH="REPLACE_WITH_PREPROCESSED_TRAIN_SET_DIR" # output directory from data_process.sh
TRAIN_SET_LIST_PATH="REPLACE_WITH_TRAIN_SET_LIST_JSON" # e.g., /path/to/flow_planner_training.json
RISKDIFFUSER_CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/"
RISK_MODEL_PATH="$PROJECT_ROOT/checkpoints/riskcheckpoint/"
USE_RISK_MODEL_INFER="true"
###################################

require_user_path "TRAIN_SET_PATH" "$TRAIN_SET_PATH"
require_user_path "TRAIN_SET_LIST_PATH" "$TRAIN_SET_LIST_PATH"

# sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
# --train_set  $TRAIN_SET_PATH \
# --train_set_list  $TRAIN_SET_LIST_PATH \

sudo -E "$RUN_PYTHON_PATH" train_predictor.py \
--train_set $TRAIN_SET_PATH \
--train_set_list $TRAIN_SET_LIST_PATH \
--resume_model_path $RISKDIFFUSER_CHECKPOINT_PATH \
--risk_model_path $RISK_MODEL_PATH \
--use_risk_model_infer $USE_RISK_MODEL_INFER

# sudo -E $RUN_PYTHON_PATH -m torch.distributed.run \
# --nnodes 1 \
# --nproc-per-node 1 \
# --standalone \
# train_predictor.py \
# --train_set $TRAIN_SET_PATH \
# --train_set_list $TRAIN_SET_LIST_PATH
