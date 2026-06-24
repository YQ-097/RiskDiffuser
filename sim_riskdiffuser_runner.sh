export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="REPLACE_WITH_NUPLAN_DEVKIT_ROOT"  # e.g., /path/to/nuplan-devkit
export NUPLAN_DATA_ROOT="REPLACE_WITH_NUPLAN_DATA_ROOT"  # e.g., /path/to/nuplan-data-root
export NUPLAN_MAPS_ROOT="REPLACE_WITH_NUPLAN_MAPS_DIR" # e.g., /path/to/nuplan-v1.1/maps
export NUPLAN_EXP_ROOT="REPLACE_WITH_NUPLAN_EXP_DIR" # e.g., /path/to/nuplan/exp

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="test14-hard"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_reactive_agents"  # e.g., "closed_loop_nonreactive_agents"

# Parallel simulation configuration
# Options:
#   WORKER_MODE="sequential"
#   WORKER_MODE="ray_distributed"
WORKER_MODE="ray_distributed"
THREADS_PER_NODE=8
GPUS_PER_SIMULATION=0.125
###################################

require_user_path() {
    local name="$1"
    local value="$2"
    if [[ "$value" == REPLACE_WITH_* ]]; then
        echo "Error: please edit $name in $0 and set it to your local path."
        exit 1
    fi
}

require_user_path "NUPLAN_DEVKIT_ROOT" "$NUPLAN_DEVKIT_ROOT"
require_user_path "NUPLAN_DATA_ROOT" "$NUPLAN_DATA_ROOT"
require_user_path "NUPLAN_MAPS_ROOT" "$NUPLAN_MAPS_ROOT"
require_user_path "NUPLAN_EXP_ROOT" "$NUPLAN_EXP_ROOT"


BRANCH_NAME=riskdiffuser_release
ARGS_FILE="$PROJECT_ROOT/checkpoints/args.json"
CKPT_FILE="$PROJECT_ROOT/checkpoints/latest.pth"

RISK_MODE="risk_net"  #"manual"              # options: manual / risk_net

MANUAL_RISK_LEVEL="0.1"         # used when RISK_MODE=manual

RISK_MODEL_PATH="$PROJECT_ROOT/checkpoints/riskcheckpoint/latest.pth"
RISK_MODEL_ENABLE_EMA="true"

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=riskdiffuser

if [ "$WORKER_MODE" = "ray_distributed" ]; then
    if command -v ray >/dev/null 2>&1; then
        echo "Stopping any existing Ray runtime to avoid cluster attach conflicts..."
        ray stop --force >/dev/null 2>&1 || true
    fi
fi

python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
    +simulation="$CHALLENGE" \
    planner="$PLANNER" \
    "planner.riskdiffuser.config.args_file='$ARGS_FILE'" \
    "planner.riskdiffuser.ckpt_path='$CKPT_FILE'" \
    "planner.riskdiffuser.risk_mode='$RISK_MODE'" \
    "planner.riskdiffuser.manual_risk_level=$MANUAL_RISK_LEVEL" \
    "planner.riskdiffuser.risk_model_path='$RISK_MODEL_PATH'" \
    "planner.riskdiffuser.risk_model_enable_ema=$RISK_MODEL_ENABLE_EMA" \
    scenario_builder="$SCENARIO_BUILDER" \
    scenario_filter="$SPLIT" \
    "experiment_uid='$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S")'" \
    verbose=true \
    worker=$WORKER_MODE \
    worker.threads_per_node=$THREADS_PER_NODE \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=$GPUS_PER_SIMULATION \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://riskdiffuser.config.scenario_filter, pkg://riskdiffuser.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"
