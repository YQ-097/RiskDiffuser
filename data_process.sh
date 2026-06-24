###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="REPLACE_WITH_NUPLAN_DATA_DIR" # e.g., /path/to/nuplan-v1.1/trainval
NUPLAN_MAP_PATH="REPLACE_WITH_NUPLAN_MAPS_DIR" # e.g., /path/to/nuplan-v1.1/maps

TRAIN_SET_PATH="REPLACE_WITH_PREPROCESSED_TRAIN_SET_DIR" # output directory for processed training data
###################################

require_user_path() {
    local name="$1"
    local value="$2"
    if [[ "$value" == REPLACE_WITH_* ]]; then
        echo "Error: please edit $name in $0 and set it to your local path."
        exit 1
    fi
}

require_user_path "NUPLAN_DATA_PATH" "$NUPLAN_DATA_PATH"
require_user_path "NUPLAN_MAP_PATH" "$NUPLAN_MAP_PATH"
require_user_path "TRAIN_SET_PATH" "$TRAIN_SET_PATH"

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \
