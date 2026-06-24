PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEST_SET_PATH="REPLACE_WITH_PREPROCESSED_TEST_SET_DIR"
TEST_SET_LIST_PATH="REPLACE_WITH_TEST_SET_LIST_JSON"
OUTPUT_DIR="$PROJECT_ROOT/risk_test_outputs"

require_user_path() {
    local name="$1"
    local value="$2"
    if [[ "$value" == REPLACE_WITH_* ]]; then
        echo "Error: please edit $name in $0 and set it to your local path."
        exit 1
    fi
}

require_user_path "TEST_SET_PATH" "$TEST_SET_PATH"
require_user_path "TEST_SET_LIST_PATH" "$TEST_SET_LIST_PATH"

########test14-hard#########
python test_risk_net.py \
  --test_set "$TEST_SET_PATH" \
  --test_set_list "$TEST_SET_LIST_PATH" \
  --risk_ckpt "$PROJECT_ROOT/checkpoints/riskcheckpoint/" \
  --batch_size 64 \
  --device cuda \
  --output_dir "$OUTPUT_DIR"


#########train#########
# python test_risk_net.py \
#   --test_set "$TEST_SET_PATH" \
#   --test_set_list "$TEST_SET_LIST_PATH" \
#   --risk_ckpt "$PROJECT_ROOT/checkpoints/riskcheckpoint/" \
#   --batch_size 64 \
#   --device cuda \
#   --output_dir "$PROJECT_ROOT/risk_test_outputs_train"

########val14#########
python test_risk_net.py \
  # --test_set "$TEST_SET_PATH" \
  # --test_set_list "$TEST_SET_LIST_PATH" \
  # --risk_ckpt "$PROJECT_ROOT/checkpoints/riskcheckpoint/" \
  # --batch_size 64 \
  # --device cuda \
  # --output_dir "$PROJECT_ROOT/risk_test_outputs_val"
