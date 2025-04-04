# /bin/bash

ROOT_DIR="/mnt/f/MIMIC"

EXP_NAME="exp8_26jun_img224_ps16_7folders"
PRED_NAME="test_16"


python eval_only.py \
    --target_path="$ROOT_DIR/checkpoints/$EXP_NAME/predictions/$PRED_NAME/target/labeled_reports.csv" \
    --prediction_path="$ROOT_DIR/checkpoints/$EXP_NAME/predictions/$PRED_NAME/prediction/labeled_reports.csv" \
    --output_path="$ROOT_DIR/checkpoints/$EXP_NAME/predictions/$PRED_NAME/nlg_scores.json"
