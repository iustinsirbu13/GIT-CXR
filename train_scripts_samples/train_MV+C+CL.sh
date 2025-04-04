#!/bin/bash
# GIT-CXR (MV+C) CL

ROOT_PATH="todo"
INPUT_DATA_PATH="${ROOT_PATH}/MIMIC/MIMIC_Dataset"
OUTPUT_DATA_PATH="${ROOT_PATH}/MIMIC/MIMIC_Checkpoints"

export TOKENIZERS_PARALLELISM=false

# for CL, all epoch params (epochs, patience, validation_frequency) are multiplied by 4 to account for the curriculum_learning_percent of 0.25
# --context_format --> remove or change to default to train without context/history
# --model_variation --> change to 'with_classification' to add the auxiliary classification loss
# --curriculum_learning --> remove or change to 'none' to train without CL; in this case, also change (epochs, patience, validation_frequency) to (30, 7, 1)

main.py \
    --mode train \
    --out ${OUTPUT_DATA_PATH}/git-base-msrvtt-qa_MVT_IndHistImpFind_sft_bs32_lr5_CLlin25 \
    --csv_path ${INPUT_DATA_PATH}/csvs_orig/cxr-record-list.csv \
    --splits_path ${INPUT_DATA_PATH}/csvs_jpg/mimic-cxr-2.0.0-split.csv \
    --findings_path ${INPUT_DATA_PATH}/csvs_orig/mimic_cxr_sectioned_full.csv \
    --metadata_path ${INPUT_DATA_PATH}/csvs_jpg/mimic-cxr-2.0.0-metadata.csv \
    --labels_path ${INPUT_DATA_PATH}/csvs_jpg/mimic-cxr-2.0.0-chexpert.csv \
    --images_path ${INPUT_DATA_PATH}/files_224 \
    --log_level INFO \
    --img_size 224 \
    --batch_size 32 \
    --num_workers 8 \
    --epochs 120 \
    --lr 0.00005 \
    --patience 28 \
    --context_format history+indication \
    --target_format impression+findings \
    --data_variation multi_view_temporal \
    --model_variation default \
    --use_pretrained yes \
    --model microsoft/git-base-msrvtt-qa \
    --curriculum_learning linear \
    --curriculum_learning_percent 0.25 \
    --validation_frequency 4
