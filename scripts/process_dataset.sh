#!/bin/bash

WORK_MODE=0
[[ $# -eq 1 ]] && WORK_MODE=$1

DATA_DIR="/data/ModelTrainData/PoseData"
TASK_DIR="datas/golf_annotations/ezgolf_task_20240418/annotations"
DATA_PREFIX="PoseData/ezgolf/task_20240418/images/"
#TASK_DIR="datas/golf_annotations/golfdb/annotations"
#DATA_PREFIX="PoseData/golfdb/images/"

INPUT_FILE="${TASK_DIR}/person_keypoints_default.json"
OUTPUT_FILE="${TASK_DIR}/person_keypoints_result.json"
ANNOT_FILE="${TASK_DIR}/person_keypoints_annot.json"

if [ $WORK_MODE == 0 ]; then
python scripts/process_dataset.py --work-mode 0 \
    --input-path $INPUT_FILE
fi

if [ $WORK_MODE == 1 ]; then
python scripts/process_dataset.py --work-mode 1 \
    --input-path $INPUT_FILE \
    --output-path $OUTPUT_FILE
fi

if [ $WORK_MODE == 2 ]; then
python scripts/process_dataset.py --work-mode 2 \
    --input-path $ANNOT_FILE \
    --data-prefix $DATA_PREFIX
fi


