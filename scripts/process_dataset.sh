#!/bin/bash

WORK_MODE=0
[[ $# -eq 1 ]] && WORK_MODE=$1

DATA_DIR="/data/ModelTrainData/PoseData"
INPUT_FILE="datas/golf_annotations/task01/person_keypoints_default.json"
OUTPUT_FILE="datas/golf_annotations/task01/person_keypoints_result.json"

if [ $WORK_MODE == 0 ]; then
python scripts/process_dataset.py --work-mode 0 \
    --input-path $INPUT_FILE
fi

if [ $WORK_MODE == 1 ]; then
python scripts/process_dataset.py --work-mode 1 \
    --input-path $INPUT_FILE \
    --output-path $OUTPUT_FILE
fi


