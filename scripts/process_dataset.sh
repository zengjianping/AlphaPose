#!/bin/bash

WORK_MODE=0
[[ $# -eq 1 ]] && WORK_MODE=$1

DATA_DIR="/data/ModelTrainData/PoseData"3

TASK_DIR="datas/golf_annotations/ezgolf_task_20240418_a01/annotations"
DATA_PREFIX="PoseData/ezgolf/task_20240418/images/"

#TASK_DIR="datas/golf_annotations/golfdb_a01/annotations"
#DATA_PREFIX="PoseData/golfdb/images/"

INPUT_FILE="${TASK_DIR}/person_keypoints_default.json"
ALGRES_FILE="${TASK_DIR}/person_keypoints_result.json"
KEYPOINT_ANNOT_FILE="${TASK_DIR}/keypoint/person_keypoints_annot.json"
DETECTION_ANNOT_FILE="${TASK_DIR}/detection/person_detections_annot.json"
GOLFCLUB_ANNOT_FILE="${TASK_DIR}/golfclub/person_detections_annot.json"
HALPE28_ANNOT_FILE="${TASK_DIR}/halpe28/person_keypoints_annot.json"

if [ $WORK_MODE == 0 ]; then
python scripts/process_dataset.py --work-mode 0 \
    --input-path $INPUT_FILE
fi

if [ $WORK_MODE == 1 ]; then
python scripts/process_dataset.py --work-mode 1 \
    --input-path $INPUT_FILE \
    --output-path $ALGRES_FILE
fi

if [ $WORK_MODE == 2 ]; then
python scripts/process_dataset.py --work-mode 2 \
    --data-prefix $DATA_PREFIX \
    --input-path $KEYPOINT_ANNOT_FILE \
    --output-path $DETECTION_ANNOT_FILE
fi

if [ $WORK_MODE == 3 ]; then
python scripts/process_dataset.py --work-mode 3 \
    --data-prefix $DATA_PREFIX \
    --input-path $KEYPOINT_ANNOT_FILE \
    --output-path $GOLFCLUB_ANNOT_FILE
fi

if [ $WORK_MODE == 4 ]; then
python scripts/process_dataset.py --work-mode 4 \
    --data-prefix $DATA_PREFIX \
    --input-path $ALGRES_FILE \
    --output-path $HALPE28_ANNOT_FILE
fi

if [ $WORK_MODE == 5 ]; then
python scripts/process_dataset.py --work-mode 5 \
    --input-path $KEYPOINT_ANNOT_FILE \
    --data-prefix $DATA_PREFIX \
    --pad-ratio 0.0
python scripts/process_dataset.py --work-mode 5 \
    --input-path $DETECTION_ANNOT_FILE \
    --data-prefix $DATA_PREFIX \
    --pad-ratio 0.0
python scripts/process_dataset.py --work-mode 5 \
    --input-path $GOLFCLUB_ANNOT_FILE \
    --data-prefix $DATA_PREFIX \
    --pad-ratio 0.0
python scripts/process_dataset.py --work-mode 5 \
    --input-path $HALPE28_ANNOT_FILE \
    --data-prefix $DATA_PREFIX \
    --pad-ratio 0.0
fi


