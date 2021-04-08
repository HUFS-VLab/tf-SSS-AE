#!/bin/bash

MAIN_DIR=${0%/*}
cd $MAIN_DIR/..
MAIN_DIR=${0%/*}

TARGET_CODE=run.py
DATASET_NAME=LSMD
MANIFEST_PATH=manifests/${DATASET_NAME}
MODEL_PATH=model
LOG_PATH=log

if [ ! -d $LOG_PATH ]; then
    mkdir $LOG_PATH
fi

TRAIN_MANIFESTS="${MANIFEST_PATH}/GT-4118.json"
#TRAIN_MANIFESTS="${MANIFEST_PATH}/ST-3214.json"
#TRAIN_MANIFESTS="${MANIFEST_PATH}/ST-3708.json"

TEST_MANIFESTS=$(echo ${MANIFEST_PATH}/*)

for TRAIN_MANIFEST in ${TRAIN_MANIFESTS}
do
    TEST_MANIFESTS=$(echo ${TEST_MANIFESTS//${TRAIN_MANIFEST}})
done

# HYPER PARAMETERS
SEQ_LEN=16
N_MELS=80
THRES_WEIGHT=2.0

#RNN
N_LAYERS=4
EPOCHS=2500

DATASET_PATH=seqlen_${SEQ_LEN}_mels_${N_MELS}

python -u $TARGET_CODE \
--train-manifests $TRAIN_MANIFESTS \
--test-manifests $TEST_MANIFESTS \
--dataset-name $DATASET_NAME \
--dataset-path $DATASET_PATH \
--seq_len $SEQ_LEN \
--n-mels $N_MELS \
--n-layers $N_LAYERS \
--epochs $EPOCHS \
--threshold-weight $THRES_WEIGHT \
--model-path $MODEL_PATH \
--model-name "SSS-AE-Baseline" \
--test # | tee $LOG_FILE
