#!/bin/bash

MAIN_DIR=${0%/*}
cd $MAIN_DIR/..
MAIN_DIR=${0%/*}

TARGET_CODE=utils/preprocessing.py
DATASET_NAME=LSMD
DATASET_PATH=dataset/
MANIFEST_PATH=manifests/${DATASET_NAME}

# PARAMETERS
SEQ_LEN=16
N_MELS=80
N_FFT=2048

for TARGET_MANIFEST in ${MANIFEST_PATH}/*.json
do
    python -u $TARGET_CODE \
    --main-dir $MAIN_DIR \
    --dataset-name $DATASET_NAME \
    --dataset-path $DATASET_PATH \
    --target-manifest $TARGET_MANIFEST \
    --seq-len $SEQ_LEN \
    --n-mels $N_MELS \
    --n-fft $N_FFT
done
