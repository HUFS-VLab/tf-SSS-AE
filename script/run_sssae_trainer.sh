#!/bin/bash

MAIN_DIR=${0%/*}
cd $MAIN_DIR/..
MAIN_DIR=${0%/*}

TARGET_CODE=run.py
DATASET_NAME=LSMD
MANIFEST_PATH=manifests/${DATASET_NAME}

TRAIN_MANIFESTS="${MANIFEST_PATH}/GT-4118.json" 
#TRAIN_MANIFESTS="${MANIFEST_PATH}/ST-3214.json" 
#TRAIN_MANIFESTS="${MANIFEST_PATH}/ST-3708.json" 
    
MODEL_PATH=model

# HYPER PARAMETERS
SEQ_LEN=16
N_MELS=80
#THRES_WEIGHT=2.0

#RNN
N_LAYERS=4
EPOCHS=2500

DATASET_PATH=seqlen_${SEQ_LEN}_mels_${N_MELS}

python -u $TARGET_CODE \
--train-manifests $TRAIN_MANIFESTS \
--dataset-name $DATASET_NAME \
--dataset-path $DATASET_PATH \
--seq_len $SEQ_LEN \
--n-mels $N_MELS \
--n-layers $N_LAYERS \
--epochs $EPOCHS \
--model-path $MODEL_PATH \
--model-name "SSS-AE-Baseline"
