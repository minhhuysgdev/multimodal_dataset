#!/bin/bash
GPUID=$1
DATASET=$2
NUM_LABELS=$3
LABEL_SOURCE=$4  # train_add_bt / train_all_bt

# model
TOKENIZER_NAME=roberta-base
STUDENT1_MODEL_NAME=roberta-base

# learning parameters
EPOCH=50
LR=3e-5

# hyper-parameters
TAU_LOSS=0.6
TAU_CONF=0.99

echo "Run on GPU $GPUID"
echo "Using source: $LABEL_SOURCE"
echo "Using dataset: $DATASET"
echo "Learning rate: $LR"
echo "Tau loss: $TAU_LOSS"
echo "Tau conf: $TAU_CONF"

# data dir
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 slm_distillation.py \
  --data_dir $PROJECT_ROOT/data/ \
  --output_dir $PROJECT_ROOT/slm_outputs/$LABEL_SOURCE-$DATASET/ \
  --cache_dir $PROJECT_ROOT/cached_models \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --student1_model_name_or_path $STUDENT1_MODEL_NAME \
  --tokenizer_name $TOKENIZER_NAME \
  --overwrite_output_dir --overwrite_cache \
  --task_name $DATASET --num_labels $NUM_LABELS --label_source $LABEL_SOURCE \
  --tau_loss $TAU_LOSS --tau_conf $TAU_CONF \