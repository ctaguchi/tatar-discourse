#!/bin/bash
#$ -M ctaguchi@nd.edu
#$ -m abe
#$ -pe smp 8
#$ -N tatar-asr
#$ -l gpu_card=1
#$ -q gpu@@nlp-a10

# example
DATASET="mozilla-foundation/common_voice_17_0"
SRC_LANG="tt"
# TGT_LANG="fy-NL"
# MODEL="facebook/wav2vec2-large-xlsr-53"
MODEL="facebook/wav2vec2-xls-r-300m"
PROJECT="wav2vec2-xls-r-300m"

OUTPUT_MODEL_DIR="models/$PROJECT/$SRC_LANG"
OUTPUT="test_results"
# OUTPUT_POSTEDIT="test_results_postedit"

# LLM="gpt-4o-mini" # test

cd ../

# check the directory
if [ ! -d "$OUTPUT_MODEL_DIR" ]; then
  mkdir -p "$OUTPUT_MODEL_DIR"
fi

# finetune
poetry run python finetune.py \
    --model $MODEL \
    --dataset $DATASET \
    --lang $SRC_LANG \
    --num_train_samples all \
    --num_valid_samples 1000 \
    --num_test_samples 1000 \
    --output_dir $OUTPUT_MODEL_DIR \
    --num_train_epochs 10 \
    --eval_strategy steps \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 100 \
    --learning_rate 1e-4 \
    --warmup_steps 500 \
    --streaming \
    --wandb_project $PROJECT \
    --wandb_run_name "finetune-$SRC_LANG" \
    --full_precision

# transcribe
# poetry run python transcribe.py \
#     --model $OUTPUT_MODEL_DIR \
#     --dataset $DATASET \
#     --lang $TGT_LANG \
#     --split test \
#     --streaming \
#     --num_samples 100 \
#     --output "$OUTPUT_MODEL_DIR/$TGT_LANG-$OUTPUT.csv" \
#     --output_cer "$OUTPUT_MODEL_DIR/$TGT_LANG-$OUTPUT-cer.txt"

# # postedit
# poetry run python postedit.py \
#     --model $LLM \
#     --input_file "$OUTPUT_MODEL_DIR/$TGT_LANG-$OUTPUT.csv" \
#     --output_csv "$OUTPUT_MODEL_DIR/$TGT_LANG-$OUTPUT_POSTEDIT.csv" \
#     --output_cer "$OUTPUT_MODEL_DIR/$TGT_LANG-$OUTPUT_POSTEDIT-cer.txt" \
#     --src_lang $SRC_LANG \
#     --tgt_lang $TGT_LANG \
#     --shot zero
