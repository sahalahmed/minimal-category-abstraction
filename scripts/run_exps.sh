#!/bin/bash                      
#SBATCH -t 2:00:00          # walltime
#SBATCH -n 1                 # one CPU (hyperthreaded) cores
#SBATCH --mem=4g
#SBATCH --gres=gpu:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate cxg

DATASET='testexp'

EPOCHS=70
BATCH_SIZE=16
LR=0.001
EXP_DIR="data/test-experiment/"

# unused numbers = [7, 63, 92, 63, 28, 10, 62, 48, 62, 40, 19, 31]

declare -a pairs=(nv)
# declare -a seeds=(111 222 333 444 555 666 777 888 999 1709)

MODEL="bert-base-uncased"
MODELNAME=$MODEL

# pass in other NUM 1 and NUM 2s for different unused pairs (do this later!)
CHECKPOINT_PATH="checkpoints/${MODELNAME}/unused_pairs_1"

for pair in ${pairs[@]}
do

    # token id of the unused token NUM1 = [unused1], ... so on. BERT only has 994 unused tokens.
    NUM1=1
    NUM2=2

    for i in {1..1} # for now just run once, but change this to 5+ for proper experiments.
    do
        # echo "Pair: ${pair}. Numbers: ${NUM1}_${NUM2}. Seed: ${i}"
        CUDA_VISIBLE_DEVICES=0 python src/run_language_modeling.py \
            --tokens "[unused${NUM1}]" "[unused${NUM2}]" \
            --learning_rate ${LR} \
            --seed ${i} \
            --model_type bert \
            --model_name_or_path ${MODEL} \
            --do_train \
            --evaluate_during_training \
            --mlm \
            --line_by_line \
            --num_train_epochs ${EPOCHS} \
            --save_steps -1 \
            --logging_steps 1 \
            --per_gpu_eval_batch_size ${BATCH_SIZE} \
            --overwrite_output_dir \
            --output_dir=${CHECKPOINT_PATH}/${DATASET}_${pair}_unused_token_numbers_${NUM1}_${NUM2}_learning_rate_${LR}_seed_${i} \
            --train_data_file=${EXP_DIR}/${pair}_f/${DATASET}_finetune.txt \
            --eval_prefix=${EXP_DIR}/${pair}_f/${DATASET} \

        # rm -rf ${CHECKPOINT_PATH}/mnli_${pair}_unused_token_numbers_${NUM1}_${NUM2}_learning_rate_${LR}_seed_${i}/checkpoint*
    done
done

conda deactivate