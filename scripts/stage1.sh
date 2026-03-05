#! /bin/bash
set -eux

ROOT_DIR=$(dirname $(dirname `readlink -f $0`))


# ====== ADD DÒNG NÀY ======
export PYTHONPATH=$ROOT_DIR/src

# =========================
export PYTHONUNBUFFERED=1
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"

config_file=$ROOT_DIR/configs/myconfig.yaml

## model
model_dir="sail/Sailor2-1B-Chat"
run_mode="init"

model_method="lamate"
encoder_method="stack"
encoder_layer_num=4
decoder_layer_num=4
decoder_hidden_size=512
decoder_intermediate_size=1376
decoder_num_attention_heads=8
decoder_num_key_value_heads=8
decoder_param_method="freeze"
tag=lamate_s1

## data
language_pairs=vi-km,vi-lo
mmt_data_path=$ROOT_DIR/data
trans_task="general_trans"
epoch=2
batch_size=2
gradient_accumulation=2

## save
output_dir=$ROOT_DIR/exps/$tag
mkdir -p $output_dir
cp $0 $output_dir

# ====== SỬA PATH SCRIPT ======
accelerate launch --config_file $config_file \
  $ROOT_DIR/src/qwen/training/training.py \
    --model_name_or_path $model_dir \
    --resume_from_checkpoint None \
    --encoder_layer_num ${encoder_layer_num} \
    --decoder_layer_num $decoder_layer_num \
    --decoder_hidden_size $decoder_hidden_size \
    --decoder_intermediate_size $decoder_intermediate_size \
    --decoder_num_attention_heads $decoder_num_attention_heads \
    --decoder_num_key_value_heads $decoder_num_key_value_heads \
    --encoder_method $encoder_method \
    --model_method ${model_method:-"norm"} \
    --run_mode ${run_mode:-""} \
    --decoder_param_method ${decoder_param_method:-"share"} \
    --mmt_data_path $mmt_data_path \
    --trans_task $trans_task \
    --test_dataname wmt23 \
    --language_pairs $language_pairs \
    --use_fast_tokenizer \
    --do_eval \
    --do_train \
    --do_predict \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end False \
    --cache_dir ./cache \
    --dataloader_num_workers 24 \
    --preprocessing_num_workers 16 \
    --max_source_length 512 \
    --max_target_length 512 \
    --output_dir  $output_dir \
    --num_train_epochs $epoch \
    --patience 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --predict_with_generate \
    --num_beams 5 \
    --max_new_tokens 256 \
    --eval_strategy steps \
    --save_strategy no \
    --logging_strategy steps \
    --eval_steps 2000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --bf16 True \
    --fp16 False \
    --seed 42 \
    --report_to "none" \
    --overwrite_output_dir True \
  | tee $output_dir/train.log
