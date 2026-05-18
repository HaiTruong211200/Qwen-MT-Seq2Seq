import csv
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
import numpy as np
import copy
import qwen.process_data.collator as collator
import qwen.utils.utils as utils
from qwen.process_data.process_data import (
    load_mmt_dataset, 
    process_mmt_data_for_seq2seq, 
    load_data_pretrain, 
    process_pretrain_data_for_seq2seq,
    process_mmt_data_for_seq2seq_ver2,
)
import re
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    GenerationConfig,
    NllbTokenizer
)
from transformers.trainer_utils import get_last_checkpoint

# from qwen.models.enc_dec import QwenCrossAttentionEncDec, QwenCrossAttentionEncDecNLLB, print_train_module
from qwen.models.llm_seq2seq import QwenForSeq2SeqConfig, QwenModelForSeq2Seq, print_train_module
from qwen.config.args import DataTrainingArguments, ModelArguments

from qwen.utils.check_weight import check_weight
from qwen.utils.initialize_model_weight import manual_fix_connector_weights

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info(f"Data arguments {data_args}")

    pairs = set(data_args.language_pairs.split(","))
    languages = set(data_args.languages.split(","))
    trans_task = data_args.trans_task.split(",")
    logger.info(f"Training lanauage pairs: {pairs}\nTraining translation task: {trans_task} Training languages: {languages}")

    ot_lambda = model_args.ot_lambda
    contrastive_lambda = model_args.contrastive_lambda
    logger.info(f"OT loss lambda: {ot_lambda}, contrastive loss lambda: {contrastive_lambda}")

    base_config = Qwen2Config.from_pretrained(model_args.llm_path, trust_remote_code=model_args.trust_remote_code)
    config = QwenForSeq2SeqConfig(
        **base_config.to_dict(),
        mt_model_path=model_args.mt_model_path,
        llm_path=model_args.llm_path,
        num_connector_layers=model_args.num_connector_layers,
        connector_hidden_size=model_args.connector_hidden_size,
        connector_intermediate_size=model_args.connector_intermediate_size,
        connector_num_attention_heads=model_args.connector_num_attention_heads,
        connector_num_key_value_heads=model_args.connector_num_key_value_heads,
        connector_model_method=model_args.connector_model_method,
        fuse_model_group_size=model_args.fuse_model_group_size
    )

    # add_eos_token = True if model_args.model_method == "default" else False
    llm_tokenizer = utils.load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=True)
    seq2seq_tokenizer = NllbTokenizer.from_pretrained(model_args.mt_model_path, trust_remote_code=model_args.trust_remote_code)

    
    if model_args.model_method == "SailorED":

        if training_args.report_to == "wandb":
            wandb.init(
                project="Low-Resource-Machine-Translation",
                name=(
                    "SailorED-pretrain"
                    if model_args.run_mode == "init"
                    else "SailorED-sft"
                )
            )

        model = QwenModelForSeq2Seq(config=config)

        # # =========================================================
        # # CONFIG
        # # =========================================================

        # decoder_config.model_name_or_path = model_args.decoder_model_name_or_path
        # config.decoder = decoder_config

        # adapter_config = copy.deepcopy(config.to_dict())

        # adapter_config["num_hidden_layers"] = model_args.decoder_layer_num
        # adapter_config["num_encoder_layers"] = config.num_hidden_layers
        # adapter_config["decoder_param_method"] = model_args.decoder_param_method
        # adapter_config["model_method"] = model_args.model_method
        # adapter_config["hidden_size"] = model_args.decoder_hidden_size
        # adapter_config["intermediate_size"] = model_args.decoder_intermediate_size
        # adapter_config["num_attention_heads"] = model_args.decoder_num_attention_heads
        # adapter_config["num_key_value_heads"] = model_args.decoder_num_key_value_heads
        # adapter_config["layer_types"] = ["full_attention"] * model_args.decoder_layer_num

        # adapter_config = Qwen2Config(**adapter_config)

        # config.adapter = adapter_config

        # config.use_cache = False
        # config.is_encoder_decoder = True
        # config.decoder_start_token_id = config.bos_token_id

        # config.encoder_method = model_args.encoder_method
        # config.encoder_layer_num = model_args.encoder_layer_num

        # if model_args.run_mode != "init":
        #     config.contrastive_lambda = model_args.contrastive_lambda
        #     config.contrastive_temperature = model_args.contrastive_temperature

        # # =========================================================
        # # LOAD MODEL
        # # =========================================================

        # if model_args.run_mode == "init":

        #     state_dict = utils.make_model_state_dict(
        #         model_path=model_args.model_name_or_path,
        #         seq2seq_model_name_or_path=model_args.decoder_model_name_or_path
        #     )

        #     model = QwenCrossAttentionEncDecNLLB.from_pretrained(
        #         None,
        #         config=config,
        #         state_dict=state_dict,
        #         ignore_mismatched_sizes=True
        #     )

        # else:

        #     state_dict = utils.load_checkpoint(
        #         model_args.model_name_or_path
        #     )

        #     model = QwenCrossAttentionEncDecNLLB.from_pretrained(
        #         None,
        #         config=config,
        #         state_dict=state_dict
        #     )

        # =========================================================
        # FREEZE
        # =========================================================

        model.freeze_model(
            freeze_llm=model_args.freeze_llm,
            freeze_decoder=model_args.freeze_decoder,
            freeze_decoder_cross_attn=model_args.freeze_decoder_cross_attn,
            freeze_mt_lm_head=model_args.freeze_mt_lm_head
        )

        # =========================================================
        # LORA
        # =========================================================

        target_modules = []

        if model_args.train_lora_llm:
            target_modules.append(
                r"encoder\.layers\..*"
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
            )

        if model_args.train_lora_mt:
            target_modules.append(
                r"mt_model\.model\.decoder\.layers\.\d+\.self_attn\."
                r"(q_proj|k_proj|v_proj|out_proj)$"
            )

        if len(target_modules) > 0:

            if model_args.freeze_decoder_cross_attn:
                modules_to_save = ["connector", "fuse_model", "encoder.embed_tokens"]
            else:
                modules_to_save = ["connector", "fuse_model", "encoder.embed_tokens", "encoder_attn"]

            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules="|".join(target_modules),
                modules_to_save=modules_to_save,
                task_type=TaskType.SEQ_2_SEQ_LM,
            )

            model = get_peft_model(model, lora_config)

            for name, param in model.mt_model.get_encoder().named_parameters():
                if "embed_tokens" in name:
                    param.requires_grad = False

        # =========================================================
        # PRINT TRAINABLE PARAMS
        # =========================================================

        print_train_module(model)

    else:
        print("Not implement this model yet!")
        exit()
    
    # model.tie_weights()

    model = utils.set_model_special_tokens(model, model_args.model_name_or_path)
    print(model.generation_config)


    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    ## Preprocessing data
    # Tokenize dataset
    if data_args.mmt_data_path is not None:
        train_raw_data, valid_raw_data, test_raw_data = load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger)
        # print(train_raw_data.keys())
        train_datasets, eval_datasets, test_datasets = process_mmt_data_for_seq2seq_ver2(train_raw_data, valid_raw_data, test_raw_data, pairs, llm_tokenizer, seq2seq_tokenizer, data_args, training_args)
        # print(">>> DEBUG: KIỂM TRA MẪU DATASET SAU KHI PROCESS")
        
        # try:
        #     # Lấy 1 mẫu đầu tiên từ tập train
        #     sample = train_datasets[0]
            
        #     # 1. Kiểm tra các Keys (Quan trọng nhất)
        #     print(f"👉 Các keys có trong dataset: {list(sample.keys())}")
            
        #     # 2. Check xem có 'labels' không?
        #     if "labels" not in sample:
        #         print("❌ LỖI NGHIÊM TRỌNG: Không thấy cột 'labels'. Collator sẽ không tạo decoder_input_ids!")
        #         # Thử đoán xem nó đang tên là gì
        #         print(f"   (Có thể nó đang tên là 'target', 'translation' hoặc 'output'?)")
        #     else:
        #         print("✅ Đã tìm thấy cột 'labels'.")

        #     # 3. In thử nội dung
        #     print("-" * 30)
        #     print(f"Input IDs (len={len(sample['input_ids'])}): {sample['input_ids']} ...")
        #     print(f"Input Text:  {llm_tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
            
        #     if "labels" in sample:
        #         print("-" * 30)
        #         print(f"Labels IDs (len={len(sample['labels'])}): {sample['labels']} ...")
        #         print(f"Labels Text: {seq2seq_tokenizer.decode(sample['labels'], skip_special_tokens=True)}")
            
        # except Exception as e:
        #     print(f"❌ Không thể in mẫu dataset: {e}")
            
        # print("!"*40 + "\n")

    ## Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else llm_tokenizer.pad_token_id
    data_collator = collator.DataCollatorForQwenNLLB(
        llm_tokenizer,
        seq2seq_tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    optimizer = None

    # if model_args.run_mode == "init":
    #     manual_fix_connector_weights(model, target_dim=model.config.decoder.hidden_size)
    check_weight(model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        tokenizer=llm_tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.patience)],
        optimizers=(optimizer, None)
    )

    logger.info(model)
    if training_args.do_train:
        utils.print_trainable_parameters(model)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=None)
        model = model.to(torch.bfloat16)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_datasets)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_datasets))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if training_args.report_to == "wandb":
        wandb.finish()

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    # model.to("cuda:0")

    # Generate predictions for test set
    predict_tasks = data_args.predict_task.split(",")
    if training_args.do_predict:
        lg_pairs = sorted(test_datasets.keys())
        for lg_pair in lg_pairs:
            cur_test_dataset = test_datasets[lg_pair]
            src_lang, tgt_lang = lg_pair.split("-")
            for task in cur_test_dataset.keys():
                print(f"\n\n{'='*30}\nPredicting for language pair: {lg_pair}, task: {task}\n{'='*30}")
                if task not in predict_tasks:
                    logger.info(f"skip predict {lg_pair}.{task}")
                    continue
                task_test_dataset = cur_test_dataset[task]
                start = time.time()
                logger.info(f"*** Prediction for {lg_pair}.{task} ***")

                batch_size = training_args.per_device_eval_batch_size
                all_inputs = []
                all_predictions = []
                dataloader = DataLoader(
                    task_test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=data_collator   
                )

                for batch in tqdm(dataloader, desc="Generating"):
                    inputs = {
                        k: v.to(model.device)
                        for k, v in batch.items()
                        if k != "labels"
                    }


                    with torch.no_grad():
                        generated_ids = model(**inputs)
                        input_ids, decoder_generate_ids_list = generated_ids

                        # ✅ giữ tensor, không phá structure
                        all_inputs.extend(input_ids.detach().cpu())
                        all_predictions.extend(decoder_generate_ids_list.detach().cpu())


                end = time.time()
                logger.info(f"Prediction completed in {end - start:.2f} seconds.")

                # all_inputs = np.where(all_inputs != -100,all_inputs,llm_tokenizer.pad_token_id)
                # all_predictions = np.where(all_predictions != -100,all_predictions,seq2seq_tokenizer.pad_token_id)
                all_inputs = [x.masked_fill(x == -100, llm_tokenizer.pad_token_id) for x in all_inputs]

                decoded_inputs = [
                    llm_tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "")
                    for x in all_inputs
                ]

                decoded_predictions = [
                    seq2seq_tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace("\n", "")
                    for x in all_predictions
                ]
                # all_inputs = torch.cat(all_inputs, dim=0)
                # all_predictions = torch.cat(all_predictions, dim=0)

                # all_inputs = np.where(all_inputs != -100,all_inputs,llm_tokenizer.pad_token_id)
                # decoded_inputs = llm_tokenizer.batch_decode(all_inputs,skip_special_tokens=True,clean_up_tokenization_spaces=True)
                # decoded_inputs = [text.replace("\n", "") for text in decoded_inputs]


                # # decode predictions bằng seq2seq tokenizer
                # predictions = np.where(all_predictions != -100, all_predictions, seq2seq_tokenizer.pad_token_id)
                # decoded_predictions = seq2seq_tokenizer.batch_decode(
                #     predictions,
                #     skip_special_tokens=True,
                #     clean_up_tokenization_spaces=True
                # )
                # decoded_predictions = [pred.replace("\n", "") for pred in decoded_predictions]

                # all_inputs = torch.where(
                #     all_inputs != -100,
                #     all_inputs,
                #     torch.full_like(all_inputs, llm_tokenizer.pad_token_id)
                # )

                # decoded_inputs = llm_tokenizer.batch_decode(
                #     all_inputs,
                #     skip_special_tokens=True,
                #     clean_up_tokenization_spaces=True
                # )

                # decoded_inputs = [x.replace("\n", "") for x in decoded_inputs]

                # ====== Seq2Seq predictions ======
                # all_predictions = torch.where(
                #     all_predictions != -100,
                #     all_predictions,
                #     torch.full_like(all_predictions, seq2seq_tokenizer.pad_token_id)
                # )

                # decoded_predictions = seq2seq_tokenizer.batch_decode(
                #     all_predictions,
                #     skip_special_tokens=True,
                #     clean_up_tokenization_spaces=True
                # )

                # decoded_predictions = [x.replace("\n", "") for x in decoded_predictions]
                # write prediction to csv
                decode_dir = os.path.join(training_args.output_dir, "decode_result")
                os.makedirs(decode_dir, exist_ok=True)
                predic_file_name = f"test-{src_lang}-{tgt_lang}-{task}"
                if task == "general_trans":
                    predic_file_name += f"-{data_args.test_dataname}"
                output_prediction_file = os.path.join(decode_dir, predic_file_name + ".csv")
                with open(output_prediction_file, "w", encoding="utf-8", newline="") as writer:
                    csv_writer = csv.writer(writer)
                    csv_writer.writerow(["input", "prediction"])
                    for src, pred in zip(decoded_inputs, decoded_predictions):
                        csv_writer.writerow([src, pred])

    

if __name__ == "__main__":
    main()