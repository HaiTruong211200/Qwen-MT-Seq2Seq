import copy
import os
from typing import Optional
import torch
from datasets import load_dataset, concatenate_datasets
import glob
import regex
import random
random.seed(42)
from collections import defaultdict

# from transformers.deepspeed import is_deepspeed_zero3_enabled
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback

LANG_TABLE = {
    "en": "English",
    "id": "Indonesian",
    "ja": "Japanese",
    "km": "Central Khmer",
    "lo": "Laos",
    "ms": "Malay",
    "my": "Burmese",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

TOKENIZER_LANG_TABLE = {
    "vi" : "vie_Latn",
    "km" : "khm_Khmr",
    "lo" : "lao_Laoo",
    "en" : "eng_Latn",
    "th" : "tha_Thai",
    "my" : "mya_Mymr",
    "zh" : "zho_Hans",
}

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control


def process_mmt_data_for_seq2seq_ver2(train_raw_data, valid_raw_data, test_raw_data, pairs, llm_tokenizer, seq2seq_tokenizer, data_args, training_args):

    def tokenize_train_eval_for_seq2seq(examples):
        inputs, targets, tgt_langs = [], [], []

        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:
            src_lang, tgt_lang = example["src_lang"], example["tgt_lang"]

            if f"{src_lang}-{tgt_lang}" in pairs:
                prompt, tgt_txt = get_prompt(src_lang, tgt_lang, example)
                inputs.append(prompt)
                targets.append(tgt_txt)
                tgt_langs.append(tgt_lang)

            # if do_data_reverse(pairs, example):
            #     prompt, tgt_txt = get_prompt(tgt_lang, src_lang, example)
            #     inputs.append(prompt)
            #     targets.append(tgt_txt)
            #     tgt_langs.append(src_lang)
        for target in targets:
            if target == None:
                print("Detect None target, replace it with empty string!")
        model_inputs = llm_tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs = dict(model_inputs)

        labels = seq2seq_tokenizer(
            targets,
            max_length=data_args.max_target_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        labels = dict(labels)

        new_labels, new_masks = [], []

        for tgt_lang, ids, mask in zip(
            tgt_langs, labels["input_ids"], labels["attention_mask"]
        ):
            bos_id = seq2seq_tokenizer.convert_tokens_to_ids(
                TOKENIZER_LANG_TABLE[tgt_lang]
            )

            if ids[0] == bos_id:
                ids[0] = bos_id
                mask = [1] + mask

            new_labels.append(ids)
            new_masks.append(mask)

        # mask padding -> -100
        final_labels = []
        for ids, mask in zip(new_labels, new_masks):
            final_labels.append([
                token if m == 1 else -100
                for token, m in zip(ids, mask)
            ])

        model_inputs["labels"] = new_labels

        return model_inputs

    def tokenize_test_for_seq2seq(examples):
        prompts, targets, tgt_langs = [], [], []

        # HF batch -> list[dict]
        examples = [
            {key: value for key, value in zip(examples.keys(), values)}
            for values in zip(*examples.values())
        ]

        for example in examples:
            src_lang, tgt_lang = example["src_lang"], example["tgt_lang"]

            if f"{src_lang}-{tgt_lang}" in pairs:
                prompt, tgt_txt = get_prompt(src_lang, tgt_lang, example)
                prompts.append(prompt)
                targets.append(tgt_txt)
                tgt_langs.append(tgt_lang)

        # =========================
        # 1. TOKENIZE INPUT (LLM)
        # =========================
        model_inputs = llm_tokenizer(
            prompts,
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        model_inputs = dict(model_inputs)

        # =========================
        # 2. TOKENIZE TARGET (for evaluation only)
        # =========================
        # labels = seq2seq_tokenizer(
        #     text_target=targets,
        #     max_length=data_args.max_target_length,
        #     padding=False,
        #     truncation=True,
        # )

        # check_add_eos(labels, seq2seq_tokenizer)

        # # =========================
        # # 3. FORCE TARGET LANGUAGE (QUAN TRỌNG)
        # # =========================
        forced_bos_token_ids = []
        for lang in tgt_langs:
            bos_token_id = seq2seq_tokenizer.convert_tokens_to_ids(TOKENIZER_LANG_TABLE[lang])
            forced_bos_token_ids.append([bos_token_id])
        # forced_bos_token_ids = [
        #     seq2seq_tokenizer.convert_tokens_to_ids(TOKENIZER_LANG_TABLE[lang])
        #     for lang in tgt_langs
        # ]

        # model_inputs["labels"] = labels["input_ids"]  # dùng để tính metric
        model_inputs["decoder_input_ids"] = forced_bos_token_ids
        # print(f"Forced BOS token ids for evaluation: {forced_bos_token_ids}")
        # print(f"model_inputs[\"forced_bos_token_id\"]: {model_inputs['forced_bos_token_id']}")

        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None
    
    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                processed_datasets.append(train_dataset)   
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets