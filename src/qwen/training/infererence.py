import sys
from qwen.models.enc_dec import QwenCrossAttentionEncDec
from qwen.config.args import DataTrainingArguments, ModelArguments
from transformers import AutoTokenizer, AutoConfig

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
    GenerationConfig
)


from datasets import concatenate_datasets, load_dataset
from collections import defaultdict


def load_mmt_dataset(path, pairs, task_type, test_data_name):
  test_raw_data = defaultdict(dict)
  for pair in pairs:
    # print(pair)
    test_datas = test_data_name.split(",")
    test_files = []
    for test_data in test_datas:
      test_files.append(os.path.join(path, pair, f"test.{pair}.{task_type}.{test_data}.json"))
    # print(test_files)
    for test_data, test_file in zip(test_datas,test_files):
      test_raw_data[pair][test_data] = load_dataset("json", data_files={"test": test_file})
  
  return test_raw_data


def prepare_data(test_raw_data, pairs, tokenizer, test_data_name):
    test_datasets = {}
    lang = {"vi": "Vietnamese", "km": "Khmer", "lo": "Laos"}

    def format_test_chat_template(row):
        row_json = [
            {
                "role": "user",
                "content": f"{translation_prompt}{row['translation'][src_lang]}"
            }
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False, add_generation_prompt=True)
        return row


    for pair in pairs:
        src_lang = pair.split("-")[0]
        tgt_lang = pair.split("-")[1]
        translation_prompt = f"Translate this sentence from {lang[src_lang]} to {lang[tgt_lang]}: "

        test_datasets[pair] = {}  

        for test in test_data_name.split(","):
            test_data = test_raw_data[pair][test]["test"].map(  
                format_test_chat_template,
                num_proc=4,
            )
            test_datasets[pair][test] = test_data


    return test_datasets

import torch
import pandas as pd
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def generate_all_tests(
    model,
    tokenizer,
    test_datasets,
    batch_size=8,
    max_new_tokens=512,
    num_beams=4,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return inputs, texts, batch

    for pair in test_datasets:
        src_lang, tgt_lang = pair.split("-")

        for test_name in test_datasets[pair]:
            print(f"\n🚀 Running {pair} - {test_name}")

            dataset = test_datasets[pair][test_name]
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

            all_src, all_preds, all_refs = [], [], []

            # 🔥 progress bar theo batch
            for inputs, texts, raw_batch in tqdm(
                dataloader,
                desc=f"{pair}-{test_name}",
                total=len(dataloader)
            ):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        # num_beams=num_beams,
                        # do_sample=False,
                    )
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # ✔️ cắt output chuẩn hơn
                results = []
                for text, prompt in zip(decoded_outputs, texts):
                    # print(text)
                    # Tách chuỗi dựa trên keyword của chat template
                    parts = text.split("assistant\n")
                    if len(parts) > 1:
                        results.append(parts[-1].strip())
                    else:
                        # Fallback: xóa prompt thủ công nếu format lạ
                        # (Lưu ý: prompt trong decoded đã mất special token nên replace có thể không chính xác tuyệt đối,
                        # nhưng với sailor/qwen thường split assistant là chuẩn nhất)
                        results.append(text.replace(prompt, "").strip())

                for i, item in enumerate(raw_batch):
                    # print(results[i])
                    all_preds.append(results[i])
                    all_src.append(item["translation"][src_lang])
                    all_refs.append(item["translation"][tgt_lang])

            df = pd.DataFrame({
                "src": all_src,
                "prediction": all_preds,
                "reference": all_refs,
                "pair": pair,
                "test_set": test_name
            })

            save_path = os.path.join(output_dir, f"{pair}_{test_name}.csv")
            df.to_csv(save_path, index=False, encoding="utf-8")

            print(f"✅ Saved: {save_path}")

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = QwenCrossAttentionEncDec.from_pretrained(model_args.model_name_or_path, config=config)
    test_raw_data = load_mmt_dataset(data_args.data_path, data_args.pairs, data_args.task_type, data_args.test_data_name)
    test_datasets = prepare_data(test_raw_data, data_args.pairs, tokenizer, data_args.test_data_name)
    generate_all_tests(
        model=model,
        tokenizer=tokenizer,
        test_datasets=test_datasets,
        batch_size=training_args.per_device_eval_batch_size,
        max_new_tokens=data_args.max_new_tokens,
        num_beams=data_args.num_beams,
        output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    main()

