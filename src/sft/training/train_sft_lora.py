import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer,SFTConfig

from src.graph_rag.utils.config_loader import load_config


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    config = load_config()
    sft_cfg = config["sft_paths"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Load tokenizer & model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # -------------------------
    # LoRA configuration
    # -------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Load datasets
    # -------------------------
    dataset = load_dataset(
        "json",
        data_files={
            "train": sft_cfg["processed_sft_path"] + "/train.jsonl",
            "validation": sft_cfg["processed_sft_path"] + "/val.jsonl",
        }
    )


    sft_config = SFTConfig(
        output_dir="checkpoints/sft_lora",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir="logs/sft",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=2048,  
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )


    # -------------------------
    # Train
    # -------------------------
    trainer.train()

    # -------------------------
    # Save final adapter
    # -------------------------
    trainer.model.save_pretrained("models/sft_lora/final")
    tokenizer.save_pretrained("models/sft_lora/final")

    print("✅ LoRA SFT training complete")


if __name__ == "__main__":
    main()
