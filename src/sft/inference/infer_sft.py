import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.graph_rag.utils.config_loader import load_config


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "models/sft_lora/final"


def build_prompt(sample):
    symptoms = sample["explicit_symptoms"] + sample["implicit_symptoms"]

    symptom_text = "\n".join([f"- {s.replace('_', ' ')}" for s in symptoms])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical support assistant for industrial machinery. "
                "Your task is to diagnose machine problems and provide causes and "
                "recommended actions based only on given symptoms."
            )
        },
        {
            "role": "user",
            "content": (
                f"The machine shows the following symptoms:\n"
                f"{symptom_text}\n\n"
                "Provide a structured troubleshooting response."
            )
        }
    ]
    return messages


def main():
    config = load_config()
    test_path = Path(config["sft_paths"]["test_raw_sft_data"])
    output_path = Path("logs/sft_predictions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Load tokenizer
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Load base model + LoRA
    # -------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # -------------------------
    # Load test data
    # -------------------------
    test_data = json.loads(test_path.read_text())

    with output_path.open("w") as f:
        for sample in test_data:
            messages = build_prompt(sample)

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # keep only assistant answer
            if "assistant" in decoded:
                decoded = decoded.split("assistant")[-1].strip()

            record = {
                "prediction": decoded,
                "ground_truth": sample
            }

            f.write(json.dumps(record) + "\n")

    print(f"✅ SFT inference complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
