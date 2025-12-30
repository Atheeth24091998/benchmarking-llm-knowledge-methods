import json
from pathlib import Path
from src.graph_rag.utils.config_loader import load_config


SYSTEM_PROMPT = (
    "You are a technical support assistant for industrial machinery. "
    "Your task is to diagnose machine problems and provide causes and "
    "recommended actions based only on given symptoms."
)


def build_prompt(sample: dict):
    symptoms = (
        sample.get("explicit_symptoms", [])
        + sample.get("implicit_symptoms", [])
    )

    user_prompt = (
        "The machine shows the following symptoms:\n"
        + "\n".join(f"- {s.replace('_', ' ')}" for s in symptoms)
        + "\n\nProvide a structured troubleshooting response."
    )

    assistant_answer = (
        "Problem:\n"
        f"- {sample['problem'].replace('_', ' ')}\n\n"
        "Possible Causes:\n"
        + "\n".join(f"- {c.replace('_', ' ')}" for c in sample["possible_causes"])
        + "\n\nRecommended Actions:\n"
        + "\n".join(f"- {a.replace('_', ' ')}" for a in sample["recommended_actions"])
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_answer},
        ]
    }


def process_split(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in data:
            record = build_prompt(sample)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(data)} samples → {output_path}")


def main():
    config = load_config()
    sft_cfg = config["sft_paths"]

    splits = {
        "train": sft_cfg["train_raw_sft_data"],
        "val": sft_cfg["val_raw_sft_data"],
        "test": sft_cfg["test_raw_sft_data"],
    }

    out_dir = Path(sft_cfg["processed_sft_path"])

    for split, in_path in splits.items():
        in_path = Path(in_path)
        out_path = out_dir / f"{split}.jsonl"
        process_split(in_path, out_path)


if __name__ == "__main__":
    main()
