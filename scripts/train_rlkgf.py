# scripts/train_rlkgf.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from src.rlkgf.kg_builder import IndustrialKGBuilder
from src.rlkgf.trainer import RLKGFTrainer
from src.graph_rag.utils.config_loader import load_config

def main():
    config = load_config()
    
    # Step 1: Build KG (if not exists)
    kg_path = Path("data/rlkgf/kg/graph.gpickle")
    if not kg_path.exists():
        print("🔨 Building Knowledge Graph...")
        builder = IndustrialKGBuilder()
        builder.build_from_json(Path(config["sft_paths"]["train_raw_sft_data"]))
        builder.compute_embeddings()
        builder.save(Path("data/rlkgf/kg"))
    else:
        print("✅ KG already exists")
    
    # Step 2: Load training data
    print("📂 Loading training data...")
    with Path(config["sft_paths"]["train_raw_sft_data"]).open() as f:
        #train_data = json.load(f)[:200]  # Start with 100 samples
        train_data = json.load(f)  # Start with 100 samples
    # Step 3: Train
    print("🚀 Starting RLKGF training...")
    trainer = RLKGFTrainer(config)
    trainer.train(train_data, epochs=3)
    trainer.save(Path("models/rlkgf_lora/final"))
    
    print("\n✅ RLKGF training complete!")

if __name__ == "__main__":
    main()
