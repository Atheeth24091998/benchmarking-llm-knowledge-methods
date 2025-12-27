import yaml
from pathlib import Path

def load_config():
    # Assumes config_loader.py is in src/rag/utils/
    config_path = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
    #print("Loading config from:", config_path)

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
