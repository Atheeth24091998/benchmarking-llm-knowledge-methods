# src/rlkgf/trainer.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from .reward_model import KGRewardModel
from src.sft.evaluation.metrics import parse_answer  # Reuse your parser

class RLKGFTrainer:
    """
    Train LLM with RLKGF: KG-based rewards + Simple Policy Gradient (REINFORCE).
    Simplified without TRL dependency issues.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load KG reward model
        kg_path = Path(config["rlkgf"]["kg_path"])
        self.reward_model = KGRewardModel(kg_path, alpha=0.3)
        
        # Load tokenizer
        model_name = config["rlkgf"]["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        
        # Add LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.model = get_peft_model(base_model, lora_config)
        
        # Optimizer - convert learning_rate to float
        learning_rate = float(config["rlkgf"]["training"].get("learning_rate", 5e-6))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        print(f"✅ Model loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def format_prompt(self, sample: Dict) -> str:
        """Format training sample as prompt."""
        symptoms = sample["explicit_symptoms"] + sample["implicit_symptoms"]
        symptom_text = "\n".join([f"- {s.replace('_', ' ')}" for s in symptoms])
        
        prompt = f"""The machine shows the following symptoms:
{symptom_text}

Provide a structured troubleshooting response with:
Problem:
- [describe the main problem]

Possible Causes:
- [list causes]

Recommended Actions:
- [list actions]

Answer:"""
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
    
    def compute_loss_with_reward(self, prompt: str, response: str, reward: float):
        """
        Compute policy gradient loss (REINFORCE).
        Loss = -log_prob(response) * reward
        """
        
        # Tokenize full sequence
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Get prompt length
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs.input_ids)
        logits = outputs.logits
        
        # Get log probs for response tokens only
        response_logits = logits[0, prompt_len-1:-1, :]  # Shift by 1
        response_ids = inputs.input_ids[0, prompt_len:]
        
        # Compute log probs
        log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs[range(len(response_ids)), response_ids]
        
        # REINFORCE loss: -log_prob * reward (maximize reward)
        # Add baseline to reduce variance
        advantage = reward - 0.5  # Simple baseline
        loss = -selected_log_probs.mean() * advantage
        
        return loss
    
    def train(self, train_data: List[Dict], epochs: int = 3):
        """Train with RLKGF using policy gradients."""
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            total_reward = 0.0
            total_loss = 0.0
            
            for i, sample in enumerate(tqdm(train_data, desc=f"Epoch {epoch+1}")):
                try:
                    # Format prompt
                    prompt = self.format_prompt(sample)
                    
                    # Generate response
                    response = self.generate_response(prompt)
                    
                    # Parse response
                    parsed = parse_answer(response)
                    
                    # Compute KG reward
                    reward = self.reward_model.compute_reward(prompt, parsed)
                    
                    # Compute loss and backprop
                    if reward > 0.1:  # Only train on reasonably good responses
                        loss = self.compute_loss_with_reward(prompt, response, reward)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                    
                    total_reward += reward
                    
                    if (i + 1) % 10 == 0:
                        avg_reward = total_reward / (i + 1)
                        avg_loss = total_loss / max(1, i + 1)
                        print(f"\nSample {i+1}/{len(train_data)} | Reward: {reward:.4f} | Avg Reward: {avg_reward:.4f} | Loss: {avg_loss:.4f}")
                
                except Exception as e:
                    print(f"\n⚠️  Error on sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            avg_epoch_reward = total_reward / len(train_data)
            avg_epoch_loss = total_loss / len(train_data)
            print(f"\nEpoch {epoch+1} complete | Avg Reward: {avg_epoch_reward:.4f} | Avg Loss: {avg_epoch_loss:.4f}")
    
    def save(self, output_path: Path):
        """Save trained model."""
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✅ Model saved to {output_path}")

# Train
if __name__ == "__main__":
    from src.graph_rag.utils.config_loader import load_config
    
    config = load_config()
    
    # Load training data
    with Path(config["sft_paths"]["train_raw_sft_data"]).open() as f:
        train_data = json.load(f)[:50]  # Start small
    
    # Train
    trainer = RLKGFTrainer(config)
    trainer.train(train_data, epochs=2)
    trainer.save(Path("models/rlkgf_lora/final"))
