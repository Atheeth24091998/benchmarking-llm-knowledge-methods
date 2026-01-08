import pandas as pd
from pathlib import Path
from datetime import datetime
import json

class EvaluationHistory:
    def __init__(self, history_file="data/evaluation_history.csv"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        
        # Initialize if doesn't exist
        if not self.history_file.exists():
            self._create_empty_history()
    
    def _create_empty_history(self):
        """Create empty history CSV with headers"""
        df = pd.DataFrame(columns=[
            'timestamp', 'method', 'test_samples', 
            'semantic_similarity', 'faithfulness', 'hallucination',
            'cause_f1', 'action_f1', 'latency', 
            'rouge_l', 'bert_score'
        ])
        df.to_csv(self.history_file, index=False)
    
    def add_entry(self, method, test_samples, metrics):
        """Add new evaluation entry to history"""
        new_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': method,
            'test_samples': test_samples,
            'semantic_similarity': metrics.get('semantic_similarity', 0.0),
            'faithfulness': metrics.get('faithfulness', 0.0),
            'hallucination': metrics.get('hallucination', 0.0),
            'cause_f1': metrics.get('cause_f1', 0.0),
            'action_f1': metrics.get('action_f1', 0.0),
            'latency': metrics.get('latency', 0.0),
            'rouge_l': metrics.get('rouge_l', metrics.get('rouge', 0.0)),
            'bert_score': metrics.get('bert_score', metrics.get('bert', 0.0))
        }
        
        # Read existing history
        df = pd.read_csv(self.history_file)
        
        # Append new entry
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(self.history_file, index=False)
        
        return df
    
    def get_history(self):
        """Retrieve full history as DataFrame"""
        return pd.read_csv(self.history_file)
    
    def clear_history(self):
        """Clear all history"""
        self._create_empty_history()
