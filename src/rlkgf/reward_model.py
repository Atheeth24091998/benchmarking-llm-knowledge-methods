# src/rlkgf/reward_model.py

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import json
from typing import List, Dict
from pathlib import Path

class KGRewardModel:
    """
    Combines RWR (structural) + Semantic (embedding similarity) rewards.
    Simplified version without GCN for now.
    """
    
    def __init__(self, kg_path: Path, alpha: float = 0.3, restart_prob: float = 0.7):
        # Load graph using pickle
        with open(kg_path / "graph.gpickle", "rb") as f:
            self.G = pickle.load(f)
        
        self.alpha = alpha  # Weight for semantic score
        self.restart_prob = restart_prob
        
        # Load node embeddings
        self.embeddings = np.load(kg_path / "embeddings.npy")
        with (kg_path / "nodes.json").open() as f:
            self.nodes = json.load(f)["nodes"]
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
    
    def compute_rwr_score(self, question_entities: List[str], candidate_entities: List[str]) -> float:
        """
        Compute RWR-based reachability score.
        """
        
        # Simple reachability: avg shortest path length
        scores = []
        for q_ent in question_entities:
            if q_ent not in self.G:
                continue
            for c_ent in candidate_entities:
                if c_ent not in self.G:
                    continue
                try:
                    path_len = nx.shortest_path_length(self.G, q_ent, c_ent)
                    # Closer = better (inverse)
                    scores.append(1.0 / (1.0 + path_len))
                except nx.NetworkXNoPath:
                    scores.append(0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_semantic_score(self, question_entities: List[str], candidate_entities: List[str]) -> float:
        """
        Compute cosine similarity between question and candidate embeddings.
        """
        
        q_indices = [self.node_to_idx[e] for e in question_entities if e in self.node_to_idx]
        c_indices = [self.node_to_idx[e] for e in candidate_entities if e in self.node_to_idx]
        
        if not q_indices or not c_indices:
            return 0.0
        
        q_emb = self.embeddings[q_indices].mean(axis=0)
        c_emb = self.embeddings[c_indices].mean(axis=0)
        
        # Cosine similarity
        sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-10)
        return float(sim)
    
    def compute_reward(self, question: str, parsed_response: Dict) -> float:
        """
        Compute combined reward for LLM response.
        
        Args:
            question: Input question text
            parsed_response: {"problem": str, "causes": List[str], "actions": List[str]}
        
        Returns:
            reward: Float in [0, 1]
        """
        
        # Extract entities from question (simple: use problem)
        question_entities = [parsed_response.get("problem", "")]
        question_entities = [e for e in question_entities if e]  # Remove empty
        
        if not question_entities:
            return 0.0
        
        # Candidate entities: causes + actions
        candidates = parsed_response.get("causes", []) + parsed_response.get("actions", [])
        
        if not candidates:
            return 0.0
        
        # RWR score
        rwr_score = self.compute_rwr_score(question_entities, candidates)
        
        # Semantic score
        sem_score = self.compute_semantic_score(question_entities, candidates)
        
        # Combined
        reward = self.alpha * sem_score + (1 - self.alpha) * rwr_score
        
        return float(np.clip(reward, 0.0, 1.0))
