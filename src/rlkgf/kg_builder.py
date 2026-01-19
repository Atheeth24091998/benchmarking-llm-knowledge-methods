# src/rlkgf/kg_builder.py

import json
import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class IndustrialKGBuilder:
    """
    Build Knowledge Graph from your train/val/test.json files.
    
    KG Schema:
    - Nodes: problems, symptoms, causes, actions
    - Edges: has_symptom, causes, solves
    """
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def build_from_json(self, json_path: Path):
        """Build KG from train.json"""
        
        with json_path.open() as f:
            data = json.load(f)
        
        print(f"📊 Processing {len(data)} samples...")
        
        for sample in data:
            problem = sample["problem"]
            
            # Add problem node
            self.G.add_node(problem, type="problem")
            
            # Add symptom nodes and edges
            all_symptoms = sample["explicit_symptoms"] + sample["implicit_symptoms"]
            for symptom in all_symptoms:
                if not self.G.has_node(symptom):
                    self.G.add_node(symptom, type="symptom")
                self.G.add_edge(symptom, problem, relation="indicates")
            
            # Add cause nodes and edges
            for cause in sample["possible_causes"]:
                if not self.G.has_node(cause):
                    self.G.add_node(cause, type="cause")
                self.G.add_edge(problem, cause, relation="caused_by")
            
            # Add action nodes and edges
            for action in sample["recommended_actions"]:
                if not self.G.has_node(action):
                    self.G.add_node(action, type="action")
                self.G.add_edge(problem, action, relation="solved_by")
        
        print(f"✅ KG built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G
    
    def compute_embeddings(self):
        """Compute embeddings for all nodes."""
        
        nodes = list(self.G.nodes())
        node_texts = [n.replace("_", " ") for n in nodes]
        
        print("🔢 Computing node embeddings...")
        embeddings = self.embedding_model.encode(node_texts, show_progress_bar=True)
        
        # Attach to nodes
        for i, node in enumerate(nodes):
            self.G.nodes[node]['embedding'] = embeddings[i]
        
        return embeddings
    
    def save(self, output_dir: Path):
        """Save KG."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use pickle directly (works in all NetworkX versions)
        with open(output_dir / "graph.gpickle", "wb") as f:
            pickle.dump(self.G, f, pickle.HIGHEST_PROTOCOL)
        
        # Save node embeddings separately
        nodes = list(self.G.nodes())
        embeddings = np.array([self.G.nodes[n]['embedding'] for n in nodes])
        np.save(output_dir / "embeddings.npy", embeddings)
        
        # Save node list
        with (output_dir / "nodes.json").open("w") as f:
            json.dump({"nodes": nodes}, f)
        
        print(f"✅ Saved KG to {output_dir}")

# Build KG
if __name__ == "__main__":
    builder = IndustrialKGBuilder()
    builder.build_from_json(Path("data/sft/raw/train.json"))
    builder.compute_embeddings()
    builder.save(Path("data/rlkgf/kg"))
