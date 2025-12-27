import json
from pathlib import Path

from src.graph_rag.graph_store.neo4j_store import Neo4jStore
from src.graph_rag.utils.config_loader import load_config
from src.graph_rag.utils.logger import get_logger

logger = get_logger(__name__)

def build_graph():
    config = load_config()
    data_path = Path(config["paths"]["graph_data"])

    store = Neo4jStore()
    store.clear()

    data = json.loads(data_path.read_text())

    for sample in data:
        problem = sample["problem"]

        # Create problem node
        store.run(
            "MERGE (p:Problem {name: $name})",
            {"name": problem}
        )

        # Explicit symptoms
        for s in sample.get("explicit_symptoms", []):
            store.run("""
                MERGE (s:Symptom {name: $s, type: 'explicit'})
                MERGE (p:Problem {name: $p})
                MERGE (p)-[:HAS_SYMPTOM]->(s)
            """, {"s": s, "p": problem})

        # Implicit symptoms
        for s in sample.get("implicit_symptoms", []):
            store.run("""
                MERGE (s:Symptom {name: $s, type: 'implicit'})
                MERGE (p:Problem {name: $p})
                MERGE (p)-[:HAS_IMPLICIT_SYMPTOM]->(s)
            """, {"s": s, "p": problem})

        # Causes
        for c in sample.get("possible_causes", []):
            store.run("""
                MERGE (c:Cause {name: $c})
                MERGE (p:Problem {name: $p})
                MERGE (p)-[:CAUSED_BY]->(c)
            """, {"c": c, "p": problem})

        # Actions
        for a in sample.get("recommended_actions", []):
            store.run("""
                MERGE (a:Action {name: $a})
                MERGE (p:Problem {name: $p})
                MERGE (p)-[:RECOMMENDED_ACTION]->(a)
            """, {"a": a, "p": problem})

        logger.info(f"Ingested problem: {problem}")

    store.close()
    logger.info("Graph construction completed successfully")

if __name__ == "__main__":
    build_graph()
