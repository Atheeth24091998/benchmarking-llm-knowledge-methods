from src.graph_rag.graph_store.neo4j_store import Neo4jStore
from src.graph_rag.utils.logger import get_logger

class SubgraphRetriever:
    def __init__(self):
        self.store = Neo4jStore()
        self.logger = get_logger(self.__class__.__name__)

    def retrieve(self, problem: str, hops: int = 1):
        query = f"""
        MATCH (p:Problem {{name: $problem}})
        OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (p)-[:HAS_IMPLICIT_SYMPTOM]->(is:Symptom)
        OPTIONAL MATCH (p)-[:CAUSED_BY]->(c:Cause)
        OPTIONAL MATCH (p)-[:RECOMMENDED_ACTION]->(a:Action)
        RETURN
            collect(DISTINCT s.name) AS explicit_symptoms,
            collect(DISTINCT is.name) AS implicit_symptoms,
            collect(DISTINCT c.name) AS causes,
            collect(DISTINCT a.name) AS actions
        """

        with self.store.driver.session() as session:
            result = session.run(query, {"problem": problem})
            record = result.single()

        if not record:
            self.logger.warning(f"No graph data for {problem}")
            return {}

        subgraph = {
            "problem": problem,
            "explicit_symptoms": record["explicit_symptoms"],
            "implicit_symptoms": record["implicit_symptoms"],
            "possible_causes": record["causes"],
            "recommended_actions": record["actions"],
        }

        self.logger.info(
            f"Retrieved subgraph for '{problem}' "
            f"(causes={len(subgraph['possible_causes'])}, "
            f"actions={len(subgraph['recommended_actions'])})"
        )

        return subgraph
