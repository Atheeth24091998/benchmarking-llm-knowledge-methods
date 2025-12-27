from neo4j import GraphDatabase
from src.graph_rag.utils.config_loader import load_config
from src.graph_rag.utils.logger import get_logger

class Neo4jStore:
    def __init__(self):
        config = load_config()
        neo = config["graph_rag"]["neo4j"]

        self.driver = GraphDatabase.driver(
            neo["uri"],
            auth=(neo["user"], neo["password"])
        )

        self.logger = get_logger(self.__class__.__name__)

    def close(self):
        self.driver.close()

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.logger.info("Cleared Neo4j database")

    def run(self, query, params=None):
        with self.driver.session() as session:
            session.run(query, params or {})
