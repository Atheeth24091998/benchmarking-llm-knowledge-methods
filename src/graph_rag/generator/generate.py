from src.graph_rag.utils.logger import get_logger
from src.rag.generator.generate import generate_graph_answer

logger = get_logger(__name__)

GRAPH_RAG_PROMPT = """
You MUST answer ONLY using the graph facts below.

PROBLEM:
{problem}

EXPLICIT SYMPTOMS:
{explicit_symptoms}

IMPLICIT SYMPTOMS:
{implicit_symptoms}

POSSIBLE CAUSES:
{possible_causes}

RECOMMENDED ACTIONS:
{recommended_actions}

QUESTION:
{question}

Respond strictly in this format:

Problem:
- <one sentence>

Causes:
- <bullet points>

Actions:
- <bullet points>
"""


class GraphAnswerGenerator:
    def __init__(self):
        logger.info("Graph RAG Answer Generator initialized")

    def generate(self, question: str, subgraph: dict) -> str:
        if not subgraph:
            return "No graph information available."

        prompt = GRAPH_RAG_PROMPT.format(
            problem=subgraph["problem"].replace("_", " "),
            explicit_symptoms=", ".join(subgraph["explicit_symptoms"]) or "None",
            implicit_symptoms=", ".join(subgraph["implicit_symptoms"]) or "None",
            possible_causes=", ".join(subgraph["possible_causes"]) or "None",
            recommended_actions=", ".join(subgraph["recommended_actions"]) or "None",
            question=question,
        )

        logger.info("Generating Graph RAG answer")
        return generate_graph_answer(prompt)
