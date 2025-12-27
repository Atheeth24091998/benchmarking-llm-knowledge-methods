# from src.graph_rag.linking.entity_linker import EntityLinker
# from src.graph_rag.retrieval.subgraph_retriever import SubgraphRetriever

# question = "The belt weave is peeling off and the belt surface is damaged. What should I do?"

# linker = EntityLinker()
# retriever = SubgraphRetriever()

# problem, score = linker.link(question)

# if problem:
#     subgraph = retriever.retrieve(problem)
#     print("\n=== GRAPH RAG RESULT ===")
#     for k, v in subgraph.items():
#         print(f"{k}: {v}")
# else:
#     print("No problem linked.")


from src.graph_rag.linking.entity_linker import EntityLinker
from src.graph_rag.retrieval.subgraph_retriever import SubgraphRetriever
from src.graph_rag.generator.generate import GraphAnswerGenerator

question = "transfer_rail_y_axis_not_level explicit_symptoms are bubble_off_center_y_axis,transfer_rail_tilted and implicit_symptoms are transfer_errors,y_axis_movement_issues,precision_loss"

linker = EntityLinker()
retriever = SubgraphRetriever()
generator = GraphAnswerGenerator()

problem, score = linker.link(question)

print(f"Linked problem: {problem} (score={score:.3f})")

subgraph = retriever.retrieve(problem)
answer = generator.generate(question, subgraph)

print("\n=== GRAPH RAG ANSWER ===\n")
print(answer)
