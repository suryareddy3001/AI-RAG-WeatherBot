from langgraph.graph import StateGraph, END
from .nodes import (
    router_node, weather_node, rag_node, synthesis_node,
    GraphState, route_intent
)

"""LangChain graph construction for AI RAG WeatherBot.

This module defines and builds a stateful graph for processing user queries,
routing them to either weather or document-based question-answering nodes,
and synthesizing the final response.
"""

def build_graph():
    """Construct and compile the LangChain state graph.

    Returns:
        A compiled LangChain StateGraph that routes queries to weather or RAG nodes
        and synthesizes the final response.
    """
    g = StateGraph(GraphState)

    g.add_node("router", router_node)
    g.add_node("weather", weather_node)
    g.add_node("rag", rag_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_intent,
        {"weather": "weather", "doc_qa": "rag"},
    )

    g.add_edge("weather", "synthesis")
    g.add_edge("rag", "synthesis")
    g.add_edge("synthesis", END)

    return g.compile()