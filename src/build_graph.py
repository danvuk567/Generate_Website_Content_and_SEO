from src.config import GraphState
from src.graph_functions import search_node, designer_node, editor_node, revision_node, seo_node, city_router, revision_router
from langgraph.graph import StateGraph, START, END

# -------------------------------------------------------
# LangGraph StateGraph Construction
# -------------------------------------------------------

def build_graph(llm1, llm2):

    # Initialize graph with shared state schema
    graph_builder = StateGraph(GraphState)

    # -------------------------------------------------------
    # 1. NODE REGISTRATION
    # -------------------------------------------------------
    # Each node represents a single transformation step
    # in the pipeline. Use wrappers to pass llm tht was picked dynamically

    def designer_node_llm(state):
        return designer_node(state, llm2)

    def editor_node_llm(state):
        return editor_node(state, llm1)

    def revision_node_llm(state):
        return revision_node(state, llm1)

    def seo_node_llm(state):
        return seo_node(state, llm1)

    graph_builder.add_node("search", search_node)
    graph_builder.add_node("designer", designer_node_llm)
    graph_builder.add_node("editor", editor_node_llm)
    graph_builder.add_node("revision", revision_node_llm)
    graph_builder.add_node("seo", seo_node_llm)

    # -------------------------------------------------------
    # 2. ENTRY POINT
    # -------------------------------------------------------
    # Graph execution always begins at the search node
    graph_builder.add_edge(START, "search")

    # -------------------------------------------------------
    # 3. SEARCH → DESIGNER ROUTING (MULTI-CITY LOOP)
    # -------------------------------------------------------
    # This conditional edge controls iterative city-based search.
    #
    # city_router returns:
    # - "continue" → process next city (loop back to search)
    # - "next"     → all cities processed → move forward

    graph_builder.add_conditional_edges(
        "search",
        city_router,
        {
            "continue": "search",
            "next": "designer",
        }
    )

    # -------------------------------------------------------
    # 4. DESIGN FLOW (LINEAR STAGE)
    # -------------------------------------------------------
    # Once search is complete, content flows into design phase

    graph_builder.add_edge("designer", "editor")

    # -------------------------------------------------------
    # 5. EDITOR → REVISION LOOP CONTROL
    # -------------------------------------------------------
    # revision_router controls iterative improvement loop:
    #
    # - "revise" → send back to revision node
    # - "next"   → approved or max revisions reached, go to seo_optimizer

    graph_builder.add_conditional_edges(
        "editor",
        revision_router,
        {
            "revise": "revision",
            "next": "seo",
        }
    )

    # After revision, return to editor for re-evaluation
    graph_builder.add_edge("revision", "editor")

    # -------------------------------------------------------
    # 6. FINAL STAGE
    # -------------------------------------------------------
    # SEO optimization is final transformation step

    graph_builder.add_edge("seo", END)

    # -------------------------------------------------------
    # 7. COMPILE GRAPH
    # -------------------------------------------------------
    # Converts graph definition into executable LangGraph object

    return graph_builder.compile()