from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    understand_intent_node,
    merge_information_node,
    check_required_information_node,
    ask_user_node,
    ask_confirmation_node,
    cancel_ride_node,
    book_ride_node,
)


def route_intent_or_confirmation(state: AgentState) -> str:
    """Decides flow based on whether we are awaiting confirmation or collecting info."""
    waiting_conf = state.get("waiting_for_confirmation", False)
    confirmation = state.get("confirmation_received")

    if waiting_conf:
        if confirmation is True:
            return "book_ride"
        elif confirmation is False:
            return "cancel_ride"

    # Otherwise route to merge information
    return "merge_information"


def route_after_check(state: AgentState) -> str:
    """Decides whether to ask user for missing fields or proceed to confirmation."""
    missing = state.get("missing_fields", [])
    if missing:
        return "ask_user"
    return "ask_confirmation"


def build_agent_graph():
    """Builds and compiles the deterministic LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("understand_intent", understand_intent_node)
    workflow.add_node("merge_information", merge_information_node)
    workflow.add_node("check_required_information", check_required_information_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("ask_confirmation", ask_confirmation_node)
    workflow.add_node("cancel_ride", cancel_ride_node)
    workflow.add_node("book_ride", book_ride_node)

    # 2. Add Linear & Conditional Edges
    workflow.set_entry_point("understand_intent")

    # Routing based on confirmation vs info collection
    workflow.add_conditional_edges(
        "understand_intent",
        route_intent_or_confirmation,
        {
            "book_ride": "book_ride",
            "cancel_ride": "cancel_ride",
            "merge_information": "merge_information",
        },
    )

    workflow.add_edge("merge_information", "check_required_information")

    # Conditional Branching on missing info
    workflow.add_conditional_edges(
        "check_required_information",
        route_after_check,
        {
            "ask_user": "ask_user",
            "ask_confirmation": "ask_confirmation",
        },
    )

    workflow.add_edge("ask_user", END)
    workflow.add_edge("ask_confirmation", END)
    workflow.add_edge("cancel_ride", END)
    workflow.add_edge("book_ride", END)

    return workflow.compile()


agent_graph = build_agent_graph()
