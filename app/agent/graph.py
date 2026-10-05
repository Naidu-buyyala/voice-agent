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
    book_food_order_node,
    book_service_node,
    book_bus_node,
    book_train_node,
    handle_greeting_node,
    handle_gratitude_node,
    handle_task_inquiry_node,
)


def route_intent_or_confirmation(state: AgentState) -> str:
    """Decides flow based on user intent, confirmation, and state."""
    waiting_conf = state.get("waiting_for_confirmation", False)
    confirmation = state.get("confirmation_received")
    raw_user_msg = (state.get("user_message") or "").strip().lower()
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or (state.get("collected_data") or {}).get("booking_result")

    # If the user message is a repeated confirmation ("book", "confirm", etc.) and the task is already COMPLETED,
    # route directly to the completed booking node so it returns the existing order facts idempotently
    if latest_task.get("status") == "COMPLETED" or (prev_result and (prev_result.get("booking_id") or prev_result.get("order_id") or prev_result.get("pnr"))):
        if raw_user_msg in ["book", "book it", "confirm", "yes", "proceed", "place it"]:
            task_type = latest_task.get("type") or (latest_task.get("collected_data") or {}).get("intent") or (state.get("collected_data") or {}).get("intent")
            if task_type == "ORDER_FOOD":
                return "book_food_order"
            elif task_type == "BOOK_SERVICE":
                return "book_service"
            elif task_type == "BOOK_BUS":
                return "book_bus"
            elif task_type == "BOOK_TRAIN":
                return "book_train"
            return "book_ride"

    # Context check: if waiting for confirmation of one task, but user explicitly switched context
    pending_intent = (state.get("collected_data") or {}).get("intent")
    extracted_intent = state.get("intent")
    if waiting_conf and extracted_intent and pending_intent and extracted_intent != pending_intent:
        # Context switched to another task! Do not treat as confirmation of the old task.
        return "merge_information"

    intent = state.get("intent") or (state.get("collected_data") or {}).get("intent")

    if waiting_conf:
        if confirmation is True:
            if intent == "ORDER_FOOD":
                return "book_food_order"
            elif intent == "BOOK_SERVICE":
                return "book_service"
            elif intent == "BOOK_BUS":
                return "book_bus"
            elif intent == "BOOK_TRAIN":
                return "book_train"
            return "book_ride"
        elif confirmation is False:
            return "cancel_ride"

    if intent == "GREETING":
        return "handle_greeting"
    elif intent == "GRATITUDE_OR_CLOSING":
        return "handle_gratitude"
    elif intent == "TASK_INQUIRY":
        return "handle_task_inquiry"
    elif intent in ["GENERAL_SUPPORT", "UNKNOWN"] and not state.get("collected_data") and not state.get("_new_fields"):
        return "handle_greeting"

    # All action workflows (rides, food, services, buses, trains) flow through the cognitive pipeline
    return "merge_information"


def route_after_check(state: AgentState) -> str:
    """Decides whether to ask user for genuinely missing fields or proceed to confirmation."""
    missing = state.get("missing_fields", [])
    if missing:
        return "ask_user"
    return "ask_confirmation"


def build_agent_graph():
    """Builds and compiles the deterministic LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("understand_intent", understand_intent_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_gratitude", handle_gratitude_node)
    workflow.add_node("handle_task_inquiry", handle_task_inquiry_node)
    workflow.add_node("merge_information", merge_information_node)
    workflow.add_node("check_required_information", check_required_information_node)
    workflow.add_node("ask_user", ask_user_node)
    workflow.add_node("ask_confirmation", ask_confirmation_node)
    workflow.add_node("cancel_ride", cancel_ride_node)
    workflow.add_node("book_ride", book_ride_node)
    workflow.add_node("book_food_order", book_food_order_node)
    workflow.add_node("book_service", book_service_node)
    workflow.add_node("book_bus", book_bus_node)
    workflow.add_node("book_train", book_train_node)

    # 2. Add Linear & Conditional Edges
    workflow.set_entry_point("understand_intent")

    # Routing based on confirmation vs intent
    workflow.add_conditional_edges(
        "understand_intent",
        route_intent_or_confirmation,
        {
            "book_ride": "book_ride",
            "book_food_order": "book_food_order",
            "book_service": "book_service",
            "book_bus": "book_bus",
            "book_train": "book_train",
            "cancel_ride": "cancel_ride",
            "handle_greeting": "handle_greeting",
            "handle_gratitude": "handle_gratitude",
            "handle_task_inquiry": "handle_task_inquiry",
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

    # Terminal node edges
    workflow.add_edge("ask_user", END)
    workflow.add_edge("ask_confirmation", END)
    workflow.add_edge("cancel_ride", END)
    workflow.add_edge("book_ride", END)
    workflow.add_edge("book_food_order", END)
    workflow.add_edge("book_service", END)
    workflow.add_edge("book_bus", END)
    workflow.add_edge("book_train", END)
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_gratitude", END)
    workflow.add_edge("handle_task_inquiry", END)

    return workflow.compile()


agent_graph = build_agent_graph()

