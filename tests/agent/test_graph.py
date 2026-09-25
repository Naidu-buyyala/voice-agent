import pytest
from app.agent.graph import agent_graph
from app.agent.state import AgentState


@pytest.mark.asyncio
async def test_graph_ask_pickup_when_missing():
    initial_state: AgentState = {
        "conversation_id": "conv_1",
        "task_id": "task_1",
        "user_message": "I want to book an Uber",
        "intent": None,
        "collected_data": {},
        "missing_fields": [],
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    final_state = await agent_graph.ainvoke(initial_state)

    assert final_state["current_step"] == "WAITING_FOR_USER"
    assert "pickup" in final_state["missing_fields"]
    assert "destination" in final_state["missing_fields"]
    assert "pick you up" in final_state["response"].lower()


@pytest.mark.asyncio
async def test_graph_ask_service_type_when_missing():
    initial_state: AgentState = {
        "conversation_id": "conv_service",
        "task_id": "task_service",
        "user_message": "from Hitech City to Hyderabad Airport",
        "intent": None,
        "collected_data": {},
        "missing_fields": [],
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    final_state = await agent_graph.ainvoke(initial_state)

    assert final_state["current_step"] == "WAITING_FOR_USER"
    assert "service_type" in final_state["missing_fields"]
    assert "Cab" in final_state["response"]
    assert "Bike" in final_state["response"]


@pytest.mark.asyncio
async def test_graph_ask_confirmation_when_complete():
    initial_state: AgentState = {
        "conversation_id": "conv_2",
        "task_id": "task_2",
        "user_message": "from Hitech City to Hyderabad Airport by cab",
        "intent": None,
        "collected_data": {},
        "missing_fields": [],
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    final_state = await agent_graph.ainvoke(initial_state)

    assert final_state["current_step"] == "WAITING_FOR_CONFIRMATION"
    assert final_state["waiting_for_confirmation"] is True
    assert final_state["collected_data"]["pickup"] == "Hitech City"
    assert final_state["collected_data"]["destination"] == "Hyderabad Airport"
    assert any(w in final_state["response"].lower() for w in ["book this", "book it", "book"])


@pytest.mark.asyncio
async def test_graph_rejects_same_location():
    # When user enters the same location for pickup and destination
    initial_state: AgentState = {
        "conversation_id": "conv_same",
        "task_id": "task_same",
        "user_message": "from DLF Cyber City to DLF Cyber City",
        "intent": None,
        "collected_data": {},
        "missing_fields": [],
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    final_state = await agent_graph.ainvoke(initial_state)

    assert final_state["current_step"] == "WAITING_FOR_USER"
    assert "same_location_destination" in final_state["missing_fields"]
    assert "can't be the same" in final_state["response"].lower() or "cannot be the same" in final_state["response"].lower()


@pytest.mark.asyncio
async def test_graph_multiturn_bike_flow():
    # Turn 1: "Book a bike"
    t1_state: AgentState = {
        "conversation_id": "conv_bike",
        "task_id": "task_bike",
        "user_message": "Book a bike",
        "intent": None,
        "collected_data": {},
        "missing_fields": [],
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    t1_final = await agent_graph.ainvoke(t1_state)
    assert t1_final["current_step"] == "WAITING_FOR_USER"
    assert t1_final["collected_data"].get("service_type") == "bike"
    assert t1_final["collected_data"].get("ride_type") == "Uber Moto"
    assert t1_final["collected_data"].get("last_asked") == "pickup"
    assert "pick you up" in t1_final["response"].lower()

    # Turn 2: User answers pickup location "DLF Cyber City"
    t2_state: AgentState = {
        "conversation_id": "conv_bike",
        "task_id": "task_bike",
        "user_message": "DLF Cyber City",
        "last_assistant_message": t1_final["response"],
        "intent": None,
        "collected_data": t1_final["collected_data"],
        "missing_fields": [],
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    t2_final = await agent_graph.ainvoke(t2_state)
    assert t2_final["current_step"] == "WAITING_FOR_USER"
    assert t2_final["collected_data"].get("pickup") == "Dlf Cyber City" or t2_final["collected_data"].get("pickup") == "DLF Cyber City"
    assert t2_final["collected_data"].get("last_asked") == "destination"
    assert "where would you like to head" in t2_final["response"].lower()

    # Turn 3: User answers same location "DLF Cyber City"
    t3_state: AgentState = {
        "conversation_id": "conv_bike",
        "task_id": "task_bike",
        "user_message": "DLF Cyber City",
        "last_assistant_message": t2_final["response"],
        "intent": None,
        "collected_data": t2_final["collected_data"],
        "missing_fields": [],
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    t3_final = await agent_graph.ainvoke(t3_state)
    assert t3_final["current_step"] == "WAITING_FOR_USER"
    assert "can't be the same" in t3_final["response"].lower()

    # Turn 4: User gives different destination "Airport"
    t4_state: AgentState = {
        "conversation_id": "conv_bike",
        "task_id": "task_bike",
        "user_message": "Airport",
        "last_assistant_message": t3_final["response"],
        "intent": None,
        "collected_data": t3_final["collected_data"],
        "missing_fields": [],
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    t4_final = await agent_graph.ainvoke(t4_state)
    assert t4_final["current_step"] == "WAITING_FOR_CONFIRMATION"
    assert t4_final["collected_data"].get("ride_type") == "Uber Moto"
    assert t4_final["collected_data"].get("fare_amount") == 120.0
    assert "120" in t4_final["response"]

