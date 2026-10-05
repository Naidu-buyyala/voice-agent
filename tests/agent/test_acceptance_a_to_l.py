import pytest
from app.agent.extractor import GeminiExtractor
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.tools.executor import tool_executor


@pytest.fixture
def extractor():
    return GeminiExtractor()


@pytest.mark.asyncio
async def test_acceptance_a_critical_bug_naturals_ice_cream():
    """
    Test A — The critical bug:
    Assistant proposes a Naturals ice cream order for delivery to Manikonda.
    Estimate: ₹350. State: WAITING_FOR_CONFIRMATION.
    User: "book".
    Expected: the pending food order is executed, not a new service-selection menu!
    """
    pending_state: AgentState = {
        "conversation_id": "c_food_crit",
        "task_id": "t_food_crit",
        "user_message": "book",
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [
            {"role": "user", "content": "ice cream"},
            {"role": "assistant", "content": "Do you have a restaurant preference?"},
            {"role": "user", "content": "Naturals"},
            {"role": "assistant", "content": "Where should I deliver it?"},
            {"role": "user", "content": "Manikonda"},
            {
                "role": "assistant",
                "content": (
                    "Great choice! I have an order for ice cream from Naturals.\n"
                    "Delivery Address: Manikonda\n"
                    "Estimated Total: ₹350 (including taxes and delivery)\n\n"
                    "Shall I go ahead and place this food order for you?"
                ),
            },
        ],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)

    # 1. State must transition to COMPLETED
    assert result["current_step"] == "COMPLETED"
    assert result["waiting_for_confirmation"] is False

    # 2. Must not ask what to book!
    assert "what would you like to book" not in result["response"].lower()
    assert "ride, food, or a home service" not in result["response"].lower()

    # 3. Must confirm the actual food order details
    assert "placed successfully" in result["response"].lower()
    assert "Naturals" in result["response"]
    assert "Manikonda" in result["response"]
    assert "Order ID" in result["response"] or "order_id" in str(result["tool_result"])


@pytest.mark.asyncio
@pytest.mark.parametrize("affirmative", ["yes", "confirm", "go ahead", "place it", "proceed", "do it"])
async def test_acceptance_b_equivalent_confirmations(affirmative: str):
    """
    Test B — Equivalent confirmations:
    Test "yes", "confirm", "go ahead", "place it", and "proceed".
    Expected: each confirms the correct pending booking.
    """
    pending_state: AgentState = {
        "conversation_id": f"c_b_{affirmative.replace(' ', '_')}",
        "task_id": f"t_b_{affirmative.replace(' ', '_')}",
        "user_message": affirmative,
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)
    assert result["current_step"] == "COMPLETED"
    assert "placed successfully" in result["response"].lower()
    assert "Naturals" in result["response"]


@pytest.mark.asyncio
async def test_acceptance_c_no_pending_booking():
    """
    Test C — No pending booking:
    User: "book".
    Expected: ask what they want to book.
    """
    state: AgentState = {
        "conversation_id": "c_no_pending",
        "task_id": None,
        "user_message": "book",
        "last_assistant_message": None,
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": None,
        "collected_data": {},
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "START",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(state)
    assert any(w in result["response"].lower() for w in ["what would you like to book", "ride, food", "service like urban clean"])


@pytest.mark.asyncio
async def test_acceptance_d_rejection():
    """
    Test D — Rejection:
    User: "no, don't place it".
    Expected: do not call the order-creation provider; cancel gracefully.
    """
    state: AgentState = {
        "conversation_id": "c_rejection",
        "task_id": "t_rej",
        "user_message": "no, don't place it",
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(state)
    assert result["current_step"] == "CANCELLED"
    assert result["waiting_for_confirmation"] is False
    assert any(w in result["response"].lower() for w in ["cancelled", "no worries"])
    assert "placed successfully" not in result["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_e_missing_information():
    """
    Test E — Missing information:
    A required detail is missing (e.g. delivery location in food flow).
    Expected: ask only for that detail and do not execute the booking.
    """
    state: AgentState = {
        "conversation_id": "c_missing_info",
        "task_id": "t_miss",
        "user_message": "Mehfil",
        "last_assistant_message": "Do you have a restaurant preference?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "biryani",
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(state)
    assert result["current_step"] == "WAITING_FOR_USER"
    assert "delivery_location" in result["missing_fields"]
    assert any(w in result["response"].lower() for w in ["deliver", "address", "where"])
    assert result["waiting_for_confirmation"] is False


@pytest.mark.asyncio
async def test_acceptance_f_duplicate_confirmation():
    """
    Test F — Duplicate confirmation:
    User sends "book" repeatedly.
    Expected: at most one order is created; second 'book' returns existing booking details.
    """
    initial_pending: AgentState = {
        "conversation_id": "c_dup",
        "task_id": "t_dup",
        "user_message": "book",
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    # First "book"
    res1 = await agent_graph.ainvoke(initial_pending)
    assert res1["current_step"] == "COMPLETED"
    order_id1 = res1["tool_result"]["order_id"]

    # Second "book" immediately after (e.g. slow client retry)
    second_pending = {
        **res1,
        "user_message": "book",
        "latest_completed_task": {
            "status": "COMPLETED",
            "type": "ORDER_FOOD",
            "collected_data": res1["collected_data"],
            "result": res1["tool_result"],
        },
    }
    res2 = await agent_graph.ainvoke(second_pending)
    order_id2 = res2["tool_result"]["order_id"]

    assert order_id1 == order_id2
    assert "already" in res2["response"].lower() or "placed" in res2["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_g_successful_booking_immediate_details():
    """
    Test G — Successful booking:
    Provider returns a confirmed order ID and actual details.
    Expected: show those details immediately (do NOT ask 'Would you like the booking details?').
    """
    pending_state: AgentState = {
        "conversation_id": "c_g",
        "task_id": "t_g",
        "user_message": "confirm",
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "pizza",
            "restaurant": "Domino",
            "delivery_location": "Gachibowli",
            "fare_amount": 499.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)
    assert result["current_step"] == "COMPLETED"
    assert "Order ID" in result["response"]
    assert "Domino" in result["response"]
    assert "Gachibowli" in result["response"]
    assert "Suresh V." in result["response"]
    assert "would you like the booking details" not in result["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_h_provider_failure(monkeypatch):
    """
    Test H — Provider failure:
    Provider rejects the booking.
    Expected: report the failure; do not claim success.
    """
    async def mock_fail_order(*args, **kwargs):
        raise RuntimeError("Restaurant is currently closed for new orders")

    monkeypatch.setattr(tool_executor, "execute", mock_fail_order)

    pending_state: AgentState = {
        "conversation_id": "c_fail",
        "task_id": "t_fail",
        "user_message": "book",
        "last_assistant_message": "Shall I place this order?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "biryani",
            "restaurant": "Closed Cafe",
            "delivery_location": "Banjara Hills",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)
    assert result["current_step"] == "FAILED"
    assert "failed" in result["response"].lower() or "something went wrong" in result["response"].lower()
    assert "placed successfully" not in result["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_i_uncertain_timeout(monkeypatch):
    """
    Test I — Uncertain timeout:
    Order creation times out.
    Expected: fail safely with actionable error, never claim false success.
    """
    async def mock_timeout(*args, **kwargs):
        raise TimeoutError("Provider API gateway timed out after 30000ms")

    monkeypatch.setattr(tool_executor, "execute", mock_timeout)

    pending_state: AgentState = {
        "conversation_id": "c_timeout",
        "task_id": "t_timeout",
        "user_message": "book",
        "last_assistant_message": "Shall I place this order?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "biryani",
            "restaurant": "Mehfil",
            "delivery_location": "Gachibowli",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)
    assert result["current_step"] == "FAILED"
    assert "placed successfully" not in result["response"].lower()
    assert "failed" in result["response"].lower() or "wrong" in result["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_j_context_switching():
    """
    Test J — Context switching:
    A food order is awaiting confirmation, but the user explicitly asks to book a ride instead.
    Expected: do not accidentally place the food order. Switch context appropriately.
    """
    pending_state: AgentState = {
        "conversation_id": "c_switch",
        "task_id": "t_switch",
        "user_message": "Actually, book a bike from Manikonda to Hitech City instead",
        "last_assistant_message": "Shall I go ahead and place this food order for you?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
            "fare_amount": 350.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(pending_state)
    # The food order must NOT have been executed
    assert "Naturals order from" not in result["response"]
    assert "ice cream order from Naturals has been placed" not in result["response"]
    # Intent switched to ride
    assert result["intent"] == "BOOK_RIDE" or result["collected_data"].get("intent") == "BOOK_RIDE"
    assert "bike" in result["response"].lower() or "fare" in result["response"].lower() or "manikonda" in result["response"].lower()


@pytest.mark.asyncio
async def test_acceptance_k_follow_up_details_and_location():
    """
    Test K — Follow-up:
    After booking, user asks "details?" or "where is my order?"
    Expected: use the existing booking context and retrieve the relevant information.
    """
    completed_task = {
        "status": "COMPLETED",
        "type": "ORDER_FOOD",
        "collected_data": {
            "intent": "ORDER_FOOD",
            "food": "ice cream",
            "restaurant": "Naturals",
            "delivery_location": "Manikonda",
        },
        "result": {
            "order_id": "food_naturals123",
            "booking_id": "food_naturals123",
            "restaurant": "Naturals",
            "items": ["ice cream"],
            "delivery_location": "Manikonda",
            "delivery_partner": "Suresh V.",
            "delivery_phone": "+91 98765 12345",
            "initial_eta_minutes": 20,
            "eta_minutes": 20,
            "status": "PREPARING",
            "simulation_enabled": True,
        },
    }

    inquiry_state: AgentState = {
        "conversation_id": "c_followup",
        "task_id": "t_followup",
        "user_message": "where is my order?",
        "last_assistant_message": "Your ice cream order has been placed!",
        "conversation_history": [],
        "latest_completed_task": completed_task,
        "intent": None,
        "collected_data": {},
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "COMPLETED",
        "waiting_for_confirmation": False,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }

    result = await agent_graph.ainvoke(inquiry_state)
    assert result["current_step"] == "INQUIRY_ANSWERED"
    assert "Suresh V." in result["response"]
    assert "Naturals" in result["response"]
    assert any(w in result["response"].lower() for w in ["preparing", "pick up", "minutes"])


@pytest.mark.asyncio
async def test_acceptance_l_regression_ride_and_service():
    """
    Test L — Regression:
    Existing ride, food, and home-service flows continue to work correctly end-to-end.
    """
    # 1. Ride booking confirmation
    ride_state: AgentState = {
        "conversation_id": "c_reg_ride",
        "task_id": "t_reg_ride",
        "user_message": "book",
        "last_assistant_message": "Shall I book this Uber Go?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "BOOK_RIDE",
        "collected_data": {
            "intent": "BOOK_RIDE",
            "pickup": "Madhapur",
            "destination": "Gachibowli",
            "ride_type": "Uber Go",
            "fare_amount": 450.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    r_ride = await agent_graph.ainvoke(ride_state)
    assert r_ride["current_step"] == "COMPLETED"
    assert "booked successfully" in r_ride["response"].lower()
    assert "Uber Go" in r_ride["response"]

    # 2. Home Service confirmation
    service_state: AgentState = {
        "conversation_id": "c_reg_serv",
        "task_id": "t_reg_serv",
        "user_message": "confirm",
        "last_assistant_message": "Shall I confirm this appointment?",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "BOOK_SERVICE",
        "collected_data": {
            "intent": "BOOK_SERVICE",
            "service_name": "Deep Cleaning",
            "package": "Standard",
            "service_location": "Kondapur",
            "preferred_date": "Today",
            "slot": "2:00 PM – 4:00 PM",
            "fare_amount": 1499.0,
            "waiting_for_confirmation": True,
        },
        "preferences": None,
        "missing_fields": [],
        "next_action": None,
        "_new_fields": None,
        "_llm_reply": None,
        "_needs_landmark_clarification": None,
        "current_step": "WAITING_FOR_CONFIRMATION",
        "waiting_for_confirmation": True,
        "confirmation_received": None,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    r_serv = await agent_graph.ainvoke(service_state)
    assert r_serv["current_step"] == "COMPLETED"
    assert "appointment is confirmed" in r_serv["response"].lower()
    assert "Kondapur" in r_serv["response"]
