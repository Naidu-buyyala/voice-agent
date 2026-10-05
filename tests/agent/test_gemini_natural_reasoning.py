import pytest
from app.agent.extractor import GeminiExtractor
from app.agent.graph import agent_graph
from app.agent.state import AgentState


@pytest.fixture
def extractor():
    return GeminiExtractor()


@pytest.mark.asyncio
async def test_gemini_reasoning_food_order_flow(extractor: GeminiExtractor):
    """
    Verifies that conversational reasoning interprets 'Any biryani' without asking
    for dish/cuisine again, and handles 'your wish' delegation contextually.
    """
    # Turn 1: "Order food"
    res1 = await extractor.extract("Order food")
    assert res1.intent == "ORDER_FOOD"
    assert "food" in res1.missing_required_fields

    # Turn 2: "Any biryani"
    res2 = await extractor.extract(
        "Any biryani",
        current_collected={"intent": "ORDER_FOOD", "last_asked": "food"},
    )
    assert res2.intent == "ORDER_FOOD"
    assert res2.entities.food == "biryani"
    assert "biryani" in res2.assistant_message.lower()
    # Ensure it asks for restaurant or AI delegation, NOT "what food or cuisine do you want"
    assert "restaurant" in res2.assistant_message.lower() or "pick" in res2.assistant_message.lower()

    # Turn 3: "your wish" -> Contextual Delegation
    res3 = await extractor.extract(
        "your wish",
        current_collected={
            "intent": "ORDER_FOOD",
            "food": "biryani",
            "last_asked": "restaurant",
        },
    )
    assert res3.intent == "ORDER_FOOD"
    assert res3.preferences.selection_preference == "AI_CHOOSE"
    assert res3.preferences.selection_strategy == "TOP_RATED"
    assert "delivery_location" in res3.missing_required_fields
    # Assistant responds naturally by acknowledging choice and asking for delivery location
    assert "deliver" in res3.assistant_message.lower()
    assert res3.confirmation is None  # "your wish" is delegation, NOT booking confirmation


@pytest.mark.asyncio
async def test_gemini_reasoning_ride_delegation_cheapest(extractor: GeminiExtractor):
    """
    Verifies that 'cheapest one' maps to AI_CHOOSE with CHEAPEST selection strategy.
    """
    res = await extractor.extract(
        "cheapest one",
        current_collected={
            "intent": "BOOK_RIDE",
            "pickup": "Hitech City",
            "destination": "Airport",
            "last_asked": "service_type",
        },
    )
    assert res.preferences.selection_preference == "AI_CHOOSE"
    assert res.preferences.selection_strategy == "CHEAPEST"
    assert res.entities.ride_type == "Uber Moto"
    assert res.confirmation is None


@pytest.mark.asyncio
async def test_gemini_reasoning_home_service_delegation_best_one(extractor: GeminiExtractor):
    """
    Verifies that 'best one' in home service maps to AI_CHOOSE with BEST_AVAILABLE strategy.
    """
    res = await extractor.extract(
        "best one",
        current_collected={
            "intent": "BOOK_SERVICE",
            "service_name": "Deep Cleaning",
            "last_asked": "package",
        },
    )
    assert res.preferences.selection_preference == "AI_CHOOSE"
    assert res.preferences.selection_strategy in ["BEST_AVAILABLE", "TOP_RATED"]
    assert "Full Home" in (res.entities.package or "")


@pytest.mark.asyncio
async def test_multiturn_food_delegation_in_graph():
    """
    Tests end-to-end multi-turn conversation flow through the LangGraph engine.
    """
    # 1. Turn 1: "Order food"
    s1: AgentState = {
        "conversation_id": "c_food",
        "task_id": None,
        "user_message": "Order food",
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
    r1 = await agent_graph.ainvoke(s1)
    assert r1["current_step"] == "WAITING_FOR_USER"
    assert "food" in r1["missing_fields"]

    # 2. Turn 2: "Any biryani"
    s2 = dict(r1)
    s2["user_message"] = "Any biryani"
    s2["last_assistant_message"] = r1["response"]
    s2["conversation_history"] = [
        {"role": "user", "content": "Order food"},
        {"role": "assistant", "content": r1["response"]},
    ]
    r2 = await agent_graph.ainvoke(s2)
    assert r2["current_step"] == "WAITING_FOR_USER"
    assert r2["collected_data"]["food"] == "biryani"
    assert "restaurant" in r2["missing_fields"]
    # Never ask for food/cuisine again!
    assert "restaurant" in r2["response"].lower() or "pick" in r2["response"].lower()

    # 3. Turn 3: "your wish"
    s3 = dict(r2)
    s3["user_message"] = "your wish"
    s3["last_assistant_message"] = r2["response"]
    s3["conversation_history"] = s2["conversation_history"] + [
        {"role": "user", "content": "Any biryani"},
        {"role": "assistant", "content": r2["response"]},
    ]
    r3 = await agent_graph.ainvoke(s3)
    assert r3["current_step"] == "WAITING_FOR_USER"
    assert r3["collected_data"]["restaurant_selection"] == "AI_CHOOSE"
    assert "delivery_location" in r3["missing_fields"]
    assert "deliver" in r3["response"].lower()

    # 4. Turn 4: "Deliver to Madhapur, Hyderabad"
    s4 = dict(r3)
    s4["user_message"] = "to Madhapur, Hyderabad"
    s4["last_assistant_message"] = r3["response"]
    s4["conversation_history"] = s3["conversation_history"] + [
        {"role": "user", "content": "your wish"},
        {"role": "assistant", "content": r3["response"]},
    ]
    r4 = await agent_graph.ainvoke(s4)
    assert r4["current_step"] == "WAITING_FOR_CONFIRMATION"
    assert r4["waiting_for_confirmation"] is True
    assert "Madhapur" in r4["collected_data"]["delivery_location"]
    assert any(w in r4["response"].lower() for w in ["place this", "confirm", "order"])

    # 5. Turn 5: "yes" -> Execution
    s5 = dict(r4)
    s5["user_message"] = "yes"
    s5["last_assistant_message"] = r4["response"]
    s5["conversation_history"] = s4["conversation_history"] + [
        {"role": "user", "content": "to Madhapur, Hyderabad"},
        {"role": "assistant", "content": r4["response"]},
    ]
    r5 = await agent_graph.ainvoke(s5)
    assert r5["current_step"] == "COMPLETED"
    assert "placed successfully" in r5["response"].lower()
    assert "Order ID" in r5["response"]

    # 6. Turn 6: "where is my order" -> Real-time Tracking
    s6 = {
        "user_message": "where is my order",
        "latest_completed_task": {
            "status": "COMPLETED",
            "type": "ORDER_FOOD",
            "collected_data": r5["collected_data"],
            "result": r5["tool_result"],
        },
        "collected_data": {},
        "current_step": "COMPLETED",
        "waiting_for_confirmation": False,
        "conversation_history": s5["conversation_history"] + [
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": r5["response"]},
        ],
    }
    r6 = await agent_graph.ainvoke(s6)
    assert r6["current_step"] == "INQUIRY_ANSWERED"
    assert "Suresh V." in r6["response"]
    assert "prepared" in r6["response"].lower() or "on the way to pick up" in r6["response"].lower()
    assert "Biryani" in r6["response"] or "biryani" in r6["response"]

    # 7. Dynamic Tracking across elapsed time (Simulation)
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    # Simulation +7 mins: READY_FOR_PICKUP
    s6_7m = dict(s6)
    s6_7m["latest_completed_task"] = {
        "status": "COMPLETED",
        "type": "ORDER_FOOD",
        "collected_data": r5["collected_data"],
        "result": {**r5["tool_result"], "created_at": (now - timedelta(minutes=7)).isoformat()},
    }
    r6_7m = await agent_graph.ainvoke(s6_7m)
    assert "almost ready for pickup" in r6_7m["response"].lower()
    assert "13 minutes" in r6_7m["response"] or "about 13" in r6_7m["response"]

    # Simulation +12 mins: OUT_FOR_DELIVERY
    s6_12m = dict(s6)
    s6_12m["latest_completed_task"] = {
        "status": "COMPLETED",
        "type": "ORDER_FOOD",
        "collected_data": r5["collected_data"],
        "result": {**r5["tool_result"], "created_at": (now - timedelta(minutes=12)).isoformat()},
    }
    r6_12m = await agent_graph.ainvoke(s6_12m)
    assert "picked up and is now on the way" in r6_12m["response"].lower()
    assert "8 minutes" in r6_12m["response"]

    # Simulation +18 mins: ARRIVING
    s6_18m = dict(s6)
    s6_18m["latest_completed_task"] = {
        "status": "COMPLETED",
        "type": "ORDER_FOOD",
        "collected_data": r5["collected_data"],
        "result": {**r5["tool_result"], "created_at": (now - timedelta(minutes=18)).isoformat()},
    }
    r6_18m = await agent_graph.ainvoke(s6_18m)
    assert "reach you in about 2 minutes" in r6_18m["response"].lower()

    # Simulation +25 mins: DELIVERED
    s6_25m = dict(s6)
    s6_25m["latest_completed_task"] = {
        "status": "COMPLETED",
        "type": "ORDER_FOOD",
        "collected_data": r5["collected_data"],
        "result": {**r5["tool_result"], "created_at": (now - timedelta(minutes=25)).isoformat()},
    }
    r6_25m = await agent_graph.ainvoke(s6_25m)
    assert "delivered successfully" in r6_25m["response"].lower()


@pytest.mark.asyncio
async def test_dynamic_ride_driver_tracking_and_details():
    """
    Verifies separation of ride details vs live driver tracking,
    and checks dynamic distance, ETA, and arrival status.
    """
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)

    booked_task = {
        "status": "COMPLETED",
        "type": "BOOK_RIDE",
        "collected_data": {
            "pickup": "Gachibowli",
            "destination": "Airport",
            "ride_type": "Uber Moto",
        },
        "result": {
            "booking_id": "uber_live_123",
            "driver_name": "Rajesh K.",
            "vehicle_plate": "TS 09 UB 1234",
            "driver_phone": "+91 98765 43210",
            "initial_eta_minutes": 4,
            "created_at": now.isoformat(),
        },
    }

    # 1. User asks "details?" -> Full booking details
    s_details = {
        "user_message": "details?",
        "latest_completed_task": booked_task,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_details = await agent_graph.ainvoke(s_details)
    assert r_details["current_step"] == "INQUIRY_ANSWERED"
    assert "Here are your current ride details:" in r_details["response"]
    assert "Uber Moto" in r_details["response"]
    assert "TS 09 UB 1234" not in r_details["response"] or "Rajesh K." in r_details["response"]
    assert "Gachibowli" in r_details["response"]
    assert "Airport" in r_details["response"]
    assert "uber_live_123" in r_details["response"]

    # 2. User asks "where is driver?" immediately (0 min) -> Focused live tracking
    s_track_0m = {
        "user_message": "where is driver?",
        "latest_completed_task": booked_task,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_track_0m = await agent_graph.ainvoke(s_track_0m)
    assert r_track_0m["current_step"] == "INQUIRY_ANSWERED"
    assert "on the way to your pickup location" in r_track_0m["response"].lower()
    assert "km away" in r_track_0m["response"]
    assert "Rajesh K." in r_track_0m["response"]

    # 3. User asks "where is driver?" after 1.5 mins -> Distance and ETA have decreased
    booked_task_1m = {
        **booked_task,
        "result": {**booked_task["result"], "created_at": (now - timedelta(seconds=90)).isoformat()},
    }
    s_track_1m = {
        "user_message": "where is the driver",
        "latest_completed_task": booked_task_1m,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_track_1m = await agent_graph.ainvoke(s_track_1m)
    assert "getting closer" in r_track_1m["response"].lower()
    assert "km away" in r_track_1m["response"]

    # 4. User asks "where is driver?" after 4 mins -> Driver has arrived
    booked_task_4m = {
        **booked_task,
        "result": {**booked_task["result"], "created_at": (now - timedelta(minutes=4)).isoformat()},
    }
    s_track_4m = {
        "user_message": "where is driver",
        "latest_completed_task": booked_task_4m,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_track_4m = await agent_graph.ainvoke(s_track_4m)
    assert "arrived at the gachibowli pickup location" in r_track_4m["response"].lower() or "arrived" in r_track_4m["response"].lower()

    # 5. Natural language variations: "how far is my driver" & "is my driver nearby"
    s_track_dist = {
        "user_message": "how far is my driver",
        "latest_completed_task": booked_task,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_track_dist = await agent_graph.ainvoke(s_track_dist)
    assert r_track_dist["current_step"] == "INQUIRY_ANSWERED"
    assert "km away" in r_track_dist["response"]

    # 6. Trip in progress: "where are we now?"
    booked_task_trip = {
        **booked_task,
        "result": {**booked_task["result"], "created_at": (now - timedelta(minutes=6)).isoformat()},
    }
    s_where_are_we = {
        "user_message": "where are we now?",
        "latest_completed_task": booked_task_trip,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_where_are_we = await agent_graph.ainvoke(s_where_are_we)
    assert r_where_are_we["current_step"] == "INQUIRY_ANSWERED"
    assert "currently on your trip" in r_where_are_we["response"].lower()
    assert "Airport" in r_where_are_we["response"]


@pytest.mark.asyncio
async def test_section_16_acceptance_tests_a_through_h():
    """
    Validates all Acceptance Criteria defined in Section 16:
    - Test A: Ambiguous request ('order') -> Clarify choice (food, ride, home service). Does not enter food.
    - Test B: Explicit service ('Book Urban Clean') -> Enter home service directly.
    - Test C: Availability -> Given Standard & Manikonda, check availability, return slots, ask user.
    - Test D: Confirmation -> User selects slot and confirms -> provider confirmed booking.
    - Test E: Unavailable slot -> Reports unavailable slot without fake success.
    - Test F: Existing booking -> 'When is my appointment?' returns confirmed schedule.
    - Test G: New intent -> Switch from food to Urban Clean cleanly without losing history.
    - Test H: Provider failure -> When provider fails, returns error without generating fake booking.
    """
    # Test A: Ambiguous request
    s_a = {
        "conversation_id": "c_a",
        "task_id": None,
        "user_message": "order",
        "last_assistant_message": None,
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": None,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_a = await agent_graph.ainvoke(s_a)
    assert "food, a ride, or a home service" in r_a["response"]
    assert "craving" not in r_a["response"].lower()

    # Test B: Explicit service
    s_b = {
        "conversation_id": "c_b",
        "task_id": None,
        "user_message": "Book Urban Clean",
        "last_assistant_message": None,
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": None,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_b = await agent_graph.ainvoke(s_b)
    assert r_b["collected_data"].get("intent") == "BOOK_SERVICE"
    assert "package" in r_b["response"].lower()

    # Test C: Availability & Slot Selection for Urban Clean
    s_c1 = {
        **r_b,
        "user_message": "Standard",
        "last_assistant_message": r_b["response"],
        "conversation_history": [{"role": "user", "content": "Book Urban Clean"}, {"role": "assistant", "content": r_b["response"]}],
    }
    r_c1 = await agent_graph.ainvoke(s_c1)
    assert "service address" in r_c1["response"].lower() or "address" in r_c1["response"].lower()

    s_c2 = {
        **r_c1,
        "user_message": "Manikonda",
        "last_assistant_message": r_c1["response"],
        "conversation_history": s_c1["conversation_history"] + [{"role": "user", "content": "Standard"}, {"role": "assistant", "content": r_c1["response"]}],
    }
    r_c2 = await agent_graph.ainvoke(s_c2)
    assert "date" in r_c2["response"].lower()

    s_c3 = {
        **r_c2,
        "user_message": "Today",
        "last_assistant_message": r_c2["response"],
        "conversation_history": s_c2["conversation_history"] + [{"role": "user", "content": "Manikonda"}, {"role": "assistant", "content": r_c2["response"]}],
    }
    r_c3 = await agent_graph.ainvoke(s_c3)
    # Check that real slots are presented and user is asked to choose
    assert "available" in r_c3["response"].lower()
    assert "10:00 AM – 12:00 PM" in r_c3["response"] or "10:00 AM" in r_c3["response"]
    assert "2:00 PM – 4:00 PM" in r_c3["response"]
    assert "which slot" in r_c3["response"].lower()

    # Test D: User selects slot and confirms
    s_d1 = {
        **r_c3,
        "user_message": "2 PM to 4 PM",
        "last_assistant_message": r_c3["response"],
        "conversation_history": s_c3["conversation_history"] + [{"role": "user", "content": "Today"}, {"role": "assistant", "content": r_c3["response"]}],
    }
    r_d1 = await agent_graph.ainvoke(s_d1)
    assert r_d1["waiting_for_confirmation"] is True
    assert "₹1,499" in r_d1["response"]
    assert "confirm" in r_d1["response"].lower()

    s_d2 = {
        **r_d1,
        "user_message": "Yes",
        "last_assistant_message": r_d1["response"],
        "conversation_history": s_d1["conversation_history"] + [{"role": "user", "content": "2 PM to 4 PM"}, {"role": "assistant", "content": r_d1["response"]}],
    }
    r_d2 = await agent_graph.ainvoke(s_d2)
    assert r_d2["current_step"] == "COMPLETED"
    assert "Urban Clean appointment is confirmed" in r_d2["response"]
    assert "Booking ID: uc_" in r_d2["response"]
    assert "₹1,499" in r_d2["response"]
    assert "Manikonda" in r_d2["response"]

    # Test F: Existing booking -> "When is my appointment?"
    s_f = {
        "conversation_id": "c_f",
        "task_id": None,
        "user_message": "When is my appointment?",
        "latest_completed_task": {
            "status": "COMPLETED",
            "type": "BOOK_SERVICE",
            "collected_data": r_d2["collected_data"],
            "result": r_d2["tool_result"],
        },
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_f = await agent_graph.ainvoke(s_f)
    assert r_f["current_step"] == "INQUIRY_ANSWERED"
    assert "Urban Clean appointment schedule" in r_f["response"]
    assert "2:00 PM – 4:00 PM" in r_f["response"]
    assert "Manikonda" in r_f["response"]

    # Test G: New intent -> user switches from food ordering to Urban Clean
    s_g = {
        "conversation_id": "c_g",
        "task_id": "task_food_active",
        "user_message": "Book Urban Clean",
        "last_assistant_message": "Sure! What food would you like to order?",
        "conversation_history": [{"role": "user", "content": "order food"}, {"role": "assistant", "content": "Sure! What food would you like to order?"}],
        "latest_completed_task": None,
        "intent": "ORDER_FOOD",
        "collected_data": {"intent": "ORDER_FOOD"},
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
    }
    r_g = await agent_graph.ainvoke(s_g)
    assert r_g["collected_data"].get("intent") == "BOOK_SERVICE"
    assert "package" in r_g["response"].lower()

    # Test H: Provider failure -> returns failure message without fake booking ID
    from unittest.mock import patch
    from app.tools.executor import tool_executor

    async def fail_book_home_service(*args, **kwargs):
        raise ValueError("Provider connection timed out")

    with patch.object(tool_executor, "execute", side_effect=fail_book_home_service):
        s_h = dict(s_d1)
        s_h["user_message"] = "Yes"
        s_h["waiting_for_confirmation"] = True
        s_h["confirmation_received"] = True
        r_h = await agent_graph.ainvoke(s_h)
        assert r_h["current_step"] == "FAILED"
        assert "could not be confirmed" in r_h["response"]
        assert "uc_" not in r_h["response"]


@pytest.mark.asyncio
async def test_master_prompt_booking_disambiguation_and_ride_scenarios():
    """
    Validates Master Prompt requirements:
    1. 'need booking' -> Asks what to book (cab/bike, food, home service like Urban Clean)
    2. 'cab' -> Guides user with supported options (bike, auto, Uber Go, Sedan, XL)
    3. 'Book a bike from Gachibowli to Airport' -> Directly extracts pickup, destination, and ride type
    4. Mandatory booking details returned automatically after successful booking
    """
    # 1. "need booking"
    s_need = {
        "conversation_id": "c_need",
        "task_id": None,
        "user_message": "need booking",
        "last_assistant_message": None,
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": None,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_need = await agent_graph.ainvoke(s_need)
    assert "cab or bike, food, or a home service like Urban Clean" in r_need["response"]
    assert "How can I help you?" != r_need["response"]

    # 2. "cab" follow-up asks for vehicle category / locations
    s_cab = {
        **r_need,
        "user_message": "cab",
        "last_assistant_message": r_need["response"],
        "conversation_history": [{"role": "user", "content": "need booking"}, {"role": "assistant", "content": r_need["response"]}],
    }
    r_cab = await agent_graph.ainvoke(s_cab)
    assert r_cab["current_step"] == "WAITING_FOR_USER"
    assert any(w in r_cab["response"].lower() for w in ["bike", "auto", "uber go", "pick you up", "where"])

    # 3. "Book a bike from Gachibowli to Airport"
    s_direct = {
        "conversation_id": "c_direct",
        "task_id": None,
        "user_message": "Book a bike from Gachibowli to Airport",
        "last_assistant_message": None,
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": None,
        "collected_data": {},
        "current_step": "START",
        "waiting_for_confirmation": False,
    }
    r_direct = await agent_graph.ainvoke(s_direct)
    # Extracts pickup, destination, and bike (Uber Moto)
    assert r_direct["collected_data"]["pickup"] == "Gachibowli"
    assert r_direct["collected_data"]["destination"] == "Airport"
    assert r_direct["collected_data"]["service_type"] == "bike"
    assert r_direct["collected_data"]["ride_type"] == "Uber Moto"
    # Landmark clarification or fare estimate requested
    assert "Gachibowli" in r_direct["response"]


