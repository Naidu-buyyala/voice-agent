import pytest
from datetime import datetime, timedelta
from app.agent.extractor import GeminiExtractor
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.providers.mock.bus import MockBusProvider


@pytest.fixture
def extractor():
    return GeminiExtractor()


@pytest.mark.asyncio
async def test_bus_booking_intent_detection_phrases(extractor: GeminiExtractor):
    """
    1. 'bus booking' triggers the bus workflow.
    Phrases must map to BOOK_BUS:
      - bus booking
      - book a bus
      - I need a bus ticket
      - I want to travel by bus
      - bus from Hyderabad to Vizag
      - book a bus ticket for tomorrow
      - find buses to Bengaluru
      - sleeper bus to Chennai
    """
    phrases = [
        "bus",
        "bus booking",
        "book a bus",
        "I need a bus ticket",
        "I want to travel by bus",
        "bus from Hyderabad to Vizag",
        "book a bus ticket for tomorrow",
        "find buses to Bengaluru",
        "sleeper bus to Chennai"
    ]
    for phrase in phrases:
        ext = await extractor.extract(phrase)
        assert ext.intent == "BOOK_BUS", f"Failed for phrase: '{phrase}' (got {ext.intent})"


@pytest.mark.asyncio
async def test_single_word_bus_prompt():
    """
    User: 'bus'
    Assistant: 'Sure! I can help you book a bus. Where would you like to travel from and to?'
    """
    state: AgentState = {
        "user_message": "bus",
        "collected_data": {},
        "current_step": "START",
        "task_id": "test_single_bus_1",
    }
    result = await agent_graph.ainvoke(state)
    assert result.get("intent") == "BOOK_BUS"
    assert "where would you like to travel from and to" in result.get("response", "").lower()
    assert "don't support" not in result.get("response", "").lower()


@pytest.mark.asyncio
async def test_bus_request_does_not_trigger_uber_go():
    """
    2. 'Book a bus from Hyderabad to Vizag' does not trigger Uber Go.
    A bus request must never silently fall back to Uber or a cab.
    """
    state: AgentState = {
        "user_message": "Book a bus from Hyderabad to Vizag",
        "collected_data": {},
        "current_step": "START",
        "task_id": "test_bus_routing_1",
    }
    result = await agent_graph.ainvoke(state)
    collected = result.get("collected_data", {})
    response = result.get("response", "")

    assert result.get("intent") == "BOOK_BUS"
    assert collected.get("travel_mode") == "bus"
    assert "Uber" not in response
    assert "Uber Go" not in response
    assert "cab" not in response.lower()
    assert collected.get("ride_type") is None
    # Must ask for missing travel date
    assert "date" in response.lower()


@pytest.mark.asyncio
async def test_progressive_inquiry_flow():
    """
    3. Missing date asked progressively.
    4. Missing departure-time asked progressively.
    5. 'Tomorrow' resolved dynamically.
    6. Specific date and time upfront not asked again.
    7. Bus search calls bus provider, not cab provider.
    10. Booking requires explicit confirmation.
    11. Repeated confirmations do not create duplicate bookings.
    """
    # Turn 1: User says "bus booking"
    s1: AgentState = {
        "user_message": "bus booking",
        "collected_data": {},
        "current_step": "START",
        "task_id": "prog_bus_turn_1",
    }
    r1 = await agent_graph.ainvoke(s1)
    assert r1.get("intent") == "BOOK_BUS"
    assert "where would you like to travel from and to" in r1.get("response", "").lower()

    # Turn 2: User says "Hyderabad to Vizag"
    s2: AgentState = {
        "user_message": "Hyderabad to Vizag",
        "collected_data": r1["collected_data"],
        "current_step": r1["current_step"],
        "last_assistant_message": r1["response"],
        "task_id": "prog_bus_turn_2",
    }
    r2 = await agent_graph.ainvoke(s2)
    assert r2["collected_data"].get("origin_city") == "Hyderabad"
    assert r2["collected_data"].get("destination_city") == "Visakhapatnam"
    assert "what date would you like to travel" in r2.get("response", "").lower()

    # Turn 3: User says "Tomorrow"
    s3: AgentState = {
        "user_message": "Tomorrow",
        "collected_data": r2["collected_data"],
        "current_step": r2["current_step"],
        "last_assistant_message": r2["response"],
        "task_id": "prog_bus_turn_3",
    }
    r3 = await agent_graph.ainvoke(s3)
    expected_tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    assert r3["collected_data"].get("travel_date") == expected_tomorrow
    assert "what time would you prefer to depart" in r3.get("response", "").lower()

    # Turn 4: User says "Around 6 PM"
    s4: AgentState = {
        "user_message": "Around 6 PM",
        "collected_data": r3["collected_data"],
        "current_step": r3["current_step"],
        "last_assistant_message": r3["response"],
        "task_id": "prog_bus_turn_4",
    }
    r4 = await agent_graph.ainvoke(s4)
    assert "passenger" in r4.get("response", "").lower()

    # Turn 5: User says "Two adults"
    s5: AgentState = {
        "user_message": "Two adults",
        "collected_data": r4["collected_data"],
        "current_step": r4["current_step"],
        "last_assistant_message": r4["response"],
        "task_id": "prog_bus_turn_5",
    }
    r5 = await agent_graph.ainvoke(s5)
    # Search buses executed: lists bus options
    assert r5["collected_data"].get("passenger_count") == 2
    assert "orange tours" in r5.get("response", "").lower() or "which bus" in r5.get("response", "").lower()

    # Turn 6: User selects bus
    s6: AgentState = {
        "user_message": "the first one",
        "collected_data": r5["collected_data"],
        "current_step": r5["current_step"],
        "last_assistant_message": r5["response"],
        "task_id": "prog_bus_turn_6",
    }
    r6 = await agent_graph.ainvoke(s6)
    assert "seats" in r6.get("response", "").lower()

    # Turn 7: User selects seats
    s7: AgentState = {
        "user_message": "U1, U2",
        "collected_data": r6["collected_data"],
        "current_step": r6["current_step"],
        "last_assistant_message": r6["response"],
        "task_id": "prog_bus_turn_7",
    }
    r7 = await agent_graph.ainvoke(s7)
    # Confirmation prompt
    assert r7["waiting_for_confirmation"] is True
    assert "shall i confirm this bus booking" in r7.get("response", "").lower()
    assert "₹" in r7.get("response", "")

    # Turn 8: Explicit confirmation
    s8: AgentState = {
        "user_message": "confirm",
        "collected_data": r7["collected_data"],
        "current_step": r7["current_step"],
        "waiting_for_confirmation": True,
        "task_id": "prog_bus_turn_8",
    }
    r8 = await agent_graph.ainvoke(s8)
    assert r8["current_step"] == "COMPLETED"
    assert "confirmed successfully" in r8.get("response", "").lower()
    assert "PNR:" in r8.get("response", "")
    pnr = r8["tool_result"].get("pnr")
    assert pnr is not None

    # Turn 9: Repeated confirmation (Idempotency check)
    s9: AgentState = {
        "user_message": "confirm",
        "collected_data": r8["collected_data"],
        "current_step": "COMPLETED",
        "latest_completed_task": {"status": "COMPLETED", "result": r8["tool_result"], "type": "BOOK_BUS"},
        "tool_result": r8["tool_result"],
        "task_id": "prog_bus_turn_8",
    }
    r9 = await agent_graph.ainvoke(s9)
    assert "already confirmed" in r9.get("response", "").lower()
    assert pnr in r9.get("response", "")


@pytest.mark.asyncio
async def test_upfront_full_details_do_not_re_ask(extractor: GeminiExtractor):
    """
    6. A specific date and time supplied upfront are not asked for again.
    """
    msg = "Book a bus from Hyderabad to Vizag tomorrow at 6 PM for two adults"
    ext = await extractor.extract(msg)
    assert ext.intent == "BOOK_BUS"
    assert ext.entities.origin_city == "Hyderabad"
    assert ext.entities.destination_city == "Visakhapatnam"
    assert ext.entities.travel_date is not None
    assert ext.entities.departure_window is not None
    assert ext.entities.passenger_count == 2
    # missing_required_fields should NOT contain origin, destination, date, window, or passenger count
    for f in ["origin_city", "destination_city", "travel_date", "departure_window", "passenger_count"]:
        assert f not in ext.missing_required_fields


@pytest.mark.asyncio
async def test_provider_results_display_correct_service_type_and_price():
    """
    9. Provider results display the correct service type and price.
    """
    provider = MockBusProvider()
    results = await provider.search_buses("Hyderabad", "Visakhapatnam", "2026-10-04")
    assert len(results) >= 2
    for b in results:
        assert "Hyderabad" in b.origin
        assert "Visakhapatnam" in b.destination
        assert b.total_fare > 0
        assert b.operator_name is not None
        assert b.bus_type is not None


@pytest.mark.asyncio
async def test_existing_services_continue_to_work():
    """
    13. Existing cab, food, and home-service flows continue to work.
    """
    # 1. Cab flow
    cab_state: AgentState = {
        "user_message": "Book a cab from Hyderabad to the airport",
        "collected_data": {},
        "current_step": "START",
        "task_id": "cab_test_1",
    }
    cab_res = await agent_graph.ainvoke(cab_state)
    assert cab_res.get("intent") == "BOOK_RIDE"
    assert "bus" not in cab_res.get("response", "").lower()

    # 2. Food flow
    food_state: AgentState = {
        "user_message": "Order biryani from Meghana Foods to Manikonda",
        "collected_data": {},
        "current_step": "START",
        "task_id": "food_test_1",
    }
    food_res = await agent_graph.ainvoke(food_state)
    assert food_res.get("intent") == "ORDER_FOOD"
    assert "biryani" in food_res.get("response", "").lower()

    # 3. Home service flow
    svc_state: AgentState = {
        "user_message": "Book deep cleaning at Kondapur tomorrow",
        "collected_data": {},
        "current_step": "START",
        "task_id": "svc_test_1",
    }
    svc_res = await agent_graph.ainvoke(svc_state)
    assert svc_res.get("intent") == "BOOK_SERVICE"
