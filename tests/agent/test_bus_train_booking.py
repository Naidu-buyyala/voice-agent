import pytest
from datetime import datetime, timedelta
from app.agent.extractor import GeminiExtractor
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.providers.mock.bus import MockBusProvider
from app.providers.mock.train import MockTrainProvider


@pytest.fixture
def extractor():
    return GeminiExtractor()


@pytest.mark.asyncio
async def test_scenario_1_bus_request_prompts_date(extractor: GeminiExtractor):
    """Scenario 1: 'Book a bus from Hyderabad to Vizag' extracts route and prompts for date."""
    res = await extractor.extract("Book a bus from Hyderabad to Vizag")
    assert res.intent == "BOOK_BUS"
    assert res.entities.origin_city == "Hyderabad"
    assert res.entities.destination_city == "Visakhapatnam"
    assert "travel_date" in res.missing_required_fields


@pytest.mark.asyncio
async def test_scenario_2_user_supplies_tomorrow_dynamic_runtime():
    """Scenario 2: User supplies 'tomorrow'; resolved at runtime to tomorrow's date."""
    now = datetime.now()
    expected_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    
    resolved_date, window = GeminiExtractor.resolve_travel_date_and_time("tomorrow")
    assert resolved_date == expected_date


@pytest.mark.asyncio
async def test_scenario_3_tomorrow_evening_preserves_time_window():
    """Scenario 3: User says 'tomorrow evening'; date and preferred time window are preserved."""
    now = datetime.now()
    expected_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    
    date_val, time_window = GeminiExtractor.resolve_travel_date_and_time("tomorrow evening")
    assert date_val == expected_date
    assert time_window is not None
    assert "evening" in time_window.lower() or "6 pm" in time_window.lower()


@pytest.mark.asyncio
async def test_scenario_4_passenger_count_in_later_turn(extractor: GeminiExtractor):
    """Scenario 4: User provides passenger count in a later turn."""
    res = await extractor.extract(
        "Two adults",
        current_collected={
            "intent": "BOOK_BUS",
            "origin_city": "Hyderabad",
            "destination_city": "Visakhapatnam",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "last_asked": "passenger_count"
        }
    )
    assert res.entities.passenger_count == 2


@pytest.mark.asyncio
async def test_scenario_5_user_selects_bus_from_results(extractor: GeminiExtractor):
    """Scenario 5: User selects a bus from search results ('The first one')."""
    res = await extractor.extract(
        "The first one",
        current_collected={
            "intent": "BOOK_BUS",
            "origin_city": "Hyderabad",
            "destination_city": "Visakhapatnam",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "passenger_count": 2,
            "last_asked": "selected_service_id"
        }
    )
    assert res.preferences.selection_preference == "AI_CHOOSE" or res.entities.selected_service_id is not None


@pytest.mark.asyncio
async def test_scenario_6_seat_selection_supported():
    """Scenario 6: User selects seats when seat selection is supported."""
    bus_provider = MockBusProvider()
    seat_map = await bus_provider.get_seat_layout("bus_orange_01")
    assert "available_seats" in seat_map
    assert "L1" in seat_map["available_seats"]
    assert "U1" in seat_map["available_seats"]


@pytest.mark.asyncio
async def test_scenario_7_user_selects_train_and_class(extractor: GeminiExtractor):
    """Scenario 7: User selects a train and class ('Godavari Express in 3A')."""
    res = await extractor.extract(
        "Godavari Express in 3A",
        current_collected={
            "intent": "BOOK_TRAIN",
            "origin_city": "Secunderabad Junction (SC)",
            "destination_city": "Visakhapatnam Junction (VSKP)",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "passenger_count": 1,
            "last_asked": "selected_service_id"
        }
    )
    assert res.intent == "BOOK_TRAIN"
    assert res.entities.train_class == "3A"


@pytest.mark.asyncio
async def test_scenario_8_ask_only_for_missing_details(extractor: GeminiExtractor):
    """Scenario 8: The agent asks only for missing details."""
    res = await extractor.extract(
        "Book a train from Secunderabad to Tirupati on next Monday for 2 passengers",
    )
    assert res.intent == "BOOK_TRAIN"
    assert res.entities.origin_city == "Secunderabad Junction (SC)"
    assert res.entities.destination_city == "Tirupati (TPTY)"
    assert res.entities.passenger_count == 2
    assert res.entities.travel_date is not None
    # Origin and destination are not in missing fields
    assert "origin_city" not in res.missing_required_fields
    assert "destination_city" not in res.missing_required_fields


@pytest.mark.asyncio
async def test_scenario_9_provider_schedules_deterministic_not_invented():
    """Scenario 9: Provider does not invent buses/trains; returns deterministic schedules."""
    bus_p = MockBusProvider()
    train_p = MockTrainProvider()
    
    buses = await bus_p.search_buses("Hyderabad", "Visakhapatnam", "2026-10-10")
    assert len(buses) > 0
    assert buses[0].operator_name in ["Orange Tours & Travels", "Morning Star Travels", "APSRTC Garuda Plus", "Kaveri Travels"]
    
    trains = await train_p.search_trains("Secunderabad Junction (SC)", "Visakhapatnam Junction (VSKP)", "2026-10-10")
    assert len(trains) > 0
    assert trains[0].train_number in ["12728", "12738"]


@pytest.mark.asyncio
async def test_scenario_10_bus_seat_unavailable_validation():
    """Scenario 10: Seats or berths become unavailable validation."""
    bus_p = MockBusProvider()
    bus_p.unavailable_seats.append("L4")
    # Try booking unavailable seat L4
    with pytest.raises(ValueError, match="no longer available"):
        await bus_p.book_bus(
            service_id="bus_orange_01",
            origin="Hyderabad",
            destination="Visakhapatnam",
            travel_date="2026-10-10",
            passenger_count=1,
            selected_seats=["L4"],
            boarding_point="Ameerpet",
            dropping_point="RTC Complex",
        )


@pytest.mark.asyncio
async def test_scenario_11_user_confirms_bus_booking():
    """Scenario 11: The user confirms the booking; provider returns confirmed status and SMS notice."""
    bus_p = MockBusProvider()
    result = await bus_p.book_bus(
        service_id="bus_orange_01",
        origin="Hyderabad",
        destination="Visakhapatnam",
        travel_date="2026-10-10",
        passenger_count=1,
        selected_seats=["U1"],
        boarding_point="Ameerpet",
        dropping_point="RTC Complex",
        idempotency_key="bus_test_idem_11"
    )
    assert result.booking_status == "CONFIRMED"
    assert result.booking_id.startswith("bus_")
    assert result.total_fare > 0
    assert "SMS" in result.sms_message


@pytest.mark.asyncio
async def test_scenario_12_user_rejects_booking(extractor: GeminiExtractor):
    """Scenario 12: The user rejects the booking ('No, cancel')."""
    res = await extractor.extract(
        "No, cancel it",
        current_collected={
            "intent": "BOOK_BUS",
            "pending_confirmation": True
        }
    )
    assert res.confirmation is False


@pytest.mark.asyncio
async def test_scenario_13_repeated_confirmation_idempotency():
    """Scenario 13: Repeated confirmation does not create duplicate tickets."""
    bus_p = MockBusProvider()
    
    res1 = await bus_p.book_bus(
        service_id="bus_orange_01",
        origin="Hyderabad",
        destination="Visakhapatnam",
        travel_date="2026-10-10",
        passenger_count=1,
        selected_seats=["U2"],
        boarding_point="Ameerpet",
        dropping_point="RTC Complex",
        idempotency_key="bus_idem_scenario_13"
    )
    
    res2 = await bus_p.book_bus(
        service_id="bus_orange_01",
        origin="Hyderabad",
        destination="Visakhapatnam",
        travel_date="2026-10-10",
        passenger_count=1,
        selected_seats=["U2"],
        boarding_point="Ameerpet",
        dropping_point="RTC Complex",
        idempotency_key="bus_idem_scenario_13"
    )
    assert res1.booking_id == res2.booking_id


@pytest.mark.asyncio
async def test_scenario_14_train_booking_confirmed_with_pnr():
    """Scenario 14: Train booking returns valid 10-digit IRCTC PNR with coach and berth."""
    train_p = MockTrainProvider()
    
    result = await train_p.book_train(
        train_number="12728",
        origin_station="Secunderabad Junction (SC)",
        destination_station="Visakhapatnam Junction (VSKP)",
        travel_date="2026-10-10",
        selected_class="3A",
        passenger_count=1,
        passenger_names=["Suresh V"],
        berth_preference="Lower"
    )
    assert result.booking_status == "CONFIRMED"
    assert len(result.pnr) == 10
    assert result.pnr.isdigit()
    assert len(result.allocated_berths) == 1
    assert "B3" in result.allocated_berths[0]


@pytest.mark.asyncio
async def test_scenario_15_provider_cancellation_and_refund():
    """Scenario 15: Provider cancels booking and returns refund details."""
    bus_p = MockBusProvider()
    b_res = await bus_p.book_bus(
        service_id="bus_orange_01",
        origin="Hyderabad",
        destination="Visakhapatnam",
        travel_date="2026-10-10",
        passenger_count=1,
        selected_seats=["U3"],
        boarding_point="Ameerpet",
        dropping_point="RTC Complex",
    )
    
    cancel_res = await bus_p.cancel_bus_booking(b_res.booking_id)
    assert cancel_res["status"] == "CANCELLED"
    assert cancel_res["refund_amount"] > 0
    assert "SMS" in cancel_res["message"]


@pytest.mark.asyncio
async def test_scenario_16_train_cancellation_clerkage_fee():
    """Scenario 16: Train cancellation deducts IRCTC clerkage fee."""
    train_p = MockTrainProvider()
    t_res = await train_p.book_train(
        train_number="12728",
        origin_station="Secunderabad Junction (SC)",
        destination_station="Visakhapatnam Junction (VSKP)",
        travel_date="2026-10-10",
        selected_class="3A",
        passenger_count=1,
    )
    
    c_res = await train_p.cancel_train_booking(t_res.booking_id)
    assert c_res["status"] == "CANCELLED"
    assert c_res["refund_amount"] == 1150.0 - 60.0


@pytest.mark.asyncio
async def test_scenario_17_station_distinction_sc_vs_hyb():
    """Scenario 17: Secunderabad (SC) and Hyderabad Deccan (HYB) are distinct stations."""
    train_p = MockTrainProvider()
    sc_station = train_p.canonicalize_station("Secunderabad")
    hyb_station = train_p.canonicalize_station("Hyderabad")
    assert sc_station != hyb_station
    assert "SC" in sc_station
    assert "HYB" in hyb_station


@pytest.mark.asyncio
async def test_scenario_18_context_switching_bus_to_train(extractor: GeminiExtractor):
    """Scenario 18: Switching from bus booking to train booking preserves route."""
    res = await extractor.extract(
        "Actually, book a train instead",
        current_collected={
            "intent": "BOOK_BUS",
            "origin_city": "Hyderabad",
            "destination_city": "Visakhapatnam",
            "travel_date": "2026-10-10",
            "passenger_count": 2
        }
    )
    assert res.intent == "BOOK_TRAIN"
    # Preserves origin, destination, travel_date
    assert res.entities.origin_city == "Hyderabad"
    assert res.entities.destination_city == "Visakhapatnam"


@pytest.mark.asyncio
async def test_scenario_19_inquiry_what_is_my_pnr(extractor: GeminiExtractor):
    """Scenario 19: Asking 'What is my PNR?' routes to TASK_INQUIRY."""
    res = await extractor.extract("What is my PNR?")
    assert res.intent in ["TASK_INQUIRY", "INQUIRY"]


@pytest.mark.asyncio
async def test_scenario_20_inquiry_cancel_my_ticket(extractor: GeminiExtractor):
    """Scenario 20: Asking 'Cancel my ticket' routes to TASK_INQUIRY."""
    res = await extractor.extract("Cancel my ticket")
    assert res.intent in ["TASK_INQUIRY", "INQUIRY"]


@pytest.mark.asyncio
async def test_scenario_21_train_class_unavailable_validation():
    """Scenario 21: Unavailable train class raises validation error."""
    train_p = MockTrainProvider()
    train_p.unavailable_classes.append("1A")
    with pytest.raises(ValueError, match="not available"):
        await train_p.book_train(
            train_number="12728",
            origin_station="Secunderabad Junction (SC)",
            destination_station="Visakhapatnam Junction (VSKP)",
            travel_date="2026-10-10",
            selected_class="1A",
            passenger_count=1,
        )


@pytest.mark.asyncio
async def test_scenario_22_end_to_end_graph_bus_booking():
    """Scenario 22: Graph execution produces booking summary and SMS message."""
    confirm_state: AgentState = {
        "conversation_id": "test_bus_e2e_session",
        "task_id": None,
        "user_message": "book it",
        "last_assistant_message": "Shall I confirm this bus booking for you?",
        "conversation_history": [
            {"role": "user", "content": "Book a bus from Hyderabad to Vizag for tomorrow"},
            {"role": "assistant", "content": "Shall I confirm this bus booking for you?"}
        ],
        "latest_completed_task": None,
        "intent": "BOOK_BUS",
        "collected_data": {
            "intent": "BOOK_BUS",
            "origin_city": "Hyderabad",
            "destination_city": "Visakhapatnam",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "passenger_count": 1,
            "selected_service_id": "bus_orange_01",
            "selected_seats": ["U1"],
            "boarding_point": "Ameerpet",
            "dropping_point": "RTC Complex",
            "fare_amount": 997.5,
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
        "confirmation_received": True,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    
    final_output = await agent_graph.ainvoke(confirm_state)
    assert final_output["current_step"] in ["COMPLETED", "BOOKING_CONFIRMED"]
    assert "booking_id" in final_output["tool_result"]
    # Check mandatory SMS notice requirement
    last_msg = final_output["response"]
    assert "SMS" in last_msg
    assert "booking details" in last_msg.lower()


@pytest.mark.asyncio
async def test_scenario_23_end_to_end_graph_train_booking():
    """Scenario 23: Graph execution for train booking produces 10-digit PNR and SMS message."""
    confirm_state: AgentState = {
        "conversation_id": "test_train_e2e_session",
        "task_id": None,
        "user_message": "confirm and book train",
        "last_assistant_message": "Shall I go ahead and book this ticket for you?",
        "conversation_history": [
            {"role": "user", "content": "Book a train from Secunderabad to Vizag"},
            {"role": "assistant", "content": "Shall I go ahead and book this ticket for you?"}
        ],
        "latest_completed_task": None,
        "intent": "BOOK_TRAIN",
        "collected_data": {
            "intent": "BOOK_TRAIN",
            "origin_city": "Secunderabad Junction (SC)",
            "destination_city": "Visakhapatnam Junction (VSKP)",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "passenger_count": 1,
            "selected_service_id": "12728",
            "train_class": "3A",
            "boarding_point": "Secunderabad Junction (SC)",
            "dropping_point": "Visakhapatnam Junction (VSKP)",
            "fare_amount": 1150.0,
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
        "confirmation_received": True,
        "tool_name": None,
        "tool_input": None,
        "tool_result": None,
        "response": None,
        "error": None,
    }
    
    final_output = await agent_graph.ainvoke(confirm_state)
    assert final_output["current_step"] in ["COMPLETED", "BOOKING_CONFIRMED"]
    assert "booking_id" in final_output["tool_result"]
    assert "pnr" in final_output["tool_result"]
    last_msg = final_output["response"]
    assert "SMS" in last_msg
    assert "PNR" in last_msg


@pytest.mark.asyncio
async def test_scenario_24_travel_booking_inquiry_and_pnr_status():
    """Scenario 24: Inquiry for active travel booking retrieves details and includes SMS notice."""
    inquiry_state: AgentState = {
        "conversation_id": "test_train_inquiry_session",
        "task_id": None,
        "user_message": "What is my PNR status?",
        "last_assistant_message": "Your booking is confirmed.",
        "conversation_history": [],
        "latest_completed_task": None,
        "intent": "TASK_INQUIRY",
        "collected_data": {
            "intent": "BOOK_TRAIN",
            "booking_id": "train_sc_vskp_123",
            "pnr": "4829104812",
            "train_number": "12728",
            "service_name": "Godavari Express (12728)",
            "travel_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "origin_city": "Secunderabad Junction (SC)",
            "destination_city": "Visakhapatnam Junction (VSKP)",
            "train_class": "3A",
            "passenger_count": 1,
            "berths": ["B1-24 (Lower Berth)"]
        },
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
    
    final_output = await agent_graph.ainvoke(inquiry_state)
    last_msg = final_output["response"]
    assert "4829104812" in last_msg
    assert "Godavari Express" in last_msg
    assert "SMS" in last_msg
