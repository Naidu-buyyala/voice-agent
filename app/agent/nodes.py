import uuid
from datetime import datetime, timezone
from typing import Any
from app.agent.state import AgentState
from app.agent.extractor import GeminiExtractor
from app.tools.executor import tool_executor

extractor = GeminiExtractor()


def calculate_dynamic_food_status(created_at_val: Any, initial_eta: int = 20) -> tuple[str, str, int]:
    """
    Calculates dynamic food delivery status and remaining ETA based on elapsed time.
    Lifecycle:
      - 0 to 5 mins: PREPARING (restaurant is preparing meal)
      - 5 to 10 mins: READY_FOR_PICKUP (almost ready for pickup)
      - 10 to 15 mins: OUT_FOR_DELIVERY (delivery partner picked up and on the way)
      - 15 to 20 mins: ARRIVING (arriving in ~5 mins)
      - > 20 mins: DELIVERED (delivered successfully)
    Returns: (status_code, status_description, remaining_eta_mins)
    """
    if not created_at_val:
        return "PREPARING", "Your order is currently being prepared at the restaurant.", initial_eta

    try:
        if isinstance(created_at_val, str):
            # Parse ISO string
            created_dt = datetime.fromisoformat(created_at_val)
        elif isinstance(created_at_val, datetime):
            created_dt = created_at_val
        else:
            return "PREPARING", "Your order is currently being prepared at the restaurant.", initial_eta

        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed_seconds = max(0.0, (now - created_dt).total_seconds())
        elapsed_mins = int(elapsed_seconds / 60)
        remaining_eta = max(0, initial_eta - elapsed_mins)

        if elapsed_mins < 5:
            return "PREPARING", "Your order is currently being prepared at the restaurant.", remaining_eta
        elif elapsed_mins < 10:
            return "READY_FOR_PICKUP", "Your meal is being prepared and is almost ready for pickup.", remaining_eta
        elif elapsed_mins < 15:
            return "OUT_FOR_DELIVERY", "Your order has been picked up and is now on the way to you.", remaining_eta
        elif elapsed_mins < 20:
            return "ARRIVING", "Your order is on the way and should reach you in about 5 minutes.", remaining_eta
        else:
            return "DELIVERED", "Your order has been delivered successfully.", 0
    except Exception:
        return "PREPARING", "Your order is currently being prepared at the restaurant.", initial_eta


def calculate_dynamic_ride_status(created_at_val: Any, initial_eta: int = 4, initial_dist: float = 3.5, pickup_loc: str = "your pickup location") -> tuple[str, str, int, float, str]:
    """
    Calculates dynamic ride tracking status, remaining ETA, distance (km), and approximate driver landmark.
    Lifecycle:
      - 0 to 1 min (3.5 km -> ~2.8 km): DRIVER_EN_ROUTE ("Near Gachibowli Flyover / Outer Ring Road")
      - 1 to 2 mins (~2.5 km -> ~1.8 km): DRIVER_CLOSER ("Approaching main junction")
      - 2 to 3.5 mins (~1.2 km -> ~0.4 km): DRIVER_NEARBY ("Just turning into your street")
      - 3.5 to 5 mins (0 km): DRIVER_ARRIVED ("Arrived at pickup location")
      - 5 to 15 mins: TRIP_IN_PROGRESS ("On trip to destination")
      - > 15 mins: TRIP_COMPLETED ("Trip completed")
    Returns: (status_code, status_description, remaining_eta_mins, distance_km, driver_landmark)
    """
    if not created_at_val:
        return "DRIVER_EN_ROUTE", "On the way to pickup", initial_eta, initial_dist, "Near main avenue"

    try:
        if isinstance(created_at_val, str):
            created_dt = datetime.fromisoformat(created_at_val)
        elif isinstance(created_at_val, datetime):
            created_dt = created_at_val
        else:
            return "DRIVER_EN_ROUTE", "On the way to pickup", initial_eta, initial_dist, "Near main avenue"

        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed_seconds = max(0.0, (now - created_dt).total_seconds())
        elapsed_mins = int(elapsed_seconds / 60)
        remaining_eta = max(0, initial_eta - elapsed_mins)

        # Distance decreases smoothly based on elapsed time (at ~0.9 km per min)
        distance_km = max(0.0, round(initial_dist - (elapsed_seconds / 60.0) * 0.9, 1))

        if elapsed_seconds < 70:
            return "DRIVER_EN_ROUTE", "On the way to pickup", remaining_eta or 4, distance_km or 3.2, "Near Gachibowli Flyover"
        elif elapsed_seconds < 140:
            return "DRIVER_CLOSER", "Getting closer to your pickup point", remaining_eta or 3, distance_km or 2.1, "Near Botanical Garden Junction"
        elif elapsed_seconds < 210:
            return "DRIVER_NEARBY", "Driver is approaching your street", remaining_eta or 1, distance_km or 0.8, "Just turning into your lane"
        elif elapsed_seconds < 330:
            return "DRIVER_ARRIVED", f"Your driver has arrived at the {pickup_loc} pickup location", 0, 0.0, f"At {pickup_loc}"
        elif elapsed_seconds < 900:
            return "TRIP_IN_PROGRESS", "Trip is currently in progress heading to your destination", 0, 0.0, "En route to destination"
        else:
            return "TRIP_COMPLETED", "Your ride has been completed successfully", 0, 0.0, "At destination"
    except Exception:
        return "DRIVER_EN_ROUTE", "On the way to pickup", initial_eta, initial_dist, "Near main avenue"


def calculate_dynamic_service_status(created_at_val: Any) -> tuple[str, str]:
    """
    Calculates dynamic home service status based on elapsed time.
    """
    if not created_at_val:
        return "CONFIRMED", "Confirmed & Dispatched"

    try:
        if isinstance(created_at_val, str):
            created_dt = datetime.fromisoformat(created_at_val)
        elif isinstance(created_at_val, datetime):
            created_dt = created_at_val
        else:
            return "CONFIRMED", "Confirmed & Dispatched"

        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed_mins = int(max(0.0, (now - created_dt).total_seconds()) / 60)

        if elapsed_mins < 30:
            return "CONFIRMED", "Confirmed & Dispatched (team preparing equipment)"
        elif elapsed_mins < 90:
            return "ON_THE_WAY", "Professional team is on the way to your address"
        elif elapsed_mins < 180:
            return "IN_PROGRESS", "Service is currently in progress at your address"
        else:
            return "COMPLETED", "Service completed"
    except Exception:
        return "CONFIRMED", "Confirmed & Dispatched"


async def understand_intent_node(state: AgentState) -> dict[str, Any]:
    """Node 1: Extract intent, entities, preferences, delegation via Gemini."""
    current_col = dict(state.get("collected_data", {}))
    if state.get("intent"):
        current_col["intent"] = state["intent"]

    extraction = await extractor.extract(
        state["user_message"],
        current_collected=current_col,
        last_assistant_message=state.get("last_assistant_message"),
        conversation_history=state.get("conversation_history"),
        current_workflow_state=state.get("current_step"),
    )

    intent = extraction.intent
    if intent == "UNKNOWN" and state.get("intent") and state.get("intent") not in ["GREETING", "GRATITUDE_OR_CLOSING"]:
        intent = state["intent"]

    updates: dict[str, Any] = {
        "intent": intent,
        "current_step": "UNDERSTOOD",
        "confirmation_received": extraction.is_confirmed,
        "preferences": extraction.preferences.model_dump(),
        "next_action": extraction.next_action,
    }

    new_fields: dict[str, Any] = {}
    # Rides
    if extraction.pickup:
        new_fields["pickup"] = extraction.pickup
    if extraction.destination:
        new_fields["destination"] = extraction.destination
    if extraction.ride_type:
        new_fields["ride_type"] = extraction.ride_type
    if extraction.service_type:
        new_fields["service_type"] = extraction.service_type

    # Food
    if extraction.entities.food:
        new_fields["food"] = extraction.entities.food
    if extraction.entities.restaurant:
        new_fields["restaurant"] = extraction.entities.restaurant
    if extraction.entities.delivery_location:
        new_fields["delivery_location"] = extraction.entities.delivery_location

    # Home Services
    if extraction.entities.service_name:
        new_fields["service_name"] = extraction.entities.service_name
    if extraction.entities.package:
        new_fields["package"] = extraction.entities.package
    if extraction.entities.service_location:
        new_fields["service_location"] = extraction.entities.service_location
    if extraction.entities.preferred_date:
        new_fields["preferred_date"] = extraction.entities.preferred_date
    if extraction.entities.slot:
        new_fields["slot"] = extraction.entities.slot

    # Bus & Train Travel
    if extraction.entities.travel_mode:
        new_fields["travel_mode"] = extraction.entities.travel_mode
    if extraction.entities.origin_city:
        new_fields["origin_city"] = extraction.entities.origin_city
    if extraction.entities.destination_city:
        new_fields["destination_city"] = extraction.entities.destination_city
    if extraction.entities.travel_date:
        new_fields["travel_date"] = extraction.entities.travel_date
    if extraction.entities.departure_window:
        new_fields["departure_window"] = extraction.entities.departure_window
    if extraction.entities.passenger_count:
        new_fields["passenger_count"] = extraction.entities.passenger_count
    if extraction.entities.passenger_names:
        new_fields["passenger_names"] = extraction.entities.passenger_names
    if extraction.entities.bus_type:
        new_fields["bus_type"] = extraction.entities.bus_type
    if extraction.entities.train_class:
        new_fields["train_class"] = extraction.entities.train_class
    if extraction.entities.selected_service_id:
        new_fields["selected_service_id"] = extraction.entities.selected_service_id
    if extraction.entities.selected_seats:
        new_fields["selected_seats"] = extraction.entities.selected_seats
    if extraction.entities.boarding_point:
        new_fields["boarding_point"] = extraction.entities.boarding_point
    if extraction.entities.dropping_point:
        new_fields["dropping_point"] = extraction.entities.dropping_point

    if intent and intent in ["ORDER_FOOD", "BOOK_SERVICE", "BOOK_RIDE", "BOOK_BUS", "BOOK_TRAIN"]:
        new_fields["intent"] = intent

    updates["_new_fields"] = new_fields
    if extraction.assistant_message:
        updates["_llm_reply"] = extraction.assistant_message
    if extraction.needs_landmark_clarification:
        updates["_needs_landmark_clarification"] = True

    return updates


async def merge_information_node(state: AgentState) -> dict[str, Any]:
    """Node 2: Merge new extractions into persistent collected_data and resolve AI delegation."""
    collected = dict(state.get("collected_data", {}))
    new_fields = state.get("_new_fields", {})
    preferences = state.get("preferences") or {}
    intent = state.get("intent") or collected.get("intent", "BOOK_RIDE")
    collected["intent"] = intent

    # 1. Merge explicitly provided fields
    for k, v in new_fields.items():
        if v:
            # If user provided a specific landmark to refine broad locality
            if k == "pickup" and collected.get("pickup"):
                is_old_broad, _ = GeminiExtractor.is_broad_locality_only(collected["pickup"])
                if is_old_broad and collected["pickup"].lower() not in v.lower():
                    collected[k] = f"{v}, {collected['pickup']}"
                else:
                    collected[k] = v
            elif k == "destination" and collected.get("destination"):
                is_old_broad, _ = GeminiExtractor.is_broad_locality_only(collected["destination"])
                if is_old_broad and collected["destination"].lower() not in v.lower():
                    collected[k] = f"{v}, {collected['destination']}"
                else:
                    collected[k] = v
            else:
                collected[k] = v

    # 2. Handle delegation and selection strategy
    sel_pref = preferences.get("selection_preference")
    sel_strat = preferences.get("selection_strategy") or "TOP_RATED"

    if sel_pref:
        collected["selection_preference"] = sel_pref
    if sel_strat:
        collected["selection_strategy"] = sel_strat

    # Handle domain-specific delegation
    if intent == "ORDER_FOOD":
        if sel_pref == "AI_CHOOSE":
            collected["restaurant_selection"] = "AI_CHOOSE"
            if not collected.get("restaurant") or new_fields.get("restaurant"):
                if sel_strat == "CHEAPEST":
                    collected["restaurant"] = "Bawarchi Express (Budget Friendly)"
                elif sel_strat == "FASTEST":
                    collected["restaurant"] = "FastBites Cloud Kitchen (15 min delivery)"
                else:
                    collected["restaurant"] = "Meghana Foods (Top Rated 4.8/5)"
        elif new_fields.get("restaurant"):
            collected["restaurant"] = new_fields["restaurant"]
            collected["restaurant_selection"] = "USER_SPECIFIED"

    elif intent == "BOOK_SERVICE":
        if sel_pref == "AI_CHOOSE":
            collected["package_selection"] = "AI_CHOOSE"
            if not collected.get("package"):
                if sel_strat == "CHEAPEST":
                    collected["package"] = "Standard Cleaning"
                elif sel_strat in ["BEST_AVAILABLE", "TOP_RATED"]:
                    collected["package"] = "Full Home Deep Cleaning (Top Rated)"
                else:
                    collected["package"] = "Premium Deep Cleaning"
        elif new_fields.get("package"):
            collected["package"] = new_fields["package"]

        # Default preferred date if address is set but date not set
        if not collected.get("preferred_date") and new_fields.get("preferred_date"):
            collected["preferred_date"] = new_fields["preferred_date"]

        # Slot selection
        if new_fields.get("slot"):
            collected["slot"] = new_fields["slot"]

    elif intent == "BOOK_BUS":
        collected["travel_mode"] = "bus"
        # Support pickup/destination remapping if user said "from Hyd to Vizag"
        if new_fields.get("pickup") and not collected.get("origin_city"):
            collected["origin_city"] = new_fields["pickup"]
        if new_fields.get("destination") and not collected.get("destination_city"):
            collected["destination_city"] = new_fields["destination"]

        if sel_pref == "AI_CHOOSE":
            if not collected.get("bus_type"):
                if sel_strat == "CHEAPEST":
                    collected["bus_type"] = "Non-AC Seater"
                else:
                    collected["bus_type"] = "AC Sleeper (2+1)"
            if not collected.get("selected_seats"):
                collected["selected_seats"] = ["U1"]
        if new_fields.get("bus_type"):
            collected["bus_type"] = new_fields["bus_type"]
        if new_fields.get("selected_service_id"):
            collected["selected_service_id"] = new_fields["selected_service_id"]
        if new_fields.get("selected_seats"):
            collected["selected_seats"] = new_fields["selected_seats"]

    elif intent == "BOOK_TRAIN":
        collected["travel_mode"] = "train"
        if new_fields.get("pickup") and not collected.get("origin_city"):
            collected["origin_city"] = new_fields["pickup"]
        if new_fields.get("destination") and not collected.get("destination_city"):
            collected["destination_city"] = new_fields["destination"]

        if sel_pref == "AI_CHOOSE":
            if not collected.get("train_class"):
                if sel_strat == "CHEAPEST":
                    collected["train_class"] = "SL"
                else:
                    collected["train_class"] = "3A"
        if new_fields.get("train_class"):
            collected["train_class"] = new_fields["train_class"]
        if new_fields.get("selected_service_id"):
            collected["selected_service_id"] = new_fields["selected_service_id"]

    elif intent == "BOOK_RIDE":
        # Invalidate old fare if ride/service type changed
        if new_fields.get("service_type") or new_fields.get("ride_type"):
            collected.pop("fare_amount", None)

        if sel_pref == "AI_CHOOSE":
            collected["ride_selection"] = "AI_CHOOSE"
            if not collected.get("ride_type"):
                if sel_strat == "CHEAPEST":
                    collected["ride_type"] = "Uber Moto"
                    collected["service_type"] = "bike"
                elif sel_strat == "FASTEST":
                    collected["ride_type"] = "Uber Premier"
                    collected["service_type"] = "cab"
                else:
                    collected["ride_type"] = "Uber Go"
                    collected["service_type"] = "cab"
        elif new_fields.get("ride_type"):
            collected["ride_type"] = new_fields["ride_type"]
        elif new_fields.get("service_type"):
            st = new_fields["service_type"]
            type_mapping = {
                "bike": "Uber Moto",
                "auto": "Uber Auto",
                "parcel": "Uber Connect",
                "cab": "Uber Go",
            }
            collected["ride_type"] = type_mapping.get(st, "Uber Go")
        elif collected.get("service_type") and not collected.get("ride_type"):
            st = collected["service_type"]
            type_mapping = {
                "bike": "Uber Moto",
                "auto": "Uber Auto",
                "parcel": "Uber Connect",
                "cab": "Uber Go",
            }
            collected["ride_type"] = type_mapping.get(st, "Uber Go")

    return {
        "collected_data": collected,
        "current_step": "INFORMATION_MERGED",
    }


async def check_required_information_node(state: AgentState) -> dict[str, Any]:
    """Node 3: Check for genuinely missing fields based on domain, availability, and context."""
    collected = dict(state.get("collected_data", {}))
    intent = state.get("intent") or collected.get("intent", "BOOK_RIDE")
    missing = []

    if intent == "BOOK_BUS":
        if not collected.get("origin_city"):
            missing.append("origin_city")
        if not collected.get("destination_city"):
            missing.append("destination_city")
        if not collected.get("travel_date"):
            missing.append("travel_date")
        if not collected.get("departure_window") and not collected.get("selected_service_id"):
            missing.append("departure_window")
        if not collected.get("passenger_count"):
            missing.append("passenger_count")
        if not missing and not collected.get("selected_service_id") and collected.get("bus_selection") != "AI_CHOOSE":
            missing.append("selected_service_id")
        if not missing and not collected.get("selected_seats") and collected.get("bus_selection") != "AI_CHOOSE":
            missing.append("selected_seats")

    elif intent == "BOOK_TRAIN":
        if not collected.get("origin_city"):
            missing.append("origin_city")
        if not collected.get("destination_city"):
            missing.append("destination_city")
        if not collected.get("travel_date"):
            missing.append("travel_date")
        if not collected.get("passenger_count"):
            missing.append("passenger_count")
        if not collected.get("train_class") and collected.get("train_selection") != "AI_CHOOSE":
            missing.append("train_class")
        if not missing and not collected.get("selected_service_id") and collected.get("train_selection") != "AI_CHOOSE":
            missing.append("selected_service_id")

    elif intent == "ORDER_FOOD":
        if not collected.get("food"):
            missing.append("food")
        elif not collected.get("restaurant") and collected.get("restaurant_selection") != "AI_CHOOSE":
            missing.append("restaurant")
        if not collected.get("delivery_location"):
            missing.append("delivery_location")

    elif intent == "BOOK_SERVICE":
        if not collected.get("service_name"):
            missing.append("service_name")
        elif not collected.get("package") and collected.get("package_selection") != "AI_CHOOSE":
            missing.append("package")
        elif not collected.get("service_location"):
            missing.append("service_location")
        elif not collected.get("preferred_date") and collected.get("package_selection") != "AI_CHOOSE":
            missing.append("preferred_date")
        elif not collected.get("slot") and collected.get("package_selection") != "AI_CHOOSE":
            missing.append("slot")

    else:  # BOOK_RIDE
        pickup = collected.get("pickup")
        dest = collected.get("destination")
        last_asked = collected.get("last_asked")

        # Check same location error
        if pickup and dest and GeminiExtractor.is_same_location(pickup, dest):
            if last_asked in ["pickup", "pickup_landmark"]:
                collected.pop("pickup", None)
                missing.append("same_location_pickup")
            else:
                collected.pop("destination", None)
                missing.append("same_location_destination")
        else:
            if not pickup:
                missing.append("pickup")
            else:
                is_broad, _ = GeminiExtractor.is_broad_locality_only(pickup)
                if is_broad:
                    missing.append("pickup_landmark")

            if not dest:
                missing.append("destination")
            else:
                is_broad, _ = GeminiExtractor.is_broad_locality_only(dest)
                if is_broad:
                    missing.append("destination_landmark")

        # If pickup & destination are known, check if ride tier is selected
        if not missing and not collected.get("service_type") and not collected.get("ride_type"):
            missing.append("service_type")

    return {
        "collected_data": collected,
        "missing_fields": missing,
        "current_step": "CHECKED_REQUIREMENTS",
    }


async def ask_user_node(state: AgentState) -> dict[str, Any]:
    """Node 4: Inquire user for genuinely missing info using warm, contextual phrasing and real availability."""
    missing = state.get("missing_fields", [])
    collected = dict(state.get("collected_data", {}))
    llm_reply = state.get("_llm_reply")
    intent = state.get("intent") or collected.get("intent", "BOOK_RIDE")

    # Bus booking prompts
    if intent == "BOOK_BUS":
        orig = collected.get("origin_city")
        dest = collected.get("destination_city")
        date_t = collected.get("travel_date")
        window = collected.get("departure_window")
        pax = collected.get("passenger_count")

        current_step = "WAITING_FOR_USER"
        if "origin_city" in missing and "destination_city" in missing:
            collected["last_asked"] = "route"
            current_step = "COLLECTING_ROUTE"
            response = "Sure! I can help you book a bus. Where would you like to travel from and to?"
        elif "origin_city" in missing:
            collected["last_asked"] = "origin_city"
            current_step = "COLLECTING_ROUTE"
            response = "Sure! Where are you traveling from?"
        elif "destination_city" in missing:
            collected["last_asked"] = "destination_city"
            current_step = "COLLECTING_ROUTE"
            response = f"Got your departure from {orig}. Where would you like to travel to?"
        elif "travel_date" in missing:
            collected["last_asked"] = "travel_date"
            current_step = "COLLECTING_TRAVEL_DATE"
            response = "What date would you like to travel?"
        elif "departure_window" in missing:
            collected["last_asked"] = "departure_window"
            current_step = "COLLECTING_DEPARTURE_TIME"
            response = "What time would you prefer to depart — morning, afternoon, evening, or a specific time?"
        elif "passenger_count" in missing:
            collected["last_asked"] = "passenger_count"
            current_step = "COLLECTING_PASSENGER_DETAILS"
            response = "How many passengers are travelling?"
        elif "selected_service_id" in missing:
            collected["last_asked"] = "selected_service_id"
            current_step = "SELECTING_BUS"
            # Query real provider for bus options
            try:
                bus_res = await tool_executor.execute(
                    "search_buses",
                    {
                        "origin": orig,
                        "destination": dest,
                        "travel_date": date_t,
                        "departure_window": window,
                    },
                )
                buses = bus_res.get("buses", [])
                collected["available_buses"] = buses
                lines = []
                for i, b in enumerate(buses[:3]):
                    lines.append(f"{i+1}. 🚌 {b['operator_name']} — {b['bus_type']} (Dep: {b['departure_time']}, Arr: {b['arrival_time']}, Total: ₹{int(b['total_fare'])}, {b['rating']}★)")
                bus_list = "\n".join(lines)
                response = (
                    f"I found these available options from {orig} to {dest} for {date_t} ({window or 'anytime'}):\n\n"
                    f"{bus_list}\n\n"
                    f"Which bus would you prefer?"
                )
            except Exception as e:
                response = f"I'll look for available buses from {orig} to {dest} for {date_t}. Which operator or bus type do you prefer?"
        elif "selected_seats" in missing:
            collected["last_asked"] = "selected_seats"
            current_step = "SELECTING_SEATS"
            response = "These seats are available: U1, U2, L3, L4. Which seats would you like?"
        else:
            response = "Could you please provide the remaining details for your bus booking?"

        return {
            "response": response,
            "collected_data": collected,
            "current_step": current_step,
            "waiting_for_confirmation": False,
        }

    # Train booking prompts
    elif intent == "BOOK_TRAIN":
        orig = collected.get("origin_city")
        dest = collected.get("destination_city")
        date_t = collected.get("travel_date")
        pax = collected.get("passenger_count")
        cls_p = collected.get("train_class")

        current_step = "WAITING_FOR_USER"
        if "origin_city" in missing and "destination_city" in missing:
            collected["last_asked"] = "origin_city"
            current_step = "COLLECTING_ROUTE"
            response = "Sure! Which railway station or city are you departing from?"
        elif "origin_city" in missing:
            collected["last_asked"] = "origin_city"
            current_step = "COLLECTING_ROUTE"
            response = "Sure! Which railway station or city are you departing from?"
        elif "destination_city" in missing:
            collected["last_asked"] = "destination_city"
            current_step = "COLLECTING_ROUTE"
            response = f"Got your origin at {orig}. What is your destination station?"
        elif "travel_date" in missing:
            collected["last_asked"] = "travel_date"
            current_step = "COLLECTING_TRAVEL_DATE"
            response = "What date would you like to travel?"
        elif "passenger_count" in missing:
            collected["last_asked"] = "passenger_count"
            current_step = "COLLECTING_PASSENGER_DETAILS"
            response = "How many passengers are travelling?"
        elif "train_class" in missing:
            collected["last_asked"] = "train_class"
            current_step = "SELECTING_CLASS"
            response = "What class do you prefer (e.g. Sleeper SL, 3A, 2A, or AC Chair Car CC)?"
        elif "selected_service_id" in missing:
            collected["last_asked"] = "selected_service_id"
            current_step = "SELECTING_TRAIN"
            # Query real provider for trains
            try:
                train_res = await tool_executor.execute(
                    "search_trains",
                    {
                        "origin": orig,
                        "destination": dest,
                        "travel_date": date_t,
                        "class_preference": cls_p,
                    },
                )
                trains = train_res.get("trains", [])
                collected["available_trains"] = trains
                lines = []
                for i, t in enumerate(trains[:3]):
                    cls_summary = " | ".join(f"{c['class_code']}: ₹{int(c['fare'])} ({c['availability_status']})" for c in t["available_classes"])
                    lines.append(f"{i+1}. 🚆 {t['train_name']} ({t['train_number']}) — Dep: {t['departure_time']}, Arr: {t['arrival_time']}\n   {cls_summary}")
                train_list = "\n".join(lines)
                response = (
                    f"Here are the available trains from {orig} to {dest} on {date_t}:\n\n"
                    f"{train_list}\n\n"
                    f"Which train would you prefer to book?"
                )
            except Exception:
                response = f"I'm searching for trains from {orig} to {dest} on {date_t}. Which train would you like?"
        else:
            response = "Could you please provide the remaining details for your train booking?"

        return {
            "response": response,
            "collected_data": collected,
            "current_step": current_step,
            "waiting_for_confirmation": False,
        }

    # Food ordering prompts
    elif intent == "ORDER_FOOD":
        current_step = "WAITING_FOR_USER"
        if "food" in missing:
            collected["last_asked"] = "food"
            response = "Sure! What delicious food or cuisine are you craving today?"
        elif "restaurant" in missing:
            collected["last_asked"] = "restaurant"
            food = collected.get("food", "food")
            if llm_reply and any(w in llm_reply.lower() for w in ["restaurant", "place", "preference", "pick", "choose"]):
                response = llm_reply
            else:
                response = f"Sure, {food} sounds good! Do you have a restaurant preference (like Mehfil, Bawarchi, Paradise), or should I pick a highly rated option for you?"
        elif "delivery_location" in missing:
            collected["last_asked"] = "delivery_location"
            food = collected.get("food", "food")
            rest = collected.get("restaurant")
            if llm_reply and any(w in llm_reply.lower() for w in ["deliver", "address", "location", "where"]):
                response = llm_reply
            elif collected.get("restaurant_selection") == "AI_CHOOSE":
                response = f"Sure, I'll pick a highly rated {food} restaurant for you. Where should I deliver it?"
            else:
                response = f"Got it, {food} from {rest or 'your preferred restaurant'}! Where should I deliver it?"
        else:
            response = "Could you please let me know your delivery address?"

        return {
            "response": response,
            "collected_data": collected,
            "current_step": "WAITING_FOR_USER",
            "waiting_for_confirmation": False,
        }

    # Home service prompts
    elif intent == "BOOK_SERVICE":
        current_step = "WAITING_FOR_USER"
        if "service_name" in missing:
            collected["last_asked"] = "service_name"
            response = "What type of cleaning or home service do you need?"
        elif "package" in missing:
            collected["last_asked"] = "package"
            srv = collected.get("service_name", "cleaning")
            if llm_reply and any(w in llm_reply.lower() for w in ["package", "tier", "standard", "premium"]):
                response = llm_reply
            else:
                response = "Sure! Which cleaning package would you prefer — Standard, Premium, or Full Home?"
        elif "service_location" in missing:
            collected["last_asked"] = "service_location"
            pkg = collected.get("package", "service")
            if llm_reply and any(w in llm_reply.lower() for w in ["address", "location", "where", "visit", "come"]):
                response = llm_reply
            elif collected.get("package_selection") == "AI_CHOOSE":
                response = f"I'll select our top-rated {pkg} package for you. Where should our service team visit?"
            else:
                response = "Got it. What's the service address?"
        elif "preferred_date" in missing:
            collected["last_asked"] = "preferred_date"
            response = "Thanks! What date would you prefer for the cleaning?"
        elif "slot" in missing:
            collected["last_asked"] = "slot"
            pkg = collected.get("package", "Standard")
            loc = collected.get("service_location", "your address")
            date_pref = collected.get("preferred_date", "Today")

            # Query real provider availability tool
            try:
                avail_res = await tool_executor.execute(
                    "check_service_availability",
                    {
                        "service_name": collected.get("service_name", "Deep Cleaning"),
                        "package": pkg,
                        "service_location": loc,
                        "preferred_date": date_pref,
                    },
                )
                available_slots = [s["label"] for s in avail_res.get("available_slots", []) if s.get("is_available")]
            except Exception:
                available_slots = [
                    "10:00 AM – 12:00 PM",
                    "12:00 PM – 2:00 PM",
                    "2:00 PM – 4:00 PM",
                    "4:00 PM – 6:00 PM",
                ]

            slot_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(available_slots))
            response = (
                f"Let me check the available {pkg} cleaning appointments for {loc}.\n\n"
                f"These slots are available for {pkg} cleaning:\n\n"
                f"{slot_lines}\n\n"
                f"Which slot works best for you?"
            )
        else:
            response = "What address should we schedule your home service for?"

        return {
            "response": response,
            "collected_data": collected,
            "current_step": "WAITING_FOR_USER",
            "waiting_for_confirmation": False,
        }

    # Ride booking prompts
    if llm_reply and ("pick you up" not in llm_reply.lower() or "pickup" in missing):
        if "same_location_pickup" not in missing and "same_location_destination" not in missing:
            return {
                "response": llm_reply,
                "collected_data": collected,
                "current_step": "WAITING_FOR_USER",
                "waiting_for_confirmation": False,
            }

    # Ride booking prompts
    else:
        if "same_location_pickup" in missing:
            collected["last_asked"] = "pickup"
            d = collected.get("destination", "your destination")
            response = (
                f"Oops! Your pickup and destination can't be the same place. "
                f"Since your destination is set to {d}, where should the driver pick you up from?"
            )
        elif "same_location_destination" in missing:
            collected["last_asked"] = "destination"
            p = collected.get("pickup", "your pickup")
            response = (
                f"Oops! Your pickup and destination can't be the same place. "
                f"Since we're picking you up at {p}, where would you like to travel to?"
            )
        elif "pickup_landmark" in missing:
            collected["last_asked"] = "pickup_landmark"
            _, prompt_msg = GeminiExtractor.is_broad_locality_only(collected.get("pickup"))
            response = prompt_msg or f"Got it, {collected.get('pickup')}! Could you let me know which building, mall, or landmark you're near?"
        elif "destination_landmark" in missing:
            collected["last_asked"] = "destination_landmark"
            _, prompt_msg = GeminiExtractor.is_broad_locality_only(collected.get("destination"))
            response = prompt_msg or f"Got your destination as {collected.get('destination')}! Where specifically should the driver drop you off (like a specific building or gate)?"
        elif "pickup" in missing and "destination" in missing:
            collected["last_asked"] = "pickup"
            service = collected.get("service_type")
            ride_tier = collected.get("ride_type")
            if service or ride_tier:
                name = ride_tier or f"Uber {service.capitalize()}"
                article = "an" if name.lower().startswith(("a", "e", "i", "o", "u")) else "a"
                response = f"Sure thing! I'd be happy to help you get {article} {name}. Where can I pick you up from today?"
            else:
                response = "Hello! I'd be happy to help you get a ride. Where can I pick you up from today?"
        elif "pickup" in missing:
            collected["last_asked"] = "pickup"
            service = collected.get("service_type")
            ride_tier = collected.get("ride_type")
            if service or ride_tier:
                name = ride_tier or f"Uber {service.capitalize()}"
                response = f"Sure thing! Where should the driver pick you up for your {name}?"
            else:
                response = "Sure thing! Where should we pick you up from?"
        elif "destination" in missing:
            collected["last_asked"] = "destination"
            p = collected.get("pickup")
            response = f"Got your pickup at {p}. Where would you like to head to today?"
        elif "service_type" in missing:
            collected["last_asked"] = "service_type"
            p = collected.get("pickup", "your pickup")
            d = collected.get("destination", "your destination")
            response = (
                f"Got it, from {p} to {d}! What kind of ride would you prefer today?\n"
                f"• 🚗 Cab / Car (Uber Go / Premier / XL)\n"
                f"• 🏍️ Bike (Uber Moto)\n"
                f"• 🛺 Auto (Uber Auto)\n"
                f"• 📦 Parcel delivery (Uber Connect)"
            )
        elif llm_reply:
            response = llm_reply
        else:
            response = "Could you please share a few more details so I can get your ride sorted?"

    return {
        "response": response,
        "collected_data": collected,
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
    }


async def ask_confirmation_node(state: AgentState) -> dict[str, Any]:
    """Node 5: Previews real quotes/estimates and asks user for explicit confirmation."""
    collected = dict(state.get("collected_data", {}))
    collected.pop("last_asked", None)
    intent = state.get("intent") or collected.get("intent", "BOOK_RIDE")

    if intent == "ORDER_FOOD":
        food = collected.get("food", "Food")
        restaurant = collected.get("restaurant", "Meghana Foods")
        delivery_loc = collected.get("delivery_location", "your address")
        total_amount = 350.0
        collected["fare_amount"] = total_amount
        response = (
            f"Great choice! I have an order for {food} from {restaurant}.\n"
            f"📍 Delivery Address: {delivery_loc}\n"
            f"💰 Estimated Total: ₹{int(total_amount)} (incl. taxes and delivery)\n\n"
            f"Shall I go ahead and place this food order for you?"
        )
        return {
            "response": response,
            "collected_data": collected,
            "tool_result": {"food": food, "restaurant": restaurant, "amount": total_amount},
            "current_step": "WAITING_FOR_CONFIRMATION",
            "waiting_for_confirmation": True,
        }

    elif intent == "BOOK_SERVICE":
        service_name = collected.get("service_name", "Deep Cleaning")
        package = collected.get("package", "Standard")
        service_loc = collected.get("service_location", "your address")
        date_pref = collected.get("preferred_date", "Today")
        slot = collected.get("slot", "2:00 PM – 4:00 PM")

        # Get actual price from provider
        fee = 1499.0
        try:
            avail_res = await tool_executor.execute(
                "check_service_availability",
                {
                    "service_name": service_name,
                    "package": package,
                    "service_location": service_loc,
                    "preferred_date": date_pref,
                },
            )
            pricing_map = avail_res.get("pricing", {})
            if package in pricing_map:
                fee = pricing_map[package]
            elif "standard" in package.lower():
                fee = 1499.0
            elif "premium" in package.lower():
                fee = 2499.0
            elif "full home" in package.lower():
                fee = 3499.0
        except Exception:
            pass

        collected["fare_amount"] = fee
        response = (
            f"{package} cleaning at {service_loc} on your selected date, from {slot}. "
            f"The total estimated price is ₹{int(fee):,}. Shall I confirm this booking?"
        )
        return {
            "response": response,
            "collected_data": collected,
            "tool_result": {"service": service_name, "package": package, "slot": slot, "amount": fee},
            "current_step": "WAITING_FOR_CONFIRMATION",
            "waiting_for_confirmation": True,
        }
    elif intent == "BOOK_BUS":
        orig = collected.get("origin_city", "Hyderabad")
        dest = collected.get("destination_city", "Visakhapatnam")
        date_t = collected.get("travel_date", "Tomorrow")
        pax = int(collected.get("passenger_count") or 1)
        seats = collected.get("selected_seats") or ["U1"]
        operator = "Orange Tours & Travels"
        bus_type = collected.get("bus_type", "AC Sleeper (2+1)")
        total_fare = round(997.5 * pax, 2)
        collected["fare_amount"] = total_fare

        response = (
            f"I have your bus booking summary ready:\n\n"
            f"🚌 Operator: {operator}\n"
            f"🚍 Bus Type: {bus_type}\n"
            f"📍 Route: {orig} to {dest}\n"
            f"📅 Travel Date: {date_t} (Dep: 20:30, Arr: 06:00)\n"
            f"👥 Passengers: {pax}\n"
            f"💺 Seats: {', '.join(seats)}\n"
            f"💰 Total Amount: ₹{int(total_fare)} (incl. taxes & fees)\n\n"
            f"Shall I confirm this bus booking for you? You will get the SMS with all the booking details."
        )
        return {
            "response": response,
            "collected_data": collected,
            "tool_result": {"service": operator, "route": f"{orig} to {dest}", "amount": total_fare},
            "current_step": "WAITING_FOR_CONFIRMATION",
            "waiting_for_confirmation": True,
        }

    elif intent == "BOOK_TRAIN":
        orig = collected.get("origin_city", "Secunderabad Junction (SC)")
        dest = collected.get("destination_city", "Visakhapatnam Junction (VSKP)")
        date_t = collected.get("travel_date", "Tomorrow")
        pax = int(collected.get("passenger_count") or 1)
        train_class = collected.get("train_class", "3A")
        train_name = "Godavari Express (12728)"
        unit_fare = 1150.0 if train_class == "3A" else (1640.0 if train_class == "2A" else 435.0)
        total_fare = round(unit_fare * pax, 2)
        collected["fare_amount"] = total_fare

        response = (
            f"I have your train reservation summary ready:\n\n"
            f"🚆 Train: {train_name}\n"
            f"🚉 Route: {orig} to {dest}\n"
            f"📅 Travel Date: {date_t} (Dep: 17:05, Arr: 05:45)\n"
            f"🎫 Class: {train_class}\n"
            f"👥 Passengers: {pax}\n"
            f"💰 Total Fare: ₹{int(total_fare)} (IRCTC service charges included)\n\n"
            f"Shall I go ahead and book this ticket for you? You will get the SMS with all the booking details."
        )
        return {
            "response": response,
            "collected_data": collected,
            "tool_result": {"train": train_name, "route": f"{orig} to {dest}", "amount": total_fare},
            "current_step": "WAITING_FOR_CONFIRMATION",
            "waiting_for_confirmation": True,
        }

    elif intent == "BOOK_RIDE":
        pickup = collected.get("pickup")
        destination = collected.get("destination")
        ride_type = collected.get("ride_type", "Uber Go")

        # Graceful re-routing: handle travel requests routed as rides
        travel_mode = collected.get("travel_mode") or collected.get("service_type")
        if travel_mode == "bus" or intent == "BOOK_BUS":
            collected["intent"] = "BOOK_BUS"
            orig = collected.get("origin_city") or collected.get("pickup", "Hyderabad")
            dest = collected.get("destination_city") or collected.get("destination", "Visakhapatnam")
            date_t = collected.get("travel_date", "Tomorrow")
            pax = int(collected.get("passenger_count") or 1)
            operator = collected.get("bus_operator", "Orange Tours & Travels")
            bus_type = collected.get("bus_type", "AC Sleeper (2+1)")
            seats = collected.get("selected_seats") or ["U1"]
            total_fare = float(collected.get("fare_amount") or 850.0)
            collected["fare_amount"] = total_fare
            response = (
                f"I have your bus booking summary ready:\n\n"
                f"🚌 Operator: {operator}\n"
                f"🚍 Bus Type: {bus_type}\n"
                f"📍 Route: {orig} to {dest}\n"
                f"📅 Travel Date: {date_t} (Dep: 20:30, Arr: 06:00)\n"
                f"👥 Passengers: {pax}\n"
                f"💺 Seats: {', '.join(seats)}\n"
                f"💰 Total Amount: ₹{int(total_fare)} (incl. taxes & fees)\n\n"
                f"Shall I confirm this bus booking for you? You will get the SMS with all the booking details."
            )
            return {
                "response": response,
                "collected_data": collected,
                "tool_result": {"service": operator, "route": f"{orig} to {dest}", "amount": total_fare},
                "current_step": "WAITING_FOR_CONFIRMATION",
                "waiting_for_confirmation": True,
            }
        elif travel_mode == "train" or intent == "BOOK_TRAIN":
            collected["intent"] = "BOOK_TRAIN"
            orig = collected.get("origin_city", "Secunderabad Junction (SC)")
            dest = collected.get("destination_city", "Visakhapatnam Junction (VSKP)")
            date_t = collected.get("travel_date", "Tomorrow")
            pax = int(collected.get("passenger_count") or 1)
            train_class = collected.get("train_class", "3A")
            train_name = "Godavari Express (12728)"
            unit_fare = 1150.0 if train_class == "3A" else (1640.0 if train_class == "2A" else 435.0)
            total_fare = round(unit_fare * pax, 2)
            collected["fare_amount"] = total_fare
            response = (
                f"I have your train reservation summary ready:\n\n"
                f"🚆 Train: {train_name}\n"
                f"🚉 Route: {orig} to {dest}\n"
                f"📅 Travel Date: {date_t} (Dep: 17:05, Arr: 05:45)\n"
                f"🎫 Class: {train_class}\n"
                f"👥 Passengers: {pax}\n"
                f"💰 Total Fare: ₹{int(total_fare)} (IRCTC service charges included)\n\n"
                f"Shall I go ahead and book this ticket for you? You will get the SMS with all the booking details."
            )
            return {
                "response": response,
                "collected_data": collected,
                "tool_result": {"train": train_name, "route": f"{orig} to {dest}", "amount": total_fare},
                "current_step": "WAITING_FOR_CONFIRMATION",
                "waiting_for_confirmation": True,
            }

        # Fetch real estimate via tool
        estimate_result = await tool_executor.execute(
            "get_ride_estimate",
            {
                "pickup_address": pickup,
                "destination_address": destination,
                "ride_type": ride_type,
            },
        )

        amount = estimate_result["amount"]
        collected["fare_amount"] = amount

        icon = "🚗"
        if "moto" in ride_type.lower() or "bike" in ride_type.lower():
            icon = "🏍️"
        elif "auto" in ride_type.lower():
            icon = "🛺"
        elif "connect" in ride_type.lower() or "parcel" in ride_type.lower():
            icon = "📦"

        response = (
            f"Great choice! I found an {icon} {ride_type} from {pickup} to {destination}. "
            f"The estimated fare is ₹{int(amount)}. Shall I go ahead and book this for you?"
        )

        return {
            "response": response,
            "collected_data": collected,
            "tool_result": estimate_result,
            "current_step": "WAITING_FOR_CONFIRMATION",
            "waiting_for_confirmation": True,
        }
    else:
        return {
            "response": "I'm sorry, that service provider is not connected yet. Please let me know if you'd like to book a bus, train, cab, food, or home cleaning!",
            "collected_data": collected,
            "current_step": "WAITING_FOR_USER",
            "waiting_for_confirmation": False,
        }


async def cancel_ride_node(state: AgentState) -> dict[str, Any]:
    """Node 6: Handles user cancellation with warm reassurance."""
    return {
        "response": "No worries at all! I've cancelled that request for you. Let me know whenever you're ready next time!",
        "current_step": "CANCELLED",
        "waiting_for_confirmation": False,
    }


async def book_ride_node(state: AgentState) -> dict[str, Any]:
    """Node 7: Executes ride booking with provider ONLY after verified confirmation."""
    collected = state.get("collected_data", {})
    travel_mode = collected.get("travel_mode") or collected.get("service_type")
    if travel_mode in ["bus", "train"] or state.get("intent") in ["BOOK_BUS", "BOOK_TRAIN"]:
        raise ValueError(f"Strict validation failure: cannot book {travel_mode or state.get('intent')} as a cab.")

    pickup = collected.get("pickup")
    destination = collected.get("destination")
    ride_type = collected.get("ride_type", "Uber Go")
    fare_amount = collected.get("fare_amount", 450.0)

    # Idempotency check: if already completed or has active booking_id in collected_data/result, reuse it
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or collected.get("booking_result")
    if prev_result and (prev_result.get("booking_id") or prev_result.get("order_id")):
        booking_id = prev_result.get("booking_id") or prev_result.get("order_id")
        driver_name = prev_result.get("driver_name", "your driver")
        vehicle_plate = prev_result.get("vehicle_plate", "")
        driver_phone = prev_result.get("driver_phone", "+91 98765 43210")
        plate_info = f" ({vehicle_plate})" if vehicle_plate else ""
        response = (
            f"Your {ride_type} is already booked!\n\n"
            f"🚕 Booking ID: {booking_id}\n"
            f"👤 Driver: {driver_name}{plate_info}\n"
            f"📞 Driver Phone: {driver_phone}\n\n"
            f"Your driver will arrive shortly (ETA: ~4 mins). You can ask me for live status or driver location anytime!"
        )
        return {
            "response": response,
            "tool_result": prev_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }

    try:
        booking_result = await tool_executor.execute(
            "book_ride",
            {
                "pickup_address": pickup,
                "destination_address": destination,
                "ride_type": ride_type,
                "fare_amount": fare_amount,
            },
        )

        booking_id = booking_result.get("booking_id")
        driver_name = booking_result.get("driver_name", "your driver")
        vehicle_plate = booking_result.get("vehicle_plate", "")
        driver_phone = booking_result.get("driver_phone", "+91 98765 43210")
        plate_info = f" ({vehicle_plate})" if vehicle_plate else ""

        now_iso = datetime.now(timezone.utc).isoformat()
        booking_result["created_at"] = now_iso
        booking_result["initial_eta_minutes"] = 4
        booking_result["simulation_enabled"] = True

        response = (
            f"All set! Your {ride_type} has been booked successfully.\n\n"
            f"🚕 Booking ID: {booking_id}\n"
            f"👤 Driver: {driver_name}{plate_info}\n"
            f"📞 Driver Phone: {driver_phone}\n\n"
            f"Your ride will arrive shortly (ETA: ~4 mins). I'll keep your booking active right here—you can ask me for live status, driver location, or assistance anytime!"
        )

        return {
            "response": response,
            "tool_result": booking_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }
    except Exception as e:
        return {
            "response": f"I'm sorry, something went wrong while booking your ride: {str(e)}. Would you like me to try again?",
            "error": str(e),
            "current_step": "FAILED",
            "waiting_for_confirmation": False,
        }


async def book_food_order_node(state: AgentState) -> dict[str, Any]:
    """Node 8: Executes food ordering with provider after user confirmation."""
    collected = state.get("collected_data", {})
    food = collected.get("food", "Food")
    restaurant = collected.get("restaurant", "Meghana Foods")
    delivery_loc = collected.get("delivery_location", "your address")
    estimated_total = float(collected.get("fare_amount", 350.0))

    # Idempotency check: if already completed or has active order_id in task/result, reuse it
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or collected.get("booking_result")
    if prev_result and (prev_result.get("order_id") or prev_result.get("booking_id")):
        order_id = prev_result.get("order_id") or prev_result.get("booking_id")
        partner = prev_result.get("delivery_partner", "Suresh V.")
        response = (
            f"Your {food} order from {restaurant} has already been placed!\n\n"
            f"🍔 Item: {food.title()}\n"
            f"🏪 Restaurant: {restaurant}\n"
            f"🎫 Order ID: {order_id}\n"
            f"📍 Delivering to: {delivery_loc}\n"
            f"🛵 Delivery partner: {partner}\n"
            f"⏱️ Status: Preparing\n"
            f"⏳ Estimated delivery: ~20 minutes\n\n"
            f"You can ask me for the latest status or live tracking anytime!"
        )
        return {
            "response": response,
            "tool_result": prev_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }

    idempotency_key = state.get("task_id") or state.get("conversation_id")

    try:
        booking_result = await tool_executor.execute(
            "place_food_order",
            {
                "food": food,
                "restaurant": restaurant,
                "delivery_location": delivery_loc,
                "estimated_total": estimated_total,
                "idempotency_key": idempotency_key,
            },
        )

        order_id = booking_result.get("order_id")
        partner = booking_result.get("delivery_partner", "Suresh V.")

        response = (
            f"Your {food} order from {restaurant} has been placed successfully!\n\n"
            f"🍔 Item: {food.title()}\n"
            f"🏪 Restaurant: {restaurant}\n"
            f"🎫 Order ID: {order_id}\n"
            f"📍 Delivering to: {delivery_loc}\n"
            f"🛵 Delivery partner: {partner}\n"
            f"⏱️ Status: Preparing\n"
            f"⏳ Estimated delivery: ~20 minutes\n\n"
            f"I'll keep the order active here. You can ask me for the latest status or live tracking anytime!"
        )

        return {
            "response": response,
            "tool_result": booking_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }
    except Exception as e:
        return {
            "response": f"I'm sorry, placing your food order failed: {str(e)}. Would you like me to try again?",
            "error": str(e),
            "current_step": "FAILED",
            "waiting_for_confirmation": False,
        }


async def book_service_node(state: AgentState) -> dict[str, Any]:
    """Node 9: Executes home service booking with provider after user confirmation."""
    collected = state.get("collected_data", {})
    service_name = collected.get("service_name", "Deep Cleaning")
    package = collected.get("package", "Standard")
    service_loc = collected.get("service_location", "your address")
    date_pref = collected.get("preferred_date", "Today")
    slot = collected.get("slot", "2:00 PM – 4:00 PM")
    fare_amount = float(collected.get("fare_amount", 1499.0))

    # Idempotency check: if already completed, reuse result
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or collected.get("booking_result")
    if prev_result and prev_result.get("booking_id"):
        booking_id = prev_result.get("booking_id")
        response = (
            f"Your Urban Clean appointment is already confirmed!\n\n"
            f"Booking ID: {booking_id}\n"
            f"Package: {package}\n"
            f"Address: {service_loc}\n"
            f"Date: {date_pref}\n"
            f"Time: {slot}\n"
            f"Price: ₹{int(fare_amount):,}\n"
            f"Status: Confirmed\n\n"
            f"You can ask me for booking updates anytime."
        )
        return {
            "response": response,
            "tool_result": prev_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }

    try:
        booking_result = await tool_executor.execute(
            "book_home_service",
            {
                "service_name": service_name,
                "package": package,
                "service_location": service_loc,
                "preferred_date": date_pref,
                "slot_label": slot,
                "price": fare_amount,
            },
        )

        booking_id = booking_result.get("booking_id")
        now_iso = datetime.now(timezone.utc).isoformat()
        booking_result["created_at"] = now_iso
        booking_result["simulation_enabled"] = True

        response = (
            f"Your Urban Clean appointment is confirmed!\n\n"
            f"Booking ID: {booking_id}\n"
            f"Package: {package}\n"
            f"Address: {service_loc}\n"
            f"Date: {date_pref}\n"
            f"Time: {slot}\n"
            f"Price: ₹{int(fare_amount):,}\n"
            f"Status: Confirmed\n\n"
            f"You can ask me for booking updates anytime."
        )

        return {
            "response": response,
            "tool_result": booking_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }
    except Exception as e:
        return {
            "response": f"I'm sorry, that appointment could not be confirmed: {str(e)}.",
            "error": str(e),
            "current_step": "FAILED",
            "waiting_for_confirmation": False,
        }


async def book_bus_node(state: AgentState) -> dict[str, Any]:
    """Node: Executes intercity bus booking with provider after verified confirmation."""
    collected = state.get("collected_data", {})
    orig = collected.get("origin_city", "Hyderabad")
    dest = collected.get("destination_city", "Visakhapatnam")
    date_t = collected.get("travel_date", "Tomorrow")
    pax = int(collected.get("passenger_count") or 1)
    seats = collected.get("selected_seats") or ["U1"]
    boarding = collected.get("boarding_point") or "Ameerpet"
    dropping = collected.get("dropping_point") or "RTC Complex"
    fare_amount = float(collected.get("fare_amount", 997.5 * pax))
    service_id = collected.get("selected_service_id") or "bus_srv_1"

    # Idempotency check: if already completed, reuse result
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or collected.get("booking_result")
    if prev_result and (prev_result.get("pnr") or (prev_result.get("booking_id") and "bus_" in str(prev_result.get("booking_id")))):
        pnr = prev_result.get("pnr") or prev_result.get("booking_id")
        op = prev_result.get("operator_name", "Orange Tours & Travels")
        response = (
            f"Your bus booking is already confirmed!\n\n"
            f"🚌 Operator: {op}\n"
            f"🎫 PNR: {pnr}\n"
            f"📍 Route: {orig} to {dest}\n"
            f"📅 Travel Date: {date_t}\n"
            f"💺 Seats: {', '.join(seats)}\n"
            f"💰 Fare: ₹{int(fare_amount)}\n"
            f"Status: Confirmed\n\n"
            f"You will get the SMS with all the booking details."
        )
        return {
            "response": response,
            "tool_result": prev_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }

    idempotency_key = state.get("task_id") or state.get("conversation_id")

    try:
        booking_result = await tool_executor.execute(
            "book_bus",
            {
                "service_id": service_id,
                "origin": orig,
                "destination": dest,
                "travel_date": date_t,
                "passenger_count": pax,
                "selected_seats": seats,
                "boarding_point": boarding,
                "dropping_point": dropping,
                "fare_amount": fare_amount,
                "idempotency_key": idempotency_key,
            },
        )

        pnr = booking_result.get("pnr")
        op = booking_result.get("operator_name", "Orange Tours & Travels")
        bus_type = booking_result.get("bus_type", "AC Sleeper (2+1)")
        total_fare = booking_result.get("total_fare", fare_amount)

        response = (
            f"Your bus ticket has been confirmed successfully!\n\n"
            f"🚌 Operator: {op} ({bus_type})\n"
            f"🎫 PNR: {pnr}\n"
            f"📍 Route: {orig} to {dest}\n"
            f"📅 Travel Date: {date_t} (Dep: 20:30, Arr: 06:00)\n"
            f"🚏 Boarding Point: {boarding}\n"
            f"👥 Passengers: {pax}\n"
            f"💺 Confirmed Seats: {', '.join(seats)}\n"
            f"💰 Total Amount: ₹{int(total_fare)}\n"
            f"⏱️ Status: Confirmed\n\n"
            f"You will get the SMS with all the booking details.\n\n"
            f"Do you need any other help?"
        )

        return {
            "response": response,
            "tool_result": booking_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }
    except Exception as e:
        return {
            "response": f"I'm sorry, bus booking could not be completed: {str(e)}.",
            "error": str(e),
            "current_step": "FAILED",
            "waiting_for_confirmation": False,
        }


async def book_train_node(state: AgentState) -> dict[str, Any]:
    """Node: Executes Indian Railways train ticket booking with provider after verified confirmation."""
    collected = state.get("collected_data", {})
    orig = collected.get("origin_city", "Secunderabad Junction (SC)")
    dest = collected.get("destination_city", "Visakhapatnam Junction (VSKP)")
    date_t = collected.get("travel_date", "Tomorrow")
    pax = int(collected.get("passenger_count") or 1)
    train_class = collected.get("train_class", "3A")
    train_number = collected.get("selected_service_id") or "12728"
    unit_fare = 1150.0 if train_class == "3A" else (1640.0 if train_class == "2A" else 435.0)
    fare_amount = float(collected.get("fare_amount", unit_fare * pax))

    # Idempotency check: if already completed, reuse result
    latest_task = state.get("latest_completed_task") or {}
    prev_result = latest_task.get("result") or state.get("tool_result") or collected.get("booking_result")
    if prev_result and (prev_result.get("pnr") or (prev_result.get("booking_id") and "train_" in str(prev_result.get("booking_id")))):
        pnr = prev_result.get("pnr")
        t_name = prev_result.get("train_name", "Godavari Express")
        response = (
            f"Your train ticket is already booked!\n\n"
            f"🚆 Train: {t_name} ({train_number})\n"
            f"🎫 IRCTC PNR: {pnr}\n"
            f"🚉 Route: {orig} to {dest}\n"
            f"📅 Travel Date: {date_t}\n"
            f"🎫 Class: {train_class}\n"
            f"💰 Fare: ₹{int(fare_amount)}\n"
            f"Status: Confirmed\n\n"
            f"You will get the SMS with all the booking details."
        )
        return {
            "response": response,
            "tool_result": prev_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }

    idempotency_key = state.get("task_id") or state.get("conversation_id")

    try:
        booking_result = await tool_executor.execute(
            "book_train",
            {
                "train_number": train_number,
                "origin_station": orig,
                "destination_station": dest,
                "travel_date": date_t,
                "selected_class": train_class,
                "passenger_count": pax,
                "fare_amount": fare_amount,
                "idempotency_key": idempotency_key,
            },
        )

        pnr = booking_result.get("pnr")
        train_name = booking_result.get("train_name", "Godavari Express")
        berths = booking_result.get("allocated_berths") or ["B3-21 (Lower)"]
        total_fare = booking_result.get("total_fare", fare_amount)

        response = (
            f"Your train ticket has been booked successfully!\n\n"
            f"🚆 Train: {train_name} ({train_number})\n"
            f"🎫 IRCTC PNR: {pnr}\n"
            f"🚉 Origin: {orig}\n"
            f"🏁 Destination: {dest}\n"
            f"📅 Travel Date: {date_t} (Dep: 17:05, Arr: 05:45)\n"
            f"🎫 Class: {train_class}\n"
            f"👥 Passengers: {pax}\n"
            f"🛌 Allocated Berths: {', '.join(berths)}\n"
            f"💰 Total Amount: ₹{int(total_fare)}\n"
            f"⏱️ Status: Confirmed\n\n"
            f"You will get the SMS with all the booking details.\n\n"
            f"Do you need any other help?"
        )

        return {
            "response": response,
            "tool_result": booking_result,
            "current_step": "COMPLETED",
            "waiting_for_confirmation": False,
        }
    except Exception as e:
        return {
            "response": f"I'm sorry, train reservation failed: {str(e)}.",
            "error": str(e),
            "current_step": "FAILED",
            "waiting_for_confirmation": False,
        }



async def handle_greeting_node(state: AgentState) -> dict[str, Any]:
    """Handles short, natural initial greeting and post-booking greetings."""
    latest_task = state.get("latest_completed_task")
    if latest_task and latest_task.get("status") == "COMPLETED":
        result = latest_task.get("result") or {}
        collected = (latest_task.get("collected_data") or {})
        task_type = latest_task.get("type") or collected.get("intent")

        if task_type == "ORDER_FOOD" or "restaurant" in collected or "order_id" in result:
            restaurant = result.get("restaurant") or collected.get("restaurant", "the restaurant")
            partner = result.get("delivery_partner", "Suresh V.")
            food = collected.get("food", "order")
            response = (
                f"Hello! 👋 Your delivery partner {partner} is on the way to pick up your {food} from {restaurant} (ETA: ~20 mins). "
                f"How can I help you right now?"
            )
        elif task_type == "BOOK_SERVICE" or "service_name" in collected:
            service = collected.get("service_name", "home service")
            team = result.get("assigned_team", "Urban Clean Professional Team")
            response = (
                f"Hello! 👋 Your {service} booking with {team} is confirmed. "
                f"How can I help you right now?"
            )
        else:
            driver_name = result.get("driver_name", "your driver")
            plate = result.get("vehicle_plate", "")
            plate_info = f" ({plate})" if plate else ""
            ride_type = collected.get("ride_type", "ride")
            response = (
                f"Hello! 👋 Your driver {driver_name}{plate_info} is on the way for your {ride_type} (ETA: ~4 mins). "
                f"How can I help you right now?"
            )
    else:
        llm_reply = state.get("_llm_reply")
        response = llm_reply or "How can I help you?"

    return {
        "response": response,
        "current_step": "GREETING_ANSWERED",
        "waiting_for_confirmation": False,
    }


async def handle_gratitude_node(state: AgentState) -> dict[str, Any]:
    """Warmly acknowledges gratitude and offers proactive post-booking assistance."""
    latest_task = state.get("latest_completed_task")
    if latest_task and latest_task.get("status") == "COMPLETED":
        result = latest_task.get("result") or {}
        collected = (latest_task.get("collected_data") or {})
        task_type = latest_task.get("type") or collected.get("intent")

        if task_type == "ORDER_FOOD" or "restaurant" in collected or "order_id" in result:
            restaurant = result.get("restaurant") or collected.get("restaurant", "the restaurant")
            partner = result.get("delivery_partner", "Suresh V.")
            food = collected.get("food", "order")
            response = (
                f"You're very welcome! Your delivery partner {partner} is on the way to pick up your {food} from {restaurant}. "
                f"Let me know if you want live updates or need any other help!"
            )
        elif task_type == "BOOK_SERVICE" or "service_name" in collected:
            service = collected.get("service_name", "home service")
            response = (
                f"You're very welcome! Your {service} is all set. Let me know if you need to make any changes or book another service!"
            )
        else:
            driver_name = result.get("driver_name", "your driver")
            plate = result.get("vehicle_plate", "")
            plate_info = f" ({plate})" if plate else ""
            ride_type = collected.get("ride_type", "ride")
            response = (
                f"You're very welcome! Your driver {driver_name}{plate_info} is on the way for your {ride_type}. "
                f"Let me know if you need to track the vehicle, get the driver's contact number, or need any other help (like ordering food or home cleaning)!"
            )
    else:
        response = (
            "You're very welcome! I'm here anytime you need to book a ride, order food, "
            "or schedule services like home cleaning. Have a wonderful day!"
        )

    return {
        "response": response,
        "current_step": "GRATITUDE_ACKNOWLEDGED",
        "waiting_for_confirmation": False,
    }


async def handle_task_inquiry_node(state: AgentState) -> dict[str, Any]:
    """Answers user inquiries about live tracking, status, driver location, and ETA dynamically."""
    latest_task = state.get("latest_completed_task")
    result = (latest_task or {}).get("result") or {}
    collected = (latest_task or {}).get("collected_data") or state.get("collected_data") or {}
    task_type = (latest_task or {}).get("type") or collected.get("intent")
    user_msg_lower = state.get("user_message", "").lower()
    created_at = result.get("created_at") or (latest_task or {}).get("created_at")

    # 1. Food Order Real-Time Dynamic Tracking
    if task_type == "ORDER_FOOD" or "restaurant" in collected or "order_id" in result:
        order_id = result.get("order_id") or result.get("booking_id") or "food_live"
        restaurant = result.get("restaurant") or collected.get("restaurant", "Paradise")
        food = collected.get("food", "biryani")
        partner = result.get("delivery_partner", "Suresh V.")
        partner_phone = result.get("delivery_phone", "+91 98765 12345")
        delivery_loc = result.get("delivery_location") or collected.get("delivery_location", "your delivery location")
        initial_eta = result.get("initial_eta_minutes", 20)

        # Calculate time-based dynamic status and remaining ETA
        status_code, status_desc, remaining_eta = calculate_dynamic_food_status(created_at, initial_eta=initial_eta)

        if "order details" in user_msg_lower or "booking details" in user_msg_lower:
            response = (
                f"Here are your current order details:\n\n"
                f"• 🍔 Item: {food.title()}\n"
                f"• 🏪 Restaurant: {restaurant}\n"
                f"• 🎫 Order ID: {order_id}\n"
                f"• 📍 Delivering to: {delivery_loc}\n"
                f"• 🛵 Delivery Partner: {partner} ({partner_phone})\n"
                f"• ⚡ Status: {status_code}\n"
                f"• ⏳ Estimated delivery: {f'about {remaining_eta} minutes' if remaining_eta > 0 else 'Arriving now / Delivered'}"
            )
        elif status_code == "DELIVERED":
            response = (
                f"Your order has been delivered successfully!\n\n"
                f"• 🍔 Item: {food.title()} from {restaurant}\n"
                f"• 📍 Delivered to: {delivery_loc}\n"
                f"• 🎫 Order ID: {order_id}\n"
                f"• ✅ Status: Delivered\n\n"
                f"Enjoy your meal! Let me know if you need help with anything else."
            )
        elif status_code == "ARRIVING":
            response = (
                f"Your order is on the way and should reach you in about {remaining_eta or 5} minutes.\n\n"
                f"• 🍔 {food.title()} from {restaurant}\n"
                f"• 🛵 Status: Out for delivery\n"
                f"• ⏳ Estimated delivery: about {remaining_eta or 5} minutes\n"
                f"• 📍 Delivering to: {delivery_loc}\n"
                f"• 🎫 Order ID: {order_id}"
            )
        elif status_code == "OUT_FOR_DELIVERY":
            response = (
                f"Your order has been picked up and is now on the way to you.\n\n"
                f"• 🍔 {food.title()} from {restaurant}\n"
                f"• 🛵 Status: Out for delivery (Partner: {partner})\n"
                f"• ⏳ Estimated delivery: about {remaining_eta} minutes\n"
                f"• 📍 Delivering to: {delivery_loc}\n"
                f"• 🎫 Order ID: {order_id}"
            )
        elif status_code == "READY_FOR_PICKUP":
            response = (
                f"I checked the latest status.\n\n"
                f"• 🍔 Your {food} is being prepared and is almost ready for pickup from {restaurant}.\n"
                f"• ⏳ Estimated delivery: about {remaining_eta} minutes\n"
                f"• 📍 Delivering to: {delivery_loc}\n"
                f"• 🎫 Order ID: {order_id}"
            )
        else:
            # PREPARING (Initial stage)
            response = (
                f"Your order is currently being prepared at the restaurant.\n\n"
                f"• 🍔 {food.title()} — {restaurant}\n"
                f"• ⏱️ Estimated delivery: about {remaining_eta} minutes\n"
                f"• 📍 Delivering to: {delivery_loc}\n"
                f"• 🎫 Order ID: {order_id}\n\n"
                f"Delivery partner {partner} is on the way to pick up your order once ready."
            )

    # 2. Bus Booking Status / Details / Cancellation
    elif task_type == "BOOK_BUS" or (result.get("booking_id") and "bus_" in str(result.get("booking_id"))) or collected.get("intent") == "BOOK_BUS":
        pnr = result.get("pnr") or collected.get("pnr") or result.get("booking_id") or collected.get("booking_id")
        op = result.get("operator_name") or collected.get("operator_name", "Orange Tours & Travels")
        orig = result.get("origin") or collected.get("origin_city", "Hyderabad")
        dest = result.get("destination") or collected.get("destination_city", "Visakhapatnam")
        date_t = result.get("travel_date") or collected.get("travel_date", "Tomorrow")
        seats = result.get("selected_seats") or collected.get("selected_seats") or ["U1"]
        fare = result.get("total_fare") or collected.get("fare_amount", 997.5)
        status = result.get("booking_status", "CONFIRMED")

        if any(w in user_msg_lower for w in ["cancel", "cancellation"]):
            # Cancellation action
            try:
                b_id = result.get("booking_id") or collected.get("booking_id") or pnr
                cancel_res = await tool_executor.execute("cancel_bus_booking", {"booking_id": b_id})
                result["booking_status"] = "CANCELLED"
                response = (
                    f"Your bus ticket (PNR: {pnr}) has been cancelled.\n\n"
                    f"💰 Refund Amount: ₹{cancel_res.get('refund_amount', fare * 0.9)}\n"
                    f"⏱️ Status: Cancelled\n\n"
                    f"You will get the SMS with all the booking details."
                )
            except Exception as e:
                response = f"I could not cancel your bus booking: {str(e)}."
        else:
            response = (
                f"Here are your bus booking details:\n\n"
                f"🚌 Operator: {op}\n"
                f"🎫 PNR: {pnr}\n"
                f"📍 Route: {orig} to {dest}\n"
                f"📅 Travel Date: {date_t} (Dep: 20:30, Arr: 06:00)\n"
                f"💺 Seats: {', '.join(seats)}\n"
                f"💰 Total Fare: ₹{int(fare)}\n"
                f"⏱️ Status: {status}\n\n"
                f"You will get the SMS with all the booking details."
            )

    # 3. Train Booking Status / Details / Cancellation
    elif task_type == "BOOK_TRAIN" or (result.get("booking_id") and "train_" in str(result.get("booking_id"))) or collected.get("intent") == "BOOK_TRAIN":
        pnr = result.get("pnr") or collected.get("pnr") or result.get("booking_id") or collected.get("booking_id")
        train_name = result.get("train_name") or collected.get("service_name") or "Godavari Express"
        train_num = result.get("train_number") or collected.get("train_number", "12728")
        orig = result.get("origin_station") or collected.get("origin_city", "Secunderabad Junction (SC)")
        dest = result.get("destination_station") or collected.get("destination_city", "Visakhapatnam Junction (VSKP)")
        date_t = result.get("travel_date") or collected.get("travel_date", "Tomorrow")
        t_class = result.get("selected_class") or collected.get("train_class", "3A")
        berths = result.get("allocated_berths") or collected.get("berths") or ["B3-21 (Lower)"]
        fare = result.get("total_fare") or collected.get("fare_amount", 1150.0)
        status = result.get("booking_status", "CONFIRMED")

        if any(w in user_msg_lower for w in ["cancel", "cancellation"]):
            try:
                b_id = result.get("booking_id") or pnr
                cancel_res = await tool_executor.execute("cancel_train_booking", {"booking_id": b_id})
                result["booking_status"] = "CANCELLED"
                response = (
                    f"Your train ticket (PNR: {pnr}) has been cancelled.\n\n"
                    f"🚆 Train: {train_name} ({train_num})\n"
                    f"💰 Refund Amount: ₹{cancel_res.get('refund_amount', fare - 60.0)}\n"
                    f"⏱️ Status: Cancelled\n\n"
                    f"You will get the SMS with all the booking details."
                )
            except Exception as e:
                response = f"I could not cancel your train ticket: {str(e)}."
        else:
            response = (
                f"Here are your train reservation details:\n\n"
                f"🚆 Train: {train_name} ({train_num})\n"
                f"🎫 IRCTC PNR: {pnr}\n"
                f"🚉 Route: {orig} to {dest}\n"
                f"📅 Travel Date: {date_t} (Dep: 17:05, Arr: 05:45)\n"
                f"🎫 Class: {t_class}\n"
                f"🛌 Berths: {', '.join(berths)}\n"
                f"💰 Total Fare: ₹{int(fare)}\n"
                f"⏱️ Status: {status}\n\n"
                f"You will get the SMS with all the booking details."
            )

    # 4. Home Service Real-Time Dynamic Tracking
    elif task_type == "BOOK_SERVICE" or "service_name" in collected:
        booking_id = result.get("booking_id") or "uc_live"
        service_name = collected.get("service_name") or result.get("service_name", "Deep Cleaning")
        package = collected.get("package") or result.get("package", "Standard")
        assigned_team = result.get("assigned_team", "Urban Clean Professional Team #4")
        team_phone = result.get("team_lead_phone", "+91 98765 88990")
        service_loc = result.get("service_location") or collected.get("service_location", "your address")
        date_pref = result.get("date") or collected.get("preferred_date", "Today")
        slot = result.get("slot", "2:00 PM – 4:00 PM")

        svc_code, svc_desc = calculate_dynamic_service_status(created_at)

        response = (
            f"Here is your confirmed Urban Clean appointment schedule:\n\n"
            f"• 🧹 Service: {service_name} ({package})\n"
            f"• 📅 Date: {date_pref}\n"
            f"• ⏰ Scheduled Time: {slot}\n"
            f"• 📍 Address: {service_loc}\n"
            f"• 👤 Assigned Team: {assigned_team} (📞 {team_phone})\n"
            f"• ⏱️ Live Status: {svc_desc}\n"
            f"• 🎫 Booking ID: {booking_id}\n\n"
            f"The team is scheduled to arrive on time with complete equipment. Let me know if you need any assistance!\n\n"
            f"You will get the SMS with all the booking details."
        )

    # 5. Ride Booking Real-Time Dynamic Tracking
    elif result.get("booking_id") or "pickup" in collected:
        booking_id = result.get("booking_id", "uber_live")
        driver_name = result.get("driver_name", "Rajesh K.")
        vehicle_plate = result.get("vehicle_plate", "TS 09 UB 1234")
        driver_phone = result.get("driver_phone", "+91 98765 43210")
        ride_type = collected.get("ride_type", "Uber Moto")
        pickup = collected.get("pickup", "your pickup location")
        dest = collected.get("destination", "your destination")
        initial_eta = result.get("initial_eta_minutes", 4)

        ride_code, ride_desc, remaining_eta, distance_km, driver_landmark = calculate_dynamic_ride_status(
            created_at, initial_eta=initial_eta, pickup_loc=pickup
        )

        is_details_query = any(w in user_msg_lower for w in ["ride details", "booking details", "details?", "details"])
        is_contact_or_plate_query = any(w in user_msg_lower for w in ["phone", "contact", "call", "number", "plate"])
        is_where_are_we = any(w in user_msg_lower for w in ["where are we", "where we are", "current trip location"])

        if is_where_are_we and ride_code in ["TRIP_IN_PROGRESS", "TRIP_COMPLETED"]:
            response = (
                f"You are currently on your trip to {dest}.\n\n"
                f"🚕 Vehicle: {ride_type} ({vehicle_plate})\n"
                f"👤 Driver: {driver_name}\n"
                f"📍 Current Status: {driver_landmark}\n"
                f"🏁 Heading to: {dest}"
            )
        elif is_details_query or is_contact_or_plate_query:
            # Full booking details / Driver contact & vehicle info
            response = (
                f"Here are your current ride details:\n\n"
                f"🛵 Vehicle: {ride_type} ({vehicle_plate})\n"
                f"👤 Driver: {driver_name}\n"
                f"📞 Driver phone: {driver_phone}\n"
                f"📍 Pickup: {pickup}\n"
                f"🏁 Destination: {dest}\n"
                f"⏱️ Driver arrival: {f'about {remaining_eta} minutes' if remaining_eta > 0 else 'Arrived at pickup'} (Status: On the way)\n"
                f"🎫 Booking ID: {booking_id}"
            )
        elif ride_code == "TRIP_COMPLETED":
            response = (
                f"Your {ride_type} trip has been completed successfully.\n\n"
                f"• 🚕 Vehicle: {ride_type} ({vehicle_plate})\n"
                f"• 👤 Driver: {driver_name}\n"
                f"• 📍 Destination: {dest}\n"
                f"• 🎫 Booking ID: {booking_id}\n\n"
                f"Thank you for riding with us! Let me know if you need anything else."
            )
        elif ride_code == "DRIVER_ARRIVED":
            response = (
                f"Your driver has arrived at the {pickup} pickup location.\n\n"
                f"🛵 {ride_type} ({vehicle_plate})\n"
                f"👤 {driver_name} (📞 {driver_phone})\n"
                f"📍 Status: Arrived at pickup"
            )
        elif ride_code == "DRIVER_NEARBY":
            response = (
                f"Your driver is getting closer.\n\n"
                f"📍 About {distance_km} km away ({driver_landmark})\n"
                f"⏱️ Estimated arrival: about {remaining_eta or 1} minute\n"
                f"🛵 Status: On the way to pickup"
            )
        elif ride_code == "DRIVER_CLOSER":
            response = (
                f"Your driver {driver_name} is getting closer to your pickup point in {pickup}.\n\n"
                f"📍 About {distance_km} km away ({driver_landmark})\n"
                f"⏱️ Estimated arrival: about {remaining_eta} minutes\n"
                f"🛵 Status: On the way to pickup"
            )
        else:
            # DRIVER_EN_ROUTE (Initial stage)
            response = (
                f"Your driver {driver_name} is on the way to your pickup location in {pickup}.\n\n"
                f"📍 About {distance_km} km away ({driver_landmark})\n"
                f"⏱️ Estimated arrival: about {remaining_eta} minutes\n"
                f"🛵 Status: On the way to pickup"
            )
    else:
        response = (
            "You don't currently have an active booking or order. "
            "I can help you hail a ride, order food, or arrange home cleaning! What would you like to do?"
        )

    return {
        "response": response,
        "current_step": "INQUIRY_ANSWERED",
        "waiting_for_confirmation": False,
    }


async def handle_food_order_node(state: AgentState) -> dict[str, Any]:
    """Direct handler for food ordering when bypassed to node."""
    return await merge_information_node(state)


async def handle_service_booking_node(state: AgentState) -> dict[str, Any]:
    """Direct handler for service booking when bypassed to node."""
    return await merge_information_node(state)
