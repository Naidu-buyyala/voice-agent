from typing import Any
from app.agent.state import AgentState
from app.agent.extractor import GeminiExtractor
from app.tools.executor import tool_executor

extractor = GeminiExtractor()


async def understand_intent_node(state: AgentState) -> dict[str, Any]:
    """Node 1: Extract intent, entities, and confirmations via Gemini."""
    extraction = await extractor.extract(
        state["user_message"],
        current_collected=state.get("collected_data", {}),
        last_assistant_message=state.get("last_assistant_message"),
    )

    updates: dict[str, Any] = {
        "intent": extraction.intent,
        "current_step": "UNDERSTOOD",
        "confirmation_received": extraction.confirmation,
    }

    new_fields = {}
    if extraction.pickup:
        new_fields["pickup"] = extraction.pickup
    if extraction.destination:
        new_fields["destination"] = extraction.destination
    if extraction.ride_type:
        new_fields["ride_type"] = extraction.ride_type
    if extraction.service_type:
        new_fields["service_type"] = extraction.service_type

    updates["_new_fields"] = new_fields
    if extraction.conversational_reply:
        updates["_llm_reply"] = extraction.conversational_reply
    if extraction.needs_landmark_clarification:
        updates["_needs_landmark_clarification"] = True

    return updates


async def merge_information_node(state: AgentState) -> dict[str, Any]:
    """Node 2: Merge new extractions into persistent collected_data."""
    collected = dict(state.get("collected_data", {}))
    new_fields = state.get("_new_fields", {})

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

    # If service_type or ride_type changed, invalidate old fare
    if new_fields.get("service_type") or new_fields.get("ride_type"):
        collected.pop("fare_amount", None)

    # If service_type or ride_type is provided in new_fields, update ride_type
    if new_fields.get("ride_type"):
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
    """Node 3: Verify required parameters (pickup, destination, landmark refinement, service_type)."""
    collected = dict(state.get("collected_data", {}))
    missing = []

    pickup = collected.get("pickup")
    dest = collected.get("destination")
    last_asked = collected.get("last_asked")

    # Check if pickup and destination are identical / same location
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

    # If pickup & destination are known, check if user has selected a service / vehicle type
    if not missing and not collected.get("service_type") and not collected.get("ride_type"):
        missing.append("service_type")

    return {
        "collected_data": collected,
        "missing_fields": missing,
        "current_step": "CHECKED_REQUIREMENTS",
    }


async def ask_user_node(state: AgentState) -> dict[str, Any]:
    """Node 4: Inquire user for missing required info using warm, helpful human phrasing."""
    missing = state.get("missing_fields", [])
    collected = dict(state.get("collected_data", {}))
    llm_reply = state.get("_llm_reply")

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
    """Node 5: Uses tool executor to fetch estimate, then asks user for confirmation."""
    collected = dict(state.get("collected_data", {}))
    collected.pop("last_asked", None)
    pickup = collected.get("pickup")
    destination = collected.get("destination")
    ride_type = collected.get("ride_type", "Uber Go")

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

    # Service icon for display
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


async def cancel_ride_node(state: AgentState) -> dict[str, Any]:
    """Node 6: Handles user cancellation with warm reassurance."""
    return {
        "response": "No worries at all! I've cancelled that booking request for you. Let me know whenever you're ready to head somewhere next time!",
        "current_step": "CANCELLED",
        "waiting_for_confirmation": False,
    }


async def book_ride_node(state: AgentState) -> dict[str, Any]:
    """Node 7: Executes booking with provider ONLY after verified confirmation."""
    collected = state.get("collected_data", {})
    pickup = collected.get("pickup")
    destination = collected.get("destination")
    ride_type = collected.get("ride_type", "Uber Go")
    fare_amount = collected.get("fare_amount", 450.0)

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

        response = (
            f"All set! Your {ride_type} has been booked successfully.\n\n"
            f"🚕 Booking ID: {booking_id}\n"
            f"👤 Driver: {driver_name}{plate_info}\n"
            f"📞 Driver Phone: {driver_phone}\n\n"
            f"Your ride will arrive shortly (ETA: ~4 mins). Let me know if you need to track your vehicle, contact the driver, or need any other help!"
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


async def handle_greeting_node(state: AgentState) -> dict[str, Any]:
    """Handles warm, human greeting and introduces multi-domain action capabilities."""
    latest_task = state.get("latest_completed_task")
    if latest_task and latest_task.get("status") == "COMPLETED":
        result = latest_task.get("result") or {}
        driver_name = result.get("driver_name", "your driver")
        plate = result.get("vehicle_plate", "")
        plate_info = f" ({plate})" if plate else ""
        ride_type = (latest_task.get("collected_data") or {}).get("ride_type", "ride")
        response = (
            f"Hello! 👋 Your driver {driver_name}{plate_info} is on the way for your {ride_type} (ETA: ~4 mins). "
            f"How can I help you right now? I can share the driver's phone number, check vehicle status, or help you order food or book home services!"
        )
    else:
        response = (
            "Hello! 👋 I'm your Universal AI Action Assistant.\n\n"
            "Here is what I can do for you:\n"
            "• 🚗 **Hail Rides**: Cabs, bikes (Uber Moto), autos, or parcel delivery\n"
            "• 🍔 **Order Food**: Meals, cuisines, or quick bites\n"
            "• 🧹 **Home Services**: Deep cleaning, plumbing, or repairs (Urban Clean)\n\n"
            "What would you like to do today? Just type naturally!"
        )

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
        driver_name = result.get("driver_name", "your driver")
        plate = result.get("vehicle_plate", "")
        plate_info = f" ({plate})" if plate else ""
        ride_type = (latest_task.get("collected_data") or {}).get("ride_type", "ride")
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
    """Answers user inquiries about vehicle location, driver phone number, ETA, and ride status."""
    latest_task = state.get("latest_completed_task")
    result = (latest_task or {}).get("result") or {}
    collected = (latest_task or {}).get("collected_data") or state.get("collected_data") or {}

    booking_id = result.get("booking_id")
    if booking_id:
        driver_name = result.get("driver_name", "Rajesh K.")
        vehicle_plate = result.get("vehicle_plate", "TS 09 UB 1234")
        driver_phone = result.get("driver_phone", "+91 98765 43210")
        ride_type = collected.get("ride_type", "Uber ride")
        pickup = collected.get("pickup", "your pickup location")
        dest = collected.get("destination", "your destination")

        response = (
            f"Here are the live details for your {ride_type}:\n\n"
            f"• 🚕 Vehicle: {ride_type} ({vehicle_plate})\n"
            f"• 👤 Driver: {driver_name}\n"
            f"• 📞 Driver Phone: {driver_phone}\n"
            f"• ⏱️ Status: On the way to {pickup} (ETA: ~4 mins)\n"
            f"• 📍 Heading to: {dest}\n"
            f"• 🎫 Booking ID: {booking_id}\n\n"
            f"Your rider is on the way! Do you want any other help or details?"
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
    """Handles food ordering requests across multi-domain architecture."""
    response = (
        "I'd love to help you order food! Which restaurant, dish, or cuisine are you craving today, "
        "and where should we deliver to?"
    )
    return {
        "response": response,
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
    }


async def handle_service_booking_node(state: AgentState) -> dict[str, Any]:
    """Handles home services like Urban Clean / Urban Company."""
    response = (
        "I'd be glad to help you arrange home services (like Urban Clean / Urban Company)! "
        "We support deep home cleaning, kitchen & bathroom cleaning, plumbing, and appliance repair. "
        "Which service do you need and for which location?"
    )
    return {
        "response": response,
        "current_step": "WAITING_FOR_USER",
        "waiting_for_confirmation": False,
    }
