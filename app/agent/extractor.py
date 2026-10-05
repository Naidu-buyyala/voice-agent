import asyncio
import json
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types

from app.agent.schemas import AgentExtraction, EntityData, PreferenceData
from app.agent.prompts import EXTRACTION_SYSTEM_PROMPT
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self._client: Optional[genai.Client] = None
        self._client_loop = None

    @staticmethod
    def resolve_travel_date_and_time(
        text: str, reference_date: Optional[date] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Deterministically resolves natural language relative dates (tomorrow, day after tomorrow,
        today, next monday, october 15) using runtime local time, and parses departure windows
        (evening, after 8 PM, morning, afternoon).
        """
        ref = reference_date or datetime.now().date()
        lower = text.strip().lower()

        resolved_date_str: Optional[str] = None
        # 1. Relative dates
        if "day after tomorrow" in lower:
            resolved_date_str = (ref + timedelta(days=2)).isoformat()
        elif "tomorrow" in lower:
            resolved_date_str = (ref + timedelta(days=1)).isoformat()
        elif "today" in lower:
            resolved_date_str = ref.isoformat()
        elif "this weekend" in lower:
            # upcoming Saturday
            days_ahead = (5 - ref.weekday()) % 7
            days_ahead = 7 if days_ahead == 0 else days_ahead
            resolved_date_str = (ref + timedelta(days=days_ahead)).isoformat()
        elif "next monday" in lower:
            days_ahead = (7 - ref.weekday()) % 7
            days_ahead = 7 if days_ahead == 0 else days_ahead
            resolved_date_str = (ref + timedelta(days=days_ahead)).isoformat()
        elif "next friday" in lower:
            days_ahead = (4 - ref.weekday()) % 7
            days_ahead = 7 if days_ahead == 0 else days_ahead
            resolved_date_str = (ref + timedelta(days=days_ahead)).isoformat()

        # 2. Preferred departure window / time
        dep_window: Optional[str] = None
        if "after 8 pm" in lower or "after 8pm" in lower or "after 20:00" in lower:
            dep_window = "After 8 PM"
        elif "after 7 pm" in lower or "after 7pm" in lower:
            dep_window = "After 7 PM"
        elif any(w in lower for w in ["after 6 pm", "after 6pm", "around 6 pm", "around 6pm", "at 6 pm", "at 6pm", "6 pm", "6pm", "6:00 pm"]):
            dep_window = "Evening (~6 PM)"
        elif any(w in lower for w in ["around 8 pm", "around 8pm", "at 8 pm", "at 8pm", "8 pm", "8pm"]):
            dep_window = "Night (~8 PM)"
        elif "evening" in lower or "night" in lower:
            dep_window = "Evening (6 PM - 11 PM)"
        elif "morning" in lower:
            dep_window = "Morning (6 AM - 12 PM)"
        elif "afternoon" in lower:
            dep_window = "Afternoon (12 PM - 5 PM)"
        elif "overnight" in lower:
            dep_window = "Overnight (8 PM - 6 AM)"
        elif "earliest" in lower:
            dep_window = "Earliest Available"

        return resolved_date_str, dep_window


    @property
    def client(self) -> Optional[genai.Client]:
        if not self.api_key:
            return None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self._client is None or self._client_loop != current_loop:
            self._client = genai.Client(api_key=self.api_key)
            self._client_loop = current_loop
        return self._client

    @staticmethod
    def _is_valid_location(val: Optional[str]) -> bool:
        if not val:
            return False
        clean = val.strip().lower().rstrip(".,!?")
        if len(clean) < 3:
            return False
        invalid_words = {
            "hi", "hello", "hey", "hola", "sup", "yo", "greetings",
            "good morning", "good evening", "good afternoon",
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "fine", "cool",
            "proceed", "confirm", "book it", "go ahead", "do it",
            "no", "nope", "cancel", "stop", "abort", "don't", "nevermind",
            "thanks", "thank you", "bye", "goodbye", "nowhere", "anywhere",
            "nothing", "none", "please", "help",
            "your wish", "you decide", "anything is fine", "whatever is good",
            "best one", "cheapest one", "fastest one"
        }
        if clean in invalid_words:
            return False
        # If it's a conversational sentence like "i want to book an uber", not a location
        conversational_phrases = ["i want", "book a", "book an", "need a", "give me", "where can", "can you", "what would"]
        if any(phrase in clean for phrase in conversational_phrases):
            return False
        return True

    BROAD_LOCALITIES = {
        "gachibowli": "Got it, Gachibowli! Could you let me know which specific building or landmark you're at (for instance, near IKEA, DLF Cyber City, or Bio-Diversity Park)?",
        "madhapur": "Got it, Madhapur! Could you share which specific building or spot you're near (like Inorbit Mall, Mindspace, or Cyber Towers)?",
        "kondapur": "Got it, Kondapur! Which landmark or building should the driver meet you at (like near Sarath City Capital Mall, RTO, or botanical gardens)?",
        "kukatpally": "Got it, Kukatpally! Could you tell me the exact spot (like near Nexus/Forum Sujana Mall, KPHB Metro, or a specific phase)?",
        "jubilee hills": "Got it, Jubilee Hills! Could you specify the road number or landmark (like Road No. 36, Apollo Hospital, or Checkpost)?",
        "banjara hills": "Got it, Banjara Hills! Could you let me know the road number or building (like near GVK One Mall, Care Hospital, or Road No. 1)?",
        "whitefield": "Got it, Whitefield! Which tech park or campus are you at (like ITPL, Phoenix Marketcity, or a specific gate)?",
        "indiranagar": "Got it, Indiranagar! Which landmark or cross street are you on (like 100 Feet Road, 12th Main, or near the Metro)?",
        "koramangala": "Got it, Koramangala! Which block or landmark should we pick you up from (like near Sony World Signal, Forum Mall, or 4th Block)?",
    }

    @classmethod
    def is_broad_locality_only(cls, text: Optional[str]) -> tuple[bool, Optional[str]]:
        if not text:
            return False, None
        clean = text.strip().lower().rstrip(".,!?")
        for locality, prompt_msg in cls.BROAD_LOCALITIES.items():
            if clean == locality or clean in [f"in {locality}", f"from {locality}", f"to {locality}", f"at {locality}"]:
                return True, prompt_msg
        return False, None

    @staticmethod
    def detect_service_type(text: str) -> tuple[Optional[str], Optional[str]]:
        lower = text.strip().lower()
        if any(w in lower for w in ["bike", "moto", "motorcycle", "two wheeler", "scooter"]):
            return "bike", "Uber Moto"
        elif any(w in lower for w in ["auto", "rickshaw", "tuk tuk", "three wheeler"]):
            return "auto", "Uber Auto"
        elif any(w in lower for w in ["parcel", "package", "courier", "delivery", "connect", "send item"]):
            return "parcel", "Uber Connect"
        elif any(w in lower for w in ["cab", "car", "taxi", "uber go", "premier", "sedan", "xl"]):
            if "premier" in lower:
                return "cab", "Uber Premier"
            elif "xl" in lower:
                return "cab", "Uber XL"
            return "cab", "Uber Go"
        return None, None

    @staticmethod
    def is_same_location(loc1: Optional[str], loc2: Optional[str]) -> bool:
        if not loc1 or not loc2:
            return False
        c1 = loc1.strip().lower().rstrip(".,!?")
        c2 = loc2.strip().lower().rstrip(".,!?")
        if c1 == c2:
            return True
        stop_words = {"near", "in", "at", "to", "from", "the", "by", "of", "and"}
        words1 = {w for w in c1.replace(",", " ").split() if w not in stop_words}
        words2 = {w for w in c2.replace(",", " ").split() if w not in stop_words}
        if not words1 or not words2:
            return False
        if words1 == words2:
            return True
        city_names = {"hyderabad", "bengaluru", "bangalore", "mumbai", "delhi", "chennai", "kolkata", "pune"}
        diff1 = words1 - words2
        diff2 = words2 - words1
        if not diff1 and words1.issubset(city_names):
            return False
        if not diff2 and words2.issubset(city_names):
            return False
        if not diff1 or not diff2:
            return True
        return False

    def _sanitize_extraction(
        self,
        extraction: AgentExtraction,
        user_message: str = "",
        current_collected: Optional[dict] = None,
        current_workflow_state: Optional[str] = None,
    ) -> AgentExtraction:
        lower = user_message.lower().strip()
        words = lower.split()
        collected = current_collected or {}

        # Sanitize locations
        if extraction.pickup and not self._is_valid_location(extraction.pickup):
            extraction.pickup = None
        if extraction.destination and not self._is_valid_location(extraction.destination):
            extraction.destination = None

        # Strict intent and service type routing across ALL primary domains
        if any(w in lower for w in [
            "bus booking", "book a bus", "need a bus ticket", "travel by bus", "sleeper bus", "volvo bus",
            "bus from", "bus ticket", "buses to", "bus to", "find buses", "book bus", "bus"
        ]):
            extraction.intent = "BOOK_BUS"
            extraction.service_type = "bus"
            extraction.entities.service_type = "bus"
            extraction.entities.travel_mode = "bus"
            extraction.ride_type = None
            extraction.entities.ride_type = None
            if not extraction.entities.origin_city and not extraction.entities.destination_city and not (current_collected or {}).get("origin_city"):
                extraction.assistant_message = "Sure! I can help you book a bus. Where would you like to travel from and to?"
        elif any(w in lower for w in [
            "train booking", "book a train", "need a train ticket", "train ticket", "railway ticket",
            "irctc", "express train", "vande bharat", "berth", "train from", "train to", "train"
        ]):
            extraction.intent = "BOOK_TRAIN"
            extraction.service_type = "train"
            extraction.entities.service_type = "train"
            extraction.entities.travel_mode = "train"
            extraction.ride_type = None
            extraction.entities.ride_type = None
            if not extraction.entities.origin_city and not extraction.entities.destination_city and not (current_collected or {}).get("origin_city"):
                extraction.assistant_message = "Sure! Which railway station or city are you departing from?"
        elif any(w in lower for w in [
            "order food", "food delivery", "order lunch", "order dinner", "order breakfast", "food",
            "biryani", "pizza", "burger", "dosa", "noodles", "shawarma"
        ]) and not any(w in lower for w in ["ride", "clean", "bus", "train"]):
            extraction.intent = "ORDER_FOOD"
            extraction.service_type = "food"
            extraction.entities.service_type = "food"
            extraction.ride_type = None
            extraction.entities.ride_type = None
            if not extraction.entities.food and not (current_collected or {}).get("food"):
                extraction.assistant_message = "Sure! What delicious food or cuisine are you craving today?"
        elif any(w in lower for w in [
            "book cleaning", "deep cleaning", "home cleaning", "urban clean", "urbanclean",
            "plumbing", "plumber", "electrician", "appliance repair", "cleaning", "home service"
        ]) and not any(w in lower for w in ["ride", "food", "bus", "train"]):
            extraction.intent = "BOOK_SERVICE"
            extraction.service_type = "cleaning"
            extraction.entities.service_type = "cleaning"
            extraction.ride_type = None
            extraction.entities.ride_type = None
            if not extraction.entities.service_name and not (current_collected or {}).get("service_name"):
                extraction.assistant_message = "What type of cleaning or home service do you need?"
        elif not extraction.service_type and user_message:
            st, rt = self.detect_service_type(user_message)
            if st:
                extraction.service_type = st
                extraction.ride_type = extraction.ride_type or rt
                extraction.intent = "BOOK_RIDE"
                extraction.entities.service_type = st
                extraction.entities.ride_type = extraction.ride_type

        # Distinguish delegation from confirmation
        delegation_phrases = [
            "your wish", "you decide", "anything is fine", "whatever is good",
            "any is fine", "up to you", "you choose", "choose for me",
            "pick for me", "best one", "cheapest one", "fastest one",
            "any good", "something nearby", "same as before"
        ]
        is_delegation = any(phrase in lower for phrase in delegation_phrases)
        if is_delegation:
            extraction.confirmation = None
            extraction.confirmation_type = None
            if not extraction.preferences.selection_preference:
                extraction.preferences.selection_preference = "AI_CHOOSE"
        else:
            # If the user did not say any delegation phrase, never allow AI_CHOOSE
            if extraction.preferences.selection_preference == "AI_CHOOSE":
                extraction.preferences.selection_preference = None
                extraction.preferences.selection_strategy = None

        # Specifying or selecting a vehicle/service type is NOT a confirmation of a previous quote
        if extraction.service_type and extraction.confirmation:
            if not any(a in lower for a in ["yes", "proceed", "confirm", "go ahead", "sure", "book it"]):
                extraction.confirmation = None
                extraction.confirmation_type = None

        # Check restaurant mention in user message
        if not extraction.entities.restaurant:
            famous_restaurants = ["mehfil", "meghana", "bawarchi", "paradise", "domino", "mcdonald", "subway", "kfc", "pista house", "kritunga", "shah ghouse", "haldiram"]
            for r in famous_restaurants:
                if r in lower:
                    extraction.entities.restaurant = r.title()
                    extraction.preferences.selection_preference = "USER_SPECIFIED"
                    break

        # Context-aware remapping: if the user was directly answering a pickup or destination question
        last_asked = collected.get("last_asked")
        if last_asked in ["pickup", "pickup_landmark"] and extraction.destination and not extraction.pickup:
            extraction.pickup = extraction.destination
            extraction.destination = None
        elif last_asked in ["destination", "destination_landmark"] and extraction.pickup and not extraction.destination:
            extraction.destination = extraction.pickup
            extraction.pickup = None

        # Context check: is a booking pending confirmation?
        is_switching_context = any(w in lower for w in [
            "instead", "rather", "switch to", "book a bike", "book bike", "book a ride", "book cab",
            "book a cab", "call a cab", "order food instead", "book cleaning instead", "clean instead",
            "book a bus instead", "book bus instead", "book a train instead", "book train instead",
            "bus instead", "train instead"
        ])
        is_awaiting_confirmation = (
            not is_switching_context and (
                current_workflow_state in ["WAITING_FOR_CONFIRMATION", "AWAITING_CONFIRMATION"]
                or collected.get("waiting_for_confirmation") is True
                or (collected.get("fare_amount") is not None and not any(w in lower for w in ["ride", "clean", "food", "order food", "bus", "train"]))
            )
        )

        clean_lower = lower.rstrip(".!?,")
        confirm_phrases = [
            "book", "book it", "confirm", "yes", "yeah", "yep", "yup", "sure",
            "go ahead", "place it", "place order", "place the order", "proceed",
            "do it", "sounds good", "please do", "cool", "ok", "okay"
        ]
        reject_phrases = [
            "no", "don't", "dont", "do not", "cancel", "stop", "nevermind", "abort",
            "not now", "don't place it", "no don't place it", "no, don't place it"
        ]

        if is_awaiting_confirmation:
            if any(term in lower for term in reject_phrases):
                extraction.confirmation = False
                extraction.confirmation_type = "REJECT"
                if extraction.intent in ["GENERAL_SUPPORT", "UNKNOWN"]:
                    extraction.intent = collected.get("intent") or "ORDER_FOOD"
            elif any(
                clean_lower == term
                or clean_lower.startswith(term + " ")
                or clean_lower.endswith(" " + term)
                or f" {term} " in f" {clean_lower} "
                or term in clean_lower
                for term in confirm_phrases
            ):
                extraction.confirmation = True
                extraction.confirmation_type = "CONFIRM"
                active_intent = collected.get("intent")
                if not active_intent:
                    if collected.get("travel_mode") == "bus" or collected.get("bus_type") or collected.get("selected_seats"):
                        active_intent = "BOOK_BUS"
                    elif collected.get("travel_mode") == "train" or collected.get("train_class"):
                        active_intent = "BOOK_TRAIN"
                    elif collected.get("restaurant") or collected.get("food"):
                        active_intent = "ORDER_FOOD"
                    elif collected.get("service_name") or collected.get("package"):
                        active_intent = "BOOK_SERVICE"
                    else:
                        active_intent = "BOOK_RIDE"
                extraction.intent = active_intent
                extraction.assistant_message = ""

        # Intent sanitization
        if lower in ["hi", "hello", "hey", "hola", "sup", "yo", "greetings", "good morning", "good evening", "good afternoon", "hey there", "hi there"]:
            extraction.intent = "GREETING"
            extraction.pickup = None
            extraction.destination = None
            if not extraction.assistant_message:
                extraction.assistant_message = "How can I help you?"
        elif any(w in words for w in ["thanks", "thank", "thx", "cheers", "bye", "goodbye"]):
            extraction.intent = "GRATITUDE_OR_CLOSING"
            extraction.pickup = None
            extraction.destination = None
        elif any(w in lower for w in [
            "where is", "where is it", "where's it", "where's my", "where is he", "where's he",
            "where are we", "driver", "vehicle", "phone number", "phone", "contact",
            "eta", "how long", "status", "track", "delivery boy", "delivery partner",
            "order id", "booking id", "my order", "my ride", "my bike", "the bike",
            "track order", "order details", "booking details", "details?", "details",
            "how far", "nearby", "has my driver", "when will he", "is my driver",
            "when is my appointment", "appointment", "cleaning appointment"
        ]) and not any(w in lower for w in ["take me", "from ", "order a", "order some"]):
            extraction.intent = "TASK_INQUIRY"
            extraction.pickup = None
            extraction.destination = None
        elif any(w in lower for w in ["pnr", "irctc", "train status", "bus status", "train details", "bus details"]):
            extraction.intent = "TASK_INQUIRY"
        # Ambiguous requests: "need booking", "need a booking", "order", "book something", "book"
        # ONLY treat "book" as ambiguous if NOT awaiting confirmation!
        elif lower in ["need booking", "need a booking", "i need a booking", "i need booking"]:
            extraction.intent = "GENERAL_SUPPORT"
            extraction.assistant_message = "Sure! What would you like to book — a cab or bike, food, or a home service like Urban Clean?"
        elif lower in ["order", "i want to order", "order please"] and not is_awaiting_confirmation:
            extraction.intent = "GENERAL_SUPPORT"
            extraction.assistant_message = "Sure! What would you like to order or book — food, a ride, or a home service?"
        elif lower in ["book something", "i want to book", "book", "help me book"] and not is_awaiting_confirmation:
            extraction.intent = "GENERAL_SUPPORT"
            extraction.assistant_message = "Sure! What would you like to book — a cab or bike, food, or a home service like Urban Clean?"
        elif any(w in lower for w in ["train", "railway", "irctc", "express", "vande bharat", "berth"]):
            extraction.intent = "BOOK_TRAIN"
        elif any(w in lower for w in ["bus", "sleeper bus", "volvo", "redbus", "abhibus", "apsrtc", "tsrtc", "ksrtc"]):
            extraction.intent = "BOOK_BUS"
        elif any(w in lower for w in ["clean", "cleaning", "urbanclean", "urban clean", "urban company", "plumber", "repair", "maid", "electrician"]):
            extraction.intent = "BOOK_SERVICE"
        elif any(w in lower for w in ["food", "order food", "burger", "pizza", "biryani", "swiggy", "zomato", "groceries", "restaurant"]):
            extraction.intent = "ORDER_FOOD"
        elif any(w in lower for w in ["bike", "moto", "cab", "uber", "taxi", "ride", "auto", "parcel"]):
            extraction.intent = "BOOK_RIDE"

        # Check broad locality
        is_broad_p, prompt_p = self.is_broad_locality_only(extraction.pickup)
        is_broad_d, prompt_d = self.is_broad_locality_only(extraction.destination)
        if is_broad_p or is_broad_d:
            extraction.needs_landmark_clarification = True
            if not extraction.assistant_message:
                extraction.assistant_message = prompt_p or prompt_d

        return extraction

    async def extract(
        self,
        user_message: str,
        current_collected: Optional[dict] = None,
        last_assistant_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_workflow_state: Optional[str] = None,
    ) -> AgentExtraction:
        """
        Invokes Gemini with structured output constraint.
        Sends conversation history, workflow state, available information, and required fields.
        Falls back to rule-based offline parsing if no API key is set or upon error.
        """
        if not self.client:
            return self._sanitize_extraction(
                self._offline_fallback_extract(
                    user_message,
                    current_collected=current_collected,
                    last_assistant_message=last_assistant_message,
                    conversation_history=conversation_history,
                    current_workflow_state=current_workflow_state,
                ),
                user_message=user_message,
                current_collected=current_collected,
                current_workflow_state=current_workflow_state,
            )

        dialogue_context = []
        if conversation_history:
            history_lines = []
            for m in conversation_history[-10:]:
                role = "User" if m.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {m.get('content')}")
            dialogue_context.append("Conversation History:\n" + "\n".join(history_lines))
        elif last_assistant_message:
            dialogue_context.append(f"Previous assistant question: {last_assistant_message}")

        if current_workflow_state:
            dialogue_context.append(f"Current workflow step: {current_workflow_state}")

        last_asked = (current_collected or {}).get("last_asked")
        if last_asked:
            dialogue_context.append(f"Field specifically being answered: {last_asked}")

        ctx_str = "\n".join(dialogue_context)
        if ctx_str:
            ctx_str = f"{ctx_str}\n"

        context_info = (
            f"{ctx_str}"
            f"Current known data/state: {json.dumps(current_collected or {})}\n"
            f"Latest User message: \"{user_message}\""
        )

        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=context_info,
                    config=types.GenerateContentConfig(
                        system_instruction=EXTRACTION_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=AgentExtraction,
                        temperature=0.0,
                    ),
                ),
                timeout=30.0,
            )
            raw_extraction = AgentExtraction.model_validate_json(response.text)
            return self._sanitize_extraction(
                raw_extraction,
                user_message=user_message,
                current_collected=current_collected,
                current_workflow_state=current_workflow_state,
            )
        except Exception as e:
            logger.error(f"Gemini API error during extraction: {e}. Using fallback.", exc_info=True)
            return self._sanitize_extraction(
                self._offline_fallback_extract(
                    user_message,
                    current_collected=current_collected,
                    last_assistant_message=last_assistant_message,
                    conversation_history=conversation_history,
                    current_workflow_state=current_workflow_state,
                ),
                user_message=user_message,
                current_collected=current_collected,
                current_workflow_state=current_workflow_state,
            )

    def _offline_fallback_extract(
        self,
        text: str,
        current_collected: Optional[dict] = None,
        last_assistant_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_workflow_state: Optional[str] = None,
    ) -> AgentExtraction:
        """Deterministic heuristic extraction for offline testing or when API key is unconfigured."""
        lower = text.strip().lower()
        collected = current_collected or {}
        active_intent = collected.get("intent", "UNKNOWN")

        entities = EntityData()
        preferences = PreferenceData()
        missing_fields = []
        next_action = "ASK_MISSING_INFORMATION"
        assistant_message = ""
        confirmation = None
        confirmation_type = None
        needs_landmark_clarification = False

        # Context check: is awaiting confirmation?
        is_switching_context = any(w in lower for w in [
            "instead", "rather", "switch to", "book a bike", "book bike", "book a ride", "book cab",
            "book a cab", "call a cab", "order food instead", "book cleaning instead", "clean instead"
        ])
        is_awaiting_conf = (
            not is_switching_context and (
                current_workflow_state in ["WAITING_FOR_CONFIRMATION", "AWAITING_CONFIRMATION"]
                or collected.get("waiting_for_confirmation") is True
                or (collected.get("fare_amount") is not None and not any(w in lower for w in ["ride", "clean", "food", "order food"]))
            )
        )

        # 1. Check Delegation & Strategy
        delegation_phrases = [
            "your wish", "you decide", "anything is fine", "whatever is good",
            "any is fine", "up to you", "you choose", "choose for me",
            "pick for me", "best one", "cheapest one", "fastest one",
            "any good", "something nearby", "same as before"
        ]
        is_delegating = any(phrase in lower for phrase in delegation_phrases)
        if is_delegating:
            preferences.selection_preference = "AI_CHOOSE"
            if "cheap" in lower:
                preferences.selection_strategy = "CHEAPEST"
            elif "fast" in lower:
                preferences.selection_strategy = "FASTEST"
            elif any(w in lower for w in ["best", "top", "good"]):
                preferences.selection_strategy = "TOP_RATED"
            elif "near" in lower:
                preferences.selection_strategy = "NEAREST"
            else:
                preferences.selection_strategy = "TOP_RATED"

        # 2. Service / vehicle type detection (Ride flow)
        service_type, ride_type = self.detect_service_type(lower)
        if service_type:
            entities.service_type = service_type
            entities.ride_type = ride_type

        # 3. Confirmation detection
        affirmative_terms = [
            "yes", "yeah", "sure", "book", "book it", "confirm", "go ahead", "proceed",
            "place it", "place order", "place the order",
            "ok", "okay", "yep", "yup", "do it", "sounds good", "please do", "cool"
        ]
        negative_terms = [
            "no", "cancel", "don't", "dont", "do not", "stop", "nevermind", "abort", "not now",
            "don't place it", "no don't place it", "no, don't place it"
        ]

        clean_lower = lower.rstrip(".!?,")
        if not is_delegating and not service_type:
            if is_awaiting_conf:
                if any(
                    clean_lower == term
                    or clean_lower.startswith(term + " ")
                    or clean_lower.endswith(" " + term)
                    or f" {term} " in f" {clean_lower} "
                    or term in clean_lower
                    for term in affirmative_terms
                ):
                    confirmation = True
                    confirmation_type = "CONFIRM"
                elif any(term in lower for term in negative_terms):
                    confirmation = False
                    confirmation_type = "REJECT"
            else:
                if any(term in lower for term in affirmative_terms if term != "book"):
                    confirmation = True
                    confirmation_type = "CONFIRM"
                elif any(term in lower for term in negative_terms):
                    confirmation = False
                    confirmation_type = "REJECT"

        # 4. Intent detection
        intent = "UNKNOWN"
        words = lower.split()
        if lower in ["hi", "hello", "hey", "hola", "sup", "yo", "greetings", "good morning", "good evening", "good afternoon", "hey there", "hi there"]:
            intent = "GREETING"
            assistant_message = "How can I help you?"
        elif any(w in words for w in ["thanks", "thank", "thx", "cheers", "bye", "goodbye"]):
            intent = "GRATITUDE_OR_CLOSING"
        elif any(w in lower for w in [
            "where is", "where is it", "where's it", "where's my", "where is he", "where's he",
            "where are we", "driver", "vehicle", "phone number", "phone", "contact",
            "eta", "how long", "status", "track", "delivery boy", "delivery partner",
            "order id", "booking id", "my order", "my ride", "my bike", "the bike",
            "track order", "order details", "booking details", "details?", "details",
            "how far", "nearby", "has my driver", "when will he", "is my driver",
            "when is my appointment", "appointment", "cleaning appointment",
            "pnr", "ticket", "cancel my ticket", "cancel ticket", "what is my pnr"
        ]) and not any(w in lower for w in ["take me", "order a", "order some", "book a bus", "book bus", "book a train", "book train"]):
            intent = "TASK_INQUIRY"
        # Ambiguous requests: "need booking", "need a booking", "order", "book something", "book"
        elif lower in ["need booking", "need a booking", "i need a booking", "i need booking"]:
            intent = "GENERAL_SUPPORT"
            assistant_message = "Sure! What would you like to book — a cab or bike, food, travel (bus/train), or a home service like Urban Clean?"
        elif lower in ["order", "i want to order", "order please"] and not is_awaiting_conf:
            intent = "GENERAL_SUPPORT"
            assistant_message = "Sure! What would you like to order or book — food, a ride, travel, or a home service?"
        elif lower in ["book something", "i want to book", "book", "help me book"] and not is_awaiting_conf:
            intent = "GENERAL_SUPPORT"
            assistant_message = "Sure! What would you like to book — a ride, food, travel (bus/train), or a home service like Urban Clean?"
        # Explicit topic switching & keyword matches
        elif any(w in lower for w in ["clean", "cleaning", "urbanclean", "urban clean", "urban company", "plumber", "repair", "maid", "electrician"]):
            intent = "BOOK_SERVICE"
        elif any(w in lower for w in ["food", "order food", "burger", "pizza", "biryani", "swiggy", "zomato", "groceries", "eat"]):
            intent = "ORDER_FOOD"
        elif any(w in lower for w in ["bus", "sleeper bus", "volvo", "apsrtc", "tsrtc", "redbus", "abhibus"]):
            intent = "BOOK_BUS"
        elif any(w in lower for w in ["train", "railway", "irctc", "express", "3a", "2a", "sleeper train", "tatkal"]):
            intent = "BOOK_TRAIN"
        elif any(w in lower for w in ["bike", "moto", "cab", "uber", "taxi", "ride", "auto", "parcel"]):
            intent = "BOOK_RIDE"
        elif active_intent == "BOOK_BUS" and not any(w in lower for w in ["book a ride", "book train", "train"]):
            intent = "BOOK_BUS"
        elif active_intent == "BOOK_TRAIN" and not any(w in lower for w in ["book a ride", "book bus", "bus"]):
            intent = "BOOK_TRAIN"
        elif active_intent == "ORDER_FOOD" and not any(w in lower for w in ["book a ride", "book cab", "need a cab", "call uber", "clean", "bus", "train"]):
            intent = "ORDER_FOOD"
        elif active_intent == "BOOK_SERVICE" and not any(w in lower for w in ["book a ride", "book cab", "order food", "bus", "train"]):
            intent = "BOOK_SERVICE"
        elif active_intent != "UNKNOWN":
            intent = active_intent
        elif collected:
            intent = collected.get("intent", "BOOK_RIDE")

        # 5. Food Ordering flow reasoning
        if intent == "ORDER_FOOD":
            if "biryani" in lower:
                entities.food = "biryani"
            elif "pizza" in lower:
                entities.food = "pizza"
            elif "burger" in lower:
                entities.food = "burger"
            elif collected.get("food"):
                entities.food = collected.get("food")

            if is_delegating:
                preferences.selection_preference = "AI_CHOOSE"
                preferences.selection_strategy = "TOP_RATED"
                entities.restaurant = "Meghana Foods (Top Rated 4.8/5)"
            elif any(r in lower for r in ["meghana", "paradise", "bawarchi", "domino", "mcdonald"]):
                entities.restaurant = text.title()
                preferences.selection_preference = "USER_SPECIFIED"

            if "to " in lower:
                entities.delivery_location = text.split("to ")[-1].strip().title()
            elif collected.get("delivery_location"):
                entities.delivery_location = collected.get("delivery_location")

            known_food = entities.food or collected.get("food")
            known_rest = entities.restaurant or collected.get("restaurant")
            known_loc = entities.delivery_location or collected.get("delivery_location")

            if not known_food:
                missing_fields.append("food")
                assistant_message = "Sure! What food or dish would you like to order?"
            elif not known_rest and preferences.selection_preference != "AI_CHOOSE":
                missing_fields.append("restaurant")
                assistant_message = f"Sure, {known_food} sounds good. Do you have a restaurant preference, or should I pick a highly rated option for you?"
            elif not known_loc:
                missing_fields.append("delivery_location")
                if preferences.selection_preference == "AI_CHOOSE":
                    assistant_message = f"Sure, I'll pick a highly rated {known_food} restaurant for you. Where should I deliver it?"
                else:
                    assistant_message = f"Got it, {known_food} from {known_rest}! Where should I deliver it?"
            else:
                next_action = "CONFIRM_ACTION"
                assistant_message = f"Great! I found top-rated {known_food} from {known_rest or 'a popular restaurant'}. Total is ₹350. Shall I place this order to {known_loc}?"

        # 6. Home Service flow reasoning
        elif intent == "BOOK_SERVICE":
            if "deep cleaning" in lower:
                entities.service_name = "Deep Cleaning"
            elif "cleaning" in lower:
                entities.service_name = "Home Cleaning"
            elif collected.get("service_name"):
                entities.service_name = collected.get("service_name")
            else:
                entities.service_name = "Deep Cleaning"

            if is_delegating:
                preferences.selection_preference = "AI_CHOOSE"
                preferences.selection_strategy = "BEST_AVAILABLE"
                entities.package = "Full Home Deep Cleaning (Top Rated)"
            elif any(p in lower for p in ["standard", "premium", "full home"]):
                for p in ["full home deep cleaning", "full home", "premium deep cleaning", "premium", "standard cleaning", "standard"]:
                    if p in lower:
                        entities.package = p.title()
                        preferences.selection_preference = "USER_SPECIFIED"
                        break

            # Date extraction
            if "today" in lower:
                entities.preferred_date = "Today"
            elif "tomorrow" in lower:
                entities.preferred_date = "Tomorrow"

            # Slot extraction
            slot_patterns = [
                ("10:00 am – 12:00 pm", "10:00 AM – 12:00 PM"),
                ("10 am to 12 pm", "10:00 AM – 12:00 PM"),
                ("10 to 12", "10:00 AM – 12:00 PM"),
                ("12:00 pm – 2:00 pm", "12:00 PM – 2:00 PM"),
                ("12 pm to 2 pm", "12:00 PM – 2:00 PM"),
                ("12 to 2", "12:00 PM – 2:00 PM"),
                ("2:00 pm – 4:00 pm", "2:00 PM – 4:00 PM"),
                ("2 pm to 4 pm", "2:00 PM – 4:00 PM"),
                ("2 to 4", "2:00 PM – 4:00 PM"),
                ("4:00 pm – 6:00 pm", "4:00 PM – 6:00 PM"),
                ("4 pm to 6 pm", "4:00 PM – 6:00 PM"),
                ("4 to 6", "4:00 PM – 6:00 PM"),
                ("slot 1", "10:00 AM – 12:00 PM"),
                ("slot 2", "12:00 PM – 2:00 PM"),
                ("slot 3", "2:00 PM – 4:00 PM"),
                ("slot 4", "4:00 PM – 6:00 PM"),
            ]
            for pattern, canonical in slot_patterns:
                if pattern in lower:
                    entities.slot = canonical
                    break

            # Location extraction
            if "at " in lower:
                candidate_loc = text.split("at ")[-1].strip()
                entities.service_location = candidate_loc.title()
            elif collected.get("last_asked") == "service_location" and self._is_valid_location(text):
                entities.service_location = text.strip().title()
            elif collected.get("service_location"):
                entities.service_location = collected.get("service_location")

            known_srv = entities.service_name or collected.get("service_name")
            known_pkg = entities.package or collected.get("package")
            known_loc = entities.service_location or collected.get("service_location")
            known_date = entities.preferred_date or collected.get("preferred_date")
            known_slot = entities.slot or collected.get("slot")

            if not known_srv:
                missing_fields.append("service_name")
                assistant_message = "What type of cleaning or home service do you need?"
            elif not known_pkg and preferences.selection_preference != "AI_CHOOSE":
                missing_fields.append("package")
                assistant_message = "Sure! Which cleaning package would you prefer — Standard, Premium, or Full Home?"
            elif not known_loc:
                missing_fields.append("service_location")
                if preferences.selection_preference == "AI_CHOOSE":
                    assistant_message = "I'll select our top-rated Full Home Deep Cleaning package for you. Where should our cleaning team come?"
                else:
                    assistant_message = f"Got it. What's the service address?"
            elif not known_date:
                missing_fields.append("preferred_date")
                assistant_message = "Thanks! What date would you prefer for the cleaning?"
            elif not known_slot:
                missing_fields.append("slot")
                assistant_message = f"Let me check the available {known_pkg} cleaning appointments for {known_loc}."
            else:
                next_action = "CONFIRM_ACTION"
                assistant_message = f"{known_pkg} cleaning at {known_loc} on your selected date, from {known_slot}. The total estimated price is ₹1,499. Shall I confirm this booking?"

        # 7. Bus Booking flow reasoning
        elif intent == "BOOK_BUS":
            # Origin & Destination parsing
            if "from " in lower and " to " in lower:
                parts = lower.split("from ")[1].split(" to ")
                entities.origin_city = parts[0].strip().title()
                dest_raw = parts[1].split()[0].strip().title()
                entities.destination_city = dest_raw
            elif " to " in lower and not entities.destination_city:
                after_to = lower.split(" to ")[1].split()[0].strip().title()
                entities.destination_city = after_to

            # Normalize cities
            if entities.origin_city:
                if entities.origin_city.lower() in ["hyd", "hyderabad"]:
                    entities.origin_city = "Hyderabad"
                elif entities.origin_city.lower() in ["vizag", "visakhapatnam"]:
                    entities.origin_city = "Visakhapatnam"
            if entities.destination_city:
                if entities.destination_city.lower() in ["vizag", "visakhapatnam"]:
                    entities.destination_city = "Visakhapatnam"
                elif entities.destination_city.lower() in ["hyd", "hyderabad"]:
                    entities.destination_city = "Hyderabad"

            if collected.get("last_asked") == "origin_city" and self._is_valid_location(text):
                entities.origin_city = text.strip().title()
            elif collected.get("last_asked") == "destination_city" and self._is_valid_location(text):
                entities.destination_city = text.strip().title()
            elif collected.get("last_asked") == "route":
                # User provided route e.g. "Hyderabad to Vizag" or "Hyd to Vizag"
                if " to " in lower:
                    p = lower.split(" to ")
                    entities.origin_city = p[0].replace("from", "").strip().title()
                    entities.destination_city = p[1].strip().title()
                    if entities.origin_city.lower() in ["hyd", "hyderabad"]:
                        entities.origin_city = "Hyderabad"
                    elif entities.origin_city.lower() in ["vizag", "visakhapatnam"]:
                        entities.origin_city = "Visakhapatnam"
                    if entities.destination_city.lower() in ["vizag", "visakhapatnam"]:
                        entities.destination_city = "Visakhapatnam"
                    elif entities.destination_city.lower() in ["hyd", "hyderabad"]:
                        entities.destination_city = "Hyderabad"

            # Date and Departure window extraction
            res_date, res_window = self.resolve_travel_date_and_time(lower)
            if res_date:
                entities.travel_date = res_date
            if res_window:
                entities.departure_window = res_window

            # Passenger count
            if "passenger" in lower or "adult" in lower or "people" in lower or "person" in lower or "tickets" in lower:
                for word, num in [("two", 2), ("three", 3), ("four", 4), ("one", 1), ("2", 2), ("3", 3), ("4", 4), ("1", 1)]:
                    if word in lower:
                        entities.passenger_count = num
                        break
            elif collected.get("last_asked") == "passenger_count":
                for word, num in [("two", 2), ("three", 3), ("four", 4), ("one", 1), ("2", 2), ("3", 3), ("4", 4), ("1", 1)]:
                    if word in lower:
                        entities.passenger_count = num
                        break

            # Bus type preference
            if "sleeper" in lower:
                entities.bus_type = "AC Sleeper (2+1)"
            elif "volvo" in lower:
                entities.bus_type = "Multi-Axle Volvo AC"
            elif "seater" in lower:
                entities.bus_type = "Non-AC Seater"

            # Service selection: "first one", "second option", "option 1"
            if any(w in lower for w in ["first one", "first option", "1st", "option 1"]):
                entities.selected_service_id = "bus_srv_1"
            elif any(w in lower for w in ["second one", "second option", "2nd", "option 2"]):
                entities.selected_service_id = "bus_srv_2"

            # Seat selection
            for seat in ["u1", "u2", "l1", "l2", "l3", "l4"]:
                if seat in lower:
                    curr_seats = entities.selected_seats or []
                    if seat.upper() not in curr_seats:
                        curr_seats.append(seat.upper())
                    entities.selected_seats = curr_seats

            known_orig = entities.origin_city or collected.get("origin_city")
            known_dest = entities.destination_city or collected.get("destination_city")
            known_date = entities.travel_date or collected.get("travel_date")
            known_pax = entities.passenger_count or collected.get("passenger_count")
            known_window = entities.departure_window or collected.get("departure_window")
            known_srv = entities.selected_service_id or collected.get("selected_service_id")
            known_seats = entities.selected_seats or collected.get("selected_seats")

            if not known_orig and not known_dest:
                missing_fields.append("origin_city")
                missing_fields.append("destination_city")
                assistant_message = "Sure! I can help you book a bus. Where would you like to travel from and to?"
            elif not known_orig:
                missing_fields.append("origin_city")
                assistant_message = "Sure! Where are you traveling from?"
            elif not known_dest:
                missing_fields.append("destination_city")
                assistant_message = f"Got your origin at {known_orig}. Where would you like to travel to?"
            elif not known_date:
                missing_fields.append("travel_date")
                assistant_message = "What date would you like to travel?"
            elif not known_window and not collected.get("departure_window"):
                missing_fields.append("departure_window")
                assistant_message = "What time would you prefer to depart — morning, afternoon, evening, or a specific time?"
            elif not known_pax and not collected.get("passenger_count"):
                missing_fields.append("passenger_count")
                assistant_message = "How many passengers are travelling?"
            elif not known_srv and not collected.get("selected_service_id"):
                missing_fields.append("selected_service_id")
                assistant_message = (
                    f"I found these available buses from {known_orig} to {known_dest} for {known_date} ({known_window or 'anytime'}):\n"
                    f"1. 🚌 Orange Tours & Travels — AC Sleeper (2+1) (Dep: 20:30, Arr: 06:00, ₹997.50, 4.8★)\n"
                    f"2. 🚌 Morning Star Travels — Bharat Benz AC Sleeper (Dep: 21:15, Arr: 06:45, ₹924.00, 4.6★)\n"
                    f"3. 🚌 APSRTC Garuda Plus — Multi-Axle Volvo AC (Dep: 18:00, Arr: 04:30, ₹787.50, 4.4★)\n\n"
                    f"Which bus would you prefer?"
                )
            elif not known_seats and not collected.get("selected_seats"):
                missing_fields.append("selected_seats")
                assistant_message = "These seats are currently available: U1, U2, L3, L4. Which seats would you like?"
            else:
                next_action = "CONFIRM_ACTION"
                assistant_message = (
                    f"I have your bus booking ready:\n"
                    f"🚌 Service: Orange Tours & Travels (AC Sleeper)\n"
                    f"📍 Route: {known_orig} to {known_dest}\n"
                    f"📅 Date: {known_date} at 20:30\n"
                    f"👥 Passengers: {known_pax or 1}\n"
                    f"💺 Seats: {', '.join(known_seats or ['U1'])}\n"
                    f"💰 Total Fare: ₹{int(997.5 * (known_pax or 1))}\n\n"
                    f"Shall I confirm this bus booking for you? You will get the SMS with all the booking details."
                )

        # 8. Train Booking flow reasoning
        elif intent == "BOOK_TRAIN":
            # Origin & Destination stations
            if "from " in lower and " to " in lower:
                parts = lower.split("from ")[1].split(" to ")
                entities.origin_city = parts[0].strip().title()
                dest_raw = parts[1].split()[0].strip().title()
                entities.destination_city = dest_raw
            elif " to " in lower and not entities.destination_city:
                after_to = lower.split(" to ")[1].split()[0].strip().title()
                entities.destination_city = after_to

            # Normalize train stations
            station_map = {
                "secunderabad": "Secunderabad Junction (SC)",
                "sc": "Secunderabad Junction (SC)",
                "hyderabad": "Hyderabad Deccan (HYB)",
                "hyd": "Hyderabad Deccan (HYB)",
                "vizag": "Visakhapatnam Junction (VSKP)",
                "visakhapatnam": "Visakhapatnam Junction (VSKP)",
                "vskp": "Visakhapatnam Junction (VSKP)",
                "tirupati": "Tirupati (TPTY)",
                "tpty": "Tirupati (TPTY)",
                "vijayawada": "Vijayawada Junction (BZA)",
                "bza": "Vijayawada Junction (BZA)",
            }
            if entities.origin_city and entities.origin_city.lower() in station_map:
                entities.origin_city = station_map[entities.origin_city.lower()]
            if entities.destination_city and entities.destination_city.lower() in station_map:
                entities.destination_city = station_map[entities.destination_city.lower()]

            if collected.get("last_asked") == "origin_city" and self._is_valid_location(text):
                entities.origin_city = text.strip().title()
            elif collected.get("last_asked") == "destination_city" and self._is_valid_location(text):
                entities.destination_city = text.strip().title()

            # Date and Departure window extraction
            res_date, res_window = self.resolve_travel_date_and_time(lower)
            if res_date:
                entities.travel_date = res_date
            if res_window:
                entities.departure_window = res_window

            # Passenger count
            if "passenger" in lower or "adult" in lower or "people" in lower or "person" in lower or "tickets" in lower:
                for word, num in [("two", 2), ("three", 3), ("four", 4), ("one", 1), ("2", 2), ("3", 3), ("4", 4), ("1", 1)]:
                    if word in lower:
                        entities.passenger_count = num
                        break
            elif collected.get("last_asked") == "passenger_count":
                for word, num in [("two", 2), ("three", 3), ("four", 4), ("one", 1), ("2", 2), ("3", 3), ("4", 4), ("1", 1)]:
                    if word in lower:
                        entities.passenger_count = num
                        break

            # Class preference
            if "sleeper" in lower or " sl " in f" {lower} ":
                entities.train_class = "SL"
            elif "3a" in lower or "3 tier" in lower or "3rd ac" in lower:
                entities.train_class = "3A"
            elif "2a" in lower or "2 tier" in lower or "2nd ac" in lower:
                entities.train_class = "2A"
            elif "chair car" in lower or " cc " in f" {lower} ":
                entities.train_class = "CC"

            # Service selection
            if any(w in lower for w in ["first one", "first train", "godavari", "1st", "option 1"]):
                entities.selected_service_id = "12728"
            elif any(w in lower for w in ["second one", "second train", "charminar", "2nd", "option 2"]):
                entities.selected_service_id = "12760"

            # Context switching entity carryover
            if not entities.origin_city and collected.get("origin_city"):
                entities.origin_city = collected.get("origin_city")
            if not entities.destination_city and collected.get("destination_city"):
                entities.destination_city = collected.get("destination_city")
            if not entities.travel_date and collected.get("travel_date"):
                entities.travel_date = collected.get("travel_date")
            if not entities.passenger_count and collected.get("passenger_count"):
                entities.passenger_count = collected.get("passenger_count")

            known_orig = entities.origin_city
            known_dest = entities.destination_city
            known_date = entities.travel_date
            known_pax = entities.passenger_count
            known_class = entities.train_class or collected.get("train_class")
            known_srv = entities.selected_service_id or collected.get("selected_service_id")

            if not known_orig:
                missing_fields.append("origin_city")
                assistant_message = "Sure! Which railway station or city are you departing from?"
            elif not known_dest:
                missing_fields.append("destination_city")
                assistant_message = f"Got your departure from {known_orig}. What is your destination station?"
            elif not known_date:
                missing_fields.append("travel_date")
                assistant_message = f"What date would you like to book train tickets for from {known_orig} to {known_dest}?"
            elif not known_pax and not collected.get("passenger_count"):
                missing_fields.append("passenger_count")
                assistant_message = "How many passengers are travelling?"
            elif not known_class and not collected.get("train_class"):
                missing_fields.append("train_class")
                assistant_message = "What class do you prefer (e.g. Sleeper SL, 3A, 2A, or AC Chair Car CC)?"
            elif not known_srv and not collected.get("selected_service_id"):
                missing_fields.append("selected_service_id")
                assistant_message = (
                    f"Here are the available trains from {known_orig} to {known_dest} on {known_date}:\n"
                    f"1. 🚆 Godavari Express (12728) — Dep: 17:05, Arr: 05:45 | 3A: ₹1,150 (AVAILABLE-18) | SL: ₹435\n"
                    f"2. 🚆 Vande Bharat Express (20834) — Dep: 15:00, Arr: 23:30 | CC: ₹1,720 (AVAILABLE-24)\n\n"
                    f"Which train would you like to book?"
                )
            else:
                next_action = "CONFIRM_ACTION"
                unit_fare = 1150.0 if (known_class or "3A") == "3A" else 435.0
                pax_count = known_pax or 1
                total_fare = unit_fare * pax_count
                assistant_message = (
                    f"I have your train booking summary:\n"
                    f"🚆 Train: Godavari Express (12728)\n"
                    f"🚉 Route: {known_orig} to {known_dest}\n"
                    f"📅 Date: {known_date} at 17:05\n"
                    f"🎫 Class: {known_class or '3A'}\n"
                    f"👥 Passengers: {pax_count}\n"
                    f"💰 Total Fare: ₹{int(total_fare)}\n\n"
                    f"Shall I go ahead and book this ticket for you? You will get the SMS with all the booking details."
                )

        # 9. Ride booking flow or Location parsing
        else:
            if is_delegating:
                intent = "BOOK_RIDE"
                if preferences.selection_strategy == "CHEAPEST":
                    entities.ride_type = "Uber Moto"
                    entities.service_type = "bike"
                elif preferences.selection_strategy == "FASTEST":
                    entities.ride_type = "Uber Premier"
                    entities.service_type = "cab"
                else:
                    entities.ride_type = "Uber Go"
                    entities.service_type = "cab"

            def clean_loc(loc_str: str) -> str:
                loc_lower = loc_str.lower().strip()
                for suffix in [
                    " instead", " please", " rather", " by cab", " by car", " by bike",
                    " by moto", " by auto", " by parcel", " by uber", " in cab", " in a cab"
                ]:
                    if loc_lower.endswith(suffix):
                        loc_lower = loc_lower[: len(loc_lower) - len(suffix)].strip()
                return loc_lower.rstrip(".!?, ").title()

            if "from " in lower and " to " in lower:
                parts = lower.split("from ")[1].split(" to ")
                candidate_pickup = clean_loc(parts[0])
                candidate_dest = clean_loc(parts[1])
                if self._is_valid_location(candidate_pickup):
                    entities.pickup = candidate_pickup
                    intent = "BOOK_RIDE"
                if self._is_valid_location(candidate_dest):
                    entities.destination = candidate_dest
                    intent = "BOOK_RIDE"
            elif lower.startswith("to "):
                candidate = text[3:].strip().title()
                if self._is_valid_location(candidate):
                    entities.destination = candidate
                    intent = "BOOK_RIDE"
            elif not entities.pickup and not entities.destination:
                last_asked = collected.get("last_asked")
                candidate = text.strip().rstrip(".").title()

                is_broad_loc, prompt_broad = self.is_broad_locality_only(candidate)

                if is_broad_loc and not is_delegating and service_type is None and confirmation is None:
                    if last_asked in ["destination", "destination_landmark"]:
                        entities.destination = candidate
                    else:
                        entities.pickup = candidate
                    intent = "BOOK_RIDE"
                    needs_landmark_clarification = True
                    assistant_message = prompt_broad
                elif last_asked in ["pickup", "pickup_landmark"] and collected:
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        entities.pickup = candidate
                        intent = "BOOK_RIDE"
                elif last_asked in ["destination", "destination_landmark"] and collected:
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        entities.destination = candidate
                        intent = "BOOK_RIDE"
                elif collected and "pickup" not in collected:
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        entities.pickup = candidate
                        intent = "BOOK_RIDE"
                elif collected and "destination" not in collected:
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        entities.destination = candidate
                        intent = "BOOK_RIDE"
                elif collected:
                    curr_p = collected.get("pickup")
                    curr_d = collected.get("destination")
                    is_broad_p, _ = self.is_broad_locality_only(curr_p)
                    is_broad_d, _ = self.is_broad_locality_only(curr_d)
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        if is_broad_p:
                            entities.pickup = candidate
                            intent = "BOOK_RIDE"
                        elif is_broad_d:
                            entities.destination = candidate
                            intent = "BOOK_RIDE"
                elif not collected:
                    if self._is_valid_location(candidate) and service_type is None and confirmation is None and not is_delegating:
                        entities.pickup = candidate
                        intent = "BOOK_RIDE"

        return AgentExtraction(
            intent=intent,
            entities=entities,
            preferences=preferences,
            confirmation=confirmation,
            confirmation_type=confirmation_type,
            missing_required_fields=missing_fields,
            next_action=next_action,
            assistant_message=assistant_message,
            needs_landmark_clarification=needs_landmark_clarification,
        )
