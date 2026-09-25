import asyncio
import json
import logging
from typing import Optional
from google import genai
from google.genai import types

from app.agent.schemas import AgentExtraction
from app.agent.prompts import EXTRACTION_SYSTEM_PROMPT
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class GeminiExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.client: Optional[genai.Client] = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

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
            "nothing", "none", "please", "help"
        }
        if clean in invalid_words:
            return False
        # If it's a conversational sentence like "i want to book an uber", not a location
        conversational_phrases = ["i want", "book a", "book an", "need a", "give me", "where can", "can you"]
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
            # Matches "gachibowli", "in gachibowli", "from gachibowli", "at gachibowli"
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
    ) -> AgentExtraction:
        if extraction.pickup and not self._is_valid_location(extraction.pickup):
            extraction.pickup = None
        if extraction.destination and not self._is_valid_location(extraction.destination):
            extraction.destination = None

        # Check service type from user message if missing
        if not extraction.service_type and user_message:
            st, rt = self.detect_service_type(user_message)
            if st:
                extraction.service_type = st
                extraction.ride_type = extraction.ride_type or rt

        # Specifying or selecting a vehicle/service type is NOT a confirmation of a previous quote
        if extraction.service_type and extraction.confirmation:
            lower = user_message.lower().strip()
            if not any(a in lower for a in ["yes", "proceed", "confirm", "go ahead", "sure"]):
                extraction.confirmation = None

        # Context-aware remapping: if the user was directly answering a pickup question
        last_asked = (current_collected or {}).get("last_asked")
        if last_asked in ["pickup", "pickup_landmark"] and extraction.destination and not extraction.pickup:
            extraction.pickup = extraction.destination
            extraction.destination = None
        elif last_asked in ["destination", "destination_landmark"] and extraction.pickup and not extraction.destination:
            extraction.destination = extraction.pickup
            extraction.pickup = None

        # Intent sanitization
        lower = user_message.lower().strip()
        words = lower.split()
        if lower in ["hi", "hello", "hey", "hola", "sup", "yo", "greetings", "good morning", "good evening", "good afternoon", "hey there", "hi there"]:
            extraction.intent = "GREETING"
            extraction.pickup = None
            extraction.destination = None
        elif any(w in words for w in ["thanks", "thank", "thx", "cheers", "bye", "goodbye"]):
            extraction.intent = "GRATITUDE_OR_CLOSING"
            extraction.pickup = None
            extraction.destination = None
        elif any(w in lower for w in ["where is", "driver", "vehicle", "phone number", "phone", "contact", "eta", "how long", "status of ride", "booking id"]) and not any(w in lower for w in ["book", "take me", "from "]):
            extraction.intent = "TASK_INQUIRY"
            extraction.pickup = None
            extraction.destination = None
        elif any(w in lower for w in ["food", "order food", "burger", "pizza", "biryani", "swiggy", "zomato", "groceries"]):
            extraction.intent = "ORDER_FOOD"
        elif any(w in lower for w in ["clean", "cleaning", "urbanclean", "urban clean", "urban company", "plumber", "repair", "maid", "electrician"]):
            extraction.intent = "BOOK_SERVICE"

        # Check broad locality
        is_broad_p, prompt_p = self.is_broad_locality_only(extraction.pickup)
        is_broad_d, prompt_d = self.is_broad_locality_only(extraction.destination)
        if is_broad_p or is_broad_d:
            extraction.needs_landmark_clarification = True
            if not extraction.conversational_reply:
                extraction.conversational_reply = prompt_p or prompt_d

        return extraction

    async def extract(
        self,
        user_message: str,
        current_collected: Optional[dict] = None,
        last_assistant_message: Optional[str] = None,
    ) -> AgentExtraction:
        """
        Invokes Gemini with structured output constraint.
        Falls back to rule-based offline parsing if no API key is set or upon error.
        """
        if not self.client:
            return self._sanitize_extraction(
                self._offline_fallback_extract(user_message, current_collected),
                user_message=user_message,
                current_collected=current_collected,
            )

        last_asked = (current_collected or {}).get("last_asked")
        dialogue_context = []
        if last_assistant_message:
            dialogue_context.append(f"Previous assistant question: {last_assistant_message}")
        if last_asked:
            dialogue_context.append(f"Field the user is currently answering: {last_asked}")

        ctx_str = "\n".join(dialogue_context)
        if ctx_str:
            ctx_str = f"\n{ctx_str}"

        context_info = (
            f"Current known data: {json.dumps(current_collected or {})}{ctx_str}\n"
            f"User message: {user_message}"
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
            )
        except Exception as e:
            logger.error(f"Gemini API error during extraction: {e}. Using fallback.", exc_info=True)
            return self._sanitize_extraction(
                self._offline_fallback_extract(user_message, current_collected),
                user_message=user_message,
                current_collected=current_collected,
            )

    def _offline_fallback_extract(
        self, text: str, current_collected: Optional[dict] = None
    ) -> AgentExtraction:
        """Deterministic heuristic extraction for offline testing or when API key is unconfigured."""
        lower = text.strip().lower()

        # Service / vehicle type detection
        service_type = None
        ride_type = None
        if any(w in lower for w in ["bike", "moto", "motorcycle", "two wheeler", "scooter"]):
            service_type = "bike"
            ride_type = "Uber Moto"
        elif any(w in lower for w in ["auto", "rickshaw", "tuk tuk", "three wheeler"]):
            service_type = "auto"
            ride_type = "Uber Auto"
        elif any(w in lower for w in ["parcel", "package", "courier", "delivery", "connect", "send item"]):
            service_type = "parcel"
            ride_type = "Uber Connect"
        elif any(w in lower for w in ["cab", "car", "taxi", "uber go", "premier", "sedan", "xl"]):
            service_type = "cab"
            if "premier" in lower:
                ride_type = "Uber Premier"
            elif "xl" in lower:
                ride_type = "Uber XL"
            else:
                ride_type = "Uber Go"

        # Confirmation detection
        confirmation = None
        affirmative_terms = [
            "yes", "yeah", "sure", "book it", "confirm", "go ahead", "proceed",
            "ok", "okay", "yep", "yup", "do it", "fine", "cool", "sounds good", "please do"
        ]
        negative_terms = [
            "no", "cancel", "don't", "stop", "nevermind", "abort", "not now"
        ]

        if not service_type:
            if any(term in lower for term in affirmative_terms):
                confirmation = True
            elif any(term in lower for term in negative_terms):
                confirmation = False
        else:
            if lower in ["yes", "proceed", "confirm", "book it", "go ahead"]:
                confirmation = True
            elif any(term in lower for term in negative_terms):
                confirmation = False

        # Intent detection
        intent = "UNKNOWN"
        words = lower.split()
        if lower in ["hi", "hello", "hey", "hola", "sup", "yo", "greetings", "good morning", "good evening", "good afternoon", "hey there", "hi there"]:
            intent = "GREETING"
        elif any(w in words for w in ["thanks", "thank", "thx", "cheers", "bye", "goodbye"]):
            intent = "GRATITUDE_OR_CLOSING"
        elif any(w in lower for w in ["where is", "driver", "vehicle", "phone number", "phone", "contact", "eta", "how long", "status of ride", "booking id"]) and not any(w in lower for w in ["book", "take me", "from "]):
            intent = "TASK_INQUIRY"
        elif any(w in lower for w in ["food", "order food", "burger", "pizza", "biryani", "swiggy", "zomato", "groceries"]):
            intent = "ORDER_FOOD"
        elif any(w in lower for w in ["clean", "cleaning", "urbanclean", "urban clean", "urban company", "plumber", "repair", "maid", "electrician"]):
            intent = "BOOK_SERVICE"
        elif any(w in lower for w in ["ride", "uber", "cab", "taxi", "go to", "from", "to", "book", "bike", "auto", "parcel"]):
            intent = "BOOK_RIDE"
        elif current_collected:  # Continue existing task
            intent = "BOOK_RIDE"

        # Simple location heuristic: "from X to Y"
        def clean_loc(loc_str: str) -> str:
            loc_lower = loc_str.lower()
            for suffix in [" by cab", " by car", " by bike", " by moto", " by auto", " by parcel", " by uber", " in cab", " in a cab"]:
                if loc_lower.endswith(suffix):
                    loc_str = loc_str[: len(loc_str) - len(suffix)]
                    break
            return loc_str.strip().title()

        pickup = None
        destination = None
        if "from " in lower and " to " in lower:
            parts = lower.split("from ")[1].split(" to ")
            candidate_pickup = clean_loc(parts[0])
            candidate_dest = clean_loc(parts[1])
            if self._is_valid_location(candidate_pickup):
                pickup = candidate_pickup
            if self._is_valid_location(candidate_dest):
                destination = candidate_dest
        elif lower.startswith("to "):
            candidate = text[3:].strip().title()
            if self._is_valid_location(candidate):
                destination = candidate
        if not pickup and not destination:
            last_asked = (current_collected or {}).get("last_asked")
            if last_asked in ["pickup", "pickup_landmark"] and current_collected is not None:
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    pickup = candidate
            elif last_asked in ["destination", "destination_landmark"] and current_collected is not None:
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    destination = candidate
            elif current_collected is not None and "pickup" not in current_collected:
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    pickup = candidate
            elif current_collected is not None and "destination" not in current_collected:
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    destination = candidate
            elif current_collected is not None:
                # Check if existing pickup or destination is a broad locality awaiting landmark refinement
                curr_p = current_collected.get("pickup")
                curr_d = current_collected.get("destination")
                is_broad_p, _ = self.is_broad_locality_only(curr_p)
                is_broad_d, _ = self.is_broad_locality_only(curr_d)
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    if is_broad_p:
                        pickup = candidate
                    elif is_broad_d:
                        destination = candidate
            elif current_collected is None or not current_collected:
                candidate = text.strip().rstrip(".").title()
                if self._is_valid_location(candidate) and service_type is None and confirmation is None:
                    pickup = candidate

        reply = None
        if not pickup and not destination:
            if any(w in lower.split() for w in ["hi", "hello", "hey"]) or any(w in lower for w in ["good morning", "good evening"]):
                reply = "Hello! Where would you like to travel today? Just give me your pickup and destination locations."

        return AgentExtraction(
            intent=intent,
            pickup=pickup,
            destination=destination,
            ride_type=ride_type,
            service_type=service_type,
            confirmation=confirmation,
            conversational_reply=reply,
        )
