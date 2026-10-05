import logging
import re
import uuid
from typing import Any, Optional

from app.services.conversation_service import ConversationService
from app.services.onevo_state import onevo_state
from app.voice.sarvam_client import SarvamClient, sarvam_client, detect_language_from_text
from app.voice.schemas import VoiceTurnResponse

logger = logging.getLogger(__name__)


class VoiceGateway:
    """
    Unified Voice Gateway for ONEVO.AI:
    - Multilingual Voice & Text ingestion (Sarvam STT & Translation)
    - Action Assistant Execution
    - Functional Onevo Wallet deductions, top-ups, & limit enforcements
    - Saved & Frequent Address management via voice
    - Active Service tracking & history population
    - Sarvam Bulbul v3 speech synthesis
    """

    def __init__(self, client: Optional[SarvamClient] = None):
        self.client = client or sarvam_client
        self._conversation_languages: dict[str, str] = {}
        self._processed_turns: dict[str, VoiceTurnResponse] = {}
        self._pending_topups: dict[str, float] = {}
        self._processed_booking_ids: set[str] = set()

    def get_conversation_language(self, conversation_id: str) -> str:
        return self._conversation_languages.get(conversation_id, "en-IN")

    def set_conversation_language(self, conversation_id: str, language_code: str) -> None:
        if language_code and language_code != "unknown":
            self._conversation_languages[conversation_id] = language_code

    async def _handle_direct_voice_intents(
        self, conversation_id: str, english_text: str
    ) -> Optional[tuple[str, str]]:
        """
        Intercepts direct voice commands for Onevo Wallet and Saved Addresses.
        Returns: Optional (reply_text, task_state)
        """
        clean_text = english_text.strip().lower()

        # 1. Address saving via voice: "Save as Brother Home", "Yes, save it as Brother Home"
        addr_match = re.search(r'save(?:\s+it)?\s+as\s+([A-Za-z0-9\s]+)', clean_text, re.I)
        if addr_match:
            name = addr_match.group(1).strip().title()
            onevo_state.save_frequent_address(name=name)
            return (
                f"Done! I've saved {name} (Sector 2, HSR Layout) to your family addresses.",
                "COMPLETED",
            )

        # 2. Check wallet balance: "How much money is in my wallet?", "Wallet balance", etc.
        if any(p in clean_text for p in [
            "how much money is in my wallet",
            "how much is in my wallet",
            "what is my wallet balance",
            "check my wallet balance",
            "check wallet balance",
            "wallet balance",
            "wallet amount",
            "balance in my wallet",
            "how much balance",
        ]):
            current_bal = onevo_state.get_state()["wallet"]["balance"]
            return (
                f"Your current Onevo Wallet balance is ₹{current_bal:,.2f}.",
                "COMPLETED",
            )

        # 3. Confirming a pending top-up
        if conversation_id in self._pending_topups:
            if clean_text in ["yes", "yeah", "yep", "sure", "confirm", "proceed", "add", "please add", "add it", "ok", "okay", "do it"]:
                amount = self._pending_topups.pop(conversation_id)
                try:
                    res = onevo_state.topup_wallet(amount, source="Voice Top-up")
                    return (
                        f"Done! Added ₹{amount:,.0f} to your Onevo Wallet. Your new balance is ₹{res['new_balance']:,.2f}.",
                        "COMPLETED",
                    )
                except ValueError as e:
                    return (str(e), "FAILED")

        # 4. Top-up request: "Add 500 rupees", "Add 500 to my wallet", "Top up 500", "Add ₹1000"
        topup_match = re.search(
            r'(?:add|top\s*up|put|deposit|load)\s*(?:₹|rs\.?|rupees)?\s*([0-9]+(?:\.[0-9]+)?)',
            clean_text,
            re.I,
        ) or re.search(
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:₹|rs\.?|rupees)\s*(?:to\s*(?:my\s*)?wallet|add)',
            clean_text,
            re.I,
        )
        if topup_match and any(w in clean_text for w in ["wallet", "add", "top", "money", "rupees", "rs"]):
            try:
                requested_amount = float(topup_match.group(1))
                if requested_amount > 0:
                    current_bal = onevo_state.get_state()["wallet"]["balance"]
                    max_allowed = 5000.0

                    if current_bal >= max_allowed:
                        return (
                            "Your wallet can hold a maximum of ₹5,000. It is already at the maximum capacity.",
                            "WAITING_FOR_USER",
                        )

                    capacity = max_allowed - current_bal
                    if requested_amount > capacity:
                        self._pending_topups[conversation_id] = capacity
                        return (
                            f"Your wallet has ₹{current_bal:,.0f}. You can add up to ₹{capacity:,.0f}. "
                            f"Would you like to add ₹{capacity:,.0f}?",
                            "WAITING_FOR_CONFIRMATION",
                        )
                    else:
                        self._pending_topups[conversation_id] = requested_amount
                        return (
                            f"Your current balance is ₹{current_bal:,.0f}. Shall I add ₹{requested_amount:,.0f}?",
                            "WAITING_FOR_CONFIRMATION",
                        )
            except Exception as e:
                logger.warning(f"Error parsing top-up amount: {e}")

        return None

    def _extract_and_apply_booking_deduction(
        self, backend_reply: str, task_state: str
    ) -> str:
        """
        Inspects completed booking and applies deduction to Onevo Wallet,
        updating Active Services and History.
        """
        is_completed = (
            task_state in ("COMPLETED", "BOOKED")
            or "booked successfully" in backend_reply.lower()
            or "ticket has been confirmed successfully" in backend_reply.lower()
            or "has been booked successfully" in backend_reply.lower()
            or "booking id:" in backend_reply.lower()
        )
        if not is_completed:
            return backend_reply

        # Deduce service type and amount
        reply_lower = backend_reply.lower()
        service_type = "CAB"
        fare = 350.0

        if "bus" in reply_lower or "pnr" in reply_lower:
            service_type = "BUS"
            fare = 850.0
        elif "food" in reply_lower or "biryani" in reply_lower or "restaurant" in reply_lower:
            service_type = "FOOD"
            fare = 420.0
        elif "train" in reply_lower:
            service_type = "TRAIN"
            fare = 680.0
        elif "flight" in reply_lower:
            service_type = "FLIGHT"
            fare = 2500.0
        elif "hotel" in reply_lower:
            service_type = "HOTEL"
            fare = 1800.0
        elif "clean" in reply_lower:
            service_type = "CLEANING"
            fare = 499.0
        else:
            service_type = "CAB"
            fare = 350.0

        # Try to extract explicit ₹ amount from reply
        amt_match = re.search(r'₹\s*([0-9,]+)', backend_reply)
        if amt_match:
            try:
                parsed_amt = float(amt_match.group(1).replace(",", ""))
                if parsed_amt > 0:
                    fare = parsed_amt
            except ValueError:
                pass

        # Generate a unique key for this booking deduction
        b_key = f"{service_type}_{fare}_{hash(backend_reply[:50])}"
        if b_key in self._processed_booking_ids:
            return backend_reply
        self._processed_booking_ids.add(b_key)

        try:
            deduct_res = onevo_state.deduct_wallet_for_booking(
                amount=fare,
                service_title=f"{service_type.title()} Booking",
                service_subtitle="Completed via Onevo Voice Assistant",
            )
            # Add to history and active services
            details = {
                "booking_id": f"ONEVO-{service_type.upper()}-{uuid.uuid4().hex[:6].upper()}",
                "booked_by": "Rahul Sharma",
                "passenger": "Rahul Sharma",
                "amount": f"₹{fare:,.2f}",
                "payment": "Onevo Wallet",
                "status": "Confirmed & Active",
            }
            onevo_state.add_booking_to_history_and_active(
                service_type=service_type,
                title=f"{service_type.title()} Booking",
                amount=fare,
                details=details,
                is_active=True,
            )
            notification = (
                f"\n\n💳 ₹{fare:,.0f} deducted from your Onevo Wallet. "
                f"Remaining balance: ₹{deduct_res['new_balance']:,.2f}."
            )
            return backend_reply + notification
        except ValueError as e:
            # Insufficient balance message
            return backend_reply + f"\n\n⚠️ {str(e)}"

    async def process_turn(
        self,
        conversation_id: str,
        conversation_service: ConversationService,
        audio_bytes: Optional[bytes] = None,
        text_input: Optional[str] = None,
        language_override: Optional[str] = None,
        turn_id: Optional[str] = None,
        audio_format: str = "wav",
    ) -> VoiceTurnResponse:
        """
        Executes a single voice turn:
        Audio/Text -> Sarvam STT -> English Backend -> Native Translation -> Bulbul v3 TTS.
        """
        if turn_id and turn_id in self._processed_turns:
            logger.info(f"Duplicate turn_id detected: {turn_id}, returning cached response.")
            return self._processed_turns[turn_id]

        active_turn_id = turn_id or str(uuid.uuid4())
        original_user_text = ""
        detected_language = language_override or self.get_conversation_language(conversation_id)
        english_text = ""

        # 1. Ingest Voice or Multilingual Text
        if audio_bytes:
            stt_res = await self.client.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=f"input.{audio_format}",
                language_code=language_override or "unknown",
            )
            original_user_text = stt_res.get("transcript", "").strip()
            detected_language = stt_res.get("language_code") or detect_language_from_text(original_user_text)
            english_text = await self.client.translate_to_english(
                text=original_user_text,
                source_language_code=detected_language,
            )
        elif text_input:
            original_user_text = text_input.strip()
            detected_language = language_override or detect_language_from_text(original_user_text)
            english_text = await self.client.translate_to_english(
                text=original_user_text,
                source_language_code=detected_language,
            )
        else:
            raise ValueError("Either audio_bytes or text_input must be provided.")

        self.set_conversation_language(conversation_id, detected_language)

        # 2. Check for direct voice commands (Wallet, Addresses)
        direct_result = await self._handle_direct_voice_intents(conversation_id, english_text)
        if direct_result:
            backend_reply, task_state = direct_result
            task_id = None
        else:
            # 3. Call existing backend action assistant with English text
            backend_reply, task_id, task_state = await conversation_service.handle_user_message(
                conversation_id=conversation_id,
                user_text=english_text,
            )
            # Apply wallet deduction & update active services if booking completed
            backend_reply = self._extract_and_apply_booking_deduction(backend_reply, task_state)

        # 4. Localize backend reply into user's detected language
        localized_reply = await self.client.translate_from_english(
            text=backend_reply,
            target_language_code=detected_language,
        )

        # 5. Synthesize speech in user's language via Bulbul v3
        audio_b64 = None
        voice_status = "success"
        try:
            audio_b64 = await self.client.synthesize_speech(
                text=localized_reply,
                language_code=detected_language,
            )
            if not audio_b64:
                voice_status = "tts_unavailable"
        except Exception as e:
            logger.warning(f"TTS synthesis failed: {e}")
            voice_status = "tts_fallback"

        response = VoiceTurnResponse(
            conversation_id=conversation_id,
            task_id=task_id,
            turn_id=active_turn_id,
            original_user_text=original_user_text,
            user_language=detected_language,
            backend_english_text=english_text,
            assistant_english_text=backend_reply,
            assistant_localized_text=localized_reply,
            task_state=task_state,
            audio_base64=audio_b64,
            audio_format="wav",
            voice_status=voice_status,
            onevo_state_snapshot=onevo_state.get_state(),
        )

        self._processed_turns[active_turn_id] = response
        return response


# Global singleton voice gateway
voice_gateway = VoiceGateway()
