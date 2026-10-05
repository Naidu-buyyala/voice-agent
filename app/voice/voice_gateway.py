import logging
import uuid
from typing import Any, Optional

from app.services.conversation_service import ConversationService
from app.voice.sarvam_client import SarvamClient, sarvam_client, detect_language_from_text
from app.voice.schemas import VoiceTurnResponse

logger = logging.getLogger(__name__)


class VoiceGateway:
    """
    Voice Gateway adapter wrapping around the existing Universal AI Action Assistant.
    Coordinates:
    - User audio / multilingual text -> Sarvam STT -> English Text
    - Invoking existing backend (ConversationService) with original conversation_id
    - Translating assistant English response to user's native language via Sarvam
    - Synthesizing speech via Sarvam Bulbul v3 TTS
    - Maintaining conversational language state & preventing duplicate turns
    """

    def __init__(self, client: Optional[SarvamClient] = None):
        self.client = client or sarvam_client
        # In-memory session tracking for active conversation language
        self._conversation_languages: dict[str, str] = {}
        # Duplicate turn protection cache
        self._processed_turns: dict[str, VoiceTurnResponse] = {}

    def get_conversation_language(self, conversation_id: str) -> str:
        """Retrieves active language code for conversation, defaulting to auto/en-IN."""
        return self._conversation_languages.get(conversation_id, "en-IN")

    def set_conversation_language(self, conversation_id: str, language_code: str) -> None:
        """Persists detected or switched language for conversation."""
        if language_code and language_code != "unknown":
            self._conversation_languages[conversation_id] = language_code

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
        # 1. Duplicate turn prevention
        if turn_id and turn_id in self._processed_turns:
            logger.info(f"Duplicate turn_id detected: {turn_id}, returning cached response.")
            return self._processed_turns[turn_id]

        active_turn_id = turn_id or str(uuid.uuid4())
        original_user_text = ""
        detected_language = language_override or self.get_conversation_language(conversation_id)
        english_text = ""

        # 2. Ingest Voice or Multilingual Text
        if audio_bytes:
            # Call Sarvam STT to get native transcript & language
            stt_res = await self.client.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=f"input.{audio_format}",
                language_code=language_override or "unknown",
            )
            original_user_text = stt_res.get("transcript", "").strip()
            detected_language = stt_res.get("language_code") or detect_language_from_text(original_user_text)
            
            # Translate user speech to English for the backend
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

        # Persist language for this conversation session
        self.set_conversation_language(conversation_id, detected_language)

        # 3. Call existing backend action assistant with English text and same session ID!
        backend_reply, task_id, task_state = await conversation_service.handle_user_message(
            conversation_id=conversation_id,
            user_text=english_text,
        )

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
        )

        # Cache for idempotency
        self._processed_turns[active_turn_id] = response
        return response


# Global singleton voice gateway
voice_gateway = VoiceGateway()
