import base64
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.repositories.conversation import ConversationRepository
from app.db.repositories.task import TaskRepository
from app.services.conversation_service import ConversationService
from app.services.task_service import TaskService
from app.voice.voice_gateway import voice_gateway
from app.voice.schemas import VoiceTurnRequest, VoiceTurnResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])


def get_conversation_service(
    session: AsyncSession = Depends(get_db),
) -> ConversationService:
    conv_repo = ConversationRepository(session)
    task_repo = TaskRepository(session)
    task_service = TaskService(task_repo)
    return ConversationService(conv_repo, task_service)


@router.post(
    "/conversations/{conversation_id}/audio",
    response_model=VoiceTurnResponse,
    summary="Process speech audio input through Sarvam STT, backend action engine, and Bulbul TTS",
)
async def process_voice_audio(
    conversation_id: str,
    file: UploadFile = File(..., description="Audio file from microphone (WAV, WebM, MP3)"),
    turn_id: Optional[str] = Form(None),
    language_code: Optional[str] = Form(None),
    service: ConversationService = Depends(get_conversation_service),
):
    conv = None
    if conversation_id and conversation_id not in ("undefined", "null", "none"):
        conv = await service.get_conversation(conversation_id)
    if not conv:
        logger.info(f"Conversation {conversation_id} not found, auto-creating session.")
        conv = await service.create_conversation(user_id="usr_rahul_sharma")
        conversation_id = conv.id

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload received.",
        )

    # Determine audio format
    filename = file.filename or "audio.webm"
    audio_format = filename.split(".")[-1].lower() if "." in filename else "webm"

    try:
        turn_response = await voice_gateway.process_turn(
            conversation_id=conversation_id,
            conversation_service=service,
            audio_bytes=audio_bytes,
            language_override=language_code,
            turn_id=turn_id,
            audio_format=audio_format,
        )
        return turn_response
    except Exception as e:
        logger.error(f"Voice processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice layer processing failed: {str(e)}",
        )


@router.post(
    "/conversations/{conversation_id}/text-turn",
    response_model=VoiceTurnResponse,
    summary="Process multilingual text turn through translation, existing backend, and Bulbul TTS",
)
@router.post(
    "/conversations/{conversation_id}/text",
    response_model=VoiceTurnResponse,
    summary="Alias for text turn",
    include_in_schema=False,
)
async def process_multilingual_text_turn(
    conversation_id: str,
    req: VoiceTurnRequest,
    service: ConversationService = Depends(get_conversation_service),
):
    conv = None
    if conversation_id and conversation_id not in ("undefined", "null", "none"):
        conv = await service.get_conversation(conversation_id)
    if not conv:
        logger.info(f"Conversation {conversation_id} not found, auto-creating session.")
        conv = await service.create_conversation(user_id="usr_rahul_sharma")
        conversation_id = conv.id

    if not req.text and not req.audio_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'audio_base64' is required.",
        )

    audio_bytes = None
    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid base64 audio data.",
            )

    try:
        turn_response = await voice_gateway.process_turn(
            conversation_id=conversation_id,
            conversation_service=service,
            audio_bytes=audio_bytes,
            text_input=req.text,
            language_override=req.language_code,
            turn_id=req.turn_id,
            audio_format=req.audio_format,
        )
        return turn_response
    except Exception as e:
        logger.error(f"Multilingual voice turn processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice processing error: {str(e)}",
        )


@router.websocket("/ws/{conversation_id}")
async def voice_realtime_websocket(
    websocket: WebSocket,
    conversation_id: str,
):
    """
    WebSocket endpoint for realtime streaming microphone audio from browser:
    - Buffers audio chunks from user speech
    - Processes turn upon silence/turn end
    - Sends native language transcript, localized response, and Bulbul v3 TTS audio
    """
    await websocket.accept()
    logger.info(f"Voice WebSocket connected for conversation: {conversation_id}")

    # Create scoped database session for WebSocket lifecycle
    from app.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        conv_repo = ConversationRepository(session)
        task_repo = TaskRepository(session)
        task_service = TaskService(task_repo)
        service = ConversationService(conv_repo, task_service)

        conv = await service.get_conversation(conversation_id)
        if not conv:
            await websocket.send_json({"type": "error", "message": f"Conversation {conversation_id} not found"})
            await websocket.close()
            return

        audio_buffer = bytearray()

        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    audio_buffer.extend(message["bytes"])
                elif "text" in message and message["text"]:
                    import json
                    try:
                        data = json.loads(message["text"])
                    except Exception:
                        continue

                    msg_type = data.get("type")

                    if msg_type == "audio_chunk":
                        raw_chunk = base64.b64decode(data.get("data", ""))
                        audio_buffer.extend(raw_chunk)

                    elif msg_type == "audio_end":
                        # Process buffered audio
                        if not audio_buffer:
                            await websocket.send_json({"type": "error", "message": "No audio received"})
                            continue

                        turn_id = data.get("turn_id")
                        lang_override = data.get("language_code")
                        audio_format = data.get("audio_format", "webm")

                        await websocket.send_json({"type": "processing", "message": "Processing voice input..."})

                        try:
                            turn_res = await voice_gateway.process_turn(
                                conversation_id=conversation_id,
                                conversation_service=service,
                                audio_bytes=bytes(audio_buffer),
                                language_override=lang_override,
                                turn_id=turn_id,
                                audio_format=audio_format,
                            )
                            # Reset audio buffer
                            audio_buffer.clear()

                            await websocket.send_json({
                                "type": "turn_complete",
                                "turn_id": turn_res.turn_id,
                                "original_user_text": turn_res.original_user_text,
                                "user_language": turn_res.user_language,
                                "backend_english_text": turn_res.backend_english_text,
                                "assistant_localized_text": turn_res.assistant_localized_text,
                                "assistant_english_text": turn_res.assistant_english_text,
                                "task_state": turn_res.task_state,
                                "audio_base64": turn_res.audio_base64,
                                "voice_status": turn_res.voice_status,
                            })
                        except Exception as ex:
                            logger.error(f"WebSocket turn processing error: {ex}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Sorry, I couldn't understand that. Please try again: {str(ex)}",
                            })
                            audio_buffer.clear()

                    elif msg_type == "text_turn":
                        # Direct text turn over WebSocket
                        text_val = data.get("text", "")
                        turn_id = data.get("turn_id")
                        lang_override = data.get("language_code")

                        try:
                            turn_res = await voice_gateway.process_turn(
                                conversation_id=conversation_id,
                                conversation_service=service,
                                text_input=text_val,
                                language_override=lang_override,
                                turn_id=turn_id,
                            )
                            await websocket.send_json({
                                "type": "turn_complete",
                                "turn_id": turn_res.turn_id,
                                "original_user_text": turn_res.original_user_text,
                                "user_language": turn_res.user_language,
                                "backend_english_text": turn_res.backend_english_text,
                                "assistant_localized_text": turn_res.assistant_localized_text,
                                "assistant_english_text": turn_res.assistant_english_text,
                                "task_state": turn_res.task_state,
                                "audio_base64": turn_res.audio_base64,
                                "voice_status": turn_res.voice_status,
                            })
                        except Exception as ex:
                            logger.error(f"WebSocket text turn error: {ex}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Error processing turn: {str(ex)}",
                            })

        except WebSocketDisconnect:
            logger.info(f"Voice WebSocket disconnected for {conversation_id}")
        except Exception as e:
            logger.error(f"Voice WebSocket unexpected error: {e}")
