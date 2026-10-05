import pytest
from httpx import AsyncClient
from app.voice.sarvam_client import detect_language_from_text, SarvamClient
from app.voice.voice_gateway import VoiceGateway


def test_script_language_detection():
    # Telugu
    assert detect_language_from_text("నాకు క్యాబ్ కావాలి") == "te-IN"
    # Hindi
    assert detect_language_from_text("मुझे कैब बुक करनी है") == "hi-IN"
    # Tamil
    assert detect_language_from_text("எனக்கு கேப் வேண்டும்") == "ta-IN"
    # Kannada
    assert detect_language_from_text("ನನಗೆ ಕ್ಯಾಬ್ ಬೇಕು") == "kn-IN"
    # English
    assert detect_language_from_text("I need a cab") == "en-IN"


@pytest.mark.asyncio
async def test_voice_text_turn_telugu(async_client: AsyncClient):
    """
    Test Telugu turn:
    User: "నాకు మణికొండ నుంచి గచ్చిబౌలికి క్యాబ్ బుక్ చేయాలి"
    Voice Gateway converts to English: "Book a cab from Manikonda to Gachibowli."
    Existing backend processes ride booking.
    UI shows original Telugu text, and receives Bulbul TTS audio.
    """
    # 1. Create conversation
    create_res = await async_client.post("/api/v1/conversations")
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # 2. Post Telugu voice turn
    payload = {
        "text": "నాకు మణికొండ నుంచి గచ్చిబౌలికి క్యాబ్ బుక్ చేయాలి",
        "turn_id": "turn_te_001",
    }
    res = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["conversation_id"] == conv_id
    assert data["user_language"] == "te-IN"
    assert data["original_user_text"] == "నాకు మణికొండ నుంచి గచ్చిబౌలికి క్యాబ్ బుక్ చేయాలి"
    assert "cab" in data["backend_english_text"].lower() or "manikonda" in data["backend_english_text"].lower()
    assert data["assistant_english_text"] is not None
    assert data["assistant_localized_text"] is not None
    assert data["audio_base64"] is not None
    assert data["task_state"] in ["WAITING_FOR_USER", "WAITING_FOR_CONFIRMATION", "IN_PROGRESS", "CHECKED_REQUIREMENTS"]


@pytest.mark.asyncio
async def test_voice_text_turn_hindi(async_client: AsyncClient):
    """
    Test Hindi turn:
    User: "मुझे कैब बुक करनी है"
    Backend receives English: "I need a cab"
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    payload = {
        "text": "मुझे कैब बुक करनी है",
        "turn_id": "turn_hi_001",
    }
    res = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()

    assert data["user_language"] == "hi-IN"
    assert data["original_user_text"] == "मुझे कैब बुक करनी है"
    assert data["audio_base64"] is not None


@pytest.mark.asyncio
async def test_voice_turn_idempotency_prevents_duplicate_calls(async_client: AsyncClient):
    """
    Test Turn ID duplicate protection:
    If the same turn_id is sent twice (e.g. streaming STT retry or duplicate final event),
    the gateway returns the cached response without re-triggering backend booking.
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    turn_id = "duplicate_test_turn_123"
    payload = {
        "text": "I need a cab from Madhapur to Gachibowli",
        "turn_id": turn_id,
    }

    res1 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json=payload,
    )
    assert res1.status_code == 200
    d1 = res1.json()

    # Second call with same turn_id
    res2 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json=payload,
    )
    assert res2.status_code == 200
    d2 = res2.json()

    assert d1["turn_id"] == d2["turn_id"]
    assert d1["assistant_english_text"] == d2["assistant_english_text"]


@pytest.mark.asyncio
async def test_voice_audio_upload_endpoint(async_client: AsyncClient):
    """
    Test binary audio upload endpoint with simulated audio WAV bytes.
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    # Minimal 44-byte WAV header
    wav_bytes = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )

    files = {"file": ("mic.wav", wav_bytes, "audio/wav")}
    res = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/audio",
        files=files,
        data={"turn_id": "audio_turn_01", "language_code": "en-IN"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["conversation_id"] == conv_id
    assert data["original_user_text"] is not None
    assert data["assistant_english_text"] is not None
    assert data["audio_base64"] is not None


@pytest.mark.asyncio
async def test_session_language_persistence_across_turns(async_client: AsyncClient):
    """
    Test that session language persists across turns:
    Turn 1: Telugu request -> session language = te-IN
    Turn 2: Follow-up response -> retains te-IN context
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    # Turn 1: Telugu
    res1 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json={"text": "నాకు క్యాబ్ కావాలి", "turn_id": "pers_01"},
    )
    assert res1.status_code == 200
    assert res1.json()["user_language"] == "te-IN"

    # Turn 2: Follow-up in Telugu
    res2 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text-turn",
        json={"text": "మీ ఇష్టం", "turn_id": "pers_02"},
    )
    assert res2.status_code == 200
    assert res2.json()["user_language"] == "te-IN"
    # Booking state should have progressed under the exact same conversation_id
    assert res2.json()["conversation_id"] == conv_id
