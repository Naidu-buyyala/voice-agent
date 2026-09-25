import pytest
from app.agent.extractor import GeminiExtractor
from app.agent.schemas import AgentExtraction


@pytest.fixture
def extractor():
    return GeminiExtractor(api_key="")  # Uses deterministic extraction fallback for tests



@pytest.mark.asyncio
async def test_extract_intent_and_locations(extractor: GeminiExtractor):
    text = "I want an Uber from Hitech City to Hyderabad Airport"
    res: AgentExtraction = await extractor.extract(text)

    assert res.intent == "BOOK_RIDE"
    assert res.pickup == "Hitech City"
    assert res.destination == "Hyderabad Airport"
    assert res.confirmation is None


@pytest.mark.asyncio
async def test_extract_confirmation_positive(extractor: GeminiExtractor):
    text = "Yes, please book it."
    res: AgentExtraction = await extractor.extract(text)
    assert res.confirmation is True

    # Test natural phrasing "proceed", "ok", "cool", "sounds good"
    res2: AgentExtraction = await extractor.extract("proceed")
    assert res2.confirmation is True

    res3: AgentExtraction = await extractor.extract("looks good, go ahead")
    assert res3.confirmation is True


@pytest.mark.asyncio
async def test_extract_confirmation_negative(extractor: GeminiExtractor):
    text = "No, cancel"
    res: AgentExtraction = await extractor.extract(text)
    assert res.confirmation is False

    res2: AgentExtraction = await extractor.extract("abort ride")
    assert res2.confirmation is False


@pytest.mark.asyncio
async def test_extract_followup_destination(extractor: GeminiExtractor):
    text = "Secunderabad Station"
    res: AgentExtraction = await extractor.extract(
        text, current_collected={"pickup": "Hitech City"}
    )

    assert res.intent == "BOOK_RIDE"
    assert res.destination == "Secunderabad Station"


@pytest.mark.asyncio
async def test_greeting_never_extracted_as_pickup(extractor: GeminiExtractor):
    for greeting in ["hi", "Hi", "hello", "Hello!", "hey", "cool", "sure"]:
        res: AgentExtraction = await extractor.extract(
            greeting, current_collected={"destination": "Hyderabad Airport"}
        )
        assert res.pickup is None, f"Expected pickup to be None for '{greeting}', got '{res.pickup}'"


@pytest.mark.asyncio
async def test_broad_locality_needs_landmark(extractor: GeminiExtractor):
    res: AgentExtraction = await extractor.extract("Gachibowli")
    assert res.pickup == "Gachibowli"
    assert res.needs_landmark_clarification is True
    assert "Gachibowli" in (res.conversational_reply or "")


@pytest.mark.asyncio
async def test_service_type_extraction(extractor: GeminiExtractor):
    bike_res = await extractor.extract("Book a bike")
    assert bike_res.service_type == "bike"
    assert bike_res.ride_type == "Uber Moto"

    auto_res = await extractor.extract("I want an auto")
    assert auto_res.service_type == "auto"
    assert auto_res.ride_type == "Uber Auto"

    parcel_res = await extractor.extract("Send parcel")
    assert parcel_res.service_type == "parcel"
    assert parcel_res.ride_type == "Uber Connect"

    cab_res = await extractor.extract("Book a cab")
    assert cab_res.service_type == "cab"
    assert cab_res.ride_type == "Uber Go"

