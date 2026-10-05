import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_investor_flow_3_wallet_voice_inquiry_and_topup(async_client: AsyncClient):
    """
    FLOW 3 — WALLET
    Voice: "How much money is in my wallet?"
    Assistant: "Your current Onevo Wallet balance is ₹1,420.00."
    Voice: "Add 500 rupees to my wallet."
    Assistant asks confirmation: "Your current balance is ₹1,420. Shall I add ₹500?"
    User: "Yes"
    Wallet: ₹1,920.00
    """
    # 0. Reset to baseline demo state
    await async_client.post("/api/v1/onevo/reset")

    # Create session
    c_res = await async_client.post("/api/v1/conversations")
    conv_id = c_res.json()["id"]

    # 1. Inquiry
    res1 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "How much money is in my wallet?", "language_code": "en-IN"},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert "₹1,420" in data1["assistant_english_text"]

    # 2. Add 500 rupees
    res2 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Add 500 rupees to my wallet", "language_code": "en-IN"},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert "Shall I add ₹500" in data2["assistant_english_text"] or "add ₹500" in data2["assistant_english_text"]

    # 3. Confirm with "Yes"
    res3 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Yes", "language_code": "en-IN"},
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert "Added ₹500" in data3["assistant_english_text"]
    assert "₹1,920" in data3["assistant_english_text"]

    # Verify state snapshot returned
    snapshot = data3["onevo_state_snapshot"]
    assert snapshot["wallet"]["balance"] == 1920.0


@pytest.mark.asyncio
async def test_wallet_capacity_exceed_rule(async_client: AsyncClient):
    """
    WALLET RULE: Maximum wallet balance is ₹5,000.
    If current balance is ₹4,200 and user asks to add ₹1,000, assistant says:
    "Your wallet has ₹4,200. You can add up to ₹800. Would you like to add ₹800?"
    Upon confirmation, adds ₹800, balance becomes ₹5,000.
    """
    await async_client.post("/api/v1/onevo/reset")
    # Bring wallet to 4200 (top up 2780 to 1420)
    await async_client.post(
        "/api/v1/onevo/wallet/topup",
        json={"amount": 2780.0, "source": "Preset"},
    )

    c_res = await async_client.post("/api/v1/conversations")
    conv_id = c_res.json()["id"]

    # User asks to add 1000
    res = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Add 1000 rupees to my wallet", "language_code": "en-IN"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "You can add up to ₹800" in data["assistant_english_text"]

    # Confirm
    res2 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Yes", "language_code": "en-IN"},
    )
    assert res2.status_code == 200
    assert "₹5,000" in res2.json()["assistant_english_text"]


@pytest.mark.asyncio
async def test_frequent_address_saving_via_voice(async_client: AsyncClient):
    """
    Voice: "Save as Brother Home"
    Assistant: "Done! I've saved Brother Home to your family addresses."
    """
    c_res = await async_client.post("/api/v1/conversations")
    conv_id = c_res.json()["id"]

    res = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Yes, save it as Brother Home", "language_code": "en-IN"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "saved Brother Home" in data["assistant_english_text"]
    assert "Brother Home" in [a["name"] for a in data["onevo_state_snapshot"]["addresses"]]


@pytest.mark.asyncio
async def test_investor_flow_1_cab_booking_and_wallet_deduction(async_client: AsyncClient):
    """
    FLOW 1 — CAB
    Voice: "Book a cab from my home to the airport."
    Assistant asks ride type.
    User: "Go."
    Assistant shows: ₹350
    User: "Yes."
    Wallet: ₹1,420 -> ₹1,070
    Booking created.
    Active ride appears on dashboard.
    """
    await async_client.post("/api/v1/onevo/reset")
    c_res = await async_client.post("/api/v1/conversations")
    conv_id = c_res.json()["id"]

    # Step 1: Initial request
    r1 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Book a cab from my home to the airport", "language_code": "en-IN"},
    )
    assert r1.status_code == 200

    # Step 2: Select Go
    r2 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Go", "language_code": "en-IN"},
    )
    assert r2.status_code == 200

    # Step 3: Confirm
    r3 = await async_client.post(
        f"/api/v1/voice/conversations/{conv_id}/text",
        json={"text": "Yes", "language_code": "en-IN"},
    )
    assert r3.status_code == 200
    d3 = r3.json()
    assert "booked successfully" in d3["assistant_english_text"].lower() or "booking id:" in d3["assistant_english_text"].lower()

    # Wallet deducted
    snap = d3["onevo_state_snapshot"]
    assert snap["wallet"]["balance"] < 1420.0
    assert len(snap["history"]) >= 4
    assert len(snap["active_services"]) >= 1
