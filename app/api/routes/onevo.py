import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.onevo_state import onevo_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onevo", tags=["Onevo Family Assistant"])


class TopupRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Amount to add to Onevo wallet")
    source: Optional[str] = Field("Voice Top-up", description="Payment source")


class DeductRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Amount to deduct")
    service_title: str
    service_subtitle: str


class AddMemberRequest(BaseModel):
    name: str
    relation: str
    role: str = "Adult Member"
    phone: str = "+91 90000 00000"
    age: int = 30


class UpdateRoleRequest(BaseModel):
    member_id: str
    new_role: str


class SaveAddressRequest(BaseModel):
    name: str
    address: Optional[str] = None


class AddAddressRequest(BaseModel):
    name: str
    address: str
    category: str = "Saved"
    icon: str = "📍"
    relevance: str = "Family location"


@router.get("/state", summary="Fetch current global Onevo state")
async def get_state() -> dict[str, Any]:
    """Returns the entire Onevo state: user, family, wallet, addresses, active services, history."""
    return onevo_state.get_state()


@router.post("/wallet/topup", summary="Add money to Onevo wallet")
async def topup_wallet(req: TopupRequest) -> dict[str, Any]:
    try:
        res = onevo_state.topup_wallet(amount=req.amount, source=req.source or "Voice Top-up")
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/wallet/deduct", summary="Deduct booking fare from wallet")
async def deduct_wallet(req: DeductRequest) -> dict[str, Any]:
    try:
        res = onevo_state.deduct_wallet_for_booking(
            amount=req.amount,
            service_title=req.service_title,
            service_subtitle=req.service_subtitle,
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/family/member", summary="Add a family member")
async def add_family_member(req: AddMemberRequest) -> dict[str, Any]:
    member = onevo_state.add_family_member(req.model_dump())
    return {"success": True, "member": member}


@router.post("/family/role", summary="Update family member role")
async def update_member_role(req: UpdateRoleRequest) -> dict[str, Any]:
    updated = onevo_state.update_member_role(req.member_id, req.new_role)
    if not updated:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"success": True, "member": updated}


@router.post("/addresses/save-frequent", summary="Save frequent address suggestion")
async def save_frequent_address(req: SaveAddressRequest) -> dict[str, Any]:
    addr = onevo_state.save_frequent_address(name=req.name, address=req.address)
    return {"success": True, "address": addr}


@router.post("/addresses/add", summary="Add a custom family address")
async def add_address(req: AddAddressRequest) -> dict[str, Any]:
    state = onevo_state.get_state()
    new_addr = {
        "id": f"addr_{len(state['addresses']) + 1}",
        "name": req.name,
        "address": req.address,
        "category": req.category,
        "icon": req.icon,
        "relevance": req.relevance,
    }
    onevo_state._state["addresses"].append(new_addr)
    return {"success": True, "address": new_addr}


@router.post("/active-service/step", summary="Advance active service tracking simulation")
async def advance_tracking() -> dict[str, Any]:
    """Steps simulated ride ETA down (8 min -> 6 min -> 4 min -> Arrived)."""
    state = onevo_state._state
    if not state["active_services"]:
        return {"status": "none", "message": "No active service to track"}

    active = state["active_services"][0]
    current_eta = active.get("eta_minutes", 8)

    if current_eta > 6:
        active["eta_minutes"] = 6
        active["progress_percent"] = 75
        active["status"] = "Nearing destination"
    elif current_eta > 4:
        active["eta_minutes"] = 4
        active["progress_percent"] = 90
        active["status"] = "Arriving in 4 min"
    elif current_eta > 0:
        active["eta_minutes"] = 0
        active["progress_percent"] = 100
        active["status"] = "Arrived at destination"
        active["badge"] = "RIDE COMPLETED"
    else:
        # Reset to 8 for continuous demo convenience
        active["eta_minutes"] = 8
        active["progress_percent"] = 65
        active["status"] = "On the way"
        active["badge"] = "RIDE IN PROGRESS"

    return {"success": True, "active_service": active}


@router.post("/reset", summary="Reset demo state back to default")
async def reset_demo_state() -> dict[str, Any]:
    onevo_state._state = onevo_state._create_initial_state()
    return {"success": True, "state": onevo_state.get_state()}
