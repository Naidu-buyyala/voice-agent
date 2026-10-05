import copy
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OnevoStateManager:
    """
    Central state manager for Onevo.ai Family Universal Action Assistant:
    - Current User & Family Account Model
    - Wallet Balance, Maximum Limits (₹5,000 max), Transactions
    - Saved Family Addresses & Frequent Address Detection
    - Active Service Tracking (Simulated ETA & Route progress)
    - Full Transaction & Service History
    - Contextual Support Tracking
    """

    def __init__(self):
        self._state: dict[str, Any] = self._create_initial_state()

    def _create_initial_state(self) -> dict[str, Any]:
        return {
            "user": {
                "id": "usr_rahul_sharma",
                "name": "Rahul Sharma",
                "role": "Primary Admin",
                "role_badge": "PRIMARY ADMIN",
                "family_name": "Sharma Household",
                "phone": "+91 98765 43210",
                "avatar_initials": "RS",
                "avatar_color": "#6366f1",
                "age": 39,
            },
            "family": {
                "name": "Sharma Household",
                "total_members": 5,
                "administrators": 2,
                "protected_accounts": 3,
                "members": [
                    {
                        "id": "mem_1",
                        "name": "Rahul Sharma",
                        "relation": "Self",
                        "role": "Primary Admin",
                        "role_badge": "PRIMARY ADMIN",
                        "phone": "+91 98765 43210",
                        "age": 39,
                        "status": "CONNECTED",
                        "avatar_initials": "RS",
                        "avatar_color": "#6366f1",
                        "can_manage": False,
                        "badge_note": "Account creator",
                    },
                    {
                        "id": "mem_2",
                        "name": "Pooja Sharma",
                        "relation": "Spouse",
                        "role": "Admin",
                        "role_badge": "ADMIN",
                        "phone": "+91 98214 56780",
                        "age": 37,
                        "status": "CONNECTED",
                        "avatar_initials": "PS",
                        "avatar_color": "#ec4899",
                        "can_manage": True,
                        "badge_note": "Manage role",
                    },
                    {
                        "id": "mem_3",
                        "name": "Aarav Sharma",
                        "relation": "Son",
                        "role": "Child Account",
                        "role_badge": "CHILD ACCOUNT",
                        "phone": "+91 90123 45678",
                        "age": 12,
                        "status": "CONNECTED",
                        "avatar_initials": "AS",
                        "avatar_color": "#0ea5e9",
                        "can_manage": True,
                        "badge_note": "Manage role",
                    },
                    {
                        "id": "mem_4",
                        "name": "Ananya Sharma",
                        "relation": "Daughter",
                        "role": "Child Account",
                        "role_badge": "CHILD ACCOUNT",
                        "phone": "Rahul's number",
                        "age": 8,
                        "status": "CONNECTED",
                        "avatar_initials": "AN",
                        "avatar_color": "#f59e0b",
                        "can_manage": True,
                        "badge_note": "Manage role",
                    },
                    {
                        "id": "mem_5",
                        "name": "Savitri Sharma",
                        "relation": "Mother",
                        "role": "Senior Citizen",
                        "role_badge": "SENIOR CITIZEN",
                        "phone": "+91 94401 23456",
                        "age": 68,
                        "status": "CONNECTED",
                        "avatar_initials": "SS",
                        "avatar_color": "#10b981",
                        "can_manage": True,
                        "badge_note": "Manage role",
                    },
                ],
            },
            "wallet": {
                "balance": 1420.0,
                "max_balance": 5000.0,
                "min_topup": 500.0,
                "auto_topup": True,
                "auto_topup_label": "₹500 min · ₹2,000 max",
                "transactions": [
                    {
                        "id": "tx_init_1",
                        "type": "DEDUCTION",
                        "title": "Cab Booking",
                        "subtitle": "Pooja Sharma · My Home to Parents Hospital",
                        "amount": 350.0,
                        "timestamp": "Today, 10:35 AM",
                        "status": "COMPLETED",
                    },
                    {
                        "id": "tx_init_2",
                        "type": "TOPUP",
                        "title": "Wallet Top-up",
                        "subtitle": "Auto Top-Up via Family Bank",
                        "amount": 1000.0,
                        "timestamp": "Yesterday, 04:15 PM",
                        "status": "COMPLETED",
                    },
                    {
                        "id": "tx_init_3",
                        "type": "DEDUCTION",
                        "title": "Food Order",
                        "subtitle": "Meghana Foods Biryani · Delivered",
                        "amount": 420.0,
                        "timestamp": "24 May, 08:30 PM",
                        "status": "COMPLETED",
                    },
                ],
            },
            "addresses": [
                {
                    "id": "addr_1",
                    "name": "My Home",
                    "address": "12th Main, Indiranagar, Bengaluru",
                    "category": "Home",
                    "icon": "🏠",
                    "relevance": "Primary Family Residence",
                },
                {
                    "id": "addr_2",
                    "name": "Parents Home",
                    "address": "Sector 4, HSR Layout, Bengaluru",
                    "category": "Family",
                    "icon": "🏡",
                    "relevance": "Savitri Sharma's Residence",
                },
                {
                    "id": "addr_3",
                    "name": "Parents Hospital",
                    "address": "Fortis Hospital, Bannerghatta Road, Bengaluru",
                    "category": "Medical",
                    "icon": "🏥",
                    "relevance": "Emergency Medical Center",
                },
                {
                    "id": "addr_4",
                    "name": "Kids Tuition",
                    "address": "Allen Institute, 5th Block, Koramangala, Bengaluru",
                    "category": "Education",
                    "icon": "🎓",
                    "relevance": "Aarav & Ananya Daily Tuition",
                },
            ],
            "frequent_address_suggestion": {
                "has_suggestion": True,
                "detected_location": "Brother's House, Sector 2, HSR Layout",
                "visit_count": 4,
                "suggested_name": "Brother Home",
                "message": "You frequently visit this location. Would you like to save it as a family address?",
            },
            "active_services": [
                {
                    "id": "ONEVO-RIDE-7821",
                    "type": "CAB",
                    "badge": "RIDE IN PROGRESS",
                    "title": "Mom is on the way",
                    "subtitle": "Pooja Sharma · Swift Dzire · KA 01 AB 1234",
                    "passenger": "Pooja Sharma",
                    "driver_name": "Suresh K.",
                    "driver_phone": "+91 98765 43210",
                    "driver_rating": "4.8",
                    "vehicle": "Swift Dzire",
                    "vehicle_plate": "KA 01 AB 1234",
                    "pickup": "My Home",
                    "destination": "Parents Hospital",
                    "eta_minutes": 8,
                    "amount": 350.0,
                    "payment_source": "Onevo Wallet",
                    "status": "On the way",
                    "progress_percent": 65,
                    "notes": "Exact gate location confirmed",
                }
            ],
            "history": [
                {
                    "id": "HIST-CAB-7821",
                    "service_type": "CAB",
                    "icon": "🚗",
                    "title": "Cab Booking",
                    "member": "Pooja Sharma",
                    "route": "Manikonda → Airport",
                    "amount": 350.0,
                    "status": "Completed",
                    "time_group": "TODAY",
                    "timestamp": "Today, 10:35 AM",
                    "details": {
                        "booking_id": "ONEVO-CAB-7821",
                        "booked_by": "Rahul Sharma",
                        "passenger": "Pooja Sharma",
                        "driver": "Suresh K.",
                        "vehicle": "Swift Dzire (KA 01 AB 1234)",
                        "pickup": "Manikonda",
                        "drop": "Rajiv Gandhi International Airport",
                        "fare": "₹350.00",
                        "payment": "Onevo Wallet",
                        "status": "Completed successfully",
                    },
                },
                {
                    "id": "HIST-FOOD-4412",
                    "service_type": "FOOD",
                    "icon": "🍔",
                    "title": "Food Order",
                    "member": "Rahul Sharma",
                    "route": "Paradise Biryani · 2x Chicken Biryani",
                    "amount": 420.0,
                    "status": "Delivered",
                    "time_group": "TODAY",
                    "timestamp": "Today, 01:15 PM",
                    "details": {
                        "order_id": "ONEVO-FOOD-4412",
                        "booked_by": "Rahul Sharma",
                        "restaurant": "Paradise Biryani",
                        "items": "2x Special Chicken Biryani, 1x Mirchi ka Salan",
                        "delivery_address": "My Home (Indiranagar)",
                        "fare": "₹420.00",
                        "payment": "Onevo Wallet",
                        "status": "Delivered on time",
                    },
                },
                {
                    "id": "HIST-BUS-9901",
                    "service_type": "BUS",
                    "icon": "🚌",
                    "title": "Bus Booking",
                    "member": "Rahul Sharma",
                    "route": "Hyderabad → Vizag (Orange Travels)",
                    "amount": 850.0,
                    "status": "Completed",
                    "time_group": "LAST WEEK",
                    "timestamp": "22 May, 07:30 PM",
                    "details": {
                        "pnr": "PNR77A991",
                        "passenger": "Rahul Sharma (Adult)",
                        "operator": "Orange Tours & Travels (AC Sleeper)",
                        "route": "Hyderabad (Ameerpet) → Visakhapatnam (RTC Complex)",
                        "seats": "U1 (Upper Deck)",
                        "fare": "₹850.00",
                        "payment": "Onevo Wallet",
                        "status": "Travel completed",
                    },
                },
            ],
        }

    def get_state(self) -> dict[str, Any]:
        """Returns deep copy of current global Onevo state."""
        return copy.deepcopy(self._state)

    def topup_wallet(self, amount: float, source: str = "Voice Top-up") -> dict[str, Any]:
        """Tops up wallet balance, adhering to maximum ₹5,000 capacity."""
        wallet = self._state["wallet"]
        current = wallet["balance"]
        max_allowed = wallet["max_balance"]

        if current >= max_allowed:
            raise ValueError(f"Your wallet is already at the maximum capacity of ₹{int(max_allowed):,}.")

        available_capacity = max_allowed - current
        if amount > available_capacity:
            raise ValueError(
                f"Your wallet has ₹{int(current):,}. You can add up to ₹{int(available_capacity):,}."
            )

        wallet["balance"] = round(current + amount, 2)
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        tx_record = {
            "id": tx_id,
            "type": "TOPUP",
            "title": "Wallet Top-up",
            "subtitle": f"{source} (via UPI)",
            "amount": float(amount),
            "timestamp": datetime.now(timezone.utc).strftime("%d %b, %I:%M %p"),
            "status": "COMPLETED",
        }
        wallet["transactions"].insert(0, tx_record)
        return {
            "success": True,
            "new_balance": wallet["balance"],
            "added_amount": amount,
            "transaction": tx_record,
        }

    def deduct_wallet_for_booking(
        self,
        amount: float,
        service_title: str,
        service_subtitle: str,
    ) -> dict[str, Any]:
        """Deducts booking fare from wallet if sufficient balance exists."""
        wallet = self._state["wallet"]
        current = wallet["balance"]

        if current < amount:
            shortfall = amount - current
            raise ValueError(
                f"Your wallet balance is ₹{int(current):,}, but this booking costs ₹{int(amount):,}. "
                f"You need ₹{int(shortfall):,} more."
            )

        wallet["balance"] = round(current - amount, 2)
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        tx_record = {
            "id": tx_id,
            "type": "DEDUCTION",
            "title": service_title,
            "subtitle": service_subtitle,
            "amount": float(amount),
            "timestamp": datetime.now(timezone.utc).strftime("%d %b, %I:%M %p"),
            "status": "COMPLETED",
        }
        wallet["transactions"].insert(0, tx_record)
        return {
            "success": True,
            "new_balance": wallet["balance"],
            "deducted_amount": amount,
            "transaction": tx_record,
        }

    def add_booking_to_history_and_active(
        self,
        service_type: str,
        title: str,
        amount: float,
        details: dict[str, Any],
        is_active: bool = True,
    ) -> None:
        """Records a confirmed booking into History and updates Active Service card."""
        hist_id = f"HIST-{service_type.upper()}-{uuid.uuid4().hex[:6]}"
        icon_map = {
            "CAB": "🚗",
            "BIKE": "🏍️",
            "AUTO": "🛺",
            "BUS": "🚌",
            "TRAIN": "🚆",
            "FLIGHT": "✈️",
            "HOTEL": "🏨",
            "FOOD": "🍔",
            "CLEANING": "🧹",
            "PARCEL": "📦",
        }
        icon = icon_map.get(service_type.upper(), "✨")
        now_str = datetime.now(timezone.utc).strftime("%d %b, %I:%M %p")

        hist_item = {
            "id": hist_id,
            "service_type": service_type.upper(),
            "icon": icon,
            "title": title,
            "member": details.get("booked_by", "Rahul Sharma"),
            "route": details.get("route") or f"{details.get('pickup', 'Origin')} → {details.get('drop', details.get('destination', 'Destination'))}",
            "amount": float(amount),
            "status": "Confirmed / Active" if is_active else "Completed",
            "time_group": "TODAY",
            "timestamp": f"Today, {now_str}",
            "details": details,
        }
        self._state["history"].insert(0, hist_item)

        if is_active:
            # Set or prepend to active services
            active_item = {
                "id": details.get("booking_id") or details.get("pnr") or hist_id,
                "type": service_type.upper(),
                "badge": f"{service_type.upper()} CONFIRMED",
                "title": f"Your {service_type.title()} is Active",
                "subtitle": f"{details.get('passenger', 'Rahul Sharma')} · {details.get('vehicle', title)}",
                "passenger": details.get("passenger", "Rahul Sharma"),
                "driver_name": details.get("driver", "Assigned Professional"),
                "driver_phone": details.get("driver_phone", "+91 98765 43210"),
                "driver_rating": "4.9",
                "vehicle": details.get("vehicle", "Standard"),
                "vehicle_plate": details.get("vehicle_plate", "KA 01 LIVE"),
                "pickup": details.get("pickup", "My Home"),
                "destination": details.get("drop", details.get("destination", "Destination")),
                "eta_minutes": details.get("eta_minutes", 10),
                "amount": float(amount),
                "payment_source": "Onevo Wallet",
                "status": "In Progress",
                "progress_percent": 30,
                "notes": "Driver / Partner assigned",
            }
            self._state["active_services"] = [active_item]

    def save_frequent_address(self, name: str, address: Optional[str] = None) -> dict[str, Any]:
        """Saves a frequently visited address confirmed by the user."""
        sugg = self._state.get("frequent_address_suggestion", {})
        addr_text = address or sugg.get("detected_location", "Sector 2, HSR Layout, Bengaluru")
        new_addr = {
            "id": f"addr_{uuid.uuid4().hex[:6]}",
            "name": name,
            "address": addr_text,
            "category": "Frequent",
            "icon": "📍",
            "relevance": "Frequently visited family location",
        }
        self._state["addresses"].append(new_addr)
        self._state["frequent_address_suggestion"]["has_suggestion"] = False
        return new_addr

    def add_family_member(self, member_data: dict[str, Any]) -> dict[str, Any]:
        """Adds a family member with specified role."""
        m_id = f"mem_{uuid.uuid4().hex[:6]}"
        name = member_data.get("name", "Family Member")
        initials = "".join(part[0].upper() for part in name.split()[:2]) or "FM"
        new_member = {
            "id": m_id,
            "name": name,
            "relation": member_data.get("relation", "Member"),
            "role": member_data.get("role", "Adult Member"),
            "role_badge": member_data.get("role", "Adult Member").upper(),
            "phone": member_data.get("phone", "+91 90000 00000"),
            "age": int(member_data.get("age", 30)),
            "status": "CONNECTED",
            "avatar_initials": initials,
            "avatar_color": "#8b5cf6",
            "can_manage": True,
            "badge_note": "Manage role",
        }
        self._state["family"]["members"].append(new_member)
        self._state["family"]["total_members"] = len(self._state["family"]["members"])
        return new_member

    def update_member_role(self, member_id: str, new_role: str) -> Optional[dict[str, Any]]:
        """Updates role of an existing family member."""
        for m in self._state["family"]["members"]:
            if m["id"] == member_id:
                m["role"] = new_role
                m["role_badge"] = new_role.upper()
                return m
        return None


# Global singleton Onevo State
onevo_state = OnevoStateManager()
