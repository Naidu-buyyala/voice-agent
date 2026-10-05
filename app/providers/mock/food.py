import uuid
from datetime import datetime, timezone
from typing import Optional, Dict
from app.providers.base import FoodProvider, FoodOrderResult


class MockFoodProvider(FoodProvider):
    """
    Deterministic mock food provider with idempotency tracking.
    """

    def __init__(self):
        self._orders_by_key: Dict[str, FoodOrderResult] = {}

    async def place_order(
        self,
        food: str,
        restaurant: str,
        delivery_location: str,
        estimated_total: float = 350.0,
        idempotency_key: Optional[str] = None,
    ) -> FoodOrderResult:
        if idempotency_key and idempotency_key in self._orders_by_key:
            return self._orders_by_key[idempotency_key]

        order_id = f"food_{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        result = FoodOrderResult(
            order_id=order_id,
            booking_id=order_id,
            restaurant=restaurant,
            items=[food],
            delivery_location=delivery_location,
            delivery_partner="Suresh V.",
            delivery_phone="+91 98765 12345",
            initial_eta_minutes=20,
            eta_minutes=20,
            status="PREPARING",
            created_at=now_iso,
            estimated_total=estimated_total,
            simulation_enabled=True,
        )

        if idempotency_key:
            self._orders_by_key[idempotency_key] = result

        return result
