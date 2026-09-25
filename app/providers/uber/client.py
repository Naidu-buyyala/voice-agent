from typing import Any, Optional
import httpx
from app.config.settings import get_settings


class UberAPIError(Exception):
    """Normalized exception for Uber API errors."""
    def __init__(self, status_code: int, message: str, error_code: Optional[str] = None):
        super().__init__(f"Uber API Error [{status_code}]: {message}")
        self.status_code = status_code
        self.message = message
        self.error_code = error_code


class UberClient:
    """
    Pure HTTP Client for Uber REST API v1.2.
    Only handles networking, headers, auth tokens, and response validation.
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        sandbox: Optional[bool] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        settings = get_settings()
        self.sandbox = settings.UBER_SANDBOX_MODE if sandbox is None else sandbox
        self.base_url = (
            "https://sandbox-api.uber.com/v1.2"
            if self.sandbox
            else "https://api.uber.com/v1.2"
        )
        self.access_token = access_token or "mock_uber_oauth_token"
        self._client = client

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept-Language": "en_US",
        }

    async def get_products(self, latitude: float, longitude: float) -> list[dict[str, Any]]:
        """GET /v1.2/products"""
        params = {"latitude": latitude, "longitude": longitude}
        return await self._request("GET", "/products", params=params)

    async def get_price_estimates(
        self,
        start_latitude: float,
        start_longitude: float,
        end_latitude: float,
        end_longitude: float,
    ) -> list[dict[str, Any]]:
        """GET /v1.2/estimates/price"""
        params = {
            "start_latitude": start_latitude,
            "start_longitude": start_longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
        }
        return await self._request("GET", "/estimates/price", params=params)

    async def create_ride_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /v1.2/requests"""
        return await self._request("POST", "/requests", json_body=payload)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        if self._client:
            resp = await self._client.request(
                method, url, headers=headers, params=params, json=json_body, timeout=10.0
            )
            return self._handle_response(resp)

        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method, url, headers=headers, params=params, json=json_body, timeout=10.0
            )
            return self._handle_response(resp)

    def _handle_response(self, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            try:
                data = resp.json()
                msg = data.get("message", resp.text)
                code = data.get("code")
            except Exception:
                msg = resp.text
                code = None
            raise UberAPIError(resp.status_code, msg, code)
        return resp.json()
