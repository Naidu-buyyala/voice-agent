from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
import httpx
import re
from pathlib import Path

from app.config.settings import get_settings

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/uber", tags=["Uber Authentication"])


@router.get("/callback", summary="Uber OAuth callback handler")
@router.get("/callback/", summary="Uber OAuth callback handler with slash")
async def uber_oauth_callback(
    request: Request,
):
    """
    Receives authorization code or errors from Uber, exchanges it for an access_token,
    and saves it to .env for the application.
    """
    settings = get_settings()
    params = dict(request.query_params)
    logger.info(f"Uber callback received with params: {params}")

    code = params.get("code")
    if not code:
        err = params.get("error_description") or params.get("error") or "No code provided in callback"
        return HTMLResponse(
            f"<body style='background:#0f172a;color:white;font-family:sans-serif;padding:40px;text-align:center;'>"
            f"<h2 style='color:#ef4444;'>OAuth Callback Notice</h2>"
            f"<p>{err}</p>"
            f"<p>Full query params: <code>{params}</code></p>"
            f"</body>"
        )

    # Exchange code for token
    token_url = (
        "https://sandbox-login.uber.com/oauth/v2/token"
        if settings.UBER_SANDBOX_MODE
        else "https://login.uber.com/oauth/v2/token"
    )

    redirect_uri = "http://localhost:8000/api/v1/auth/uber/callback"

    data = {
        "client_id": settings.UBER_CLIENT_ID,
        "client_secret": settings.UBER_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code": code,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data, timeout=15.0)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to exchange code for token: {resp.text}",
        )

    token_data = resp.json()
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in")
    scope = token_data.get("scope")

    # Update .env file automatically
    env_path = Path(".env")
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if "UBER_ACCESS_TOKEN=" in content:
            content = re.sub(r"UBER_ACCESS_TOKEN=.*", f"UBER_ACCESS_TOKEN={access_token}", content)
        else:
            content += f"\nUBER_ACCESS_TOKEN={access_token}"
        env_path.write_text(content, encoding="utf-8")

    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Uber Authentication Successful</title>
            <style>
                body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: white; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
                .card {{ background: #1e293b; padding: 40px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.4); max-width: 500px; text-align: center; }}
                h2 {{ color: #10b981; margin-bottom: 12px; }}
                p {{ color: #94a3b8; font-size: 15px; line-height: 1.6; }}
                code {{ background: #334155; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #38bdf8; word-break: break-all; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Authentication Successful!</h2>
                <p>Your Uber sandbox account has been linked. The access token has been saved into your <code>.env</code> file.</p>
                <p><strong>Scopes:</strong> {scope}</p>
                <p>You can now close this tab and proceed with testing.</p>
            </div>
        </body>
        </html>
        """
    )
