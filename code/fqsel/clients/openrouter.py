"""One OpenRouter chat-completions request, shared by the strong-LLM and judge clients.

Maps HTTP 429 to RateLimited (honouring retry-after) and 5xx, timeouts and dropped connections to
TransientError, so that callers can wrap the request in with_retries.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

from fqsel.clients.retry import RateLimited, TransientError


class ProviderMismatch(RuntimeError):
    """A response came from a provider other than the pinned one."""


def post_chat(endpoint: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            ra = e.headers.get("retry-after")
            raise RateLimited(float(ra) if ra else None) from e
        if e.code >= 500:
            raise TransientError(f"HTTP {e.code}") from e
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
        # includes a connection dropped mid-response (http.client.IncompleteRead)
        raise TransientError(str(e)) from e
