import os
import unittest
from unittest.mock import patch

from whisper_ui.config import get_settings
from whisper_ui.security import LocalRequestGuardMiddleware


class SecurityMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def _request(self, host: str) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []

        async def app(scope, receive, send):
            messages.append({"type": "app.reached"})

        middleware = LocalRequestGuardMiddleware(app, settings_provider=get_settings)
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "scheme": "http",
            "headers": [(b"host", host.encode())],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(scope, receive, send)
        return messages

    async def test_localhost_remains_allowed_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            try:
                messages = await self._request("localhost:8000")
            finally:
                get_settings.cache_clear()

        self.assertIn({"type": "app.reached"}, messages)

    async def test_configured_vps_host_is_allowed(self):
        with patch.dict(os.environ, {"ALLOWED_HOSTS": "app.example.com"}, clear=True):
            get_settings.cache_clear()
            try:
                messages = await self._request("app.example.com")
            finally:
                get_settings.cache_clear()

        self.assertIn({"type": "app.reached"}, messages)

    async def test_unconfigured_vps_host_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            try:
                messages = await self._request("app.example.com")
            finally:
                get_settings.cache_clear()

        self.assertNotIn({"type": "app.reached"}, messages)
        self.assertEqual(messages[0]["status"], 400)
