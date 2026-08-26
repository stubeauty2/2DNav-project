"""Dependency-light HTTP client used by the navigation process."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class VisionToolError(RuntimeError):
    def __init__(self, message: str, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class VisionToolClient:
    def __init__(self, base_url: str, timeout: Optional[float] = None):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = None if timeout is None else float(timeout)

    def _request(self, method: str, path: str, payload=None) -> Dict[str, Any]:
        try:
            response = requests.request(
                method,
                self.base_url + path,
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise VisionToolError(f"vision tool request failed for {path}: {exc}") from exc
        try:
            value = response.json()
        except Exception as exc:
            raise VisionToolError(
                f"vision tool returned invalid JSON for {path}: HTTP {response.status_code}",
                status_code=response.status_code,
            ) from exc
        if response.status_code >= 400:
            detail = value.get("error") if isinstance(value, dict) else response.reason
            raise VisionToolError(
                f"vision tool rejected {path}: HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
            )
        if not isinstance(value, dict):
            raise VisionToolError(f"vision tool returned non-object for {path}")
        if value.get("error"):
            raise VisionToolError(str(value["error"]))
        return value

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/candidates", payload)

    def matches(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/matches", payload)
