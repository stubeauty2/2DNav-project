"""Small dependency-light client for the local visual grounding service."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class VisionToolError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class VisionToolClient:
    def __init__(self, base_url: str, timeout: Optional[float] = 120.0):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = requests.request(method, self.base_url + path, json=payload, timeout=self.timeout)
            value = response.json()
        except Exception as exc:
            raise VisionToolError(f"vision tool request failed for {path}: {exc}") from exc
        if response.status_code >= 400:
            detail = value.get("error") if isinstance(value, dict) else response.reason
            raise VisionToolError(f"vision tool rejected {path}: HTTP {response.status_code}: {detail}", status_code=response.status_code)
        if not isinstance(value, dict):
            raise VisionToolError(f"vision tool returned non-object for {path}")
        if value.get("error"):
            raise VisionToolError(str(value["error"]))
        return value

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/v1/candidates", payload)

