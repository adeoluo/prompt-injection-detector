"""HTTP service: scores prompts with the linear detector.

Standard library only (``http.server``). Routes:

- ``POST /v1/detect`` with ``{"text": "..."}`` or ``{"texts": [...]}`` returns scores,
  flags and the threshold; add ``"explain": true`` for the top n-grams.
- ``GET /health`` (liveness), ``GET /ready`` (model loaded), ``GET /metrics``
  (Prometheus text: requests, errors, detections and latency per route).
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from prompt_injection_detector import __version__
from prompt_injection_detector.model import LinearDetector

MAX_BODY_BYTES = 1 << 20
MAX_TEXTS = 256
LATENCY_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)


@dataclass
class Metrics:
    """Thread-safe request counters and latency histograms."""

    requests: dict[tuple[str, int], int] = field(default_factory=lambda: defaultdict(int))
    buckets: dict[str, list[int]] = field(default_factory=dict)
    latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    latency_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    scanned: int = 0
    flagged: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self.lock:
            self.requests[(route, status)] += 1
            counts = self.buckets.setdefault(route, [0] * len(LATENCY_BUCKETS))
            for i, b in enumerate(LATENCY_BUCKETS):
                if seconds <= b:
                    counts[i] += 1
            self.latency_sum[route] += seconds
            self.latency_count[route] += 1

    def count_scans(self, scanned: int, flagged: int) -> None:
        with self.lock:
            self.scanned += scanned
            self.flagged += flagged

    def render(self) -> str:
        with self.lock:
            lines = [
                "# HELP pid_requests_total HTTP requests by route and status.",
                "# TYPE pid_requests_total counter",
            ]
            for (route, status), n in sorted(self.requests.items()):
                lines.append(f'pid_requests_total{{route="{route}",status="{status}"}} {n}')
            lines += [
                "# HELP pid_request_seconds Request latency.",
                "# TYPE pid_request_seconds histogram",
            ]
            for route, counts in sorted(self.buckets.items()):
                for b, n in zip(LATENCY_BUCKETS, counts, strict=True):
                    lines.append(f'pid_request_seconds_bucket{{route="{route}",le="{b}"}} {n}')
                total = self.latency_count[route]
                lines.append(f'pid_request_seconds_bucket{{route="{route}",le="+Inf"}} {total}')
                lines.append(
                    f'pid_request_seconds_sum{{route="{route}"}} {self.latency_sum[route]:.6f}'
                )
                lines.append(f'pid_request_seconds_count{{route="{route}"}} {total}')
            lines += [
                "# HELP pid_texts_scanned_total Texts scored.",
                "# TYPE pid_texts_scanned_total counter",
                f"pid_texts_scanned_total {self.scanned}",
                "# HELP pid_texts_flagged_total Texts at or above the threshold.",
                "# TYPE pid_texts_flagged_total counter",
                f"pid_texts_flagged_total {self.flagged}",
            ]
        return "\n".join(lines) + "\n"


class ValidationError(ValueError):
    """A client error, reported with HTTP 400."""


def parse_request(body: bytes) -> tuple[list[str], bool, float | None]:
    """``(texts, explain, threshold)`` from a JSON request body."""
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc.msg}") from None
    if not isinstance(payload, dict):
        raise ValidationError("body must be a JSON object")
    if "text" in payload:
        texts = [payload["text"]]
    elif "texts" in payload:
        texts = payload["texts"]
        if not isinstance(texts, list) or not texts:
            raise ValidationError("texts must be a non-empty list")
    else:
        raise ValidationError("provide text or texts")
    if len(texts) > MAX_TEXTS:
        raise ValidationError(f"at most {MAX_TEXTS} texts per request")
    if not all(isinstance(t, str) for t in texts):
        raise ValidationError("every text must be a string")
    explain = payload.get("explain", False)
    if not isinstance(explain, bool):
        raise ValidationError("explain must be a boolean")
    threshold = payload.get("threshold")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValidationError("threshold must be a number in [0, 1]")
    return texts, explain, None if threshold is None else float(threshold)


class App:
    """Routing and request handling, independent of the socket layer."""

    def __init__(self, detector: LinearDetector) -> None:
        self.detector = detector
        self.metrics = Metrics()

    def detect(self, body: bytes) -> dict[str, Any]:
        texts, explain, threshold = parse_request(body)
        thr = self.detector.threshold if threshold is None else threshold
        scores = self.detector.score_batch(texts)
        results: list[dict[str, Any]] = []
        for text, s in zip(texts, scores, strict=True):
            item: dict[str, Any] = {"score": round(float(s), 6), "injection": bool(s >= thr)}
            if explain:
                item["top_ngrams"] = [
                    {"kind": c.kind, "ngram": c.ngram, "weight": round(c.weight, 4)}
                    for c in self.detector.explain(text, top_k=5)
                ]
            results.append(item)
        flagged = sum(r["injection"] for r in results)
        self.metrics.count_scans(len(results), flagged)
        return {"threshold": thr, "results": results}

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, str]:
        """``(status, content type, body)`` for one request."""
        route = path.split("?", 1)[0]
        t0 = time.perf_counter()
        status, ctype, out = 404, "application/json", json.dumps({"error": "not found"})
        try:
            if method == "GET" and route == "/health":
                status, out = 200, json.dumps({"status": "ok", "version": __version__})
            elif method == "GET" and route == "/ready":
                status, out = 200, json.dumps({"ready": True, "dim": self.detector.cfg.dim})
            elif method == "GET" and route == "/metrics":
                status, ctype, out = 200, "text/plain; version=0.0.4", self.metrics.render()
            elif method == "POST" and route == "/v1/detect":
                status, out = 200, json.dumps(self.detect(body))
            elif route in ("/health", "/ready", "/metrics", "/v1/detect"):
                status, out = 405, json.dumps({"error": "method not allowed"})
        except ValidationError as exc:
            status, out = 400, json.dumps({"error": str(exc)})
        self.metrics.observe(route if status != 404 else "other", status, time.perf_counter() - t0)
        return status, ctype, out


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"prompt-injection-detector/{__version__}"

        def _reply(self, status: int, ctype: str, body: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(*app.handle("GET", self.path, b""))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                self._reply(413, "application/json", json.dumps({"error": "body too large"}))
                return
            self._reply(*app.handle("POST", self.path, self.rfile.read(length)))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(
    detector: LinearDetector, host: str = "127.0.0.1", port: int = 8000
) -> None:  # pragma: no cover - network loop
    server = ThreadingHTTPServer((host, port), make_handler(App(detector)))
    print(f"listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
