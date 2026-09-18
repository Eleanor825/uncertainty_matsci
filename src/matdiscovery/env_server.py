"""JSONL process RPC for one isolated official benchmark environment.

Launch with the benchmark's Python, not the policy Python::

    PYTHONPATH=/fresh/project/src /made/venv/bin/python -m matdiscovery.env_server --benchmark made
    PYTHONPATH=/fresh/project/src /crystal/venv/bin/python -m matdiscovery.env_server --benchmark crystalgym

Requests: {"id": 1, "op": "init|reset|observe|tool|step|close", "args": {...}}.
Every response has the same id and either ok/result or ok=false/error. Init creates
the initial episode; reset starts a subsequent deterministically seeded episode.
No LLM client, scientific fallback, or alternate evaluator is implemented here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import traceback
from typing import Any, TextIO

from .benchmark_adapters import AdapterError, create_adapter, jsonable


class RPCSession:
    def __init__(self, benchmark: str | None = None, factory=create_adapter):
        self.benchmark = benchmark
        self.factory = factory
        self.adapter = None
        self.closed = False

    def handle(self, request: Any) -> dict[str, Any]:
        raw_id = request.get("id") if isinstance(request, dict) else None
        request_id = raw_id if not isinstance(raw_id, bool) and isinstance(raw_id, (int, str, type(None))) else None
        try:
            if not isinstance(request, dict) or "id" not in request:
                raise AdapterError("invalid_request", "Request must be an object with an id.")
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str, type(None))):
                raise AdapterError("invalid_request", "id must be a string, integer or null.")
            op, args = request.get("op"), request.get("args", {})
            if not isinstance(args, dict):
                raise AdapterError("invalid_request", "args must be a JSON object.")
            if op not in {"init", "reset", "observe", "tool", "step", "close"}:
                raise AdapterError("unknown_operation", f"Unknown operation: {op!r}")
            if self.closed:
                raise AdapterError("session_closed", "This server session is closed.")
            if op == "init":
                if self.adapter is not None:
                    raise AdapterError("already_initialized", "Start another process for another task.")
                name = args.get("benchmark", self.benchmark)
                if self.benchmark and name != self.benchmark:
                    raise AdapterError("benchmark_mismatch", "A process cannot mix benchmark dependencies.")
                self.adapter = self.factory(name, args)
                result = self.adapter.initial_result()
            elif op == "close":
                result = self.adapter.dispatch(op, args) if self.adapter else {"closed": True}
                self.closed = True
            elif self.adapter is None:
                raise AdapterError("not_initialized", "Call init first.")
            else:
                result = self.adapter.dispatch(op, args)
            return {"id": request_id, "ok": True, "result": jsonable(result)}
        except AdapterError as exc:
            return {"id": request_id, "ok": False, "error": exc.as_dict()}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            details = self.adapter.status() if self.adapter else {}
            return {"id": request_id, "ok": False, "error": {
                "code": "unexpected_error", "type": type(exc).__name__,
                "message": str(exc), "details": jsonable(details),
            }}


def serve(input_stream: TextIO, output_stream: TextIO, session: RPCSession) -> None:
    """Transport loop; contract tests can use StringIO without scientific packages."""
    for line in input_stream:
        try:
            def reject_constant(value):
                raise ValueError(f"Non-standard JSON constant: {value}")
            request = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            response = {"id": None, "ok": False, "error": {
                "code": "invalid_json", "type": type(exc).__name__, "message": str(exc),
            }}
        else:
            with contextlib.redirect_stdout(sys.stderr):
                response = session.handle(request)
        output_stream.write(json.dumps(response, allow_nan=False, ensure_ascii=False) + "\n")
        output_stream.flush()
        if session.closed:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("made", "crystalgym"))
    args = parser.parse_args()
    # redirect_stdout alone cannot catch native library writes to file descriptor 1.
    # Preserve a private protocol descriptor and send all library/native stdout to stderr.
    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    with os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1) as protocol:
        serve(sys.stdin, protocol, RPCSession(args.benchmark))


if __name__ == "__main__":
    main()
