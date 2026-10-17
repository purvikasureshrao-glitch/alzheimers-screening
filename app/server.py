"""Local app server (standard library only). Run from the repo root:  python -m app.server   then open http://127.0.0.1:8765

Everything stays on this machine: no result is uploaded and no name is stored. Results live in the browser page until it is closed.
Clock / Eye / Speech capture + ONNX runners are not wired yet; they are reported as 'not run' (see app/runners.py)."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import motor
from .combine import combine
from .evidence import CAVEAT, EVIDENCE
from .schema import ORDER, Result
from .summary import summarize

STATIC = Path(__file__).resolve().parent / "static"
LABEL_NOT_RUN = {"clock": "Clock drawing: not run yet", "motor": "Finger tapping: not run yet",
                 "eye": "Eye movement: not run yet", "speech": "Speech: not run (research only)"}


def not_run(m):
    return Result(m, "not_run", LABEL_NOT_RUN[m], evidence=EVIDENCE[m], caveat=CAVEAT[m])


def result_from_dict(d):
    return Result(d["modality"], d["status"], d.get("label", ""), d.get("measured", {}), d.get("evidence", ""), d.get("caveat", ""))


def handle(path, body):
    """Pure routing function (testable without sockets). Returns (status, dict)."""
    if path == "/api/evidence":
        return 200, {"order": list(ORDER), "not_run": {m: not_run(m).to_dict() for m in ORDER}}
    if path == "/api/motor":
        return 200, motor.score([float(t) for t in body["taps"]], float(body["age"])).to_dict()
    if path == "/api/summary":
        by = {d["modality"]: result_from_dict(d) for d in body["results"]}
        results = [by.get(m) or not_run(m) for m in ORDER]
        return 200, {"combined": combine(results), "summary": summarize(results, model=body.get("model", "llama3.1:8b"))}
    return 404, {"error": "not found"}


class H(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        self._send(*handle(self.path, {}))

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._send(*handle(self.path, body))
        except (KeyError, ValueError) as e:
            self._send(400, {"error": f"bad request: {e}"})

    def log_message(self, *a):   # no request log: nothing about the person is written anywhere
        pass


if __name__ == "__main__":
    print("Alzheimer's screening app on http://127.0.0.1:8765  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
