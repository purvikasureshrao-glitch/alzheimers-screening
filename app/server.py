"""Local app server (standard library only). Run from the repo root:  python -m app.server   then open http://127.0.0.1:8765

Everything stays on this machine: no result is uploaded and no name is stored. Results live in the browser page until it is closed.
Clock takes an uploaded photo (app/clock.py). Eye / Speech capture is not wired; they are reported as 'not run' (see app/runners.py)."""
import base64
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import motor
from .combine import combine
from .evidence import CAVEAT, EVIDENCE
from .schema import ORDER, Result
from .summary import summarize

STATIC = Path(__file__).resolve().parent / "static"
TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".task": "application/octet-stream"}
LABEL_NOT_RUN = {"clock": "Clock drawing: not run yet", "motor": "Finger tapping: not run yet",
                 "eye": "Eye movement: not run yet", "speech": "Speech: not run yet"}


def not_run(m):
    return Result(m, "not_run", LABEL_NOT_RUN[m], evidence=EVIDENCE[m], caveat=CAVEAT[m])


def result_from_dict(d):
    return Result(d["modality"], d["status"], d.get("label", ""), d.get("measured", {}), d.get("evidence", ""), d.get("caveat", ""),
                  d.get("counts", True))


NEEDED = {"clock": ["clock_vit.onnx", "clock_cropnet.onnx", "clock_features.json"], "motor": ["motor_reference.json"],
          "eye": ["eye_flag.onnx", "eye_features.json"], "speech": ["speech_whisper.onnx", "speech_whisper_features.json"]}


def status():
    """Which model files exist and whether Ollama answers (for the banner at the top of the page)."""
    models = Path(__file__).resolve().parent.parent / "models"
    files = {m: [f for f in fs if not (models / f).exists()] for m, fs in NEEDED.items()}
    ollama = {"running": False, "model": False}
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as r:
            ollama = {"running": True, "model": any(m["name"].startswith("llama3.1:8b") for m in json.load(r)["models"])}
    except Exception:
        pass
    return {"missing_files": {m: f for m, f in files.items() if f}, "ollama": ollama,
            "vendor": (STATIC / "vendor" / "face_landmarker.task").exists()}


def handle(path, body):
    """Pure routing function (testable without sockets). Returns (status, dict)."""
    if path == "/api/evidence":
        return 200, {"order": list(ORDER), "not_run": {m: not_run(m).to_dict() for m in ORDER}}
    if path == "/api/status":
        return 200, status()
    if path == "/api/clock":
        from . import clock   # imported on first use: needs onnxruntime/opencv and models/clock_*.onnx
        r, png = clock.analyse(base64.b64decode(body["image"]))
        return 200, {**r.to_dict(), "preview": base64.b64encode(png).decode() if png else None}
    if path == "/api/eye":
        from . import eye
        cal = body.get("calibration")
        return 200, eye.score(body["tasks"], None if cal is None else float(cal) * 100).to_dict()
    if path == "/api/speech":
        from . import speech   # first use loads Whisper `small` (downloads ~460 MB once) and models/speech_whisper.onnx
        r, text = speech.analyse(base64.b64decode(body["audio"]))
        return 200, {**r.to_dict(), "transcript": text}
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
        f = (STATIC / self.path[len("/static/"):]).resolve() if self.path.startswith("/static/") else None
        if f and STATIC in f.parents and f.is_file():   # vendored MediaPipe + eye.js; nothing is fetched from the internet
            return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
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
