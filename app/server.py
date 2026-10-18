"""Local app server (standard library only). Run from the repo root:  python -m app.server   then open http://127.0.0.1:8765

Patients and screenings are saved through app/store.py (Supabase if configured in .env, else data/local_store.json); the doctor view needs the
doctor login. Models run on this machine. Clock takes a JPEG of the drawing (drawn on screen or uploaded)."""
import base64
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import motor, store
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
    return {"missing_files": {m: f for m, f in files.items() if f}, "ollama": ollama, "storage": db().name,
            "cookie_picture": any((STATIC / ("cookie_theft" + e)).exists() for e in (".jpg", ".png")),
            "vendor": (STATIC / "vendor" / "face_landmarker.task").exists()}


DB = None


def db():
    global DB
    DB = DB or store.backend()
    return DB


def handle(path, body):
    """Pure routing function (testable without sockets). Returns (status, dict)."""
    if path == "/api/patient":
        return 200, {"code": store.create_patient(db(), body.get("name"), body.get("questionnaire", {}))}
    if path == "/api/save":
        store.save_screening(db(), body["code"], body["results"], body.get("combined"), body.get("summary"), body.get("clock_jpeg"))
        return 200, {"saved": True, "where": db().name}
    if path == "/api/doctor/login":
        t = store.doctor_login(str(body.get("user", "")), str(body.get("password", "")))
        return (200, {"token": t}) if t else (401, {"error": "wrong username or password, or no doctor account set in .env"})
    if path.startswith("/api/doctor/"):
        if not store.doctor_ok(body.get("token")):
            return 401, {"error": "please log in again"}
        if path == "/api/doctor/search":
            return 200, {"patients": db().search(str(body.get("q", "")))}
        if path == "/api/doctor/record":
            p, s = db().record(str(body.get("code", "")))
            return (200, {"patient": p, "screenings": s}) if p else (404, {"error": "no such patient"})
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
        except RuntimeError as e:   # storage failure (e.g. Supabase rejected the request)
            self._send(502, {"error": str(e)})

    def log_message(self, *a):   # no request log: nothing about the person is written anywhere
        pass


if __name__ == "__main__":
    print("Alzheimer's screening app on http://127.0.0.1:8765  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
