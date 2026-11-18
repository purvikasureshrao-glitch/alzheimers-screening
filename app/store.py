"""Saving patients and screenings, and doctor sign-up / login / password reset.

Supabase is used when SUPABASE_URL and SUPABASE_SERVICE_KEY are set in `.env` (see supabase/SETUP.md). The service key stays on this
server and is never sent to the browser. Without them, records go to data/local_store.json so the app still works.
Doctors sign up in the app (name, username, password; stored as salted scrypt hashes in the `doctors` table / local file). A one-time recovery
code (also hashed) lets a doctor reset a forgotten password without e-mail. Simple by design, not high security."""
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_FILE = ROOT / "data" / "local_store.json"
_tokens = {}          # doctor session tokens -> (username, expiry time) (in memory; a restart logs doctors out)
_fails = {}           # username -> times of recent failed logins / resets
TOKEN_SECONDS = 8 * 3600
MAX_FAILS, FAIL_WINDOW = 5, 300
RECOVERY_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I
QUESTIONS = {"age": "Age", "gender": "Gender", "ethnicity": "Ethnicity", "education": "Education", "systolic_bp": "Blood pressure (top number)",
             "diastolic_bp": "Blood pressure (bottom number)", "diabetes": "Diabetes", "high_blood_pressure": "High blood pressure",
             "high_cholesterol": "High cholesterol", "family_history_dementia": "Family history of dementia", "smoking": "Smoking",
             "head_injury": "Serious head injury", "hearing_loss": "Hearing loss", "depression": "Depression or low mood",
             "physical_activity": "Physical activity"}
NUMERIC = {"age": (18, 110), "systolic_bp": (70, 250), "diastolic_bp": (40, 150)}


def load_env(path=ROOT / ".env"):
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Local:
    name = "local file"

    def __init__(self, path=LOCAL_FILE):
        self.path = Path(path)

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {"patients": [], "screenings": [], "doctors": []}

    def _write(self, d):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d), encoding="utf-8")

    def add_patient(self, row):
        d = self._read(); d["patients"].append(row); self._write(d)

    def add_screening(self, row):
        d = self._read(); d["screenings"].append(row); self._write(d)

    def get_doctor(self, user):
        return next((d for d in self._read().get("doctors", []) if d["username"] == user), None)

    def add_doctor(self, row):
        d = self._read(); d.setdefault("doctors", []).append(row); self._write(d)

    def update_doctor(self, user, fields):
        d = self._read()
        for row in d.get("doctors", []):
            if row["username"] == user:
                row.update(fields)
        self._write(d)

    def search(self, q):
        q = q.strip().lower()
        hits = [p for p in self._read()["patients"] if q and (q in p["code"].lower() or q in p["name"].lower())]
        return [{"code": p["code"], "name": p["name"], "created_at": p.get("created_at")} for p in hits[:20]]

    def record(self, code):
        d = self._read()
        p = next((p for p in d["patients"] if p["code"] == code), None)
        return p, [s for s in d["screenings"] if s["code"] == code]


class Supabase:
    name = "Supabase"

    def __init__(self, url, key):
        u = urllib.parse.urlparse(url.strip())
        self.url, self.key = f"{u.scheme}://{u.netloc}", key   # keep only the project address; a pasted /rest/v1/ suffix is ignored

    def _call(self, method, table, body=None, query=""):
        req = urllib.request.Request(f"{self.url}/rest/v1/{table}{query}", method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"apikey": self.key, "Authorization": "Bearer " + self.key,
                                              "Content-Type": "application/json", "Prefer": "return=minimal"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase {e.code}: {e.read().decode(errors='replace')[:200]}") from None
        except (urllib.error.URLError, TimeoutError) as e:   # offline, DNS, timeout, or a paused project
            raise RuntimeError(f"cannot reach Supabase ({e}); check the internet connection and that the project is not paused") from None
        return json.loads(raw) if raw else None

    def add_patient(self, row):
        self._call("POST", "patients", row)

    def add_screening(self, row):
        self._call("POST", "screenings", row)

    def get_doctor(self, user):
        r = self._call("GET", "doctors", query="?username=eq." + urllib.parse.quote(user, safe="") + "&select=*")
        return r[0] if r else None

    def add_doctor(self, row):
        self._call("POST", "doctors", row)

    def update_doctor(self, user, fields):
        self._call("PATCH", "doctors", fields, query="?username=eq." + urllib.parse.quote(user, safe=""))

    def search(self, q):
        q = q.strip()
        for ch in ",*()":
            q = q.replace(ch, " ")
        q = q.strip()
        if not q:
            return []
        flt = urllib.parse.quote(f"(code.ilike.*{q}*,name.ilike.*{q}*)", safe="()*,.")
        return self._call("GET", "patients", query=f"?select=code,name,created_at&or={flt}&order=created_at.desc&limit=20")

    def record(self, code):
        c = urllib.parse.quote(code, safe="")
        p = self._call("GET", "patients", query=f"?code=eq.{c}&select=*")
        s = self._call("GET", "screenings", query=f"?code=eq.{c}&select=*&order=created_at.desc")
        return (p[0] if p else None), s


def backend():
    load_env()
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    return Supabase(url, key) if url and key else Local()


def new_code():
    return "AD-" + "".join(secrets.choice("0123456789") for _ in range(6))


def check_questionnaire(name, q):
    """Every answer is required. Raises ValueError listing what is missing or out of range."""
    q = q or {}
    missing = [("Full name" if k == "name" else QUESTIONS[k]) for k in ["name", *QUESTIONS]
               if not str(name if k == "name" else q.get(k, "")).strip()]
    if missing:
        raise ValueError("Please fill in: " + ", ".join(missing))
    for k, (lo, hi) in NUMERIC.items():
        try:
            ok = lo <= float(q[k]) <= hi
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{QUESTIONS[k]} must be a number between {lo} and {hi}")


def create_patient(db, name, questionnaire):
    name = (name or "").strip()
    check_questionnaire(name, questionnaire)
    code = new_code()
    db.add_patient({"code": code, "name": name, "questionnaire": questionnaire, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return code


def save_screening(db, code, results, combined, summary, clock_jpeg):
    db.add_screening({"code": code, "results": results, "combined": combined, "summary": summary, "clock_jpeg": clock_jpeg,
                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ---------- doctor accounts ----------

def _hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def _session(user):
    t = secrets.token_urlsafe(24)
    _tokens[t] = (user, time.time() + TOKEN_SECONDS)
    return t


def _check_attempts(user):
    now = time.time()
    _fails[user] = [t for t in _fails.get(user, []) if now - t < FAIL_WINDOW]
    if len(_fails[user]) >= MAX_FAILS:
        raise PermissionError("too many wrong attempts; wait 5 minutes and try again")


def _fail(user):
    _fails.setdefault(user, []).append(time.time())


def new_recovery_code():
    return "-".join("".join(secrets.choice(RECOVERY_ALPHABET) for _ in range(4)) for _ in range(3))


def _norm_code(code):
    return "".join(c for c in (code or "").upper() if c.isalnum())


def _recovery_fields(code):
    salt = secrets.token_bytes(16)
    return {"recovery_salt": salt.hex(), "recovery_hash": _hash(_norm_code(code), salt)}


def _check_password(password, confirm):
    if len(password or "") < 8:
        raise ValueError("password must be at least 8 characters")
    if confirm is not None and confirm != password:
        raise ValueError("the two passwords do not match")


def doctor_signup(db, user, password, full_name="", confirm=None):
    """Creates a doctor account. Returns (session token, one-time recovery code). Raises ValueError with a message for the page."""
    user, full_name = (user or "").strip(), (full_name or "").strip()
    if len(full_name) < 2:
        raise ValueError("please enter your full name")
    if len(user) < 3:
        raise ValueError("username must be at least 3 characters")
    _check_password(password, confirm)
    if db.get_doctor(user):
        raise ValueError("that username is taken; log in instead")
    salt, code = secrets.token_bytes(16), new_recovery_code()
    db.add_doctor({"username": user, "full_name": full_name, "salt": salt.hex(), "pw_hash": _hash(password, salt),
                   **_recovery_fields(code), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return _session(user), code


def doctor_login(db, user, password):
    """Returns (session token, display name), or None for wrong details."""
    user = (user or "").strip()
    _check_attempts(user)
    d = db.get_doctor(user)
    if d and hmac.compare_digest(_hash(password or "", bytes.fromhex(d["salt"])), d["pw_hash"]):
        return _session(user), d.get("full_name") or user
    _fail(user)
    return None


def doctor_reset(db, user, recovery_code, new_password, confirm=None):
    """Forgot password: username + recovery code -> new password. Returns (token, NEW recovery code); the old code stops working."""
    user = (user or "").strip()
    _check_attempts(user)
    d = db.get_doctor(user)
    ok = bool(d and d.get("recovery_hash") and hmac.compare_digest(
        _hash(_norm_code(recovery_code), bytes.fromhex(d["recovery_salt"])), d["recovery_hash"]))
    if not ok:
        _fail(user)
        raise ValueError("username or recovery code is not right")
    _check_password(new_password, confirm)
    salt, code = secrets.token_bytes(16), new_recovery_code()
    db.update_doctor(user, {"salt": salt.hex(), "pw_hash": _hash(new_password, salt), **_recovery_fields(code)})
    _fails.pop(user, None)
    return _session(user), code


def doctor_new_recovery(db, token):
    """Logged-in doctor asks for a fresh recovery code (also how accounts made before recovery codes existed get one)."""
    user = doctor_user(token)
    code = new_recovery_code()
    db.update_doctor(user, _recovery_fields(code))
    return code


def doctor_user(token):
    row = _tokens.get(token or "")
    return row[0] if row and row[1] > time.time() else None


def doctor_ok(token):
    return doctor_user(token) is not None
