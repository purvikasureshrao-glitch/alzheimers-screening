"""Saving patients and screenings, and the doctor login.

Supabase is used when SUPABASE_URL and SUPABASE_SERVICE_KEY are set in `.env` (see supabase/SETUP.md). The service key stays on this
server and is never sent to the browser. Without them, records go to data/local_store.json so the app still works.
The doctor login is one username and password from `.env` (DOCTOR_USER, DOCTOR_PASSWORD): simple by design, not high security."""
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
_tokens = {}          # doctor session tokens -> expiry time (in memory; a restart logs doctors out)
TOKEN_SECONDS = 8 * 3600


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
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {"patients": [], "screenings": []}

    def _write(self, d):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d), encoding="utf-8")

    def add_patient(self, row):
        d = self._read(); d["patients"].append(row); self._write(d)

    def add_screening(self, row):
        d = self._read(); d["screenings"].append(row); self._write(d)

    def search(self, q):
        q = q.strip().lower()
        return [p for p in self._read()["patients"] if q and (q == p["code"].lower() or q in p["name"].lower())]

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
            with urllib.request.urlopen(req, timeout=20) as r:
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

    def search(self, q):
        q = q.strip()
        if not q:
            return []
        flt = urllib.parse.quote(f"(code.eq.{q.upper()},name.ilike.*{q.replace(',', ' ').replace('*', '')}*)", safe="()*,.")
        return self._call("GET", "patients", query=f"?select=code,name,created_at&or={flt}&limit=20")

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


def create_patient(db, name, questionnaire):
    name = (name or "").strip()
    if not name:
        raise ValueError("a name is required")
    code = new_code()
    db.add_patient({"code": code, "name": name, "questionnaire": questionnaire, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return code


def save_screening(db, code, results, combined, summary, clock_jpeg):
    db.add_screening({"code": code, "results": results, "combined": combined, "summary": summary, "clock_jpeg": clock_jpeg,
                      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def doctor_login(user, password):
    """Returns a session token, or None for wrong details / no doctor account configured."""
    load_env()
    u, p = os.environ.get("DOCTOR_USER"), os.environ.get("DOCTOR_PASSWORD")
    if not u or not p:
        return None
    if hmac.compare_digest(user.encode(), u.encode()) & hmac.compare_digest(password.encode(), p.encode()):
        t = secrets.token_urlsafe(24)
        _tokens[t] = time.time() + TOKEN_SECONDS
        return t
    return None


def doctor_ok(token):
    exp = _tokens.get(token or "")
    return bool(exp and exp > time.time())
