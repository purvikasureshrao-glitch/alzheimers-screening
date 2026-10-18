"""Checks that saving and finding a record works with the storage in .env. Prints only OK / the error, never a key.

Run from the repo root:  python -m app.store_check     (set PYTHONPATH=. first on Windows if needed)
It adds one test patient 'Storage Check' (code AD-xxxxxx); delete it later in the Supabase Table Editor."""
from . import store


def main():
    db = store.backend()
    print("Storage:", db.name)
    try:
        code = store.create_patient(db, "Storage Check", {"note": "created by app.store_check"})
        store.save_screening(db, code, [], {"text": "check"}, None, None)
        found = db.search(code)
        p, s = db.record(code)
        ok = bool(found) and p is not None and len(s) == 1
        print("OK: saved and found", code if ok else "-> but the record could not be read back")
    except Exception as e:   # message only; urllib errors carry the server's reply, not our key
        print("FAILED:", str(e)[:300])


if __name__ == "__main__":
    main()
