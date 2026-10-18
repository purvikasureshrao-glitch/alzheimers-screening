"""Storage and doctor-login tests (local file backend, temp directory): python -m unittest tests.test_store"""
import os
import tempfile
import unittest
from pathlib import Path

from app import store
from app.server import handle


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = store.Local(Path(self.dir.name) / "s.json")
        import app.server as server
        server.DB = self.db
        os.environ["DOCTOR_USER"], os.environ["DOCTOR_PASSWORD"] = "dr", "pw"

    def tearDown(self):
        self.dir.cleanup()

    def test_patient_round_trip_and_search(self):
        _, r = handle("/api/patient", {"name": "Jane Doe", "questionnaire": {"age": "70"}})
        code = r["code"]
        self.assertRegex(code, r"^AD-\d{6}$")
        handle("/api/save", {"code": code, "results": [], "combined": {"text": "x"}, "summary": None, "clock_jpeg": "abc"})
        tok = handle("/api/doctor/login", {"user": "dr", "password": "pw"})[1]["token"]
        self.assertEqual(handle("/api/doctor/search", {"token": tok, "q": "jane"})[1]["patients"][0]["code"], code)
        self.assertEqual(handle("/api/doctor/search", {"token": tok, "q": code.lower()})[1]["patients"][0]["code"], code)
        rec = handle("/api/doctor/record", {"token": tok, "code": code})[1]
        self.assertEqual(rec["patient"]["name"], "Jane Doe"); self.assertEqual(len(rec["screenings"]), 1)

    def test_doctor_routes_need_login(self):
        self.assertEqual(handle("/api/doctor/search", {"q": "a"})[0], 401)
        self.assertEqual(handle("/api/doctor/search", {"token": "fake", "q": "a"})[0], 401)
        self.assertEqual(handle("/api/doctor/login", {"user": "dr", "password": "bad"})[0], 401)

    def test_name_required(self):
        with self.assertRaises(ValueError):
            handle("/api/patient", {"name": " ", "questionnaire": {}})


if __name__ == "__main__":
    unittest.main()
