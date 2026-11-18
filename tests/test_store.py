"""Storage and doctor-login tests (local file backend, temp directory): python -m unittest tests.test_store"""
import os
import tempfile
import unittest
from pathlib import Path

from app import store
from app.server import handle


Q = {"age": "70", "gender": "Female", "ethnicity": "Asian", "education": "Postgraduate", "systolic_bp": "130", "diastolic_bp": "80", "diabetes": "No",
     "high_blood_pressure": "No", "high_cholesterol": "No", "family_history_dementia": "No", "smoking": "Never", "head_injury": "No", "hearing_loss": "No",
     "depression": "No", "physical_activity": "Rarely"}
SIGNUP = {"full_name": "Dr Smith", "user": "drsmith", "password": "password1", "confirm": "password1"}


class Store(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = store.Local(Path(self.dir.name) / "s.json")
        import app.server as server
        server.DB = self.db

    def tearDown(self):
        self.dir.cleanup()

    def test_patient_round_trip_and_search(self):
        _, r = handle("/api/patient", {"name": "Jane Doe", "questionnaire": Q})
        code = r["code"]
        self.assertRegex(code, r"^AD-\d{6}$")
        handle("/api/save", {"code": code, "results": [], "combined": {"text": "x"}, "summary": None, "clock_jpeg": "abc"})
        handle("/api/doctor/signup", SIGNUP)
        tok = handle("/api/doctor/login", {"user": "drsmith", "password": "password1"})[1]["token"]
        self.assertEqual(handle("/api/doctor/search", {"token": tok, "q": "jane"})[1]["patients"][0]["code"], code)
        self.assertEqual(handle("/api/doctor/search", {"token": tok, "q": code.lower()})[1]["patients"][0]["code"], code)
        self.assertEqual(handle("/api/doctor/search", {"token": tok, "q": code[:5]})[1]["patients"][0]["name"], "Jane Doe")   # partial ID
        rec = handle("/api/doctor/record", {"token": tok, "code": code})[1]
        self.assertEqual(rec["patient"]["name"], "Jane Doe"); self.assertEqual(len(rec["screenings"]), 1)

    def test_doctor_routes_need_login(self):
        self.assertEqual(handle("/api/doctor/search", {"q": "a"})[0], 401)
        self.assertEqual(handle("/api/doctor/search", {"token": "fake", "q": "a"})[0], 401)
        self.assertEqual(handle("/api/doctor/login", {"user": "dr", "password": "bad"})[0], 401)

    def test_signup_rules(self):
        self.assertEqual(handle("/api/doctor/signup", SIGNUP)[0], 200)
        self.assertEqual(handle("/api/doctor/signup", {**SIGNUP, "password": "password2", "confirm": "password2"})[0], 400)   # taken
        self.assertEqual(handle("/api/doctor/signup", {**SIGNUP, "user": "x"})[0], 400)
        self.assertEqual(handle("/api/doctor/signup", {**SIGNUP, "user": "drjones", "password": "short", "confirm": "short"})[0], 400)
        self.assertEqual(handle("/api/doctor/signup", {**SIGNUP, "user": "drjones", "confirm": "different1"})[0], 400)       # passwords differ
        self.assertEqual(handle("/api/doctor/signup", {**SIGNUP, "user": "drjones", "full_name": ""})[0], 400)
        self.assertEqual(handle("/api/doctor/login", {"user": "drsmith", "password": "password2"})[0], 401)
        self.assertNotIn("password1", self.db.path.read_text())

    def test_forgot_password_reset(self):
        code = handle("/api/doctor/signup", SIGNUP)[1]["recovery_code"]
        self.assertRegex(code, r"^[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$")
        self.assertNotIn(code.replace("-", ""), self.db.path.read_text())                                                        # only a hash is stored
        bad = handle("/api/doctor/reset", {"user": "drsmith", "recovery_code": "AAAA-AAAA-AAAA", "password": "newpass99", "confirm": "newpass99"})
        self.assertEqual(bad[0], 400)
        ok = handle("/api/doctor/reset", {"user": "drsmith", "recovery_code": code.lower(), "password": "newpass99", "confirm": "newpass99"})
        self.assertEqual(ok[0], 200); new_code = ok[1]["recovery_code"]; self.assertNotEqual(new_code, code)
        self.assertEqual(handle("/api/doctor/login", {"user": "drsmith", "password": "password1"})[0], 401)                       # old password dead
        self.assertEqual(handle("/api/doctor/login", {"user": "drsmith", "password": "newpass99"})[0], 200)
        self.assertEqual(handle("/api/doctor/reset", {"user": "drsmith", "recovery_code": code, "password": "again1234"})[0], 400)   # old code dead

    def test_lockout_after_wrong_attempts(self):
        store._fails.clear(); handle("/api/doctor/signup", SIGNUP)
        for _ in range(5):
            handle("/api/doctor/login", {"user": "drsmith", "password": "wrongpass"})
        self.assertEqual(handle("/api/doctor/login", {"user": "drsmith", "password": "password1"})[0], 429)
        store._fails.clear()

    def test_new_recovery_code_for_logged_in_doctor(self):
        tok = handle("/api/doctor/signup", SIGNUP)[1]["token"]
        self.assertRegex(handle("/api/doctor/new_recovery", {"token": tok})[1]["recovery_code"], r"^\w{4}-\w{4}-\w{4}$")
        self.assertEqual(handle("/api/doctor/new_recovery", {"token": "fake"})[0], 401)

    def test_name_required(self):
        with self.assertRaises(ValueError):
            handle("/api/patient", {"name": " ", "questionnaire": Q})

    def test_every_answer_required(self):
        with self.assertRaises(ValueError) as cm:
            handle("/api/patient", {"name": "Jane", "questionnaire": {"systolic_bp": "120", "diastolic_bp": "80"}})   # only name and BP filled
        self.assertIn("Gender", str(cm.exception)); self.assertIn("Smoking", str(cm.exception))
        with self.assertRaises(ValueError):
            handle("/api/patient", {"name": "Jane", "questionnaire": {**Q, "systolic_bp": "9999"}})
        self.assertEqual(self.db._read()["patients"], [])


if __name__ == "__main__":
    unittest.main()
