"""App logic tests (standard library only): python -m unittest tests.test_app"""
import json
import unittest

from app import motor
from app.combine import combine
from app.schema import ORDER, Result
from app.server import handle, not_run
from app.summary import summarize, validate


def res(m, status):
    return Result(m, status, f"{m} label", {"x": 1.5}, "evidence 0.5", "caveat")


REF = {"tps": {"intercept": 6.3, "slope_per_year": -0.0187, "res_p5": -1.17, "res_p95": 1.2},
       "cv": {"intercept": 0.055, "slope_per_year": 0.0028, "res_p5": -0.135, "res_p95": 0.227},
       "fatigue": {"intercept": 0.7, "slope_per_year": 0.0103, "res_p5": -0.273, "res_p95": 0.201}}


class Combine(unittest.TestCase):
    def test_rule_two_of_four(self):
        r = [res("clock", "flagged"), res("motor", "not_flagged"), res("eye", "flagged"), res("speech", "not_run")]
        c = combine(r)
        self.assertTrue(c["follow_up"]); self.assertEqual(c["flagged"], ["clock", "eye"]); self.assertEqual(c["completed"], ["clock", "motor", "eye"])

    def test_one_flag_no_follow_up(self):
        self.assertFalse(combine([res(m, "flagged" if m == "eye" else "not_flagged") for m in ORDER])["follow_up"])

    def test_no_accuracy_or_score(self):
        c = combine([res(m, "flagged") for m in ORDER])
        self.assertNotIn("score", json.dumps(c).lower().replace("no accuracy", ""))
        self.assertIn("no accuracy claim", c["text"])


class Motor(unittest.TestCase):
    def test_normal_taps_not_flagged(self):
        taps = [0.8 + i * 0.17 for i in range(170)]          # 5.9 taps/s, steady, no slowdown
        r = motor.score([t for t in taps if t <= 30], 70, REF)
        self.assertEqual(r.status, "not_flagged")

    def test_slow_irregular_fading_taps_flagged(self):
        t, taps = 0.5, []
        while t < 30:
            taps.append(t); t += 0.45 + (0.25 if len(taps) % 2 else 0) + t * 0.02    # slow, irregular, slowing
        self.assertEqual(motor.score(taps, 70, REF).status, "flagged")

    def test_too_few_taps_not_run(self):
        self.assertEqual(motor.score([1.0, 2.0, 3.0], 70, REF).status, "not_run")


class Summary(unittest.TestCase):
    def setUp(self):
        self.r = [res(m, "not_flagged") for m in ORDER]
        self.c = combine(self.r)
        self.good = json.dumps({"modalities": {m: f"{m}: evidence 0.5, caveat noted, value 1.5." for m in ORDER}, "overall": self.c["text"]})

    def test_good_reply_accepted(self):
        self.assertEqual(summarize(self.r, post=lambda *a: self.good)["source"], "llama3.1:8b")

    def test_rejects_diagnosis_wording(self):
        bad = json.loads(self.good); bad["overall"] = "The patient is diagnosed with early stage 2 Alzheimer's."
        out = summarize(self.r, post=lambda *a: json.dumps(bad))
        self.assertEqual(out["source"], "template"); self.assertIn("not used", out["note"])

    def test_rejects_new_numbers(self):
        bad = json.loads(self.good); bad["modalities"]["eye"] = "Accuracy was 97.3 percent."
        self.assertEqual(summarize(self.r, post=lambda *a: json.dumps(bad))["source"], "template")

    def test_rejects_bad_json_and_offline(self):
        self.assertEqual(summarize(self.r, post=lambda *a: "not json")["source"], "template")
        def boom(*a): raise OSError("no ollama")
        self.assertEqual(summarize(self.r, post=boom)["source"], "template")

    def test_incomplete_set_uses_template_without_calling_model(self):
        r = [not_run(m) for m in ORDER]
        out = summarize(r, post=lambda *a: self.fail("model must not be called"))
        self.assertEqual(out["source"], "template")

    def test_validate_shape(self):
        with self.assertRaises(ValueError):
            validate(json.dumps({"overall": "x"}), self.r, self.c)


class Routes(unittest.TestCase):
    def test_evidence_lists_four_not_run(self):
        code, d = handle("/api/evidence", {})
        self.assertEqual(code, 200); self.assertEqual(d["order"], list(ORDER))
        self.assertTrue(all(v["status"] == "not_run" for v in d["not_run"].values()))

    def test_unknown_route(self):
        self.assertEqual(handle("/nope", {})[0], 404)


if __name__ == "__main__":
    unittest.main()
