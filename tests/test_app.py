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


class Clock(unittest.TestCase):
    """Needs onnxruntime/opencv/scipy and models/clock_*.onnx (run with the Python 3.13 that has them)."""

    @classmethod
    def setUpClass(cls):
        try:
            import io
            import cv2
            import numpy as np
            from app import clock
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        if not (clock.MODELS / "clock_vit.onnx").exists():
            raise unittest.SkipTest("clock models missing")
        cls.clock, cls.np = clock, np
        img = np.full((900, 700, 3), 235, np.uint8)
        cv2.circle(img, (350, 450), 250, (20, 20, 20), 6)
        cv2.line(img, (350, 450), (350, 300), (20, 20, 20), 6)
        cv2.line(img, (350, 450), (450, 450), (20, 20, 20), 6)
        cls.png = cv2.imencode(".png", img)[1].tobytes()

    def test_cleaning_shape(self):
        a = self.clock.rough_512(self.png)
        self.assertEqual((a.shape, a.dtype), ((512, 512), self.np.uint8))
        self.assertGreater(a.max(), 200)                      # ink is bright

    def test_predict_ranges(self):
        cs, p, thr, crop = self.clock.predict(self.png)
        self.assertTrue(0 <= p <= 1 and -1 < cs < 6 and thr == 0.06 and crop.shape == (512, 512))

    def test_score_result_and_route(self):
        import base64
        r = self.clock.score(self.png)
        self.assertIn(r.status, ("flagged", "not_flagged")); self.assertIn("estimated clock score (0-5)", r.measured)
        code, d = handle("/api/clock", {"image": base64.b64encode(self.png).decode()})
        self.assertEqual((code, d["modality"]), (200, "clock"))
        self.assertEqual(base64.b64decode(d["preview"])[1:4], b"PNG")

    def test_unreadable_picture_is_not_run(self):
        self.assertEqual(self.clock.score(b"not an image").status, "not_run")


class Eye(unittest.TestCase):
    """Webcam eye path. Needs onnxruntime + models/eye_flag.onnx (Python 3.13)."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            from app import eye
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        if not (eye.MODELS / "eye_flag.onnx").exists():
            raise unittest.SkipTest("eye model missing")
        cls.np, cls.eye = np, eye

    def tasks(self, jump=True):
        t = [i * 33.3 for i in range(10)]                       # ~30 Hz, 0.3 s
        x = [400.0] * 4 + ([900.0] * 6 if jump else [400.0] * 6)
        return {k: {"t": t, "x": x, "y": [300.0] * 10} for k in ("antisaccade1", "visual_attention1", "visual_search1")}

    def test_vector_is_30_with_disconj_missing(self):
        vec, kept = self.eye.feature_vector(self.tasks())
        self.assertEqual(len(vec), 30); self.assertEqual(kept["antisaccade1"], 10)
        order = self.eye._session()["order"]
        self.assertTrue(all(self.np.isnan(v) for c, v in zip(order, vec) if "disconj" in c))

    def test_result_is_experimental_and_not_counted(self):
        r = self.eye.score(self.tasks())
        self.assertIn(r.status, ("flagged", "not_flagged")); self.assertFalse(r.counts)
        self.assertIn("features measured (of 30)", r.measured); self.assertIn("Experimental", r.label)

    def test_too_little_gaze_is_not_run(self):
        self.assertEqual(self.eye.score({}).status, "not_run")
        self.assertEqual(self.eye.score({"antisaccade1": {"t": [0, 33], "x": [1, 2], "y": [1, 2]}}).status, "not_run")

    def test_route(self):
        code, d = handle("/api/eye", {"tasks": self.tasks()})
        self.assertEqual((code, d["modality"], d["counts"]), (200, "eye", False))


class EyeNotCounted(unittest.TestCase):
    def test_eye_flag_excluded_from_rule(self):
        eye = Result("eye", "flagged", "x", counts=False)
        c = combine([res("clock", "flagged"), res("motor", "not_flagged"), eye, res("speech", "not_run")])
        self.assertEqual(c["flagged"], ["clock"]); self.assertFalse(c["follow_up"]); self.assertEqual(c["excluded"], ["eye"])
        self.assertIn("not counted", c["text"])

    def test_two_counted_flags_still_trigger(self):
        eye = Result("eye", "flagged", "x", counts=False)
        self.assertTrue(combine([res("clock", "flagged"), res("motor", "flagged"), eye, res("speech", "not_run")])["follow_up"])


class Speech(unittest.TestCase):
    """Pure parts of the speech path (no Whisper, no model download)."""

    def words(self, n, gap=0.1):
        return [("word%d" % i, i * (0.3 + gap), i * (0.3 + gap) + 0.3) for i in range(n)]

    def test_features_from_words(self):
        from speech.whisper_features import features_from_words
        w = [("um", 0.0, 0.3), ("the", 0.4, 0.6), ("the", 0.7, 0.9), ("dog", 2.0, 2.3)]      # 1 filler, 1 repeat, 1 pause >= 0.5 s
        f = features_from_words(w, 10)
        self.assertEqual([round(v, 3) for v in f[:4]], [1.0, 0.0, 1.0, 1.0]); self.assertGreater(f[4], 0)

    def test_too_few_words_is_not_run_and_not_counted(self):
        from app import speech
        r = speech.score_words(self.words(5), 60)
        self.assertEqual((r.status, r.counts), ("not_run", False))

    def test_unreadable_audio_is_not_run(self):
        from app import speech
        from unittest import mock
        with mock.patch.object(speech, "transcribe", side_effect=RuntimeError("bad audio")):
            self.assertEqual(speech.score(b"not audio").status, "not_run")

    def test_model_result_is_experimental(self):
        from app import speech
        if not (speech.MODELS / "speech_whisper.onnx").exists():
            self.skipTest("models/speech_whisper.onnx not exported yet (Colab: notebooks/speech_live_export.ipynb)")
        r = speech.score_words(self.words(80), 60)
        self.assertIn(r.status, ("flagged", "not_flagged")); self.assertFalse(r.counts); self.assertIn("Experimental", r.label)


class SummaryOverall(unittest.TestCase):
    def test_overall_is_always_the_rule_text(self):
        rs = [res(m, "flagged" if m == "clock" else "not_flagged") for m in ORDER]
        reply = json.dumps({"modalities": {m: "ok" for m in ORDER}, "overall": "Everything looks fine."})
        out = summarize(rs, post=lambda url, payload, timeout: reply)
        self.assertEqual(out["source"], "llama3.1:8b"); self.assertIn("no accuracy claim", out["overall"])
        self.assertNotIn("Everything looks fine", out["overall"])


class Status(unittest.TestCase):
    def test_status_shape(self):
        code, d = handle("/api/status", {})
        self.assertEqual(code, 200); self.assertEqual(set(d), {"missing_files", "ollama", "vendor"})
        self.assertEqual(set(d["ollama"]), {"running", "model"})

    def test_page_has_four_cards_in_order(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parent.parent / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index('const ORDER = ["clock", "motor", "eye", "speech"]')
        self.assertGreater(i, 0)
        for fn in ("buildClock", "buildMotor", "buildEye", "buildSpeech"):
            self.assertIn("function " + fn, html)


if __name__ == "__main__":
    unittest.main()
