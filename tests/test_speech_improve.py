"""Pure-function tests for speech/improve_features.py (no audio, no models): python -m unittest tests.test_speech_improve"""
import unittest
from pathlib import Path

from speech import improve_features as imp

CHA = "*INV:\ttell me . 0_2360\n%mor:\tv|tell\n*PAR:\twell there's a mother &uh\n\tan(d) the sink . 4266_13310\n%mor:\tx\n*PAR:\tand the window . 13310_20608\n"


class Improve(unittest.TestCase):
    def test_par_spans_skip_investigator_and_handle_wrapped_lines(self):
        self.assertEqual(imp.par_spans(CHA), [(4.266, 13.31), (13.31, 20.608)])

    def test_keep_in_spans(self):
        w = [("tell", 0.5, 0.9), ("well", 5.0, 5.3), ("window", 14.0, 14.4), ("later", 30.0, 30.4)]
        self.assertEqual([x[0] for x in imp.keep_in_spans(w, imp.par_spans(CHA))], ["well", "window"])

    def test_features_ext_shape_and_info_units(self):
        w = [("um", 0.0, 0.3), ("the", 0.4, 0.6), ("mother", 0.7, 1.1), ("is", 1.2, 1.3), ("at", 1.4, 1.5), ("the", 1.6, 1.7), ("sink", 3.0, 3.4), ("water", 3.5, 3.9)]
        f = imp.features_ext(w, 10)
        self.assertEqual(len(f), len(imp.FEXT_NAMES))
        self.assertEqual(f[imp.FEXT_NAMES.index("info_units")], 3.0)       # mother, sink, water
        self.assertEqual(f[imp.FEXT_NAMES.index("long_pause")], 1.0)        # the 1.3 s gap before "sink"
        self.assertEqual(f[imp.FEXT_NAMES.index("content_words")], 7.0)

    def test_empty_words_do_not_crash(self):
        self.assertEqual(len(imp.features_ext([], 10)), len(imp.FEXT_NAMES))

    def test_real_transcript_has_par_spans(self):
        p = Path("datasets/ADReSS-IS2020-train/ADReSS-IS2020-data/train/transcription/cc/S001.cha")
        if not p.is_file():
            self.skipTest("ADReSS not on this machine")
        sp = imp.par_spans(p.read_text(encoding="utf-8", errors="replace"))
        self.assertGreater(len(sp), 3); self.assertGreater(imp.spans_seconds(sp), 20)


if __name__ == "__main__":
    unittest.main()
