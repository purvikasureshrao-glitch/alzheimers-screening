"""Clock drawing: photo/scan bytes -> cleaning -> 3-step Crop-Net crop -> ViT ONNX -> Result.

Reuses clock.preprocess / clock.crop (the code that produced the training images) so the app cannot drift from training.
Photo scale [OURS]: NHATS TIFFs are 2560x3312 and were area-averaged 4x (to about 640x828) before cleaning. A photo is resized so its
long side is 828 px, which puts name-box size and crop margins on the same scale. Photo lighting and shadows are not corrected.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from clock import crop as cropmod
from clock import preprocess

from .evidence import CAVEAT, EVIDENCE
from .schema import Result

MODELS = Path(__file__).resolve().parent.parent / "models"
LONG_SIDE = 828
_S = {}


def _sessions():
    if not _S:
        import onnxruntime as ort
        _S["vit"] = ort.InferenceSession(str(MODELS / "clock_vit.onnx"))
        _S["crop"] = ort.InferenceSession(str(MODELS / "clock_cropnet.onnx"))
        _S["thr"] = float(json.loads((MODELS / "clock_features.json").read_text(encoding="utf-8"))["dementia_threshold"])
    return _S


def rough_512(img_bytes):
    """Photo bytes -> uint8 (512,512), ink bright: same steps as preprocess.rough_crop_512 after the 4x reduction."""
    im = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(img_bytes))).convert("L")
    k = LONG_SIDE / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BOX)
    a = preprocess.remove_name_boxes(preprocess.invert(np.asarray(im, np.float32) / 255.0), min_side=10)
    a = preprocess.pad_square(preprocess.crop_to_mass(a, margin=12), 512)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def multi_step_crop(img, crop_sess, iterations=3):
    cur = img
    for _ in range(iterations):
        x = (cur.astype(np.float32) / 255.0)[None, None]
        heat = crop_sess.run(None, {"x": x})[0][0, 0]
        heat = cv2.resize(heat, (512, 512), interpolation=cv2.INTER_LINEAR)
        cur = cropmod._crop_one(cur, heat)
    return cur


def predict(img_bytes):
    """Returns (clock score 0-5, P(dementia), threshold, cropped 512 image)."""
    s = _sessions()
    crop = multi_step_crop(rough_512(img_bytes), s["crop"])
    x = (cropmod.to224(crop).astype(np.float32) / 255.0)[None, None]
    score, p = s["vit"].run(None, {"x": x})[0][0]
    return float(score) * 5, float(p), s["thr"], crop


def score(img_bytes):
    return analyse(img_bytes)[0]


def preview_png(crop):
    """The 224x224 image the ViT saw, as PNG bytes (shown to the user: 'what was captured')."""
    ok, buf = cv2.imencode(".png", cropmod.to224(crop))
    return buf.tobytes()


def analyse(img_bytes):
    """(Result, PNG of the cropped drawing or None)."""
    try:
        cs, p, thr, crop = predict(img_bytes)
    except Exception:
        return Result("clock", "not_run", "Picture could not be read", evidence=EVIDENCE["clock"], caveat=CAVEAT["clock"]), None
    flagged = p >= thr
    return Result(
        "clock", "flagged" if flagged else "not_flagged",
        "Drawing resembles those of people with dementia in the training data" if flagged
        else "Drawing does not resemble those of people with dementia in the training data",
        measured={"estimated clock score (0-5)": round(cs, 1)},
        evidence=EVIDENCE["clock"], caveat=CAVEAT["clock"]), preview_png(crop)
