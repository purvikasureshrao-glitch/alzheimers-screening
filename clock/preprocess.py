"""H3 preprocessing: invert, remove name boxes [OURS], rough crop, pad square.

Images are float32 arrays in [0, 1]. After `invert`, ink is bright on a black background.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None


def load_tif(src):
    """Read a TIFF (path or bytes) as float32 [0, 1], one channel. White paper = 1."""
    im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def invert(img):
    return 1.0 - img


def remove_name_boxes(img, min_side=40, min_fill=0.9, thresh=0.5):
    """Set solid filled rectangles to 0 [OURS: the paper's filter is unpublished].

    A connected ink component is a name box if its bounding box is at least `min_side`
    px on both sides and the component fills at least `min_fill` of that box. Thin
    pen strokes fill far less of their bounding box.
    """
    out = img.copy()
    lab, _ = ndimage.label(img > thresh)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h < min_side or w < min_side:
            continue
        if (lab[sl] == i).sum() / (h * w) >= min_fill:
            out[sl][lab[sl] == i] = 0.0
    return out


def _min_window(profile, mass):
    """Shortest [start, stop) holding at least `mass` of the profile's sum."""
    cs = np.concatenate([[0.0], np.cumsum(profile, dtype=np.float64)])
    target = mass * cs[-1]
    starts = np.arange(len(profile))
    ends = np.searchsorted(cs, cs[:-1] + target - 1e-9, side="left")
    widths = np.where(ends <= len(profile), ends - starts, np.iinfo(np.int64).max)
    s = int(np.argmin(widths))
    return s, int(ends[s])


def crop_to_mass(img, mass=0.96, margin=50):
    """Smallest box holding `mass` of the pixel sum (per axis [OURS]), plus `margin` px."""
    if img.sum() <= 0:
        return img
    r0, r1 = _min_window(img.sum(axis=1), mass)
    c0, c1 = _min_window(img.sum(axis=0), mass)
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(img.shape[0], r1 + margin), min(img.shape[1], c1 + margin)
    return img[r0:r1, c0:c1]


def pad_square(img, size=None):
    """Zero-pad to a centred square; optionally resize to `size` x `size`."""
    h, w = img.shape
    s = max(h, w)
    out = np.zeros((s, s), dtype=img.dtype)
    r, c = (s - h) // 2, (s - w) // 2
    out[r:r + h, c:c + w] = img
    if size and size != s:
        out = np.asarray(Image.fromarray(out).resize((size, size), Image.BILINEAR))
    return out


def rough_crop(img):
    """H3 in one call: invert -> name boxes off -> 96% / +50 px crop."""
    return crop_to_mass(remove_name_boxes(invert(img)))
