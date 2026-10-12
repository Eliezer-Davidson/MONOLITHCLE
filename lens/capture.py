# MONOLITHCLE (package `lens`). External components: THIRD_PARTY_NOTICES.md.
"""Input handling. Phones store rotation in EXIF, and 12MP is wasted on the
VLM/segmenter — normalize both up front."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


@dataclass
class Capture:
    image: np.ndarray
    source: str
    orig_size: tuple[int, int]


def load_capture(photo: str | Path, max_side: int = 2048) -> Capture:
    im = Image.open(photo)
    im = ImageOps.exif_transpose(im)
    # Record the corrected (post-rotation) full-resolution size so it always
    # describes the same orientation as the returned .image.
    orig = im.size

    has_alpha = im.mode in ("RGBA", "LA") or (
        im.mode == "P" and "transparency" in im.info
    )
    if has_alpha:
        im = im.convert("RGBA")
        background = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(background, im)
    im = im.convert("RGB")

    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return Capture(image=np.asarray(im, dtype=np.uint8),
                   source=str(photo), orig_size=orig)
