import numpy as np
from PIL import Image
from lens.capture import load_capture

def test_downscale_and_rgb(tmp_path):
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(tmp_path / "p.jpg")
    c = load_capture(tmp_path / "p.jpg", max_side=2048)
    assert max(c.image.shape[:2]) == 2048 and c.image.shape[2] == 3
    assert c.image.dtype == np.uint8 and c.orig_size == (4000, 3000)


def test_exif_orientation_orig_size_matches_returned_image(tmp_path):
    """A phone held in portrait stores a landscape pixel buffer plus an EXIF
    orientation tag (5/6/7/8 all swap width/height on correction). orig_size
    must describe the SAME orientation as the corrected .image, not the raw
    on-disk buffer, or downstream code that maps between them gets an
    inverted aspect ratio."""
    path = tmp_path / "portrait.jpg"
    # Raw landscape buffer, as a phone sensor writes it.
    Image.new("RGB", (4032, 3024), (10, 20, 30)).save(path)

    # Re-open, stamp EXIF orientation=6 (90-degree rotation), and re-save —
    # this is the standard way to attach an Exif() object via save(exif=...).
    exif = Image.Exif()
    exif[274] = 6  # 274 = Orientation tag
    Image.open(path).save(path, exif=exif)

    # Verify the tag actually round-trips before relying on it.
    reopened = Image.open(path)
    assert reopened.getexif().get(274) == 6, "orientation tag did not round-trip"

    c = load_capture(path, max_side=2048)

    img_h, img_w = c.image.shape[:2]
    orig_w, orig_h = c.orig_size

    # The corrected image is portrait (w < h); orig_size must agree.
    assert img_w < img_h
    assert orig_w < orig_h, (
        f"orig_size={c.orig_size} is landscape but image.shape={c.image.shape} "
        "is portrait -- orig_size was recorded before EXIF transposition"
    )
    # orig_size must be the full-resolution, EXIF-corrected size.
    assert c.orig_size == (3024, 4032)


def test_rgba_fully_transparent_composites_onto_white(tmp_path):
    """A fully transparent pixel must not leak the color hidden underneath
    when flattened to RGB -- it should composite onto opaque white."""
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (64, 64), (0, 255, 0, 0)).save(path)  # transparent green

    c = load_capture(path, max_side=2048)

    assert c.image.shape == (64, 64, 3)
    assert c.image.dtype == np.uint8
    assert tuple(int(v) for v in c.image[32, 32]) == (255, 255, 255)
