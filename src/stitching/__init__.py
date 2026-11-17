"""Local, quality-gated shelf-image stitching."""

from .two_image import StitchConfig, StitchResult, stitch_two_images
from .sequence import stitch_image_sequence

__all__ = ["StitchConfig", "StitchResult", "stitch_image_sequence", "stitch_two_images"]
