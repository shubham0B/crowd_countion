from dataclasses import dataclass, field
from typing import Tuple, List
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_PERSON_WIDTH, MIN_PERSON_HEIGHT, MAX_FRAME_FRACTION

@dataclass
class RejectionStats:
    too_small   : int = 0
    too_large   : int = 0
    too_wide    : int = 0
    too_thin    : int = 0
    low_quality : int = 0

    def total(self):
        return self.too_small + self.too_large + self.too_wide + self.too_thin + self.low_quality

    def to_dict(self):
        return {
            "Too Small"   : self.too_small,
            "Too Large"   : self.too_large,
            "Too Wide"    : self.too_wide,
            "Too Thin"    : self.too_thin,
            "Low Quality" : self.low_quality,
        }

def validate_person(x1, y1, x2, y2, conf, frame_w, frame_h, raw_faces):
    """
    Minimal filtering — only reject obvious noise.
    We TRUST YOLO's confidence threshold. Extra rejection was deleting
    valid front-facing people. Keep this permissive.
    """
    w = x2 - x1
    h = y2 - y1

    # Reject only truly tiny noise (smaller than a real distant person)
    if w < 8 or h < 16:
        return False, "too_small"

    # Reject only a box covering almost the ENTIRE frame (hallucination)
    if w > frame_w * 0.95 and h > frame_h * 0.95:
        return False, "too_large"

    # Reject extreme shapes only (clearly not a person)
    aspect = w / max(1, h)
    if aspect > 2.5:           # very wide = horizontal object
        return False, "too_wide"
    if aspect < 0.08:          # extremely thin = pole/line
        return False, "too_thin"

    return True, "valid"
