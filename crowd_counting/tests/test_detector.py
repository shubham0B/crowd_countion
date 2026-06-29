# ============================================================
# tests/test_detector.py — Unit Tests for Every Module
# ============================================================
# HOW TO RUN:
#   cd crowd_counting
#   python -m pytest tests/ -v
#
# WHAT EACH TEST COVERS:
#   TestValidator     — validate_person() geometric filter
#   TestNMSFaces      — nms_faces() deduplication math
#   TestCLAHE         — Contrast enhancement on real pixels
#   TestSpatialMerge  — Face vs body deduplication logic
#   TestHeatmap       — Accumulator update and overlay
#   TestUtils         — Color conversion helpers
# ============================================================

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2

from src.validator   import validate_person, RejectionStats
from src.face_recovery import nms_faces
from src.utils       import apply_clahe, pil_to_bgr, bgr_to_rgb
from src.heatmap     import HeatmapGenerator
from src.detector    import _find_orphan_faces, _split_group_boxes


# ─────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """800×600 black frame — simulates a blank image."""
    return np.zeros((600, 800, 3), dtype=np.uint8)

@pytest.fixture
def person_frame():
    """800×600 frame with a simple white rectangle simulating a person."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 100), (280, 400), (255, 255, 255), -1)
    return frame

@pytest.fixture
def no_faces():
    """Empty face list for tests that don't need face validation."""
    return []


# ─────────────────────────────────────────
# TEST: validate_person() — Geometric Filter
# ─────────────────────────────────────────

class TestValidator:
    """
    Tests all 5 checks inside validate_person().
    Frame is 800×600. All coordinates are in pixels.
    """

    def test_valid_standing_person(self, no_faces):
        """
        A normal standing person box (80px wide, 200px tall).
        aspect_ratio = 80/200 = 0.40 — well within [0.15, 1.30].
        Should PASS all checks.
        """
        is_valid, reason = validate_person(
            100, 100, 180, 300,   # 80×200 box
            conf=0.85,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is True, f"Expected valid, got rejected: {reason}"

    def test_reject_too_small_width(self, no_faces):
        """
        Box is only 10px wide (less than MIN_PERSON_WIDTH=15).
        Check 1 should reject it as "too_small".
        """
        is_valid, reason = validate_person(
            100, 100, 110, 200,   # 10×100 — too narrow
            conf=0.9,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is False
        assert reason == "too_small"

    def test_reject_too_small_height(self, no_faces):
        """Box is only 20px tall (less than MIN_PERSON_HEIGHT=30)."""
        is_valid, reason = validate_person(
            100, 100, 160, 120,   # 60×20 — too short
            conf=0.9,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is False
        assert reason == "too_small"

    def test_reject_too_large(self, no_faces):
        """
        Box covers 80% of the 800px frame width (> MAX_FRAME_FRACTION=0.70).
        Should be rejected as a hallucination ("too_large").
        """
        is_valid, reason = validate_person(
            0, 0, 660, 500,   # 660/800 = 82.5% of frame width
            conf=0.9,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is False
        assert reason == "too_large"

    def test_reject_too_wide(self, no_faces):
        """
        Box is 400px wide, 100px tall.
        aspect_ratio = 400/100 = 4.0 > MAX_ASPECT_RATIO (1.30).
        Represents an outstretched arm or horizontal object.
        """
        is_valid, reason = validate_person(
            50, 200, 450, 300,   # 400×100 — much wider than tall
            conf=0.75,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is False
        assert reason == "too_wide"

    def test_reject_too_thin(self, no_faces):
        """
        Box is 5px wide, 300px tall.
        aspect_ratio = 5/300 = 0.017 < MIN_ASPECT_RATIO (0.15).
        Represents a lamp post or door frame edge.
        """
        is_valid, reason = validate_person(
            100, 50, 105, 350,   # 5×300 — vertical line
            conf=0.80,
            frame_w=800, frame_h=600,
            raw_faces=no_faces
        )
        assert is_valid is False
        assert reason == "too_thin"

    def test_low_conf_with_face_passes(self):
        """
        Low confidence (0.25 < 0.35 threshold) BUT a face exists
        in the upper 50% of the box.
        The head validation should RESCUE this detection.

        Body box: (100, 100) to (200, 400) → h=300
        Upper 50%: y=100 to y=250
        Face at (140, 160) → center (150, 175) → inside upper 50%
        Expected: is_valid = True
        """
        face = (130, 140, 40, 40)   # (x1, y1, width, height) → center=(150, 160)
        is_valid, reason = validate_person(
            100, 100, 200, 400,
            conf=0.25,   # Below LOW_CONF_THRESHOLD
            frame_w=800, frame_h=600,
            raw_faces=[face]
        )
        assert is_valid is True, f"Should be rescued by face, got: {reason}"

    def test_low_conf_without_face_rejected(self):
        """
        Low confidence + NO face in upper region.
        Should be rejected as "low_quality".
        """
        is_valid, reason = validate_person(
            100, 100, 200, 400,
            conf=0.20,
            frame_w=800, frame_h=600,
            raw_faces=[]   # No faces at all
        )
        assert is_valid is False
        assert reason == "low_quality"

    def test_low_conf_face_in_lower_half_rejected(self):
        """
        Low confidence. Face exists BUT its center is in the LOWER 50%
        of the body box. A head can't be at hip level — likely a different person.
        Should still be rejected.

        Body box: y=100 to y=400 → upper 50% boundary = y=250
        Face center at y=310 → in lower half → no rescue
        """
        face = (130, 280, 40, 40)   # center_y = 300 → below y=250
        is_valid, reason = validate_person(
            100, 100, 200, 400,
            conf=0.20,
            frame_w=800, frame_h=600,
            raw_faces=[face]
        )
        assert is_valid is False
        assert reason == "low_quality"


# ─────────────────────────────────────────
# TEST: nms_faces() — Face Deduplication
# ─────────────────────────────────────────

class TestNMSFaces:
    """
    Tests the custom NMS algorithm for face boxes.
    Format: (x1, y1, width, height)
    """

    def test_empty_input(self):
        """No faces → empty list returned without error."""
        assert nms_faces([]) == []

    def test_single_face_kept(self):
        """Only one face → must be returned unchanged."""
        face = (100, 100, 50, 50)
        result = nms_faces([face])
        assert result == [face]

    def test_non_overlapping_both_kept(self):
        """
        Two faces far apart (no overlap → IoU=0).
        Both should be kept.
        """
        face1 = (0,   0,  50, 50)
        face2 = (400, 0,  50, 50)
        result = nms_faces([face1, face2])
        assert len(result) == 2

    def test_identical_boxes_deduped(self):
        """
        Two identical boxes (IoU=1.0 > threshold=0.30).
        Only one should survive.
        """
        face = (100, 100, 60, 60)
        result = nms_faces([face, face])
        assert len(result) == 1

    def test_high_overlap_smaller_removed(self):
        """
        Large face (80×80) overlaps significantly with small face (30×30).
        Small face should be removed; large face kept.
        IoU will be high because the small box is mostly inside the large one.
        """
        large = (100, 100, 80, 80)   # Area = 6400
        small = (110, 110, 30, 30)   # Area = 900 — mostly inside 'large'
        result = nms_faces([large, small])
        assert len(result) == 1
        assert result[0] == large   # Largest area survives

    def test_slight_overlap_both_kept(self):
        """
        Two faces that barely touch at their edges (IoU well below 0.30).
        Both should survive NMS.
        """
        face1 = (0,   0,  60, 60)
        face2 = (55,  0,  60, 60)   # 5px overlap on right edge of face1
        result = nms_faces([face1, face2])
        # IoU = 5*60 / (3600 + 3600 - 300) ≈ 0.043 < 0.30 → both kept
        assert len(result) == 2


# ─────────────────────────────────────────
# TEST: CLAHE Contrast Enhancement
# ─────────────────────────────────────────

class TestCLAHE:

    def test_output_same_shape(self, blank_frame):
        """CLAHE must return an image with identical dimensions."""
        result = apply_clahe(blank_frame)
        assert result.shape == blank_frame.shape

    def test_output_is_uint8(self, person_frame):
        """Output must be uint8 (0–255 range) for OpenCV compatibility."""
        result = apply_clahe(person_frame)
        assert result.dtype == np.uint8

    def test_dark_image_gets_brighter(self):
        """
        A very dark frame (mean pixel = 5) should get brighter after CLAHE.
        CLAHE redistributes the histogram, increasing mean brightness in dark images.
        """
        dark = np.full((100, 100, 3), 5, dtype=np.uint8)
        result = apply_clahe(dark)
        assert result.mean() >= dark.mean()

    def test_clahe_does_not_invert_colors(self, person_frame):
        """
        After CLAHE, bright areas should still be brighter than dark areas.
        The white rectangle (value≈255) should remain lighter than the background (value=0).
        """
        result  = apply_clahe(person_frame)
        rect_area = result[150:350, 210:270]      # Inside the white rectangle
        bg_area   = result[400:500, 400:600]      # Dark background area
        assert rect_area.mean() > bg_area.mean()


# ─────────────────────────────────────────
# TEST: Color Conversion Utilities
# ─────────────────────────────────────────

class TestUtils:

    def test_pil_to_bgr_rgb_image(self):
        """PIL RGB image must be converted to BGR NumPy array."""
        from PIL import Image as PILImage
        # Create a red PIL image (R=255, G=0, B=0)
        pil_img = PILImage.fromarray(
            np.array([[[255, 0, 0]]], dtype=np.uint8)
        )
        bgr = pil_to_bgr(pil_img)
        # In BGR format, red pixel should be [0, 0, 255]
        assert bgr[0, 0, 0] == 0    # B
        assert bgr[0, 0, 1] == 0    # G
        assert bgr[0, 0, 2] == 255  # R

    def test_bgr_to_rgb_round_trip(self, person_frame):
        """Converting BGR → RGB → BGR should return identical values."""
        rgb = bgr_to_rgb(person_frame)
        bgr_again = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        assert np.array_equal(person_frame, bgr_again)

    def test_pil_handles_rgba(self):
        """RGBA PNG images (4 channels) should be converted to BGR (3 channels)."""
        from PIL import Image as PILImage
        rgba_array = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba_array[:, :, 3] = 200   # Set alpha channel
        pil_img = PILImage.fromarray(rgba_array, 'RGBA')
        result = pil_to_bgr(pil_img)
        assert result.shape[2] == 3, "RGBA should be converted to 3-channel BGR"


# ─────────────────────────────────────────
# TEST: Heatmap Generator
# ─────────────────────────────────────────

class TestHeatmap:

    def test_accumulator_starts_zero(self):
        """Freshly created heatmap should have all-zero accumulator."""
        hm = HeatmapGenerator(640, 480)
        assert np.all(hm.accumulator == 0.0)

    def test_update_adds_heat(self):
        """After adding one detection, accumulator max should be > 0."""
        hm = HeatmapGenerator(640, 480)
        hm.update([(100, 100, 200, 300)])  # One person
        assert np.max(hm.accumulator) > 0

    def test_multiple_updates_accumulate(self):
        """10 updates at the same location should have higher heat than 1 update."""
        hm1 = HeatmapGenerator(640, 480)
        hm1.update([(100, 100, 200, 300)])
        heat_after_1 = np.max(hm1.accumulator)

        hm10 = HeatmapGenerator(640, 480, decay=1.0)   # No decay
        for _ in range(10):
            hm10.update([(100, 100, 200, 300)])
        heat_after_10 = np.max(hm10.accumulator)

        assert heat_after_10 > heat_after_1

    def test_reset_clears_accumulator(self):
        """After reset(), all values should return to zero."""
        hm = HeatmapGenerator(640, 480)
        hm.update([(100, 100, 200, 300)])
        hm.reset()
        assert np.all(hm.accumulator == 0.0)

    def test_overlay_returns_frame_copy(self):
        """
        get_overlay() should return a modified copy, not modify the original.
        The original frame must remain unchanged.
        """
        hm    = HeatmapGenerator(640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hm.update([(100, 100, 200, 300)])
        overlay = hm.get_overlay(frame)

        # Original frame must be unchanged
        assert np.all(frame == 0), "get_overlay() must not modify the original frame"
        # Overlay should be different (heatmap was applied)
        assert not np.all(overlay == frame)

    def test_no_heat_returns_copy(self):
        """If no detections yet, get_overlay() returns a copy of the frame."""
        hm    = HeatmapGenerator(640, 480)
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = hm.get_overlay(frame)
        assert np.array_equal(result, frame)


# ─────────────────────────────────────────
# TEST: Spatial Merge Logic
# ─────────────────────────────────────────

class TestSpatialMerge:

    def test_face_inside_body_is_not_orphan(self):
        """
        Face center (150, 150) is inside body box (100,100)→(200,300).
        This face should NOT appear in orphan_faces (YOLO already counted this person).
        """
        bodies = [((100, 100, 200, 300), 0.85)]   # body box + confidence
        faces  = [(130, 130, 40, 40)]              # face at (130,130) w=40 h=40 → center=(150,150)
        orphans = _find_orphan_faces(faces, bodies)
        assert len(orphans) == 0, "Face inside body box should not be an orphan"

    def test_face_outside_body_is_orphan(self):
        """
        Face center (500, 200) is completely outside body box (100,100)→(200,300).
        This face represents a person YOLO missed → should be an orphan.
        """
        bodies = [((100, 100, 200, 300), 0.85)]
        faces  = [(480, 180, 40, 40)]   # center=(500, 200) — far from the body box
        orphans = _find_orphan_faces(faces, bodies)
        assert len(orphans) == 1, "Face outside all body boxes should be an orphan"

    def test_face_in_lower_half_is_orphan(self):
        """
        Face center is inside the LOWER 40% of the body box.
        Anatomically, a head can't be at knee level → treated as orphan.

        Body: (100,100)→(200,400) → h=300 → upper 60% boundary = y=280
        Face center at y=330 → in lower 40% → orphan
        """
        bodies = [((100, 100, 200, 400), 0.85)]
        faces  = [(130, 310, 40, 40)]   # center_y = 330 → below y=280
        orphans = _find_orphan_faces(faces, bodies)
        assert len(orphans) == 1, "Face in lower half of body should still be orphan"

    def test_no_faces_no_orphans(self):
        """Empty face list → no orphans."""
        bodies = [((100, 100, 200, 300), 0.85)]
        orphans = _find_orphan_faces([], bodies)
        assert orphans == []

    def test_no_bodies_all_orphans(self):
        """
        If YOLO detected nothing but faces exist,
        ALL faces are orphans (YOLO missed everyone).
        """
        bodies = []
        faces  = [(100, 100, 40, 40), (300, 200, 40, 40)]
        orphans = _find_orphan_faces(faces, bodies)
        assert len(orphans) == 2


# ─────────────────────────────────────────
# TEST: RejectionStats
# ─────────────────────────────────────────

class TestRejectionStats:

    def test_initial_all_zeros(self):
        s = RejectionStats()
        assert s.total() == 0

    def test_total_sums_correctly(self):
        s = RejectionStats(too_small=3, too_wide=2, low_quality=1)
        assert s.total() == 6

    def test_to_dict_has_all_keys(self):
        s = RejectionStats()
        d = s.to_dict()
        assert len(d) == 5
        assert all(v == 0 for v in d.values())
