# ============================================================
# src/heatmap.py — Density Heatmap Generator
# ============================================================
# WHY THIS FILE EXISTS:
# When counting people in videos, drawing 200 bounding boxes
# becomes visually cluttered and computationally heavy.
# A heatmap shows WHERE people are concentrated over time
# by accumulating position data into a color-coded grid.
# Red = many people have been here. Blue = sparse or empty.
#
# HOW IT WORKS:
#   For every frame, each detected person's center point
#   adds a "heat dot" to an accumulator matrix.
#   After N frames, the matrix is normalized and colorized
#   using OpenCV's COLORMAP_JET (blue→green→yellow→red).
#   The heatmap is overlaid on the video frame.
# ============================================================

import cv2
import numpy as np
from typing import List, Tuple


class HeatmapGenerator:
    """
    Maintains a running density accumulator across video frames.

    INTERNAL DATA STRUCTURE:
        self.accumulator — A float32 NumPy array of shape (height, width).
        Each pixel stores a cumulative "heat" value.
        When a person is detected at center (cx, cy), a filled circle
        of radius `dot_radius` is drawn at that point on the accumulator
        with an intensity value of `heat_intensity`.
        Pixels that many people have visited accumulate large values.

    WHY FLOAT32 AND NOT UINT8:
        uint8 can only store values 0–255.
        If 300 people visit the same pixel area, values overflow to 0.
        float32 has no practical upper limit → accurate accumulation.
    """

    def __init__(self, width: int, height: int,
                 dot_radius: int = 30,
                 heat_intensity: float = 3.0,
                 decay: float = 0.995):
        """
        PARAMETERS:
            width, height   — Frame dimensions (same as video resolution)
            dot_radius      — Radius of the heat dot per person (pixels)
                              Larger = smoother heatmap, less precise location
            heat_intensity  — How much heat each detection adds
                              Higher = heatmap saturates faster
            decay           — Multiplier applied to accumulator each frame
                              0.995 = the heatmap slowly fades over time
                              1.0   = no decay (all frames accumulated equally)
                              Use decay < 1.0 for moving crowds (shows recent positions)
                              Use decay = 1.0 for static counting (total occupancy)
        """
        self.width          = width
        self.height         = height
        self.dot_radius     = dot_radius
        self.heat_intensity = heat_intensity
        self.decay          = decay

        # The accumulator starts at all zeros
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self.frame_count = 0

    def update(self, detections: List[Tuple[int, int, int, int]]) -> None:
        """
        Adds heat to the accumulator for each detected person.

        PARAMETERS:
            detections — List of (x1, y1, x2, y2) body bounding boxes

        EXECUTION:
            1. Apply decay to the entire accumulator (fades older heat)
            2. For each detection, compute center point (cx, cy)
            3. Draw a filled circle at (cx, cy) on the accumulator
               cv2.circle() with -1 = filled circle
        """
        # Step 1: Decay fades old heat so recent movement dominates
        self.accumulator *= self.decay

        # Step 2: Add heat for each detected person
        for (x1, y1, x2, y2) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Clamp to frame boundaries (detections can have coords at edge)
            cx = max(self.dot_radius, min(cx, self.width  - self.dot_radius))
            cy = max(self.dot_radius, min(cy, self.height - self.dot_radius))

            cv2.circle(
                self.accumulator,
                (cx, cy),
                self.dot_radius,
                self.heat_intensity,
                -1   # Filled circle
            )

        self.frame_count += 1

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """
        Generates the heatmap and blends it onto the video frame.

        PARAMETERS:
            frame — Current video frame (BGR NumPy array)
            alpha — Heatmap opacity (0.0 = invisible, 1.0 = fully opaque)
                    0.55 = heatmap is visible but underlying frame shows through

        EXECUTION:
            1. Check if there's any heat yet (avoid dividing by zero)
            2. Normalize accumulator to 0–255 uint8
               cv2.normalize(NORM_MINMAX): stretches min→0, max→255
            3. Apply Gaussian blur to smooth the heatmap
               (hard circles look unnatural; blur creates gradients)
            4. Apply COLORMAP_JET:
               0   (cold) = Blue
               128 (warm) = Green → Yellow
               255 (hot)  = Red
            5. Create a mask: only apply heatmap where there is heat
            6. Blend heatmap + original frame using cv2.addWeighted()

        RETURNS:
            A new frame with the heatmap overlaid. Does NOT modify the original.
        """
        if np.max(self.accumulator) < 0.01:
            return frame.copy()

        output = frame.copy()

        # Step 2: Normalize to 0–255
        heatmap = cv2.normalize(
            self.accumulator, None, 0, 255, cv2.NORM_MINMAX
        )
        heatmap = heatmap.astype(np.uint8)

        # Step 3: Gaussian blur for smooth gradients
        blur_radius = max(1, self.dot_radius // 2)
        if blur_radius % 2 == 0:
            blur_radius += 1  # cv2.GaussianBlur kernel must be odd
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)

        # Step 4: Colorize
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Step 5: Mask — only overlay pixels with meaningful heat
        mask = heatmap > 10   # Ignore near-zero background noise

        # Step 6: Blend
        output[mask] = cv2.addWeighted(
            frame[mask], 1.0 - alpha,
            heatmap_color[mask], alpha,
            0
        )

        return output

    def reset(self) -> None:
        """Clears all accumulated heat. Call this to start a fresh session."""
        self.accumulator[:] = 0
        self.frame_count = 0

    def get_peak_density(self) -> float:
        """
        Returns the peak density value in the accumulator.
        Useful for the analytics dashboard to show "hotspot intensity".
        """
        return float(np.max(self.accumulator))
