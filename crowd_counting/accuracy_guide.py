"""
============================================================
accuracy_guide.py — Interactive accuracy tuning tool
============================================================
Run this to understand exactly which settings to use
for your specific crowd scenario. It asks questions
and gives you the exact config to paste into the sidebar.

HOW TO RUN:
    python accuracy_guide.py
============================================================
"""

import sys


def separator():
    print("\n" + "─" * 55)


def header(text):
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")


def recommend(crowd_size, lighting, distance, has_gpu):

    separator()
    print("\n  📋  RECOMMENDED SETTINGS FOR YOUR SCENARIO\n")

    # Model recommendation
    if crowd_size < 20:
        model = "Nano  — Fastest  (0–30 people)"
        model_reason = "Small crowd — Nano is fast and accurate enough"
    elif crowd_size < 60:
        model = "Small — Fast     (0–60 people)"
        model_reason = "Medium crowd — Small gives good accuracy"
    elif crowd_size < 150:
        model = "Large — Accurate (100+ people)"
        model_reason = "Large crowd — Large model catches more occluded people"
    else:
        model = "X-Large — Maximum (extreme crowds)"
        model_reason = "Very large crowd — X-Large for maximum detection recall"

    # SAHI recommendation
    use_sahi = distance in ["medium", "far"] or crowd_size > 50
    sahi_reason = (
        "Distant people become tiny — SAHI tiles preserve pixel detail"
        if distance in ["medium", "far"]
        else "Large crowds — SAHI finds people YOLO misses at edges"
        if crowd_size > 50
        else "Close-up crowd — SAHI optional (can skip for speed)"
    )

    # Confidence threshold
    if crowd_size > 100 or distance == "far":
        conf = 0.25
        conf_reason = "Low threshold maximises recall. validate_person() handles false positives."
    elif lighting == "poor":
        conf = 0.30
        conf_reason = "Poor lighting reduces YOLO confidence — lower threshold compensates."
    else:
        conf = 0.35
        conf_reason = "Standard starting point. Adjust down if you see missed people."

    # Resolution
    if has_gpu and crowd_size > 50:
        resolution = "Full HD   1280p (recommended)"
        res_reason = "1280p on GPU gives best accuracy with acceptable speed"
    elif has_gpu:
        resolution = "HD        960p"
        res_reason = "960p sufficient for smaller crowds, faster than 1280p"
    else:
        resolution = "Standard  640p  (fastest)"
        res_reason = "CPU only — 640p is the only practical choice for speed"

    # Face recovery
    use_face = crowd_size > 30 or distance in ["medium", "far"]
    face_reason = (
        "Background people are occluded — face recovery finds their heads"
        if use_face
        else "Small/close crowd — YOLO body detection is sufficient"
    )

    face_sensitivity = 8 if crowd_size > 100 else 7 if crowd_size > 50 else 5

    print(f"  🧠  Model           : {model}")
    print(f"       Reason          : {model_reason}\n")
    print(f"  🔍  Confidence      : {conf}")
    print(f"       Reason          : {conf_reason}\n")
    print(f"  🖼️   Resolution      : {resolution}")
    print(f"       Reason          : {res_reason}\n")
    print(f"  🚀  SAHI            : {'✅ ON' if use_sahi else '❌ OFF'}")
    print(f"       Reason          : {sahi_reason}\n")
    print(f"  👤  Face Recovery   : {'✅ ON' if use_face else '❌ OFF'}")
    print(f"       Reason          : {face_reason}")
    if use_face:
        print(f"       Sensitivity     : {face_sensitivity}/10\n")

    separator()
    print("\n  🛠️   IF YOU'RE STILL UNDER-COUNTING:\n")
    print(f"    1. Lower confidence to {max(0.15, conf - 0.10):.2f}")
    print(f"    2. {'SAHI is already ON — try larger model' if use_sahi else 'Turn SAHI ON'}")
    print(f"    3. Face sensitivity → {min(10, face_sensitivity + 1)}/10")
    print(f"    4. Resolution → {'Ultra HD 1920p (GPU)' if has_gpu else 'need GPU for higher res'}")

    separator()
    print("\n  🛠️   IF YOU'RE OVER-COUNTING (too many false detections):\n")
    print(f"    1. Raise confidence to {min(0.75, conf + 0.15):.2f}")
    print(f"    2. Raise NMS IoU to 0.50 (removes more overlapping boxes)")
    print(f"    3. Turn Face Recovery {'OFF (may be hallucinating)' if use_face else 'already OFF'}")

    separator()
    print("\n  📊  WHAT EACH METRIC TELLS YOU:\n")
    print("    • 'Raw YOLO Detections' >> 'YOLO Count'")
    print("      → validate_person() is filtering a lot")
    print("      → Confidence too LOW or aspect ratio filter too strict")
    print()
    print("    • 'Face Recovered' is very high")
    print("      → YOLO is missing occluded people; this is expected in crowds")
    print("      → Consider larger model or SAHI")
    print()
    print("    • Count fluctuates a lot between similar images")
    print("      → Enable SAHI for consistency")
    print("      → Try yolov8l.pt or yolov8x.pt")
    print()


def main():
    header("CROWD COUNTER — ACCURACY TUNING GUIDE")

    print("""
  This tool helps you pick the RIGHT settings for your image.
  It asks 4 questions, then gives you exact sidebar settings.
    """)

    # Q1
    print("  Q1: How many people do you EXPECT in the image?")
    print("      1 = Under 20")
    print("      2 = 20 to 60")
    print("      3 = 60 to 150")
    print("      4 = 150+")
    q1 = input("  Your answer (1/2/3/4): ").strip()
    crowd_map = {"1": 10, "2": 40, "3": 100, "4": 300}
    crowd_size = crowd_map.get(q1, 40)

    # Q2
    print("\n  Q2: What is the lighting like in the image?")
    print("      1 = Good (outdoor daylight, well-lit indoor)")
    print("      2 = Mixed (some shadows, uneven lighting)")
    print("      3 = Poor (night, heavy shadows, backlight)")
    q2 = input("  Your answer (1/2/3): ").strip()
    lighting_map = {"1": "good", "2": "mixed", "3": "poor"}
    lighting = lighting_map.get(q2, "good")

    # Q3
    print("\n  Q3: How far away are the people from the camera?")
    print("      1 = Close (full body visible, takes up most of frame)")
    print("      2 = Medium (half body visible, background people visible)")
    print("      3 = Far (overhead or wide-angle, people are small)")
    q3 = input("  Your answer (1/2/3): ").strip()
    distance_map = {"1": "close", "2": "medium", "3": "far"}
    distance = distance_map.get(q3, "medium")

    # Q4
    print("\n  Q4: Do you have an NVIDIA GPU?")
    print("      1 = Yes (CUDA GPU)")
    print("      2 = No (CPU only)")
    q4 = input("  Your answer (1/2): ").strip()
    has_gpu = q4 == "1"

    recommend(crowd_size, lighting, distance, has_gpu)

    print("\n  ✅  Open the app:  streamlit run app.py")
    print("  ✅  Apply settings in the left sidebar\n")


if __name__ == "__main__":
    main()
