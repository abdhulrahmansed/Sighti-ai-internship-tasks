import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 1. Load YOLO11 Model
model = YOLO("yolo11s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

video_path = "C:/Sighti/videos/cctv1.mp4"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Video dimensions
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 2. Fast Shadow/Dark Clothing Preprocessing (LUT)
gamma = 1.35
inv_gamma = 1.0 / gamma
gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")

# 3. Tilted Line Definition (w * ratio, h * ratio)
# Defines angled threshold across the entrance glass doors
x1, y1 = int(w * 0.18), int(h * 0.35)
x2, y2 = int(w * 0.39), int(h * 0.25)

def get_side(px, py):
    """Calculates 2D vector cross product to identify side of angled line."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

# Tracking state and counters
count_in = 0
count_out = 0
counted_ids = {}     # {track_id: "IN" or "OUT"}
prev_centroids = {}  # {track_id: (center_x, center_y)}
smoothed_boxes = {}  # EMA box coordinates to kill boundary flicker
ALPHA = 0.70

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Instant gamma shadow boost (no CPU lag)
    enhanced_frame = cv2.LUT(frame, gamma_lut)

    # YOLO11 Tracking with BoT-SORT
    results = model.track(
        source=enhanced_frame,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],             # Person only
        conf=0.25,               # Catches dark clothing/abayas under glare
        iou=0.45,
        imgsz=800,               # Balances high silhouette detail with zero lag
        half=(device == "cuda"), # 16-bit half precision on GPU
        verbose=False
    )

    current_frame_ids = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            current_frame_ids.add(track_id)
            bx1, by1, bx2, by2 = box

            # Smooth box boundary jitter
            if track_id in smoothed_boxes:
                px1, py1, px2, py2 = smoothed_boxes[track_id]
                bx1 = ALPHA * bx1 + (1 - ALPHA) * px1
                by1 = ALPHA * by1 + (1 - ALPHA) * py1
                bx2 = ALPHA * bx2 + (1 - ALPHA) * px2
                by2 = ALPHA * by2 + (1 - ALPHA) * py2

            smoothed_boxes[track_id] = (bx1, by1, bx2, by2)

            # Torso anchor (prevents flapping robe hemline jitter)
            box_h = by2 - by1
            cx = int((bx1 + bx2) / 2)
            cy = int(by1 + box_h * 0.75)

            # Compact Tilted Crossing Logic
            if track_id in prev_centroids:
                prev_x, prev_y = prev_centroids[track_id]
                prev_s = get_side(prev_x, prev_y)
                curr_s = get_side(cx, cy)

                # Keep crossing active only within doorway horizontal width
                if min(x1, x2) - 30 <= cx <= max(x1, x2) + 30:
                    # Crossed into the hall
                    if prev_s < 0 <= curr_s and counted_ids.get(track_id) != "IN":
                        count_in += 1
                        counted_ids[track_id] = "IN"

                    # Crossed out towards the door
                    elif prev_s > 0 >= curr_s and counted_ids.get(track_id) != "OUT":
                        count_out += 1
                        counted_ids[track_id] = "OUT"

            prev_centroids[track_id] = (cx, cy)

            # Draw person box & center anchor
            ix1, iy1, ix2, iy2 = map(int, [bx1, by1, bx2, by2])
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 120), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {track_id}", (ix1, max(18, iy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)

    # Clean inactive tracks
    smoothed_boxes = {k: v for k, v in smoothed_boxes.items() if k in current_frame_ids}
    prev_centroids = {k: v for k, v in prev_centroids.items() if k in current_frame_ids}

    # Draw Tilted Line
    cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
    cv2.putText(frame, "Entrance Line", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # In/Out Dashboard
    cv2.rectangle(frame, (20, 20), (250, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"IN  : {count_in}", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT : {count_out}", (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    cv2.imshow("CCTV IN/OUT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()