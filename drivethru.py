import os
import csv
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = r"C:\sighti\videos\carcctv3.mp4"

OUTPUT_VIDEO = r"C:\Sighti\drivethru_output.mp4"
OUTPUT_CSV = r"C:\Sighti\drivethru_summary.csv"

MODEL_PATH = "yolo11s.pt"

# COCO class 2 = car
VEHICLE_CLASSES = [2]

CONFIDENCE = 0.35
IOU = 0.50
IMAGE_SIZE = 640

# Number of consecutive frames a car can disappear
# before we consider it to have exited.
MISSING_FRAME_GRACE = 12

# Minimum service duration required before logging
MIN_SERVICE_TIME = 3.0


# ============================================================
# CHECK INPUT VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    print("=" * 60)
    print("[ERROR] Video file not found:")
    print(VIDEO_PATH)
    print("=" * 60)
    raise SystemExit


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("=" * 60)
print(f"[INFO] Device: {DEVICE.upper()}")

if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA: {torch.version.cuda}")
else:
    print("[WARNING] CUDA not available. Using CPU.")

print("=" * 60)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Could not open video.")
    raise SystemExit


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration_seconds = total_frames / fps


print(f"[INFO] Resolution : {width} x {height}")
print(f"[INFO] FPS        : {fps:.2f}")
print(f"[INFO] Frames     : {total_frames}")
print(f"[INFO] Duration   : {duration_seconds:.2f} sec")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("[INFO] Loading vehicle model...")

model = YOLO(MODEL_PATH)

model.to(DEVICE)

print("[INFO] Vehicle model loaded.")


# ============================================================
# OUTPUT VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    print("[ERROR] Could not create output video.")
    cap.release()
    raise SystemExit

# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Car_ID",
    "Entry_Time_Sec",
    "Exit_Time_Sec",
    "Service_Duration_Sec"
])


# ============================================================
# DRIVE-THRU SERVICE ZONE
# ============================================================
#
# Adjust these points according to your CCTV camera.
#
# Current zone:
# left side -> 2%
# right side -> 70%
# top -> 20%
# bottom -> 98%
#
# You can change these later.
# ============================================================

zone_pts = np.array(
    [
        [30, 200],    # top-left
        [500, 70],   # top-right
        [610, 300],   # bottom-right
        [150, 470]     # bottom-left
    ],
    dtype=np.int32
)


# ============================================================
# CAR RECORDS
# ============================================================
#
# Example:
#
# car_records[7] =
# {
#     "entry_frame": 120,
#     "last_seen_frame": 130,
#     "logged": False
# }
#
# ============================================================

car_records = {}


total_entered = 0
total_exited = 0

frame_number = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def frame_to_seconds(frame_number):
    """
    Convert video frame number to video time.
    """
    return frame_number / fps


def format_time(seconds):
    """
    Convert seconds into HH:MM:SS.mmm
    """

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def point_inside_zone(x, y):
    """
    Check whether a point is inside the drive-thru zone.
    """

    result = cv2.pointPolygonTest(
        zone_pts,
        (int(x), int(y)),
        False
    )

    return result >= 0


# ============================================================
# START PROCESSING
# ============================================================

print()
print("=" * 60)
print("[INFO] Starting processing...")
print("[INFO] Live preview enabled.")
print(f"[INFO] Output video saving disabled.")
print(f"[INFO] Output CSV  : {OUTPUT_CSV}")
print("=" * 60)


while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # CURRENT VIDEO TIME
    # --------------------------------------------------------

    current_video_time = frame_to_seconds(frame_number)

    # --------------------------------------------------------
    # YOLO + BOT-SORT
    # --------------------------------------------------------

    results = model.track(
        source=frame,
        persist=True,
        tracker="botsort.yaml",

        # Car only
        classes=VEHICLE_CLASSES,

        conf=CONFIDENCE,
        iou=IOU,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        half=(DEVICE == "cuda"),

        verbose=False
    )

    result = results[0]

    current_cars_in_zone = set()

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and result.boxes.id is not None
    ):

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, track_id, confidence in zip(
            boxes,
            track_ids,
            confidences
        ):

            bx1, by1, bx2, by2 = map(
                int,
                box
            )

            track_id = int(track_id)

            # ------------------------------------------------
            # CAR CENTER
            # ------------------------------------------------

            center_x = int(
                (bx1 + bx2) / 2
            )

            center_y = int(
                (by1 + by2) / 2
            )

            # ------------------------------------------------
            # CHECK SERVICE ZONE
            # ------------------------------------------------

            inside_zone = point_inside_zone(
                center_x,
                center_y
            )

            # ------------------------------------------------
            # CAR IS INSIDE SERVICE ZONE
            # ------------------------------------------------

            if inside_zone:

                current_cars_in_zone.add(track_id)

                # --------------------------------------------
                # NEW CAR
                # --------------------------------------------

                if track_id not in car_records:

                    car_records[track_id] = {

                        "entry_frame": frame_number,

                        "last_seen_frame": frame_number,

                        "logged": False
                    }

                    total_entered += 1

                    print(
                        f"[ENTER] Car ID {track_id} "
                        f"at {format_time(current_video_time)}"
                    )

                else:

                    car_records[track_id][
                        "last_seen_frame"
                    ] = frame_number

                # --------------------------------------------
                # SERVICE TIME
                # --------------------------------------------

                entry_frame = car_records[
                    track_id
                ]["entry_frame"]

                service_time = (
                    frame_number - entry_frame
                ) / fps

                # --------------------------------------------
                # DRAW CAR BOX
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 255, 0),
                    2
                )

                # --------------------------------------------
                # CAR INFORMATION
                # --------------------------------------------

                label = (
                    f"CAR {track_id} | "
                    f"{service_time:.1f}s"
                )

                cv2.putText(
                    frame,
                    label,
                    (bx1, max(25, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # --------------------------------------------
                # CENTER POINT
                # --------------------------------------------

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1
                )

            else:

                # Car detected outside the service zone
                # but don't immediately treat it as exited.

                if track_id in car_records:

                    car_records[track_id][
                        "last_seen_frame"
                    ] = frame_number

                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"CAR {track_id}",
                    (bx1, max(25, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


    # ========================================================
    # CHECK FOR EXIT
    # ========================================================

    for track_id in list(car_records.keys()):

        record = car_records[track_id]

        if record["logged"]:
            continue

        last_seen_frame = record[
            "last_seen_frame"
        ]

        missing_frames = (
            frame_number - last_seen_frame
        )

        # ----------------------------------------------------
        # Car disappeared for enough frames
        # ----------------------------------------------------

        if missing_frames > MISSING_FRAME_GRACE:

            entry_frame = record[
                "entry_frame"
            ]

            exit_frame = last_seen_frame

            service_duration = (
                exit_frame - entry_frame
            ) / fps

            # ------------------------------------------------
            # LOG ONLY VALID SERVICE TIMES
            # ------------------------------------------------

            if service_duration >= MIN_SERVICE_TIME:

                total_exited += 1

                entry_time = (
                    entry_frame / fps
                )

                exit_time = (
                    exit_frame / fps
                )

                csv_writer.writerow([
                    track_id,
                    f"{entry_time:.3f}",
                    f"{exit_time:.3f}",
                    f"{service_duration:.3f}"
                ])

                csv_file.flush()

                print(
                    f"[EXIT] Car ID {track_id} | "
                    f"Service: {service_duration:.2f}s"
                )

            else:

                print(
                    f"[IGNORE] Car ID {track_id} | "
                    f"Duration too short: "
                    f"{service_duration:.2f}s"
                )

            record["logged"] = True


    # ========================================================
    # DRAW SERVICE ZONE
    # ========================================================

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [zone_pts],
        (0, 255, 255)
    )

    cv2.addWeighted(
        overlay,
        0.12,
        frame,
        0.88,
        0,
        frame
    )

    cv2.polylines(
        frame,
        [zone_pts],
        True,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    active_cars = len(
        current_cars_in_zone
    )

    # Dashboard background

    cv2.rectangle(
        frame,
        (10, 10),
        (320, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "DRIVE-THRU ANALYSIS",
        (30, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars Entered : {total_entered}",
        (30, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars Exited  : {total_exited}",
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars Active  : {active_cars}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        2
    )


    # ========================================================
    # VIDEO TIME
    # ========================================================

    cv2.putText(
        frame,
        f"Timer: {format_time(current_video_time)}",
        (width - 550, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE OUTPUT FRAME
    # ========================================================

    writer.write(frame)
    cv2.imshow("Drive-Thru Live", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 200 == 0:

        percentage = (
            frame_number / total_frames
        ) * 100

        print(
            f"[PROGRESS] "
            f"{frame_number}/{total_frames} "
            f"({percentage:.1f}%) | "
            f"Entered: {total_entered} | "
            f"Exited: {total_exited}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()
csv_file.close()

print()
print("=" * 60)
print("[INFO] Processing completed.")
print("=" * 60)

print(f"[INFO] Total cars entered : {total_entered}")
print(f"[INFO] Total cars exited  : {total_exited}")

print()
print(f"[INFO] Output video:")
print(OUTPUT_VIDEO)

print()
print(f"[INFO] CSV summary:")
print(OUTPUT_CSV)

print("=" * 60)