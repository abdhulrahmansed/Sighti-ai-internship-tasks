import os
import csv
import re
from collections import Counter

import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO


CAMERA_NAME = "janburger"


CAMERA_CONFIG = {
    "janburger": {
        "video": r"C:\Sighti\videos\exit_driv.mp4",
        "output_video": r"C:\Sighti\janburger_output.mp4",
        "output_csv": r"C:\Sighti\janburger_summary.csv",
    },

    "wooden": {
        "video": r"C:\Sighti\videos\drive_thru6.mp4",
        "output_video": r"C:\Sighti\wooden_output.mp4",
        "output_csv": r"C:\Sighti\wooden_summary.csv",
    }
}


MODEL_PATH = r"C:\Sighti\yolo11s.pt"

CAR_CLASS = 2

CONFIDENCE = 0.35
IOU = 0.50
IMAGE_SIZE = 320

TRACKER = "botsort.yaml"

MISSING_FRAME_GRACE = 15

OCR_INTERVAL = 60
OCR_MIN_CONF = 0.35

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789٠١٢٣٤٥٦٧٨٩"


CAMERA_ZONES = {

    "janburger": {

        "entry_zone": np.array([
            [10, 100],
            [300, 100],
            [610, 470],
            [20, 470]
        ], dtype=np.int32),

        "service_zone": np.array([
            [300, 180],
            [450, 180],
            [600, 470],
            [100, 470]
        ], dtype=np.int32),

        "exit_zone": np.array([
            [500, 100],
            [700, 100],
            [700, 470],
            [550, 470]
        ], dtype=np.int32),

        "plate_relative_roi": (
            0.20,
            0.55,
            0.80,
            0.95
        ),
    },


    "wooden": {

        "entry_zone": np.array([
            [650, 70],    # Top-left (near top bollard)
            [1180, 70],   # Top-right (background entrance)
            [1250, 420],  # Bottom-right
            [620, 420]    # Bottom-left (near mid bollards)
        ], dtype=np.int32),

        "service_zone": np.array([
            [60, 140],    # Top-left (along left wall)
            [800, 140],   # Top-right (toward lane divider)
            [680, 710],   # Bottom-right
            [800, 810]     # Bottom-left (foreground exit threshold)
        ], dtype=np.int32),

        "exit_zone": np.array([
            [650, 70],    # Top-left (near top bollard)
            [1180, 70],   # Top-right (background entrance)
            [1250, 420],  # Bottom-right
            [620, 420]    # Bottom-left (near mid bollards)
        ], dtype=np.int32),
        "plate_relative_roi": (
            0.15,
            0.45,
            0.85,
            0.95
        ),
    }
}


def point_inside_polygon(point, polygon):

    x, y = point

    return cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        False
    ) >= 0


def get_car_center(box):

    x1, y1, x2, y2 = box

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return cx, cy


def format_seconds(seconds):

    if seconds is None:
        return "--:--"

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def clean_plate_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


def preprocess_plate(plate):

    if plate is None or plate.size == 0:
        return None

    plate = cv2.resize(
        plate,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def crop_plate_from_car(
    frame,
    box,
    relative_roi
):

    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        box
    )

    x1 = max(
        0,
        min(x1, frame_w - 1)
    )

    x2 = max(
        0,
        min(x2, frame_w - 1)
    )

    y1 = max(
        0,
        min(y1, frame_h - 1)
    )

    y2 = max(
        0,
        min(y2, frame_h - 1)
    )

    car_w = x2 - x1
    car_h = y2 - y1

    if car_w <= 10 or car_h <= 10:
        return None

    rx1, ry1, rx2, ry2 = relative_roi

    px1 = int(
        x1 + car_w * rx1
    )

    py1 = int(
        y1 + car_h * ry1
    )

    px2 = int(
        x1 + car_w * rx2
    )

    py2 = int(
        y1 + car_h * ry2
    )

    px1 = max(
        0,
        min(px1, frame_w - 1)
    )

    px2 = max(
        0,
        min(px2, frame_w)
    )

    py1 = max(
        0,
        min(py1, frame_h - 1)
    )

    py2 = max(
        0,
        min(py2, frame_h)
    )

    if px2 <= px1 or py2 <= py1:
        return None

    return frame[
        py1:py2,
        px1:px2
    ]


def read_plate(reader, plate):

    if plate is None or plate.size == 0:
        return None, 0.0

    processed = preprocess_plate(
        plate
    )

    if processed is None:
        return None, 0.0

    try:

        results = reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist=OCR_ALLOWLIST
        )

    except Exception:

        return None, 0.0

    best_text = None
    best_conf = 0.0

    for result in results:

        if len(result) < 3:
            continue

        text = result[1]

        confidence = float(
            result[2]
        )

        text = clean_plate_text(
            text
        )

        if not text:
            continue

        if confidence > best_conf:

            best_text = text
            best_conf = confidence

    if best_conf < OCR_MIN_CONF:
        return None, best_conf

    return best_text, best_conf


def main():

    config = CAMERA_CONFIG[
        CAMERA_NAME
    ]

    zones = CAMERA_ZONES[
        CAMERA_NAME
    ]

    VIDEO_PATH = config["video"]
    OUTPUT_VIDEO = config["output_video"]
    OUTPUT_CSV = config["output_csv"]

    if torch.cuda.is_available():

        DEVICE = "cuda"

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        DEVICE = "cpu"

        print("Using CPU")

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 25.0

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Frames:",
        total_frames
    )

    os.makedirs(
        os.path.dirname(
            OUTPUT_VIDEO
        ) or ".",
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (1280, 720)
    )

    model = YOLO(
        MODEL_PATH
    )

    model.to(
        DEVICE
    )

    reader = easyocr.Reader(
        ["ar","en"],
        gpu=(DEVICE == "cuda")
    )

    entry_zone = zones[
        "entry_zone"
    ]

    service_zone = zones[
        "service_zone"
    ]

    exit_zone = zones[
        "exit_zone"
    ]

    plate_relative_roi = zones[
        "plate_relative_roi"
    ]

    car_records = {}

    total_entered = 0
    total_exited = 0

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "Camera",
        "Car_ID",
        "Number_Plate",
        "Entry_Time_Sec",
        "Service_Start_Sec",
        "Exit_Time_Sec",
        "Total_Time_Sec",
        "Service_Time_Sec"
    ])

    frame_number = 0

    print(
        "Processing:",
        CAMERA_NAME
    )
    cv2.namedWindow("Drive-Thru Analytics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drive-Thru Analytics", 1280, 720)

    try:

        while True:

            ret, frame = cap.read()
            

            if not ret:
                break
            frame = cv2.resize(frame, (1280, 720))

            frame_number += 1

            current_time = (
                frame_number / fps
            )

            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                classes=[CAR_CLASS],
                conf=CONFIDENCE,
                iou=IOU,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            cv2.polylines(
                frame,
                [entry_zone],
                True,
                (255, 255, 0),
                2
            )

            cv2.polylines(
                frame,
                [service_zone],
                True,
                (0, 255, 255),
                2
            )

            cv2.polylines(
                frame,
                [exit_zone],
                True,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "ENTRY",
                tuple(entry_zone[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SERVICE",
                tuple(service_zone[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "EXIT",
                tuple(exit_zone[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for box, track_id in zip(
                    boxes,
                    ids
                ):

                    track_id = int(
                        track_id
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    center = get_car_center(
                        box
                    )

                    if track_id not in car_records:

                        car_records[
                            track_id
                        ] = {

                            "entry_frame": None,

                            "service_start_frame": None,

                            "exit_frame": None,

                            "last_seen_frame":
                                frame_number,

                            "logged": False,

                            "plate_readings": [],

                            "last_ocr_frame":
                                -999999
                        }

                    record = car_records[
                        track_id
                    ]

                    record[
                        "last_seen_frame"
                    ] = frame_number

                    inside_entry = (
                        point_inside_polygon(
                            center,
                            entry_zone
                        )
                    )

                    inside_service = (
                        point_inside_polygon(
                            center,
                            service_zone
                        )
                    )

                    inside_exit = (
                        point_inside_polygon(
                            center,
                            exit_zone
                        )
                    )

                    if (
                        inside_entry
                        and
                        record[
                            "entry_frame"
                        ] is None
                    ):

                        record[
                            "entry_frame"
                        ] = frame_number

                        total_entered += 1

                        print(
                            "ENTER:",
                            track_id
                        )

                    if (
                        inside_service
                        and
                        record[
                            "service_start_frame"
                        ] is None
                    ):

                        record[
                            "service_start_frame"
                        ] = frame_number

                        print(
                            "SERVICE:",
                            track_id
                        )

                    if (
                        inside_exit
                        and
                        record[
                            "entry_frame"
                        ] is not None
                        and
                        record[
                            "exit_frame"
                        ] is None
                    ):

                        record[
                            "exit_frame"
                        ] = frame_number

                        total_exited += 1

                    if (
                        record[
                            "entry_frame"
                        ] is not None
                        and
                        (
                            frame_number -
                            record[
                                "last_ocr_frame"
                            ]
                        ) >= OCR_INTERVAL
                    ):

                        record[
                            "last_ocr_frame"
                        ] = frame_number

                        plate_crop = (
                            crop_plate_from_car(
                                frame,
                                box,
                                plate_relative_roi
                            )
                        )

                        plate_text, plate_conf = (
                            read_plate(
                                reader,
                                plate_crop
                            )
                        )

                        if plate_text:

                            record[
                                "plate_readings"
                            ].append(
                                (
                                    plate_text,
                                    plate_conf
                                )
                            )

                    plate = "UNKNOWN"

                    if record[
                        "plate_readings"
                    ]:

                        values = [
                            x[0]
                            for x in record[
                                "plate_readings"
                            ]
                        ]

                        plate = (
                            Counter(
                                values
                            )
                            .most_common(1)[0][0]
                        )

                    total_time = None

                    if record[
                        "entry_frame"
                    ] is not None:

                        total_time = (
                            frame_number -
                            record[
                                "entry_frame"
                            ]
                        ) / fps

                    service_time = None

                    if record[
                        "service_start_frame"
                    ] is not None:

                        service_time = (
                            frame_number -
                            record[
                                "service_start_frame"
                            ]
                        ) / fps

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        center,
                        4,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"CAR {track_id}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Plate: {plate}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2
                    )

                    if service_time is not None:

                        cv2.putText(
                            frame,
                            f"Service: {format_seconds(service_time)}",
                            (x1, y2 + 42),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (0, 255, 255),
                            2
                        )

                    if total_time is not None:

                        cv2.putText(
                            frame,
                            f"Total: {format_seconds(total_time)}",
                            (x1, y2 + 64),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (255, 255, 255),
                            2
                        )

            for track_id, record in car_records.items():

                if record["logged"]:
                    continue

                if record["entry_frame"] is None:
                    continue

                missing_frames = (
                    frame_number -
                    record["last_seen_frame"]
                )

                if (
                    missing_frames >=
                    MISSING_FRAME_GRACE
                ):

                    record["logged"] = True

                    entry_frame = record[
                        "entry_frame"
                    ]

                    service_start_frame = record[
                        "service_start_frame"
                    ]

                    exit_frame = record[
                        "exit_frame"
                    ]

                    if exit_frame is None:

                        exit_frame = record[
                            "last_seen_frame"
                        ]

                    entry_time = (
                        entry_frame / fps
                    )

                    exit_time = (
                        exit_frame / fps
                    )

                    total_time = (
                        exit_frame -
                        entry_frame
                    ) / fps

                    service_time = None

                    if service_start_frame is not None:

                        service_time = (
                            exit_frame -
                            service_start_frame
                        ) / fps

                    plate = "UNKNOWN"

                    if record[
                        "plate_readings"
                    ]:

                        values = [
                            x[0]
                            for x in record[
                                "plate_readings"
                            ]
                        ]

                        plate = (
                            Counter(
                                values
                            )
                            .most_common(1)[0][0]
                        )

                    csv_writer.writerow([
                        CAMERA_NAME,
                        track_id,
                        plate,
                        round(
                            entry_time,
                            2
                        ),
                        (
                            round(
                                service_start_frame / fps,
                                2
                            )
                            if service_start_frame
                            is not None
                            else ""
                        ),
                        round(
                            exit_time,
                            2
                        ),
                        round(
                            total_time,
                            2
                        ),
                        (
                            round(
                                service_time,
                                2
                            )
                            if service_time
                            is not None
                            else ""
                        )
                    ])

                    csv_file.flush()

                    print(
                        "FINISHED:",
                        track_id,
                        plate,
                        format_seconds(
                            total_time
                        )
                    )

            cv2.putText(
                frame,
                f"CAMERA: {CAMERA_NAME.upper()}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Cars Entered: {total_entered}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Cars Exited: {total_exited}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Video Time: {format_seconds(current_time)}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            writer.write(
                frame
            )

            

            preview = cv2.resize(frame, (1280, 720))

            cv2.imshow(
                "Drive-Thru Analytics",
                 preview
            )
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        cap.release()

        writer.release()

        csv_file.close()

        cv2.destroyAllWindows()
    print("Processing:", CAMERA_NAME)

    print()
    print("PROCESSING COMPLETE")
    print("Camera:", CAMERA_NAME)
    print("Cars Entered:", total_entered)
    print("Cars Exited:", total_exited)
    print("Output Video:", OUTPUT_VIDEO)
    print("Output CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()