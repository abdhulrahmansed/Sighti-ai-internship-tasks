import os
import re
from collections import Counter

import cv2
import numpy as np
import torch
import easyocr
from openpyxl import Workbook
from ultralytics import YOLO


# ============================================================
# CAMERA SELECTION
# ============================================================

CAMERA_NAME = "wooden"


CAMERA_CONFIG = {

    "janburger": {
        "video": r"C:\Sighti\videos\exit_driv.mp4",
        "output_video": r"C:\Sighti\janburger_output.mp4",
        "output_xlsx": r"C:\Sighti\janburger_summary.xlsx",
    },

    "wooden": {
        "video": r"C:\Sighti\videos\drive_thru6.mp4",
        "output_video": r"C:\Sighti\wooden_output.mp4",
        "output_xlsx": r"C:\Sighti\wooden_summary.xlsx",
    }
}


# ============================================================
# MODELS
# ============================================================

MODEL_PATH = r"C:\Sighti\yolo11s.pt"

PLATE_MODEL_PATH = r"C:\Sighti\license_plate_detector.pt"


# YOLO COCO car class
CAR_CLASS = 2


CONFIDENCE = 0.45
IOU = 0.50

IMAGE_SIZE = 640

TRACKER = "botsort.yaml"

MISSING_FRAME_GRACE = 15


# OCR every 20 frames
# At ~20 FPS = approximately every 1 second
OCR_INTERVAL = 5

OCR_MIN_CONF = 0.25


# ============================================================
# CAMERA ZONES
# Coordinates are based on 1280 x 720 processing frame
# ============================================================

CAMERA_ZONES = {

    "janburger": {

        "entry_zone": np.array([
            [25, 504],
            [1240, 504],
            [1240, 706],
            [25, 706]
        ], dtype=np.int32),

        "service_zone": np.array([
            [60, 240],
            [1040, 240],
            [1040, 720],
            [40, 710]
        ], dtype=np.int32),

        "exit_zone": np.array([
            [25, 204],
            [640, 204],
            [868, 306],
            [25, 306]
        ], dtype=np.int32),
    },


"wooden": {
    "entry_zone": np.array([
        [620, 60],
        [1280, 60],
        [1280, 380],
        [620, 420]
    ], dtype=np.int32),

    "service_zone": np.array([
        [100, 60],
        [870, 60],
        [870, 600],
        [100, 700]
    ], dtype=np.int32),

    "exit_zone": np.array([
        [0, 250],
        [320, 250],
        [320, 700],
        [0, 700]
    ], dtype=np.int32),
}
}


# ============================================================
# FUNCTIONS
# ============================================================

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

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def clean_plate_text(text):
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9\u0600-\u06FF]", "", text).upper()

    # Blacklist known on-screen CCTV watermarks
    if text in ["IPC", "IP", "PC", "CAMERA", "WOODEN", "JANBURGER"]:
        return ""

    # Reject camera date/timestamps (e.g. 2026012..., 20260120, 191521)
    if "2026" in text or len(text) > 8 or (text.isdigit() and len(text) >= 6):
        return ""

    # Ignore isolated single-character false hits (like '؟', '5', 'ق')
    if len(text) < 2:
        return ""

    return text


def preprocess_plate(plate):

    if plate is None or plate.size == 0:
        return []

    # Add a little padding around the detected plate
    h, w = plate.shape[:2]

    pad_x = max(3, int(w * 0.08))
    pad_y = max(3, int(h * 0.15))

    x1 = max(0, -pad_x)
    y1 = max(0, -pad_y)
    x2 = min(w, w + pad_x)
    y2 = min(h, h + pad_y)

    # Actual padding using border
    plate = cv2.copyMakeBorder(
        plate,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REPLICATE
    )

    # Make the plate much larger
    plate = cv2.resize(
        plate,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Sharpen
    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.0
    )

    sharp = cv2.addWeighted(
        enhanced,
        1.6,
        blur,
        -0.6,
        0
    )

    # Threshold version
    threshold = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    return [
        plate,
        enhanced,
        sharp,
        threshold
    ]
def read_plate(reader, plate):

    if plate is None or plate.size == 0:
        return None, 0.0

    processed_images = preprocess_plate(
        plate
    )

    if not processed_images:
        return None, 0.0

    all_results = []

    for processed in processed_images:

        try:

            results = reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.15,
                mag_ratio=1.5
            )

        except Exception as e:

            print(
                "OCR error:",
                e
            )

            continue

        if not results:
            continue

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

            if confidence < OCR_MIN_CONF:
                continue

            all_results.append(
                (
                    text,
                    confidence
                )
            )

    if not all_results:
        return None, 0.0

    # Highest-confidence OCR result
    best_text, best_conf = max(
        all_results,
        key=lambda x: x[1]
    )

    return (
        best_text,
        best_conf
    )

def detect_plate(
    plate_model,
    car_crop,
    device
):

    if car_crop is None:
        return None, None, 0.0

    if car_crop.size == 0:
        return None, None, 0.0

    try:

        results = plate_model.predict(
            source=car_crop,
            imgsz=640,
            conf=0.15,
            device=device,
            verbose=False
        )

    except Exception as e:

        print(
            "Plate detection error:",
            e
        )

        return None, None, 0.0

    result = results[0]

    if (
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        return None, None, 0.0

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    best_box = boxes[
        best_index
    ]

    best_conf = float(
        confidences[
            best_index
        ]
    )

    px1, py1, px2, py2 = map(
        int,
        best_box
    )

    px1 = max(
        0,
        px1
    )

    py1 = max(
        0,
        py1
    )

    px2 = min(
        car_crop.shape[1],
        px2
    )

    py2 = min(
        car_crop.shape[0],
        py2
    )

    if px2 <= px1 or py2 <= py1:
        return None, None, 0.0

    # Add padding around detected plate
    pad_x = int((px2 - px1) * 0.10)
    pad_y = int((py2 - py1) * 0.20)

    crop_x1 = max(0, px1 - pad_x)
    crop_y1 = max(0, py1 - pad_y)

    crop_x2 = min(
        car_crop.shape[1],
        px2 + pad_x
    )

    crop_y2 = min(
        car_crop.shape[0],
        py2 + pad_y
    )

    plate_crop = car_crop[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    return (
        plate_crop,
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        ),
        best_conf
    )


# ============================================================
# MAIN
# ============================================================

def main():

    config = CAMERA_CONFIG[
        CAMERA_NAME
    ]

    PLATE_TEST_DIR = r"C:\Sighti\plate_tests"

    os.makedirs(
        PLATE_TEST_DIR,
        exist_ok=True
    )

    zones = CAMERA_ZONES[
        CAMERA_NAME
    ]

    VIDEO_PATH = config[
        "video"
    ]

    OUTPUT_VIDEO = config[
        "output_video"
    ]


    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():

        DEVICE = "cuda"

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        DEVICE = "cpu"

        print(
            "Using CPU"
        )


    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{VIDEO_PATH}"
        )


    original_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
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
        "Original Resolution:",
        original_width,
        "x",
        original_height
    )

    print(
        "Processing Resolution:",
        "1280 x 720"
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Frames:",
        total_frames
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_VIDEO
        ) or ".",
        exist_ok=True
    )


    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (1280, 720)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not open output video writer."
        )


    # ========================================================
    # CAR MODEL
    # ========================================================

    print(
        "Loading vehicle model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    model.to(
        DEVICE
    )


    # ========================================================
    # LICENSE PLATE MODEL
    # ========================================================

    print(
        "Loading license plate model..."
    )

    if not os.path.exists(
        PLATE_MODEL_PATH
    ):

        raise FileNotFoundError(
            "License plate model not found:\n"
            f"{PLATE_MODEL_PATH}"
        )

    plate_model = YOLO(
        PLATE_MODEL_PATH
    )

    plate_model.to(
        DEVICE
    )


    # ========================================================
    # OCR
    # ========================================================

    print(
        "Loading Arabic + English OCR..."
    )

    reader = easyocr.Reader(
        ["ar", "en"],
        gpu=(DEVICE == "cuda")
    )


    # ========================================================
    # ZONES
    # ========================================================

    entry_zone = zones[
        "entry_zone"
    ]

    service_zone = zones[
        "service_zone"
    ]

    exit_zone = zones[
        "exit_zone"
    ]


    # ========================================================
    # CAR RECORDS
    # ========================================================

    car_records = {}

    total_entered = 0

    total_exited = 0


    # ========================================================
    # EXCEL WORKBOOK
    # ========================================================

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = "Drive-Thru Summary"


    # Excel header
    worksheet.append([
        "Camera",
        "Track ID",
        "License Plate",
        "Entry Time",
        "Service Start",
        "Exit Time",
        "Total Time",
        "Service Time"
    ])


    frame_number = 0


    print()

    print(
        "Processing:",
        CAMERA_NAME
    )


    # ========================================================
    # GUI
    # ========================================================

    cv2.namedWindow(
        "Drive-Thru Analytics",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Drive-Thru Analytics",
        1280,
        720
    )


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    try:

        while True:

            ret, orig_frame = cap.read()

            if not ret:
               break
            # Black-out CCTV watermarks (Bottom-left "IPC" and Top timestamp)
            # Mask bottom-left IPC logo:
            orig_frame[-100:, :180] = 0
            # Mask top timestamp:
            orig_frame[:80, -450:] = 0

            # Keep orig_frame in full resolution for OCR; scale frame for display and tracking
            frame = cv2.resize(
               orig_frame,
               (1280, 720),
               interpolation=cv2.INTER_AREA
            )
            scale_x = orig_frame.shape[1] / 1280.0
            scale_y = orig_frame.shape[0] / 720.0


            frame_number += 1


            current_time = (
                frame_number / fps
            )


            # =================================================
            # VEHICLE DETECTION + TRACKING
            # =================================================

            results = model.track(

                source=frame,

                persist=True,

                tracker=TRACKER,

                classes=[
                    CAR_CLASS
                ],

                conf=CONFIDENCE,

                iou=IOU,

                imgsz=IMAGE_SIZE,

                device=DEVICE,

                verbose=False
            )


            result = results[0]


            # =================================================
            # DRAW ZONES
            # =================================================

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


            # =================================================
            # TRACKED CARS
            # =================================================

            if (
                result.boxes is not None
                and
                result.boxes.id is not None
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


                    # Clamp car coordinates

                    x1 = max(
                        0,
                        min(
                            x1,
                            frame.shape[1] - 1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            frame.shape[0] - 1
                        )
                    )

                    x2 = max(
                        0,
                        min(
                            x2,
                            frame.shape[1]
                        )
                    )

                    y2 = max(
                        0,
                        min(
                            y2,
                            frame.shape[0]
                        )
                    )


                    if x2 <= x1 or y2 <= y1:
                        continue


                    center = get_car_center(
                        box
                    )


                    # =================================================
                    # CREATE RECORD
                    # =================================================

                    if track_id not in car_records:

                        car_records[
                            track_id
                        ] = {

                            "entry_frame":
                                None,

                            "service_start_frame":
                                None,

                            "exit_frame":
                                None,

                            "last_seen_frame":
                                frame_number,

                            "logged":
                                False,

                            "plate_readings":
                                [],

                            "last_ocr_frame":
                                -999999,

                            "plate_box":
                                None
                        }


                    record = car_records[
                        track_id
                    ]


                    record[
                        "last_seen_frame"
                    ] = frame_number


                    # =================================================
                    # ZONE CHECK
                    # =================================================

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


                    # =================================================
                    # ENTRY
                    # =================================================

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


                    # =================================================
                    # SERVICE
                    # =================================================

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


                    # =================================================
                    # EXIT
                    # =================================================

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

                        print(
                            "EXIT:",
                            track_id
                        )


                    # =================================================
                    # LICENSE PLATE DETECTION + OCR
                    # =================================================

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


                        # ---------------------------------------------
                        # Crop detected car
                        # ---------------------------------------------

                        # Crop from the uncompressed original frame
                        ox1 = int(x1 * scale_x)
                        oy1 = int(y1 * scale_y)
                        ox2 = int(x2 * scale_x)
                        oy2 = int(y2 * scale_y)

                        car_crop = orig_frame[
                            oy1:oy2,
                            ox1:ox2
                        ]


                        if (
                            car_crop is not None
                            and
                            car_crop.size > 0
                        ):

                            (
                                plate_crop,
                                plate_relative_box,
                                plate_detection_conf
                            ) = detect_plate(

                                plate_model,

                                car_crop,

                                DEVICE
                            )


                            if (
                                plate_crop is not None
                                and
                                plate_relative_box is not None
                            ):

                                px1, py1, px2, py2 = (
                                    plate_relative_box
                                )


                                # -------------------------------------
                                # Convert plate coordinates to
                                # full-frame coordinates
                                # -------------------------------------

                                full_px1 = x1 + px1
                                full_py1 = y1 + py1
                                full_px2 = x1 + px2
                                full_py2 = y1 + py2


                                record[
                                    "plate_box"
                                ] = (

                                    full_px1,
                                    full_py1,
                                    full_px2,
                                    full_py2
                                )


                                # -------------------------------------
                                # OCR
                                # -------------------------------------

                                plate_text, plate_conf = (
                                    read_plate(
                                        reader,
                                        plate_crop
                                    )
                                )


                                # -------------------------------------
                                # Save plate crop for testing
                                # -------------------------------------

                                plate_test_path = os.path.join(
                                    PLATE_TEST_DIR,
                                    f"car_{track_id}_frame_{frame_number}.jpg"
                                )


                                cv2.imwrite(
                                    plate_test_path,
                                    plate_crop
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


                                    print(
                                        "PLATE:",
                                        track_id,
                                        plate_text,
                                        round(
                                            plate_conf,
                                            2
                                        )
                                    )


                    # =================================================
                    # CURRENT PLATE TEXT
                    # =================================================

                    plate = "UNKNOWN"


                    if record[
                        "plate_readings"
                    ]:

                        values = [

                            item[0]

                            for item in record[
                                "plate_readings"
                            ]
                        ]


                        plate = (
                            Counter(
                                values
                            )
                            .most_common(1)[0][0]
                        )


                    # =================================================
                    # TIME CALCULATIONS
                    # =================================================

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


                    # =================================================
                    # DRAW CAR BOX
                    # =================================================

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # DRAW CAR CENTER
                    # =================================================

                    cv2.circle(
                        frame,
                        center,
                        4,
                        (0, 0, 255),
                        -1
                    )


                    # =================================================
                    # DRAW CAR ID
                    # =================================================

                    cv2.putText(
                        frame,
                        f"CAR {track_id}",
                        (
                            x1,
                            max(
                                20,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # DRAW PLATE BOX
                    # =================================================

                    if (
                        record[
                            "plate_box"
                        ] is not None
                    ):

                        (
                            px1,
                            py1,
                            px2,
                            py2
                        ) = record[
                            "plate_box"
                        ]


                        cv2.rectangle(
                            frame,
                            (px1, py1),
                            (px2, py2),
                            (0, 0, 255),
                            3
                        )


                        cv2.putText(
                            frame,
                            "PLATE",
                            (
                                px1,
                                max(
                                    20,
                                    py1 - 8
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2
                        )


                    # =================================================
                    # DRAW PLATE TEXT
                    # =================================================

                    cv2.putText(
                        frame,
                        f"Plate: {plate}",
                        (
                            x1,
                            min(
                                frame.shape[0] - 60,
                                y2 + 20
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2
                    )


                    # =================================================
                    # SERVICE TIME
                    # =================================================

                    if service_time is not None:

                        cv2.putText(
                            frame,
                            (
                                "Service: "
                                f"{format_seconds(service_time)}"
                            ),
                            (
                                x1,
                                min(
                                    frame.shape[0] - 35,
                                    y2 + 42
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (0, 255, 255),
                            2
                        )


                    # =================================================
                    # TOTAL TIME
                    # =================================================

                    if total_time is not None:

                        cv2.putText(
                            frame,
                            (
                                "Total: "
                                f"{format_seconds(total_time)}"
                            ),
                            (
                                x1,
                                min(
                                    frame.shape[0] - 10,
                                    y2 + 64
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (255, 255, 255),
                            2
                        )


            # =========================================================
            # LOG FINISHED CARS
            # =========================================================

            for finished_track_id, record in car_records.items():

                if record[
                    "logged"
                ]:

                    continue


                if record[
                    "entry_frame"
                ] is None:

                    continue


                missing_frames = (

                    frame_number -

                    record[
                        "last_seen_frame"
                    ]
                )


                if (
                    missing_frames >=
                    MISSING_FRAME_GRACE
                ):

                    record[
                        "logged"
                    ] = True


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

                            item[0]

                            for item in record[
                                "plate_readings"
                            ]
                        ]


                        plate = (
                            Counter(
                                values
                            )
                            .most_common(1)[0][0]
                        )


                    # =================================================
                    # WRITE EXCEL
                    # =================================================

                    worksheet.append([

                        CAMERA_NAME,

                        finished_track_id,

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
                            if service_start_frame is not None
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
                            if service_time is not None
                            else ""
                        )
                    ])


            # =========================================================
            # HUD
            # =========================================================

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


            # =========================================================
            # SAVE 720P OUTPUT
            # =========================================================

            writer.write(
                frame
            )


            # =========================================================
            # LIVE PREVIEW
            # =========================================================

            preview = cv2.resize(
                frame,
                (1280, 720)
            )


            cv2.imshow(
                "Drive-Thru Analytics",
                preview
            )


            key = cv2.waitKey(
                1
            ) & 0xFF


            if key == ord("q"):

                break


    finally:

        cap.release()

        writer.release()

        workbook.save(
            config["output_xlsx"]
        )

        cv2.destroyAllWindows()


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "Camera:",
        CAMERA_NAME
    )

    print(
        "Cars Entered:",
        total_entered
    )

    print(
        "Cars Exited:",
        total_exited
    )

    print(
        "Output Video:",
        OUTPUT_VIDEO
    )

    print(
        "Output Excel:",
        config["output_xlsx"]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()