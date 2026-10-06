# Footfall Analysis using Computer Vision

A people counting system that detects and tracks individuals in CCTV footage, counting entries and exits in real time.

## Overview
This project analyzes CCTV video to automatically count how many people enter and exit a monitored space. It's designed for use cases like retail footfall analytics, smart building occupancy tracking, and surveillance insights.

## Tech Stack
- Python
- YOLO (Ultralytics) — object detection for identifying people in each frame
- OpenCV — video processing, drawing, and display

## How It Works
1. Each video frame is processed through a YOLO model, filtered to detect only the "person" class.
2. Each detected person is assigned a unique tracking ID that persists across frames.
3. A virtual line is drawn across the frame at the entrance/exit point.
4. As each tracked person crosses the line, their direction of movement determines whether they are counted as an entry (IN) or an exit (OUT).
5. Each person is counted only once, even as they continue to appear in later frames.
6. Live IN/OUT counts are displayed on screen during processing, with a final summary printed at the end.

## Key Concepts Used
- Object detection (YOLO)
- Multi-object tracking (assigning persistent IDs across frames)
- Line-crossing logic for directional counting
- Real-time video processing with OpenCV

## Possible Improvements
- Support for angled/tilted entrance lines instead of a straight horizontal line
- Lighting correction for low-light or shadowed footage
- Multi-camera / multi-zone counting
- Live camera stream support instead of pre-recorded video
- Logging results to a database with a simple analytics dashboard

## Use Case
Bui