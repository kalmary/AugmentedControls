from contextlib import contextmanager
import logging
import os
import shutil
import subprocess
import sys
import time

import cv2

from detectors import DetectionFlags, DetectionThresholds, LandmarkDetector


WINDOW_POSITIONS = (
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)

_LAST_TOPMOST_REFRESH: dict[str, float] = {}


def detect_screen_size() -> tuple[int, int] | None:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return None


def configure_camera_window(
    window_title: str,
    frame_width: int | None = None,
    window_position: str = "top-right",
) -> None:
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    window_width = 480
    window_height = 360
    if frame_width:
        window_width = min(frame_width, 480)
        window_height = int(window_width * 0.75)
    cv2.resizeWindow(window_title, window_width, window_height)

    screen_size = detect_screen_size()
    if screen_size:
        screen_width, screen_height = screen_size
        x = 12 if "left" in window_position else max(0, screen_width - window_width - 12)
        y = 12 if "top" in window_position else max(0, screen_height - window_height - 48)
        cv2.moveWindow(window_title, x, y)
    else:
        cv2.moveWindow(window_title, 100000, 0)

    keep_camera_window_topmost(window_title)


def keep_camera_window_topmost(window_title: str) -> None:
    try:
        cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        logging.debug("OpenCV topmost window property is not supported on this platform.")

    now = time.monotonic()
    if now - _LAST_TOPMOST_REFRESH.get(window_title, 0.0) < 1.0:
        return
    _LAST_TOPMOST_REFRESH[window_title] = now

    if not shutil.which("wmctrl"):
        return

    try:
        subprocess.run(
            ["wmctrl", "-r", window_title, "-b", "add,above"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        logging.debug("wmctrl topmost refresh failed.", exc_info=True)


@contextmanager
def suppress_native_stderr(enabled: bool):
    if not enabled:
        yield
        return

    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(devnull_fd)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def run_camera_feed(
    cap: cv2.VideoCapture,
    window_title: str,
    thresholds: DetectionThresholds,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    flags: DetectionFlags,
    window_position: str = "top-right",
) -> int:
    logging.info("Camera started. Press 'q' to quit.")
    if not flags.any_enabled:
        logging.info("No detectors enabled. Add --pose, --hand, or --face to draw landmarks.")
    configure_camera_window(window_title, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), window_position)

    with LandmarkDetector(
        thresholds,
        flags,
        min_detection_confidence,
        min_tracking_confidence,
    ) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                return 1

            frame = cv2.flip(frame, 1)
            frame = detector.process_frame(frame)

            cv2.imshow(window_title, frame)
            keep_camera_window_topmost(window_title)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Quit requested")
                return 0


def cleanup(cap: cv2.VideoCapture) -> None:
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera closed")
