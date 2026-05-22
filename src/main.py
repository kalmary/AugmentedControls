import argparse
import logging
import os
import sys
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

from camera import cleanup, open_camera, run_camera_feed, suppress_native_stderr
from config import ConfigError, config_value, load_config
from detectors import DetectionConfidences, DetectionFlags, DetectionThresholds


CONTROL_MODES = (
    "viewer",
    "hand-control",
    "hand-control-precise",
    "eye-control",
    "steering-wheel",
)

MODE_CONFIGS = {
    "viewer": "viewer",
    "hand-control": "hand_control",
    "hand-control-precise": "hand_control_precise",
    "eye-control": "eye_control",
    "steering-wheel": "steering_wheel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the default camera and detect close pose/hand skeletons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=CONTROL_MODES,
        default="viewer",
        help="Application mode",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Enable pose skeleton detection and drawing",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Enable hand skeleton detection and drawing",
    )
    parser.add_argument(
        "--face",
        action="store_true",
        help="Enable face mesh detection and drawing",
    )
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def load_detector_config(name: str, enabled: bool) -> dict:
    if not enabled:
        return {}
    return load_config(name)


def confidence_value(config: dict, key: str) -> float:
    return float(config_value(config, key, 0.5))


def main() -> int:
    args = parse_args()
    flags = DetectionFlags(
        pose=args.pose,
        hand=args.hand,
        face=args.face,
    )
    try:
        pose_config = load_detector_config("pose", flags.pose or args.mode == "hand-control")
        hand_config = load_detector_config("hand", flags.hand)
        face_config = load_detector_config("face", flags.face)
        mode_config = load_config(MODE_CONFIGS[args.mode])
    except ConfigError as error:
        logging.error(error)
        return 2

    configure_logging(bool(config_value(mode_config, "verbose", False)))
    thresholds = DetectionThresholds(
        pose_area=float(config_value(pose_config, "close_area", 0.1)),
        hand_area=float(config_value(hand_config, "close_area", 0.01)),
        face_area=float(config_value(face_config, "close_area", 0.025)),
    )
    confidences = DetectionConfidences(
        pose_detection=confidence_value(pose_config, "min_detection_confidence"),
        pose_tracking=confidence_value(pose_config, "min_tracking_confidence"),
        hand_detection=confidence_value(hand_config, "min_detection_confidence"),
        hand_tracking=confidence_value(hand_config, "min_tracking_confidence"),
        face_detection=confidence_value(face_config, "min_detection_confidence"),
        face_tracking=confidence_value(face_config, "min_tracking_confidence"),
    )
    if args.mode not in {"viewer", "hand-control"}:
        logging.error("%s mode is not implemented yet.", args.mode)
        return 2

    try:
        cap = open_camera(
            int(config_value(mode_config, "camera_index", 0)),
            int(config_value(mode_config, "width", 640)),
            int(config_value(mode_config, "height", 480)),
        )
    except RuntimeError as error:
        logging.error(error)
        return 1

    try:
        with suppress_native_stderr(not bool(config_value(mode_config, "native_logs", False))):
            if args.mode == "viewer":
                return_code = run_camera_feed(
                    cap,
                    str(config_value(mode_config, "window_title", "Camera Feed")),
                    thresholds,
                    confidences,
                    flags,
                    str(config_value(mode_config, "window_position", "top-right")),
                )
            elif args.mode == "hand-control":
                from control_modes import HandControlConfig, HandControlMode

                return_code = HandControlMode(
                    HandControlConfig(
                        min_detection_confidence=confidence_value(pose_config, "min_detection_confidence"),
                        min_tracking_confidence=confidence_value(pose_config, "min_tracking_confidence"),
                        click_hold_seconds=float(config_value(mode_config, "click_hold_seconds", 1.5)),
                        click_radius=float(config_value(mode_config, "click_radius", 0.025)),
                        click_grace_seconds=float(config_value(mode_config, "click_grace_seconds", 0.25)),
                        click_cooldown_seconds=float(config_value(mode_config, "click_cooldown_seconds", 3.0)),
                        wave_window_seconds=float(config_value(mode_config, "wave_window_seconds", 1.2)),
                        wave_min_span=float(config_value(mode_config, "wave_min_span", 0.12)),
                        wave_min_direction_changes=int(config_value(mode_config, "wave_min_direction_changes", 4)),
                        mouse_smoothing=float(config_value(mode_config, "mouse_smoothing", 0.4)),
                        mouse_deadzone=float(config_value(mode_config, "mouse_deadzone", 0.002)),
                        mouse_edge_padding=float(config_value(mode_config, "mouse_edge_padding", 0.005)),
                        mouse_edge_padding_pixels=(
                            int(mode_config["mouse_edge_padding_pixels"])
                            if "mouse_edge_padding_pixels" in mode_config
                            else None
                        ),
                        dwell_cursor_enabled=bool(config_value(mode_config, "dwell_cursor_enabled", True)),
                        dwell_cursor_radius=(
                            int(mode_config["dwell_cursor_radius"])
                            if mode_config.get("dwell_cursor_radius") is not None
                            else None
                        ),
                        dwell_cursor_diameter_pixels=int(
                            config_value(mode_config, "dwell_cursor_diameter_pixels", 28)
                        ),
                        dwell_cursor_base_alpha=float(config_value(mode_config, "dwell_cursor_base_alpha", 0.4)),
                        dwell_cursor_fill_alpha=float(config_value(mode_config, "dwell_cursor_fill_alpha", 0.9)),
                        dwell_cursor_workspace_alpha=float(
                            config_value(mode_config, "dwell_cursor_workspace_alpha", 0.1)
                        ),
                        control_margin=float(config_value(mode_config, "control_margin", 0.08)),
                        control_gain=float(config_value(mode_config, "control_gain", 1.25)),
                        control_acceleration=float(config_value(mode_config, "control_acceleration", 1.0)),
                        control_acceleration_threshold=float(
                            config_value(mode_config, "control_acceleration_threshold", 0.16)
                        ),
                        window_position=str(config_value(mode_config, "window_position", "top-right")),
                        exit_hold_seconds=float(config_value(mode_config, "exit_hold_seconds", 0.6)),
                    )
                ).run(cap, str(config_value(mode_config, "window_title", "Camera Feed")))
    finally:
        cleanup(cap)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
