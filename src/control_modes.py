from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time

import cv2
import mediapipe as mp

from mouse_controller import MouseController, temporary_remote_control_permission
from camera import configure_camera_window, keep_camera_window_topmost
from dwell_cursor import DwellOverlay


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


@dataclass(frozen=True)
class HandControlConfig:
    min_detection_confidence: float
    min_tracking_confidence: float
    click_hold_seconds: float
    click_radius: float
    click_grace_seconds: float
    click_cooldown_seconds: float
    wave_window_seconds: float
    wave_min_span: float
    wave_min_direction_changes: int
    mouse_smoothing: float
    mouse_deadzone: float
    mouse_edge_padding: float
    mouse_edge_padding_pixels: int | None
    dwell_cursor_enabled: bool
    dwell_cursor_radius: int | None
    dwell_cursor_diameter_pixels: int
    dwell_cursor_base_alpha: float
    dwell_cursor_fill_alpha: float
    dwell_cursor_workspace_alpha: float
    control_margin: float
    control_gain: float
    control_acceleration: float
    control_acceleration_threshold: float
    window_position: str
    exit_hold_seconds: float


@dataclass(frozen=True)
class TrackedPoint:
    x: float
    y: float
    visibility: float


@dataclass
class WaveTracker:
    samples: list[tuple[float, float]] = field(default_factory=list)

    def add(self, now: float, x: float, window_seconds: float) -> None:
        self.samples.append((now, x))
        cutoff = now - window_seconds
        self.samples = [(sample_time, sample_x) for sample_time, sample_x in self.samples if sample_time >= cutoff]

    def is_wave(self, min_span: float, min_direction_changes: int) -> bool:
        if len(self.samples) < 6:
            return False

        positions = [x for _, x in self.samples]
        if max(positions) - min(positions) < min_span:
            return False

        direction_changes = 0
        previous_direction = 0
        for previous_x, current_x in zip(positions, positions[1:]):
            delta = current_x - previous_x
            if abs(delta) < 0.015:
                continue
            direction = 1 if delta > 0 else -1
            if previous_direction and direction != previous_direction:
                direction_changes += 1
            previous_direction = direction

        return direction_changes >= min_direction_changes

    def reset(self) -> None:
        self.samples.clear()


class DwellClicker:
    def __init__(
        self,
        hold_seconds: float,
        radius: float,
        grace_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self.hold_seconds = hold_seconds
        self.radius = radius
        self.grace_seconds = grace_seconds
        self.cooldown_seconds = cooldown_seconds
        self.anchor: tuple[float, float] | None = None
        self.anchor_time = 0.0
        self.outside_since: float | None = None
        self.last_click_time = 0.0

    def update(self, now: float, point: tuple[float, float]) -> bool:
        if self.anchor is None:
            self.anchor = point
            self.anchor_time = now
            self.outside_since = None
            return False

        if self._distance(self.anchor, point) > self.radius:
            if self.outside_since is None:
                self.outside_since = now
            if now - self.outside_since < self.grace_seconds:
                return False
            self.anchor = point
            self.anchor_time = now
            self.outside_since = None
            return False

        self.outside_since = None

        if now - self.last_click_time < self.cooldown_seconds:
            return False

        if now - self.anchor_time < self.hold_seconds:
            return False

        self.last_click_time = now
        self.anchor = point
        self.anchor_time = now
        return True

    def progress(self, now: float, point: tuple[float, float]) -> float:
        if self.anchor is None or self._distance(self.anchor, point) > self.radius:
            if self.outside_since is None or now - self.outside_since < self.grace_seconds:
                return min(max((now - self.anchor_time) / self.hold_seconds, 0.0), 1.0)
            return 0.0
        if now - self.last_click_time < self.cooldown_seconds:
            return 0.0
        return min(max((now - self.anchor_time) / self.hold_seconds, 0.0), 1.0)

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


class AdaptiveControlMapper:
    def __init__(
        self,
        margin: float,
        gain: float,
        acceleration: float,
        acceleration_threshold: float,
    ) -> None:
        if not 0.0 <= margin < 0.5:
            raise ValueError("control margin must be between 0.0 and 0.5")
        if gain <= 0.0:
            raise ValueError("control gain must be positive")
        if acceleration < 1.0:
            raise ValueError("control acceleration must be at least 1.0")
        if not 0.0 < acceleration_threshold < 1.0:
            raise ValueError("control acceleration threshold must be between 0.0 and 1.0")

        self.margin = margin
        self.gain = gain
        self.acceleration = acceleration
        self.acceleration_threshold = acceleration_threshold
        self.bounds_alpha = 0.2
        self.min_x: float | None = None
        self.max_x: float | None = None
        self.min_y: float | None = None
        self.max_y: float | None = None

    def reset(self) -> None:
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

    def map(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        self._update_bounds(x, y)
        return (
            self._map_axis(x, self.min_x, self.max_x),
            self._map_axis(y, self.min_y, self.max_y),
        )

    def _update_bounds(self, x: float, y: float) -> None:
        self.min_x = self._smooth_bound(self.min_x, x, min)
        self.max_x = self._smooth_bound(self.max_x, x, max)
        self.min_y = self._smooth_bound(self.min_y, y, min)
        self.max_y = self._smooth_bound(self.max_y, y, max)

    def _smooth_bound(self, current: float | None, value: float, choose) -> float:
        if current is None:
            return value

        target = choose(current, value)
        return current + (target - current) * self.bounds_alpha

    def _map_axis(self, value: float, axis_min: float | None, axis_max: float | None) -> float:
        if axis_min is None or axis_max is None:
            return value

        center = (axis_min + axis_max) / 2.0
        span = max(axis_max - axis_min, 0.12)
        control_span = max(0.05, (span + self.margin * 2.0) / self.gain)
        normalized_delta = (value - center) / control_span
        mapped = 0.5 + self._accelerate_delta(normalized_delta)
        mapped = min(max(mapped, 0.0), 1.0)
        if mapped <= 0.025:
            return 0.0
        if mapped >= 0.975:
            return 1.0
        return mapped

    def _accelerate_delta(self, delta: float) -> float:
        sign = 1.0 if delta >= 0.0 else -1.0
        magnitude = abs(delta)
        if magnitude <= self.acceleration_threshold:
            return sign * magnitude / self.acceleration

        slow_part = self.acceleration_threshold / self.acceleration
        fast_part = (magnitude - self.acceleration_threshold) * self.acceleration
        return sign * (slow_part + fast_part)


class HandControlMode:
    def __init__(self, config: HandControlConfig) -> None:
        self.config = config
        self.user_accepted = False
        self.active_hand_label: str | None = None
        self.pose_visible = False
        self.wave_trackers = {
            "Left": WaveTracker(),
            "Right": WaveTracker(),
        }
        self.dwell_clicker = DwellClicker(
            config.click_hold_seconds,
            config.click_radius,
            config.click_grace_seconds,
            config.click_cooldown_seconds,
        )
        self.control_mapper = AdaptiveControlMapper(
            config.control_margin,
            config.control_gain,
            config.control_acceleration,
            config.control_acceleration_threshold,
        )
        self.crossed_arms_since: float | None = None

    def run(self, cap: cv2.VideoCapture, window_title: str) -> int:
        logging.info("Hand-control mode started. Press 'q' to quit.")
        logging.info("Waiting for remote-control permission in the terminal before opening the camera window.")
        logging.info("Raise your right hand to accept the visible person as the controller.")

        with temporary_remote_control_permission():
            logging.info("Remote-control permission granted. Opening camera window.")
            mouse_controller = MouseController(
                smoothing=self.config.mouse_smoothing,
                deadzone=self.config.mouse_deadzone,
                edge_padding=self.config.mouse_edge_padding,
                edge_padding_pixels=self.config.mouse_edge_padding_pixels,
            )
            screen_width, screen_height = mouse_controller.screen_dimensions()
            logging.info("Detected screen size: %sx%s", screen_width, screen_height)
            dwell_overlay = (
                DwellOverlay(
                    (screen_width, screen_height),
                    mouse_controller.workspace_padding(),
                    diameter_pixels=self._dwell_cursor_diameter_pixels(),
                    base_alpha=self.config.dwell_cursor_base_alpha,
                    fill_alpha=self.config.dwell_cursor_fill_alpha,
                    workspace_alpha=self.config.dwell_cursor_workspace_alpha,
                )
                if self.config.dwell_cursor_enabled
                else None
            )
            try:
                configure_camera_window(
                    window_title,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    self.config.window_position,
                )

                with mp_pose.Pose(
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence,
                ) as pose:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            logging.error("Failed to read frame from camera")
                            return 1

                        frame = cv2.flip(frame, 1)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame.flags.writeable = False
                        pose_results = pose.process(rgb_frame)
                        rgb_frame.flags.writeable = True

                        self.pose_visible = bool(pose_results.pose_landmarks)
                        if pose_results.pose_landmarks:
                            should_exit = self._handle_pose(
                                frame,
                                pose_results.pose_landmarks,
                                mouse_controller,
                                dwell_overlay,
                            )
                            if should_exit:
                                logging.info("Exit gesture detected.")
                                return 0
                        else:
                            if dwell_overlay:
                                dwell_overlay.hide()
                            self._forget_active_hand("Body tracking lost; wave again to select a hand.")

                        self._draw_mode_status(frame)
                        cv2.imshow(window_title, frame)
                        keep_camera_window_topmost(window_title)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logging.info("Quit requested")
                            return 0
            finally:
                if dwell_overlay:
                    dwell_overlay.close()

    def _handle_pose(
        self,
        frame,
        pose_landmarks,
        mouse_controller: MouseController,
        dwell_overlay: DwellOverlay | None,
    ) -> bool:
        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        if not self.user_accepted and self._right_hand_is_raised(pose_landmarks.landmark):
            self.user_accepted = True
            logging.info("User accepted: right hand raised.")
            logging.info("Wave one hand to select it for mouse control.")

        now = time.monotonic()
        if self._crossed_arms_exit_held(pose_landmarks.landmark, now):
            return True

        arms = {
            "Left": (
                pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                self._hand_point(pose_landmarks.landmark, "Left"),
            ),
            "Right": (
                pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                self._hand_point(pose_landmarks.landmark, "Right"),
            ),
        }

        self._forget_active_hand_if_lost(arms)

        for label, (_, hand) in arms.items():
            if hand.visibility < 0.5:
                continue

            tracker = self.wave_trackers.get(label)
            if tracker:
                tracker.add(now, hand.x, self.config.wave_window_seconds)
                if self.user_accepted and tracker.is_wave(
                    self.config.wave_min_span,
                    self.config.wave_min_direction_changes,
                ):
                    if self.active_hand_label == label:
                        tracker.reset()
                        continue

                    self.active_hand_label = label
                    tracker.reset()
                    mouse_controller.reset_smoothing()
                    self.control_mapper.reset()
                    self._reset_click_state()
                    logging.info("%s hand selected for mouse control.", label)

        if not self.user_accepted or not self.active_hand_label:
            if dwell_overlay:
                dwell_overlay.hide()
            return False

        active_arm = arms.get(self.active_hand_label)
        if not active_arm:
            if dwell_overlay:
                dwell_overlay.hide()
            return False

        normalized_point = self._arm_pointing_position(*active_arm)
        if normalized_point is None:
            if dwell_overlay:
                dwell_overlay.hide()
            return False

        mapped_point = self.control_mapper.map(normalized_point)
        mouse_controller.move_to_normalized(*mapped_point)
        click_progress = self.dwell_clicker.progress(now, mapped_point)

        if dwell_overlay:
            pixel_position = mouse_controller.last_pixel_position()
            if pixel_position:
                dwell_overlay.update(pixel_position, click_progress)

        if self.dwell_clicker.update(now, mapped_point):
            mouse_controller.click_left()
            logging.info("Mouse click.")
        return False

    def _dwell_cursor_diameter_pixels(self) -> int:
        if self.config.dwell_cursor_radius is not None:
            return self.config.dwell_cursor_radius * 2

        return max(self.config.dwell_cursor_diameter_pixels, 14)

    def _forget_active_hand_if_lost(self, arms) -> None:
        if not self.active_hand_label:
            return

        active_arm = arms.get(self.active_hand_label)
        if not active_arm:
            self._forget_active_hand("Selected hand tracking lost; wave again to select a hand.")
            return

        _, hand = active_arm
        if hand.visibility < 0.5:
            self._forget_active_hand("Selected hand tracking lost; wave again to select a hand.")

    def _forget_active_hand(self, message: str) -> None:
        if not self.active_hand_label:
            return

        self.active_hand_label = None
        self.control_mapper.reset()
        self._reset_click_state()
        for tracker in self.wave_trackers.values():
            tracker.reset()
        logging.info(message)

    def _reset_click_state(self) -> None:
        self.dwell_clicker.anchor = None
        self.dwell_clicker.outside_since = None

    @staticmethod
    def _hand_point(pose_landmarks, physical_label: str) -> TrackedPoint:
        if physical_label == "Left":
            indices = (
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.RIGHT_INDEX,
                mp_pose.PoseLandmark.RIGHT_PINKY,
            )
        else:
            indices = (
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_INDEX,
                mp_pose.PoseLandmark.LEFT_PINKY,
            )

        visible_points = [
            pose_landmarks[index]
            for index in indices
            if pose_landmarks[index].visibility >= 0.35
        ]
        if not visible_points:
            wrist = pose_landmarks[indices[0]]
            return TrackedPoint(wrist.x, wrist.y, wrist.visibility)

        total_visibility = sum(point.visibility for point in visible_points)
        return TrackedPoint(
            sum(point.x * point.visibility for point in visible_points) / total_visibility,
            sum(point.y * point.visibility for point in visible_points) / total_visibility,
            max(point.visibility for point in visible_points),
        )

    @staticmethod
    def _arm_pointing_position(shoulder, hand) -> tuple[float, float] | None:
        if shoulder.visibility < 0.5 or hand.visibility < 0.5:
            return None

        direction_x = hand.x - shoulder.x
        direction_y = hand.y - shoulder.y
        arm_length = max(abs(direction_x), abs(direction_y))
        if arm_length < 0.08:
            return None

        projection = 1.7
        return (
            min(max(shoulder.x + direction_x * projection, 0.0), 1.0),
            min(max(shoulder.y + direction_y * projection, 0.0), 1.0),
        )

    def _draw_mode_status(self, frame) -> None:
        if not self.pose_visible:
            status = "HAND CONTROL: no body detected"
            color = (60, 180, 255)
        elif not self.user_accepted:
            status = "HAND CONTROL: body detected - raise right hand"
            color = (60, 180, 255)
        elif not self.active_hand_label:
            status = "HAND CONTROL: accepted - wave a wrist"
            color = (60, 180, 255)
        else:
            status = f"HAND CONTROL: {self.active_hand_label} hand active"
            color = (20, 240, 120)

        cv2.putText(
            frame,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _right_hand_is_raised(pose_landmarks) -> bool:
        right_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        if right_wrist.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return False
        return right_wrist.y < right_shoulder.y - 0.04

    def _crossed_arms_exit_held(self, pose_landmarks, now: float) -> bool:
        if self._arms_are_crossed_near_body(pose_landmarks):
            if self.crossed_arms_since is None:
                self.crossed_arms_since = now
            return now - self.crossed_arms_since >= self.config.exit_hold_seconds

        self.crossed_arms_since = None
        return False

    @staticmethod
    def _arms_are_crossed_near_body(pose_landmarks) -> bool:
        left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if left_wrist.visibility < 0.55 or right_wrist.visibility < 0.55:
            return False
        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return False

        shoulder_left_x = min(left_shoulder.x, right_shoulder.x)
        shoulder_right_x = max(left_shoulder.x, right_shoulder.x)
        shoulder_width = max(shoulder_right_x - shoulder_left_x, 0.1)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        chest_center_x = (left_shoulder.x + right_shoulder.x) / 2.0

        wrists_inside_chest_width = (
            shoulder_left_x - shoulder_width * 0.2 <= left_wrist.x <= shoulder_right_x + shoulder_width * 0.2
            and shoulder_left_x - shoulder_width * 0.2 <= right_wrist.x <= shoulder_right_x + shoulder_width * 0.2
        )
        wrists_near_chest = (
            shoulder_y - 0.14 <= left_wrist.y <= shoulder_y + 0.55
            and shoulder_y - 0.14 <= right_wrist.y <= shoulder_y + 0.55
        )
        wrists_close_together = (
            abs(left_wrist.x - right_wrist.x) <= shoulder_width * 0.65
            and abs(left_wrist.y - right_wrist.y) <= shoulder_width * 0.45
            and abs(((left_wrist.x + right_wrist.x) / 2.0) - chest_center_x) <= shoulder_width * 0.35
        )
        return wrists_inside_chest_width and wrists_near_chest and wrists_close_together

    @staticmethod
    def _segments_intersect(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> bool:
        def orientation(a, b, c) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        first_orientation = orientation(first_start, first_end, second_start)
        second_orientation = orientation(first_start, first_end, second_end)
        third_orientation = orientation(second_start, second_end, first_start)
        fourth_orientation = orientation(second_start, second_end, first_end)
        return first_orientation * second_orientation <= 0 and third_orientation * fourth_orientation <= 0
