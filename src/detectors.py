from contextlib import ExitStack
import logging
from dataclasses import dataclass

import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


@dataclass(frozen=True)
class DetectionThresholds:
    pose_area: float
    hand_area: float
    face_area: float


@dataclass(frozen=True)
class DetectionConfidences:
    pose_detection: float
    pose_tracking: float
    hand_detection: float
    hand_tracking: float
    face_detection: float
    face_tracking: float


@dataclass(frozen=True)
class DetectionFlags:
    pose: bool
    hand: bool
    face: bool

    @property
    def any_enabled(self) -> bool:
        return self.pose or self.hand or self.face


def normalized_landmark_area(landmarks, visibility_cutoff: float | None = None) -> float:
    visible_landmarks = [
        landmark
        for landmark in landmarks
        if visibility_cutoff is None or landmark.visibility >= visibility_cutoff
    ]
    if not visible_landmarks:
        return 0.0

    min_x = max(0.0, min(landmark.x for landmark in visible_landmarks))
    max_x = min(1.0, max(landmark.x for landmark in visible_landmarks))
    min_y = max(0.0, min(landmark.y for landmark in visible_landmarks))
    max_y = min(1.0, max(landmark.y for landmark in visible_landmarks))
    return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)


def draw_status(
    frame,
    pose_area: float,
    hand_areas: list[float],
    face_areas: list[float],
    thresholds: DetectionThresholds,
    flags: DetectionFlags,
) -> None:
    status_parts = []
    is_close = False

    if flags.pose:
        pose_close = pose_area >= thresholds.pose_area
        is_close = is_close or pose_close
        status_parts.append(f"POSE: {'CLOSE' if pose_close else 'far'} ({pose_area:.2f})")

    if flags.hand:
        hand_close = any(area >= thresholds.hand_area for area in hand_areas)
        is_close = is_close or hand_close
        status_parts.append(f"HAND: {'CLOSE' if hand_close else 'far'}")

    if flags.face:
        face_close = any(area >= thresholds.face_area for area in face_areas)
        is_close = is_close or face_close
        status_parts.append(f"FACE: {'CLOSE' if face_close else 'far'}")

    if not status_parts:
        return

    status = " | ".join(status_parts)
    cv2.putText(
        frame,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 240, 120) if is_close else (60, 180, 255),
        2,
        cv2.LINE_AA,
    )


def detection_state(detected: bool, areas: list[float], threshold: float) -> str:
    if not detected:
        return "no detection"
    if any(area >= threshold for area in areas):
        return "close"
    return "far"


def log_detection_changes(previous_states: dict[str, str], current_states: dict[str, str]) -> None:
    for label, current_state in current_states.items():
        previous_state = previous_states.get(label)
        if previous_state == current_state:
            continue

        logging.info("%s: %s", label.upper(), current_state)
        previous_states[label] = current_state


class LandmarkDetector:
    def __init__(
        self,
        thresholds: DetectionThresholds,
        flags: DetectionFlags,
        confidences: DetectionConfidences,
    ) -> None:
        self.thresholds = thresholds
        self.flags = flags
        self.confidences = confidences
        self.detection_states: dict[str, str] = {}
        self._stack = ExitStack()
        self.pose = None
        self.hands = None
        self.face_mesh = None

    def __enter__(self):
        if self.flags.pose:
            self.pose = self._stack.enter_context(
                mp_pose.Pose(
                    min_detection_confidence=self.confidences.pose_detection,
                    min_tracking_confidence=self.confidences.pose_tracking,
                )
            )
        if self.flags.hand:
            self.hands = self._stack.enter_context(
                mp_hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=self.confidences.hand_detection,
                    min_tracking_confidence=self.confidences.hand_tracking,
                )
            )
        if self.flags.face:
            self.face_mesh = self._stack.enter_context(
                mp_face.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.confidences.face_detection,
                    min_tracking_confidence=self.confidences.face_tracking,
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        return self._stack.__exit__(exc_type, exc_value, traceback)

    def process_frame(self, frame):
        if not self.flags.any_enabled:
            return frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        pose_results = self.pose.process(rgb_frame) if self.pose else None
        hand_results = self.hands.process(rgb_frame) if self.hands else None
        face_results = self.face_mesh.process(rgb_frame) if self.face_mesh else None
        rgb_frame.flags.writeable = True

        pose_area = self._draw_pose(frame, pose_results)
        face_areas = self._draw_faces(frame, face_results)
        hand_areas = self._draw_hands(frame, hand_results)
        self._log_current_states(pose_area, hand_areas, face_areas, pose_results)
        draw_status(frame, pose_area, hand_areas, face_areas, self.thresholds, self.flags)
        return frame

    def _draw_pose(self, frame, pose_results) -> float:
        if not pose_results or not pose_results.pose_landmarks:
            return 0.0

        pose_area = normalized_landmark_area(
            pose_results.pose_landmarks.landmark,
            visibility_cutoff=0.5,
        )
        if pose_area >= self.thresholds.pose_area:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return pose_area

    def _draw_faces(self, frame, face_results) -> list[float]:
        face_areas = []
        if not face_results or not face_results.multi_face_landmarks:
            return face_areas

        for face_landmarks in face_results.multi_face_landmarks:
            face_area = normalized_landmark_area(face_landmarks.landmark)
            face_areas.append(face_area)
            if face_area >= self.thresholds.face_area:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
        return face_areas

    def _draw_hands(self, frame, hand_results) -> list[float]:
        hand_areas = []
        if not hand_results or not hand_results.multi_hand_landmarks:
            return hand_areas

        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_area = normalized_landmark_area(hand_landmarks.landmark)
            hand_areas.append(hand_area)
            if hand_area >= self.thresholds.hand_area:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        return hand_areas

    def _log_current_states(self, pose_area, hand_areas, face_areas, pose_results) -> None:
        current_detection_states = {}
        pose_detected = bool(pose_results and pose_results.pose_landmarks)

        if self.flags.pose:
            current_detection_states["pose"] = detection_state(
                pose_detected,
                [pose_area] if pose_detected else [],
                self.thresholds.pose_area,
            )
        if self.flags.hand:
            current_detection_states["hand"] = detection_state(
                bool(hand_areas),
                hand_areas,
                self.thresholds.hand_area,
            )
        if self.flags.face:
            current_detection_states["face"] = detection_state(
                bool(face_areas),
                face_areas,
                self.thresholds.face_area,
            )
        log_detection_changes(self.detection_states, current_detection_states)
