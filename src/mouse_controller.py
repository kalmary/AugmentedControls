from __future__ import annotations

from contextlib import contextmanager
import subprocess
import sys
import time
from dataclasses import dataclass

from pynput.mouse import Button, Controller


REMOTE_DESKTOP_KEYS = (
    ("org.gnome.desktop.remote-desktop.rdp", "view-only"),
    ("org.gnome.desktop.remote-desktop.vnc", "view-only"),
)


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


class MouseController:
    def __init__(
        self,
        smoothing: float = 0.35,
        deadzone: float = 0.002,
        edge_padding: float = 0.02,
        min_interval_seconds: float = 0.0,
        screen_size: tuple[int, int] | None = None,
        mouse_backend=None,
        clock=time.monotonic,
    ) -> None:
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be between 0.0 and 1.0")
        if not 0.0 <= deadzone <= 1.0:
            raise ValueError("deadzone must be between 0.0 and 1.0")
        if not 0.0 <= edge_padding < 0.5:
            raise ValueError("edge_padding must be between 0.0 and 0.5")
        if min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be non-negative")

        self.smoothing = smoothing
        self.deadzone = deadzone
        self.edge_padding = edge_padding
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock
        self.mouse = mouse_backend or self._create_mouse_controller()
        self.screen_size = screen_size or self._detect_screen_size(self.mouse)
        self._last_position: NormalizedPoint | None = None
        self._last_move_time = 0.0

    def move_to_normalized(self, x: float, y: float) -> tuple[int, int] | None:
        now = self.clock()
        if now - self._last_move_time < self.min_interval_seconds:
            return None

        target = self._clamp_point(NormalizedPoint(x, y))
        if self._last_position and self._distance(self._last_position, target) < self.deadzone:
            return None

        smoothed = self._smooth_point(target)
        pixel_position = self._to_screen_position(smoothed)
        self.mouse.position = pixel_position
        self._last_position = smoothed
        self._last_move_time = now
        return pixel_position

    def reset_smoothing(self) -> None:
        self._last_position = None

    def click_left(self) -> None:
        self.mouse.click(Button.left)

    def screen_dimensions(self) -> tuple[int, int]:
        return self.screen_size

    def _smooth_point(self, target: NormalizedPoint) -> NormalizedPoint:
        if self._last_position is None:
            return target

        previous = self._last_position
        alpha = 1.0 - self.smoothing
        return NormalizedPoint(
            x=previous.x + (target.x - previous.x) * alpha,
            y=previous.y + (target.y - previous.y) * alpha,
        )

    def _clamp_point(self, point: NormalizedPoint) -> NormalizedPoint:
        minimum = self.edge_padding
        maximum = 1.0 - self.edge_padding
        return NormalizedPoint(
            x=min(max(point.x, minimum), maximum),
            y=min(max(point.y, minimum), maximum),
        )

    def _to_screen_position(self, point: NormalizedPoint) -> tuple[int, int]:
        width, height = self.screen_size
        return (
            round(point.x * (width - 1)),
            round(point.y * (height - 1)),
        )

    @staticmethod
    def _distance(first: NormalizedPoint, second: NormalizedPoint) -> float:
        return max(abs(first.x - second.x), abs(first.y - second.y))

    @staticmethod
    def _create_mouse_controller():
        return Controller()

    @staticmethod
    def _detect_screen_size(mouse) -> tuple[int, int]:
        screen_size = getattr(mouse, "screen_size", None)
        if screen_size:
            width, height = screen_size
            return int(width), int(height)

        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return int(width), int(height)
        except Exception as error:
            raise RuntimeError(
                "Could not detect screen size. Pass screen_size=(width, height) "
                "when creating MouseController."
            ) from error


class _FakeMouse:
    def __init__(self) -> None:
        self.position = (0, 0)


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _run_gsettings(args: list[str]) -> str:
    result = subprocess.run(
        ["gsettings", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _set_remote_control_allowed(allowed: bool) -> None:
    view_only = "false" if allowed else "true"
    for schema, key in REMOTE_DESKTOP_KEYS:
        _run_gsettings(["set", schema, key, view_only])


@contextmanager
def temporary_remote_control_permission():
    print()
    print("Mouse control needs temporary GNOME remote-control permission.", flush=True)
    sys.stdout.write("Allow remote mouse control for this run? [y/N] ")
    sys.stdout.flush()
    answer = input().strip().lower()
    if answer not in {"y", "yes"}:
        raise RuntimeError("Remote mouse control was not allowed.")

    print("Enabling remote mouse control for this run.", flush=True)
    _set_remote_control_allowed(True)
    try:
        yield
    finally:
        print("Disabling remote mouse control.", flush=True)
        _set_remote_control_allowed(False)


def run_self_test() -> None:
    mouse = _FakeMouse()
    controller = MouseController(
        smoothing=0.0,
        deadzone=0.0,
        edge_padding=0.0,
        screen_size=(1000, 500),
        mouse_backend=mouse,
    )
    assert controller.move_to_normalized(0.5, 0.25) == (500, 125)
    assert mouse.position == (500, 125)

    mouse = _FakeMouse()
    controller = MouseController(
        smoothing=0.0,
        deadzone=0.0,
        edge_padding=0.1,
        screen_size=(100, 100),
        mouse_backend=mouse,
    )
    assert controller.move_to_normalized(-1.0, 2.0) == (10, 89)

    mouse = _FakeMouse()
    controller = MouseController(
        smoothing=0.5,
        deadzone=0.0,
        edge_padding=0.0,
        screen_size=(101, 101),
        mouse_backend=mouse,
    )
    assert controller.move_to_normalized(0.0, 0.0) == (0, 0)
    assert controller.move_to_normalized(1.0, 1.0) == (50, 50)
    controller.reset_smoothing()
    assert controller.move_to_normalized(1.0, 1.0) == (100, 100)

    mouse = _FakeMouse()
    controller = MouseController(
        smoothing=0.0,
        deadzone=0.1,
        edge_padding=0.0,
        screen_size=(100, 100),
        mouse_backend=mouse,
    )
    assert controller.move_to_normalized(0.5, 0.5) == (50, 50)
    assert controller.move_to_normalized(0.55, 0.55) is None
    assert mouse.position == (50, 50)

    clock = _FakeClock(now=10.0)
    mouse = _FakeMouse()
    controller = MouseController(
        smoothing=0.0,
        deadzone=0.0,
        edge_padding=0.0,
        min_interval_seconds=0.5,
        screen_size=(100, 100),
        mouse_backend=mouse,
        clock=clock,
    )
    assert controller.move_to_normalized(0.1, 0.1) == (10, 10)
    clock.advance(0.25)
    assert controller.move_to_normalized(0.9, 0.9) is None
    clock.advance(0.25)
    assert controller.move_to_normalized(0.9, 0.9) == (89, 89)

    for kwargs in (
        {"smoothing": -0.1},
        {"smoothing": 1.1},
        {"deadzone": -0.1},
        {"deadzone": 1.1},
        {"edge_padding": -0.1},
        {"edge_padding": 0.5},
        {"min_interval_seconds": -0.1},
    ):
        try:
            MouseController(
                screen_size=(100, 100),
                mouse_backend=_FakeMouse(),
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")

    print("The cursor will move to center, left, right, and center again.")
    print("Starting in 2 seconds...")
    time.sleep(2.0)

    with temporary_remote_control_permission():
        controller = MouseController(
            smoothing=0.0,
            deadzone=0.0,
            edge_padding=0.05,
            min_interval_seconds=0.0,
        )

        points = (
            (0.5, 0.5),
            (0.15, 0.5),
            (0.85, 0.5),
            (0.5, 0.5),
        )
        for point in points:
            pixel_position = controller.move_to_normalized(*point)
            print(f"moved normalized={point} pixel={pixel_position}")
            time.sleep(0.75)

    print("mouse_controller self-test passed")


if __name__ == "__main__":
    run_self_test()
