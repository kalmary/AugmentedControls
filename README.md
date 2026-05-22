# Augmented Controls

OpenCV camera feed with MediaPipe pose, hand, and face mesh tracking. Skeletons
and face meshes are detected continuously, then drawn only when they are close
enough to the camera based on their visible landmark area.

## Setup

```bash
sudo dnf install python3.12-devel #on fedora
sudo dnf install wmctrl #on fedora
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --mode viewer --pose --hand --face
```

Press `q` to quit.

Tune the close-distance thresholds if the skeleton appears too early or too
late by editing `configs/pose.json`, `configs/hand.json`, or
`configs/face.json`. Those files also contain separate MediaPipe detection and
tracking confidence values for each body part.

## Modes

Viewer mode draws enabled skeletons only:

```bash
python src/main.py --mode viewer --pose --hand --face
```

Hand-control mode is implemented:

```bash
python src/main.py --mode hand-control
```

It asks for remote mouse-control permission, tracks the body skeleton, then:

- raise your right hand to accept the visible person as the controller
- wave either wrist to select it for mouse control
- point with the selected arm to steer the mouse
- hold the pointer steady for `click_hold_seconds` from
  `configs/hand_control.json` to click
- watch the dwell cursor circle fill to see when a click is about to fire

The dwell cursor is a PyQt fullscreen transparent overlay. It is marked
transparent for mouse input, so clicks should pass through to windows behind it.

The camera window is marked always-on-top when OpenCV/your window manager
supports it. If `wmctrl` is installed, the app also refreshes the window's
`above` hint while it runs. Place it with `window_position` in the mode config
file.

Mouse control adapts the selected arm's pointing range to the detected monitor
resolution, so small camera-space movements can still reach the full screen.
Tune the mapping with `control_gain`, `control_margin`, and
`mouse_edge_padding_pixels` in `configs/hand_control.json`. Tune the circular
click indicator with `dwell_cursor_enabled`, `dwell_cursor_diameter_pixels`,
`dwell_cursor_base_alpha`, `dwell_cursor_fill_alpha`, and
`dwell_cursor_workspace_alpha`. Tune click forgiveness with `click_radius` and
`click_grace_seconds`. Tune pointer acceleration with `control_acceleration`
and `control_acceleration_threshold`.

Put both hands near your chest in an X sign for a moment to close the app
without pressing `q`.

`hand-control-precise`, `eye-control`, and `steering-wheel` are accepted mode
names but are not implemented yet. The current MediaPipe Pose path tracks one
visible pose, so multi-person selection is approximated by accepting the visible
person only after the right-hand-raised gesture.

Set `verbose` or `native_logs` in the active mode config to change logging.

## Mouse Controller Test

```bash
python src/mouse_controller.py
```

This runs controller math checks, then moves the real cursor to center, left,
right, and center again using `pynput`. The test asks for permission before
enabling GNOME remote mouse control, and turns remote control off when it
finishes.
