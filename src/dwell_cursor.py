from __future__ import annotations

import logging
import os
import sys


class DwellOverlay:
    def __init__(
        self,
        screen_size: tuple[int, int],
        workspace_padding: tuple[float, float],
        diameter_pixels: int = 28,
        base_alpha: float = 0.4,
        fill_alpha: float = 0.9,
        workspace_alpha: float = 0.1,
    ) -> None:
        self.enabled = False
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
            from PyQt6.QtCore import QPointF, QRectF, Qt
            from PyQt6.QtGui import QColor, QPainter, QPen
            from PyQt6.QtWidgets import QApplication, QWidget
        except ImportError:
            logging.error("PyQt overlay requires PyQt6. Install project requirements with uv/pip.")
            return

        self.QApplication = QApplication
        self.QColor = QColor
        self.QPainter = QPainter
        self.QPen = QPen
        self.QPointF = QPointF
        self.QRectF = QRectF
        self.Qt = Qt

        self.app = QApplication.instance() or QApplication(sys.argv[:1])

        class OverlayWidget(QWidget):
            def __init__(widget_self) -> None:
                super().__init__()

            def paintEvent(widget_self, _event) -> None:
                painter = QPainter(widget_self)
                try:
                    self.paint(painter)
                finally:
                    painter.end()

        self.widget = OverlayWidget()
        self.screen_width, self.screen_height = screen_size
        self.workspace_padding = workspace_padding
        self.diameter_pixels = max(diameter_pixels, 14)
        self.base_alpha = min(max(base_alpha, 0.0), 1.0)
        self.fill_alpha = min(max(fill_alpha, 0.0), 1.0)
        self.workspace_alpha = min(max(workspace_alpha, 0.0), 1.0)
        self.pixel_position: tuple[int, int] | None = None
        self.progress = 0.0

        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
        self.widget.setWindowFlags(flags)
        self.widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.widget.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.widget.setAutoFillBackground(False)
        self.widget.setStyleSheet("background: transparent;")
        self.widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        screen = self.app.primaryScreen()
        geometry = screen.geometry() if screen else None
        self.origin_x = geometry.x() if geometry else 0
        self.origin_y = geometry.y() if geometry else 0
        width = geometry.width() if geometry else self.screen_width
        height = geometry.height() if geometry else self.screen_height
        self.widget.setGeometry(self.origin_x, self.origin_y, width, height)
        self.widget.showFullScreen()
        self._keep_on_top()
        self.app.processEvents()
        self.enabled = True
        logging.info("PyQt dwell overlay started.")

    def update(self, pixel_position: tuple[int, int], progress: float) -> None:
        if not self.enabled:
            return

        self.pixel_position = pixel_position
        self.progress = min(max(progress, 0.0), 1.0)
        self._keep_on_top()
        self.widget.update()
        self.app.processEvents()

    def hide_cursor(self) -> None:
        if not self.enabled:
            return

        self.pixel_position = None
        self._keep_on_top()
        self.widget.update()
        self.app.processEvents()

    def hide(self) -> None:
        if not self.enabled:
            return

        self.hide_cursor()

    def close(self) -> None:
        if not self.enabled:
            return

        self.widget.close()
        self.app.processEvents()
        self.enabled = False

    def paint(self, painter) -> None:
        width = self.widget.width()
        height = self.widget.height()
        x_padding, y_padding = self.workspace_padding
        left = x_padding * width
        top = y_padding * height
        right = (1.0 - x_padding) * width
        bottom = (1.0 - y_padding) * height

        painter.setRenderHint(self.QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(self.QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.QRectF(0, 0, width, height), self.QColor(0, 0, 0, 0))
        painter.setCompositionMode(self.QPainter.CompositionMode.CompositionMode_SourceOver)

        painter.fillRect(
            self.QRectF(left, top, right - left, bottom - top),
            self.QColor(15, 23, 42, round(255 * self.workspace_alpha)),
        )

        border_pen = self.QPen(self.QColor(56, 189, 248, 210))
        border_pen.setWidth(4)
        painter.setPen(border_pen)
        painter.drawRect(self.QRectF(left, top, right - left, bottom - top))

        if self.pixel_position is None:
            return

        x = self.pixel_position[0] - self.origin_x
        y = self.pixel_position[1] - self.origin_y
        radius = self.diameter_pixels / 2.0
        thickness = max(round(self.diameter_pixels * 0.14), 2)
        circle = self.QRectF(x - radius, y - radius, radius * 2.0, radius * 2.0)

        base_pen = self.QPen(self.QColor(54, 211, 153, round(255 * self.base_alpha)))
        base_pen.setWidth(thickness)
        painter.setPen(base_pen)
        painter.drawEllipse(circle)

        if self.progress <= 0.0:
            return

        fill_pen = self.QPen(self.QColor(34, 197, 94, round(255 * self.fill_alpha)))
        fill_pen.setWidth(thickness)
        painter.setPen(fill_pen)
        painter.drawArc(circle, 90 * 16, round(-360 * 16 * self.progress))

    def _keep_on_top(self) -> None:
        self.widget.setWindowFlag(self.Qt.WindowType.WindowStaysOnTopHint, True)
        if not self.widget.isVisible():
            self.widget.showFullScreen()
        self.widget.raise_()
