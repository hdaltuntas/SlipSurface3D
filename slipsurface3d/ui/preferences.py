"""Application theme (Dark / Light) and QSettings preferences for SlipSurfaceIQ-3D."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ORG_NAME = "SlipSurfaceIQ"
APP_SETTINGS_NAME = "SlipSurfaceIQ-3D"


def _settings() -> QSettings:
    return QSettings(ORG_NAME, APP_SETTINGS_NAME)


def current_theme() -> str:
    val = str(_settings().value("ui/theme", "dark"))
    return val if val in ("dark", "light") else "dark"


def set_theme_preference(theme: str) -> None:
    if theme in ("dark", "light"):
        _settings().setValue("ui/theme", theme)


def recent_files() -> list[str]:
    raw = _settings().value("files/recent", [])
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str) and raw:
        return [raw]
    return []


def push_recent_file(path: str, max_items: int = 8) -> list[str]:
    items = [p for p in recent_files() if p != path]
    items.insert(0, path)
    items = items[:max_items]
    _settings().setValue("files/recent", items)
    return items


DARK_QSS = """
QMainWindow, QDialog { background-color: #131722; color: #e2e8f0; }
QMenuBar { background-color: #1a2030; color: #e2e8f0; border-bottom: 1px solid #2b354d; }
QMenuBar::item:selected { background-color: #2b354d; }
QMenu { background-color: #1a2030; color: #e2e8f0; border: 1px solid #2b354d; }
QMenu::item:selected { background-color: #2563eb; color: #ffffff; }
QToolBar { background-color: #181e2c; border-bottom: 1px solid #2b354d; spacing: 5px; padding: 3px; }
QDockWidget { color: #cbd5e1; font-weight: bold; }
QDockWidget::title { background-color: #1d2436; padding: 6px; border-bottom: 1px solid #2b354d; }
QGroupBox { border: 1px solid #2b354d; border-radius: 5px; margin-top: 9px; padding-top: 8px; color: #cbd5e1; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { background-color: #263147; color: #f1f5f9; border: 1px solid #384766; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background-color: #32415e; }
QPushButton:pressed { background-color: #1d4ed8; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget, QTextEdit, QListWidget {
    background-color: #0f131d; color: #e2e8f0; border: 1px solid #334155; border-radius: 4px; padding: 3px 5px;
}
QHeaderView::section { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 4px; }
QStatusBar { background-color: #181e2c; color: #94a3b8; border-top: 1px solid #2b354d; }
QTabWidget::pane { border: 1px solid #2b354d; }
QTabBar::tab { background-color: #1a2030; color: #94a3b8; padding: 6px 12px; border: 1px solid #2b354d; }
QTabBar::tab:selected { background-color: #2563eb; color: #ffffff; }
"""

LIGHT_QSS = """
QMainWindow, QDialog { background-color: #f8fafc; color: #0f172a; }
QMenuBar { background-color: #f1f5f9; color: #0f172a; border-bottom: 1px solid #cbd5e1; }
QMenuBar::item:selected { background-color: #e2e8f0; }
QMenu { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; }
QMenu::item:selected { background-color: #2563eb; color: #ffffff; }
QToolBar { background-color: #f1f5f9; border-bottom: 1px solid #cbd5e1; spacing: 5px; padding: 3px; }
QDockWidget { color: #1e293b; font-weight: bold; }
QDockWidget::title { background-color: #e2e8f0; padding: 6px; border-bottom: 1px solid #cbd5e1; }
QGroupBox { border: 1px solid #cbd5e1; border-radius: 5px; margin-top: 9px; padding-top: 8px; color: #1e293b; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { background-color: #e2e8f0; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background-color: #cbd5e1; }
QPushButton:pressed { background-color: #2563eb; color: #ffffff; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget, QTextEdit, QListWidget {
    background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px 5px;
}
QHeaderView::section { background-color: #e2e8f0; color: #0f172a; border: 1px solid #cbd5e1; padding: 4px; }
QStatusBar { background-color: #f1f5f9; color: #334155; border-top: 1px solid #cbd5e1; }
QTabWidget::pane { border: 1px solid #cbd5e1; }
QTabBar::tab { background-color: #e2e8f0; color: #475569; padding: 6px 12px; border: 1px solid #cbd5e1; }
QTabBar::tab:selected { background-color: #2563eb; color: #ffffff; }
"""


def apply_theme(app: QApplication, theme: str = "dark") -> None:
    """Apply Dark or Light stylesheet and palette without scanning top-level QObject properties."""
    set_theme_preference(theme)
    pal = QPalette()
    if theme == "dark":
        pal.setColor(QPalette.ColorRole.Window, QColor("#131722"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#e2e8f0"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#0f131d"))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a2030"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#e2e8f0"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#263147"))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor("#f1f5f9"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(pal)
        app.setStyleSheet(DARK_QSS)
    else:
        pal.setColor(QPalette.ColorRole.Window, QColor("#f8fafc"))
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#0f172a"))
        pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#f1f5f9"))
        pal.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))
        pal.setColor(QPalette.ColorRole.Button, QColor("#e2e8f0"))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor("#0f172a"))
        pal.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        app.setPalette(pal)
        app.setStyleSheet(LIGHT_QSS)

    for w in app.topLevelWidgets():
        if w.isVisible() and hasattr(w, "on_theme_changed"):
            w.on_theme_changed(theme)
