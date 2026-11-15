"""Pytest configuration and cross-platform Qt/VTK shutdown cleanup for SlipSurfaceIQ-3D."""

from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication([])
    yield app


def pytest_sessionfinish(session, exitstatus) -> None:
    """Close lingering Qt top-level widgets and flush QtAwesome/VTK caches before Py_FinalizeEx."""
    try:
        from PySide6.QtWidgets import QApplication, QDialog

        app = QApplication.instance()
        if isinstance(app, QApplication):
            for w in list(app.topLevelWidgets()):
                try:
                    if hasattr(w, "viewport") and hasattr(w.viewport, "cleanup"):
                        w.viewport.cleanup()
                    if isinstance(w, QDialog):
                        w.done(0)
                    w.close()
                    w.deleteLater()
                except Exception:
                    pass
            app.processEvents()
            try:
                import qtawesome as qta

                if hasattr(qta, "_resource") and isinstance(qta._resource, dict):
                    qta._resource["iconic"] = None
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
