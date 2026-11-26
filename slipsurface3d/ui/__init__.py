"""Desktop UI package for SlipSurfaceIQ-3D."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping

_LC_VARS = (
    "LC_ALL",
    "LANG",
    "LC_CTYPE",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_MESSAGES",
)


def configure_qt_vtk_env(
    env: MutableMapping[str, str] | None = None,
    platform_str: str | None = None,
) -> None:
    """Configure Qt6 + VTK environment variables before QApplication initializes.

    * Sets ``QT_API=pyside6`` so ``qtawesome`` never binds ``PyQt6`` when both are installed.
    * Upgrades ``C`` / ``POSIX`` locale settings to ``C.UTF-8`` so Qt6 never emits
      ``Detected locale "C" with character encoding "ANSI_X3.4-1968"`` warnings, and
      enforces ``LC_NUMERIC=C.UTF-8`` so VTK/OpenGL shaders always parse ``.`` decimals.
    * On Linux Wayland sessions, ``vtkXOpenGLRenderWindow`` requires an X11 32-bit Window XID
      (whereas Qt's native ``wayland`` plugin returns a 64-bit ``wl_surface*`` pointer, triggering
      ``vtkXOpenGLRenderWindow: The result is out of range, failed to get the converted tmp``).
      Automatically selects ``QT_QPA_PLATFORM=xcb`` via XWayland when ``DISPLAY`` is present,
      or ``offscreen`` when running on pure headless Wayland without XWayland.
    * On Windows, explicitly set ``QT_QPA_PLATFORM_PLUGIN_PATH`` to the PySide6 platforms
      directory so the ``windows`` plugin is found (avoids "Could not find the Qt platform
      plugin 'windows'" errors) and do **not** force offscreen.
    """
    target = os.environ if env is None else env
    plat = sys.platform if platform_str is None else platform_str

    target.setdefault("QT_API", "pyside6")

    for var in _LC_VARS:
        if target.get(var) in ("C", "POSIX"):
            target[var] = "C.UTF-8"
    if not target.get("LANG") and not target.get("LC_ALL"):
        target["LANG"] = "C.UTF-8"
    if not target.get("LC_ALL"):
        target["LC_NUMERIC"] = "C.UTF-8"

    # Explicit platform plugin path for Windows (critical for finding qwindows.dll)
    if plat.startswith("win") and not target.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        try:
            import PySide6  # noqa: WPS433

            plugin_path = os.path.join(os.path.dirname(PySide6.__file__), "plugins", "platforms")
            if os.path.isdir(plugin_path):
                target["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
        except ImportError:
            pass  # Fall back to Qt's default search

    if plat.startswith("linux") and not target.get("QT_QPA_PLATFORM"):
        is_wayland = bool(target.get("WAYLAND_DISPLAY")) or (
            target.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        )
        if is_wayland:
            target["QT_QPA_PLATFORM"] = "xcb" if target.get("DISPLAY") else "offscreen"

    # Do NOT force offscreen on Windows — let the normal "windows" plugin be used.
    # Users who still hit VTK crashes can set QT_QPA_PLATFORM=offscreen manually.


configure_qt_vtk_env()

from slipsurface3d.ui.main_window import MainWindow3D  # noqa: E402
from slipsurface3d.ui.viewport3d import Viewport3D  # noqa: E402

__all__ = ["MainWindow3D", "Viewport3D", "configure_qt_vtk_env"]
