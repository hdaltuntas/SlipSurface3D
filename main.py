#!/usr/bin/env python3
"""Entry point for SlipSurfaceIQ-3D desktop application and CLI verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slipsurface3d import APP_NAME, __version__
from slipsurface3d.analysis import report_text_3d, run_analysis_3d
from slipsurface3d.project import (
    Project3D,
    hungr_1987_hemisphere_project,
    zhang_1988_homog_ellipsoid_project,
)


def _run_headless_check(project_file: str | None = None) -> int:
    if project_file:
        proj = Project3D.load(project_file)
        res = run_analysis_3d(proj)
        print(report_text_3d(proj, res))
        return 0 if res.primary_result.ok else 1

    print(f"{APP_NAME} {__version__} — Headless Core Verification (--check)")
    p1 = hungr_1987_hemisphere_project()
    r1 = run_analysis_3d(p1)
    print(report_text_3d(p1, r1))

    p2 = zhang_1988_homog_ellipsoid_project()
    r2 = run_analysis_3d(p2)
    print(report_text_3d(p2, r2))

    ok = r1.primary_result.ok and r2.primary_result.ok
    return 0 if ok else 1


def _run_gui(project_file: str | None = None, smoke_seconds: float | None = None) -> int:
    import locale

    from slipsurface3d.ui import configure_qt_vtk_env

    configure_qt_vtk_env()

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from slipsurface3d.ui.main_window import MainWindow3D
    from slipsurface3d.ui.preferences import apply_theme, current_theme

    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    for loc_name in ("C.UTF-8", "en_US.UTF-8", "C"):
        try:
            locale.setlocale(locale.LC_NUMERIC, loc_name)
            break
        except locale.Error:
            continue
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    apply_theme(app, current_theme())

    win = MainWindow3D()
    if project_file and Path(project_file).is_file():
        win.open_file(project_file)
    win.show()

    smoke_failed: list[Exception] = []
    if smoke_seconds is not None:
        ms = max(100, int(float(smoke_seconds) * 1000.0))

        def _smoke_step() -> None:
            try:
                win.load_builtin_example("zhang_1988_homog")
                win.compute_analysis(synchronous=True)
            except Exception as exc:
                smoke_failed.append(exc)
            finally:
                win.close()
                app.quit()

        QTimer.singleShot(ms, _smoke_step)

    rc = int(app.exec())
    if smoke_failed:
        raise smoke_failed[0]
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} ({__version__})")
    parser.add_argument(
        "project", nargs="?", help="Optional .slp3d.json or 2D .slp.json project file"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run headless 3D solver verification (or solve given project file) without launching GUI.",
    )
    parser.add_argument(
        "--smoke",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Launch 3D GUI, run a benchmark compute, and close cleanly after SECONDS.",
    )
    args = parser.parse_args(argv)

    if args.check:
        return _run_headless_check(args.project)
    return _run_gui(args.project, smoke_seconds=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
