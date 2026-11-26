"""Embedded PyVista / VTK 3D Viewport widget for PySide6.

Operates in two seamless modes:
* **Interactive Desktop Mode** (`QT_QPA_PLATFORM` not offscreen/minimal): embeds
  ``QVTKRenderWindowInteractor`` backed by ``pyvista.BasePlotter`` with full 3D orbit/pan/zoom
  and interactive cell picking of 3D columns.
* **Headless Offscreen Mode** (`QT_QPA_PLATFORM == 'offscreen'`): renders using
  ``pyvista.Plotter(off_screen=True)`` into a responsive ``QLabel`` ``QPixmap`` so headless CI,
  ``--smoke``, and unit tests execute real 3D PyVista rendering without requiring X11/GLX.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget
from pyvista.plotting.render_window_interactor import RenderWindowInteractor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from slipsurface3d.analysis import AnalysisResults3D
from slipsurface3d.geometry.surfaces import SlipSurface3D
from slipsurface3d.project import Project3D

_log = logging.getLogger(__name__)

SCALAR_FIELD_OPTIONS: list[tuple[str, str]] = [
    ("SigmaN_Eff_kPa", "Effective Normal Stress σ'n [kPa]"),
    ("TauMob_kPa", "Mobilized Shear Stress τ_mob [kPa]"),
    ("PorePressure_kPa", "Basal Pore Pressure u [kPa]"),
    ("Height_m", "Column Height h [m]"),
    ("Weight_kN", "Column Weight W [kN]"),
    ("BaseDip_deg", "Basal Dip Angle α_s [deg]"),
    ("MaterialID", "Basal Material ID"),
]


LOCAL_MINIMA_COLORS: list[str] = ["#f59e0b", "#a855f7", "#06b6d4", "#84cc16", "#ec4899"]


class _EmbeddedQtPlotter(pv.BasePlotter):
    """Lightweight PyVista BasePlotter bound directly to a QVTKRenderWindowInteractor."""

    def __init__(self, qvtk_widget: QVTKRenderWindowInteractor) -> None:
        super().__init__(shape=(1, 1), lighting="light kit")
        self._initialized = False
        self.off_screen = False
        self.notebook = False
        self.qvtk_widget = qvtk_widget
        self.ren_win = qvtk_widget.GetRenderWindow()
        self.ren_win.SetMultiSamples(4)
        for renderer in self.renderers:
            self.ren_win.AddRenderer(renderer)

        num_layers = self.ren_win.GetNumberOfLayers()
        cur_layer = self.renderer.GetLayer()
        self.ren_win.SetNumberOfLayers(num_layers + 1)
        self.ren_win.AddRenderer(self.renderers.shadow_renderer)
        self.renderers.shadow_renderer.SetLayer(cur_layer + 1)
        self.renderers.shadow_renderer.SetInteractive(False)

        self.iren = RenderWindowInteractor(
            self, light_follow_camera=False, interactor=qvtk_widget._Iren
        )
        self.iren.set_render_window(self.ren_win)
        self.enable_trackball_style()
        self._initialized = True


class Viewport3D(QWidget):
    """3D Geotechnical Viewport rendering SlopeModel3D, loads, bolts, slip bowl, and columns."""

    columnSelected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._project: Project3D | None = None
        self._results: AnalysisResults3D | None = None
        self._active_method: str = "bishop"
        self._scalar_field: str = "SigmaN_Eff_kPa"
        self._camera_preset: str = "iso"
        self._strata_opacity: float = 0.45
        self._show_columns: bool = True
        self._show_slip_surface: bool = True
        self._show_water: bool = True
        self._show_loads_supports: bool = True
        self._clip_mid_y: bool = False
        self._show_safety_map: bool = False
        self._show_local_minima: bool = True
        self._show_force_vectors: bool = True
        self._live_trial_surface: SlipSurface3D | None = None
        self._selected_col: int = -1
        self._theme: str = "dark"

        platform_name = QGuiApplication.platformName().lower()
        self._offscreen_mode = platform_name in (
            "offscreen",
            "minimal",
        ) or platform_name.startswith("wayland")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._qvtk: QVTKRenderWindowInteractor | None = None
        self._plotter: _EmbeddedQtPlotter | None = None
        self._offscreen_label: QLabel | None = None

        if self._offscreen_mode:
            lbl = QLabel("3D Viewport (Offscreen Render)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            lbl.setMinimumSize(420, 320)
            layout.addWidget(lbl)
            self._offscreen_label = lbl
        else:
            qvtk = QVTKRenderWindowInteractor(self)
            qvtk.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            qvtk.setMinimumSize(420, 320)
            layout.addWidget(qvtk)
            self._qvtk = qvtk
            self._plotter = _EmbeddedQtPlotter(qvtk)
            qvtk.Initialize()

    @property
    def is_offscreen(self) -> bool:
        return self._offscreen_mode

    @property
    def scalar_field(self) -> str:
        return self._scalar_field

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.rebuild_scene(reset_camera=False)

    def set_scalar_field(self, field_name: str) -> None:
        self._scalar_field = field_name
        self.rebuild_scene(reset_camera=False)

    def set_active_method(self, method: str) -> None:
        self._active_method = method
        self.rebuild_scene(reset_camera=False)

    def set_strata_opacity(self, opacity: float) -> None:
        self._strata_opacity = float(np.clip(opacity, 0.05, 1.0))
        self.rebuild_scene(reset_camera=False)

    def set_layer_visibility(
        self,
        *,
        columns: bool | None = None,
        slip_surface: bool | None = None,
        water: bool | None = None,
        loads_supports: bool | None = None,
        clip_mid_y: bool | None = None,
        safety_map: bool | None = None,
        local_minima: bool | None = None,
        force_vectors: bool | None = None,
    ) -> None:
        if columns is not None:
            self._show_columns = columns
        if slip_surface is not None:
            self._show_slip_surface = slip_surface
        if water is not None:
            self._show_water = water
        if loads_supports is not None:
            self._show_loads_supports = loads_supports
        if clip_mid_y is not None:
            self._clip_mid_y = clip_mid_y
        if safety_map is not None:
            self._show_safety_map = safety_map
        if local_minima is not None:
            self._show_local_minima = local_minima
        if force_vectors is not None:
            self._show_force_vectors = force_vectors
        self.rebuild_scene(reset_camera=False)

    def preview_trial_surface(self, surface: SlipSurface3D | None) -> None:
        """Render or clear a live best-so-far trial slip bowl during global optimization."""
        self._live_trial_surface = surface
        self.rebuild_scene(reset_camera=False)

    def select_column(self, col_idx: int) -> None:
        self._selected_col = col_idx
        self.rebuild_scene(reset_camera=False)
        if col_idx >= 0:
            self.columnSelected.emit(col_idx)

    def set_camera_view(self, preset: str) -> None:
        self._camera_preset = preset
        if self._plotter is not None:
            self._apply_camera(self._plotter, preset)
            self._plotter.render()
        else:
            self.rebuild_scene(reset_camera=True)

    def set_scene_data(
        self,
        project: Project3D,
        results: AnalysisResults3D | None = None,
        active_method: str | None = None,
        reset_camera: bool = True,
    ) -> None:
        self._project = project
        self._results = results
        self._live_trial_surface = None
        if active_method is not None:
            self._active_method = active_method
        elif results is not None:
            self._active_method = results.primary_method
        self._selected_col = -1
        self.rebuild_scene(reset_camera=reset_camera)

    def _apply_camera(self, pl: pv.BasePlotter, preset: str) -> None:
        if preset == "xy":
            pl.view_xy()
        elif preset == "xz":
            pl.view_xz()
        elif preset == "yz":
            pl.view_yz()
        else:
            pl.view_isometric()
            pl.camera.zoom(1.06)

    def _populate_plotter(self, pl: pv.BasePlotter, reset_camera: bool = True) -> None:
        pl.clear()
        if self._theme == "dark":
            pl.set_background("#0f172a", top="#1e293b")
            fg_col = "#e2e8f0"
            wire_col = "#94a3b8"
        else:
            pl.set_background("#f8fafc", top="#dbeafe")
            fg_col = "#0f172a"
            wire_col = "#475569"

        if self._project is None:
            return

        proj = self._project
        res = self._results
        if res is None and proj.ground is None and not proj.boreholes:
            pl.add_text(
                "Empty 3D Project\nUse File → New Parametric 3D Slope, Import, or Boreholes.",
                position="upper_left",
                font_size=10,
                color=fg_col,
            )
            pl.add_axes()
            if reset_camera:
                self._apply_camera(pl, self._camera_preset)
            return

        model = res.model if res is not None else proj.build_model()
        mat_colors = {m.name: m.color for m in proj.materials}
        y_mid = 0.5 * (model.y_min + model.y_max)

        # 1. Stratified 3D solid horizons
        eff_op = self._strata_opacity if res is not None else max(0.75, self._strata_opacity)
        for mat_name, sgrid in model.layer_solid_meshes(nx=48, ny=34):
            mesh_to_add: Any = sgrid
            if self._clip_mid_y:
                mesh_to_add = sgrid.clip(normal=(0.0, 1.0, 0.0), origin=(0.0, y_mid, 0.0))
            if mesh_to_add.n_points > 0:
                pl.add_mesh(
                    mesh_to_add,
                    color=mat_colors.get(mat_name, "#d4a84b"),
                    opacity=eff_op,
                    smooth_shading=True,
                )

        # Ground wireframe overlay
        g_mesh = model.ground_mesh(nx=44, ny=32)
        pl.add_mesh(g_mesh, color=wire_col, style="wireframe", opacity=0.25, line_width=1.0)

        # Borehole sticks if present
        for bh in proj.boreholes:
            z_bot = min((lay.z_bottom for lay in bh.layers), default=bh.collar_z - 5.0)
            cyl = pv.Cylinder(
                center=(bh.x, bh.y, 0.5 * (bh.collar_z + z_bot)),
                direction=(0.0, 0.0, 1.0),
                radius=max(0.25, 0.008 * model.span_x),
                height=max(0.5, bh.collar_z - z_bot),
            )
            pl.add_mesh(cyl, color="#f59e0b", opacity=0.95)

        # 2. Water table surface
        if self._show_water:
            wt = model.water_mesh(nx=40, ny=30)
            if wt is not None:
                pl.add_mesh(wt, color="#3b82f6", opacity=0.35, smooth_shading=True)

        # Weak layers (material colour) and piezometric surfaces (cyan wireframe)
        for wl in proj.weak_layers:
            wl_grid = wl.horizon.to_mesh(model.x_bounds, model.y_bounds, nx=60, ny=44)
            # Only the part of the layer that is actually buried is meaningful
            pts = np.asarray(wl_grid.points)
            buried = pts[:, 2] <= np.asarray(model.ground_z(pts[:, 0], pts[:, 1])) + 1e-6
            wl_grid["Buried"] = buried.astype(float)
            wl_mesh = wl_grid.threshold(0.5, scalars="Buried", preference="point")
            if wl_mesh.n_cells == 0:
                continue
            pl.add_mesh(
                wl_mesh,
                color=mat_colors.get(wl.material, "#f43f5e"),
                opacity=0.55,
                show_edges=True,
                edge_color=wire_col,
                line_width=0.5,
                pickable=False,
            )
        if self._show_water:
            for _pz_name, pz_hor in proj.piezo_surfaces:
                pz_mesh = pz_hor.to_mesh(model.x_bounds, model.y_bounds, nx=30, ny=22)
                pl.add_mesh(
                    pz_mesh, color="#22d3ee", style="wireframe", opacity=0.6, pickable=False
                )

        # 3. External patch loads, point loads & 3D bolts
        if self._show_loads_supports:
            for p_ld in proj.patch_loads:
                xs = np.linspace(p_ld.x1, p_ld.x2, 8)
                ys = np.linspace(p_ld.y1, p_ld.y2, 8)
                xx, yy = np.meshgrid(xs, ys, indexing="ij")
                zz = model.ground_z(xx, yy) + 0.12
                pl.add_mesh(pv.StructuredGrid(xx, yy, zz), color="#f97316", opacity=0.55)

            for pt_ld in proj.point_loads:
                f_dir = pt_ld.force_xyz()
                norm = float(np.linalg.norm(f_dir))
                if norm > 1e-9:
                    u_dir = f_dir / norm
                    scale = max(2.0, 0.06 * model.span_x)
                    start = np.array([pt_ld.x, pt_ld.y, pt_ld.z]) - scale * u_dir
                    arr = pv.Arrow(start=start, direction=u_dir, scale=scale)
                    pl.add_mesh(arr, color="#ef4444")

            # Geotextile sheets as translucent panels, pile rows as vertical bars
            for sheet in proj.geotextiles:
                panel = pv.PolyData(sheet.corners(), faces=np.array([4, 0, 1, 2, 3]))
                pl.add_mesh(panel, color="#0ea5e9", opacity=0.45, show_edges=True, pickable=False)
            for row in proj.pile_rows:
                for px, py in row.pile_positions():
                    z_head = float(model.ground_z(float(px), float(py)))
                    pl.add_mesh(
                        pv.Line((px, py, z_head), (px, py, z_head - row.length_m)),
                        color="#cbd5e1",
                        line_width=4.0,
                        pickable=False,
                    )

            for b in proj.bolts:
                head = np.asarray(b["head"], dtype=float)
                tip = np.asarray(b["tip"], dtype=float)
                pl.add_mesh(pv.Line(head, tip), color="#ef4444", line_width=3.5)
                pl.add_mesh(
                    pv.Sphere(radius=max(0.18, 0.005 * model.span_x), center=head), color="#facc15"
                )

        # 4. User preview or live-search trial slip surface (before final results)
        if res is None and self._live_trial_surface is not None:
            t_poly = self._live_trial_surface.to_mesh(
                model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=44, ny=44
            )
            if t_poly.n_points > 0:
                pl.add_mesh(t_poly, color="#fb923c", opacity=0.55, smooth_shading=True)
        elif res is None and proj.surface_config.mode == "user":
            u_surf = proj.user_slip_surface()
            if u_surf is not None:
                u_poly = u_surf.to_mesh(
                    model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=56, ny=56
                )
                if u_poly.n_points > 0:
                    pl.add_mesh(u_poly, color="#ef4444", opacity=0.55, smooth_shading=True)

        # 5. Analysis Results: Safety Map, 3D Active Columns, Force Vectors & Critical Slip Bowl
        if res is not None:
            if self._show_safety_map and res.safety_map is not None:
                smap_mesh = res.safety_map.to_mesh(z_offset=0.06)
                if smap_mesh.n_points > 0:
                    pl.add_mesh(
                        smap_mesh,
                        scalars="MinFoS",
                        cmap="RdYlGn",
                        clim=(0.8, 2.0),
                        opacity=0.85,
                        smooth_shading=True,
                        scalar_bar_args={"title": "Slope Safety Map (Min FoS)", "color": fg_col},
                        pickable=False,
                    )

            m_res = res.method_results.get(self._active_method, res.primary_result)
            ugrid = res.columns.to_hex_mesh(
                sigma_n_eff=m_res.sigma_n_eff,
                tau_mob=m_res.tau_mob,
            )
            if self._show_columns and ugrid.n_cells > 0:
                grid_draw: Any = ugrid
                if self._clip_mid_y:
                    grid_draw = ugrid.clip(normal=(0.0, 1.0, 0.0), origin=(0.0, y_mid, 0.0))
                if grid_draw.n_cells > 0:
                    s_name = (
                        self._scalar_field
                        if self._scalar_field in grid_draw.cell_data
                        else "Height_m"
                    )
                    pl.add_mesh(
                        grid_draw,
                        scalars=s_name,
                        cmap="turbo" if s_name != "MaterialID" else "tab10",
                        opacity=0.92,
                        show_edges=True,
                        edge_color="#0f172a",
                        line_width=0.6,
                        scalar_bar_args=(
                            {"title": s_name, "color": fg_col}
                            if not self._show_safety_map
                            else None
                        ),
                        pickable=True,
                    )
                if 0 <= self._selected_col < ugrid.n_cells:
                    sel_cell = ugrid.extract_cells([self._selected_col])
                    pl.add_mesh(
                        sel_cell,
                        color="#fde047",
                        style="wireframe",
                        line_width=4.0,
                        pickable=False,
                    )
                    if self._show_force_vectors:
                        self._add_column_force_vectors(
                            pl, res, m_res, self._selected_col, model.span_x
                        )

            if self._show_slip_surface:
                slip_mesh = res.critical_surface.to_mesh(
                    model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=64, ny=64
                )
                if slip_mesh.n_points > 0:
                    pl.add_mesh(slip_mesh, color="#ef4444", opacity=0.60, smooth_shading=True)

            # Secondary failure modes from the multi-modal search (index 0 is the critical one)
            if self._show_local_minima and len(res.local_minima) > 1:
                label_pts: list[list[float]] = []
                label_txt: list[str] = []
                for rank, lm in enumerate(res.local_minima[1:], start=2):
                    lm_mesh = lm.surface.to_mesh(
                        model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=44, ny=44
                    )
                    if lm_mesh.n_points == 0:
                        continue
                    pl.add_mesh(
                        lm_mesh,
                        color=LOCAL_MINIMA_COLORS[(rank - 2) % len(LOCAL_MINIMA_COLORS)],
                        opacity=0.45,
                        smooth_shading=True,
                        pickable=False,
                    )
                    cx, cy = lm.centroid_xy
                    label_pts.append([cx, cy, float(model.ground_z(cx, cy)) + 0.5])
                    label_txt.append(f"#{rank}  {res.result_label} {lm.fos:.3f}")
                if label_pts:
                    pl.add_point_labels(
                        np.asarray(label_pts, dtype=float),
                        label_txt,
                        font_size=12,
                        text_color=fg_col,
                        shape_opacity=0.35,
                        show_points=False,
                        always_visible=True,
                    )

        pl.show_bounds(
            grid="back",
            location="outer",
            color=fg_col,
            xtitle="X [m]",
            ytitle="Y [m]",
            ztitle="Z [m]",
            font_size=9,
        )
        if reset_camera:
            self._apply_camera(pl, self._camera_preset)

    def _add_column_force_vectors(
        self,
        pl: pv.BasePlotter,
        res: AnalysisResults3D,
        m_res: Any,
        idx: int,
        span_x: float,
    ) -> None:
        """Render 3D free-body force arrows (W, N', S_mob, U) on selected column ``idx``."""
        cols = res.columns
        w_val = float(cols.W[idx])
        n_val = float(m_res.base_normal_eff[idx]) if m_res.base_normal_eff is not None else 0.0
        s_val = float(m_res.base_shear_mob[idx]) if m_res.base_shear_mob is not None else 0.0
        u_val = float(cols.u[idx] * cols.area_base[idx])
        f_ref = max(w_val, n_val, s_val, u_val, 1e-6)
        max_len = max(2.2, 0.09 * span_x)

        cx, cy = float(cols.x[idx]), float(cols.y[idx])
        zb = float(cols.z_base[idx])
        zcg = float(cols.z_cg[idx])
        base_pt = np.array([cx, cy, zb], dtype=float)

        # 1. Weight W (downward -Z, gold)
        if w_val > 0.0:
            lw = max_len * (0.35 + 0.65 * (w_val / f_ref))
            pl.add_mesh(
                pv.Arrow(
                    start=(cx, cy, zcg + 0.5 * lw),
                    direction=(0.0, 0.0, -1.0),
                    scale=lw,
                ),
                color="#facc15",
                pickable=False,
            )

        # 2. Effective Normal N' (along upward basal normal, sky-blue)
        n_dir = np.asarray(cols.normal_xyz[idx], dtype=float)
        if n_val > 0.0 and np.linalg.norm(n_dir) > 1e-6:
            ln = max_len * (0.35 + 0.65 * (n_val / f_ref))
            pl.add_mesh(
                pv.Arrow(start=base_pt - ln * n_dir, direction=n_dir, scale=ln),
                color="#38bdf8",
                pickable=False,
            )

        # 3. Mobilized Shear S_mob (opposing sliding along basal plane, emerald)
        psi = float(np.radians(cols.azimuth_deg))
        ca, sa = float(cols.cos_alpha_s[idx]), float(cols.sin_alpha_s[idx])
        s_dir = np.array([-ca * np.cos(psi), -ca * np.sin(psi), sa], dtype=float)
        if s_val > 0.0 and np.linalg.norm(s_dir) > 1e-6:
            ls = max_len * (0.35 + 0.65 * (s_val / f_ref))
            pl.add_mesh(
                pv.Arrow(start=base_pt, direction=s_dir, scale=ls),
                color="#22c55e",
                pickable=False,
            )

        # 4. Basal Pore-Water Force U (cyan, slightly offset laterally when > 0)
        if u_val > 0.02 * f_ref and np.linalg.norm(n_dir) > 1e-6:
            lu = max_len * (0.30 + 0.60 * (u_val / f_ref))
            off = np.array([0.15 * cols.dx, 0.15 * cols.dy, 0.0], dtype=float)
            pl.add_mesh(
                pv.Arrow(start=base_pt + off - lu * n_dir, direction=n_dir, scale=lu),
                color="#06b6d4",
                pickable=False,
            )

    def rebuild_scene(self, reset_camera: bool = False) -> None:
        if self._offscreen_mode:
            assert self._offscreen_label is not None
            w = max(480, self.width())
            h = max(340, self.height())
            pl = pv.Plotter(off_screen=True, window_size=[w, h])
            self._populate_plotter(pl, reset_camera=True)
            img = np.asarray(pl.screenshot(return_img=True), dtype=np.uint8)
            pl.close()
            h_i, w_i, ch = img.shape
            bytes_per_line = ch * w_i
            qimg = QImage(
                img.tobytes(), w_i, h_i, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()
            self._offscreen_label.setPixmap(QPixmap.fromImage(qimg))
            return

        assert self._plotter is not None
        self._populate_plotter(self._plotter, reset_camera=reset_camera)
        if self._results is not None and self._show_columns:
            try:
                self._plotter.disable_picking()
            except Exception:
                _log.debug("disable_picking failed", exc_info=True)

            def _on_cell_picked(picked_grid: Any) -> None:
                if picked_grid is not None and "ColumnIndex" in picked_grid.cell_data:
                    arr = picked_grid.cell_data["ColumnIndex"]
                    if len(arr) > 0:
                        idx = int(arr[0])
                        self._selected_col = idx
                        self.rebuild_scene(reset_camera=False)
                        self.columnSelected.emit(idx)

            try:
                self._plotter.enable_cell_picking(
                    callback=_on_cell_picked, through=False, show=True, show_message=False
                )
            except Exception:
                _log.warning("3D column picking is unavailable", exc_info=True)
        self._plotter.render()

    def export_screenshot(self, path: str, width: int = 1400, height: int = 900) -> None:
        pl = pv.Plotter(off_screen=True, window_size=[width, height])
        self._populate_plotter(pl, reset_camera=True)
        pl.screenshot(path)
        pl.close()

    def cleanup(self) -> None:
        """Explicitly release VTK/OpenGL resources prior to widget destruction."""
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                _log.debug("Plotter close failed", exc_info=True)
            self._plotter = None
        if self._qvtk is not None:
            try:
                self._qvtk.Finalize()
            except Exception:
                _log.debug("QVTK finalize failed", exc_info=True)
            self._qvtk = None

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self.cleanup()
        super().closeEvent(event)
