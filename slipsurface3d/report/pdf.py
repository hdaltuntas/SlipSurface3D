"""ReportLab PDF Engineering Report generator for SlipSurfaceIQ-3D with PyVista offscreen renders."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from slipsurface3d import __version__
from slipsurface3d.analysis import AnalysisResults3D
from slipsurface3d.lem.solvers import METHODS_3D
from slipsurface3d.materials.models import MODEL_LABELS
from slipsurface3d.project import Project3D


def render_offscreen_scene_png(
    project: Project3D,
    results: AnalysisResults3D | None = None,
    camera_view: str = "iso",
    scalar_field: str = "SigmaN_Eff_kPa",
    size: tuple[int, int] = (1100, 680),
) -> bytes:
    """Render a crisp offscreen 3D PyVista scene (isometric or plan view) to PNG bytes."""
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("#f8fafc", top="#e2e8f0")

    model = results.model if results is not None else project.build_model()
    mat_colors = {m.name: m.color for m in project.materials}

    # Stratified layer solids (translucent) + ground wireframe
    for mat_name, sgrid in model.layer_solid_meshes(nx=44, ny=32):
        col = mat_colors.get(mat_name, "#d4a84b")
        plotter.add_mesh(
            sgrid,
            color=col,
            opacity=0.40 if results is not None else 0.82,
            smooth_shading=True,
            show_edges=False,
        )

    g_mesh = model.ground_mesh(nx=48, ny=36)
    plotter.add_mesh(g_mesh, color="#475569", style="wireframe", opacity=0.28, line_width=1.0)

    wt_mesh = model.water_mesh(nx=36, ny=28)
    if wt_mesh is not None:
        plotter.add_mesh(wt_mesh, color="#2563eb", opacity=0.30, smooth_shading=True)

    # Bolts
    for b in project.bolts:
        head = np.asarray(b["head"], dtype=float)
        tip = np.asarray(b["tip"], dtype=float)
        line = pv.Line(head, tip)
        plotter.add_mesh(line, color="#dc2626", line_width=3.0)

    if results is not None:
        prim = results.primary_result
        ugrid = results.columns.to_hex_mesh(
            sigma_n_eff=prim.sigma_n_eff,
            tau_mob=prim.tau_mob,
        )
        if ugrid.n_cells > 0:
            field_use = scalar_field if scalar_field in ugrid.cell_data else "Height_m"
            plotter.add_mesh(
                ugrid,
                scalars=field_use,
                cmap="viridis",
                opacity=0.92,
                show_edges=True,
                edge_color="#1e293b",
                line_width=0.6,
                scalar_bar_args={"title": field_use, "color": "#0f172a"},
            )
        slip_poly = results.critical_surface.to_mesh(
            model.x_bounds, model.y_bounds, ground_fn=model.ground_z, nx=60, ny=60
        )
        if slip_poly.n_points > 0:
            plotter.add_mesh(slip_poly, color="#dc2626", opacity=0.55, smooth_shading=True)

    plotter.show_bounds(
        grid="back",
        location="outer",
        color="#334155",
        xtitle="X [m]",
        ytitle="Y [m]",
        ztitle="Z [m]",
        font_size=9,
    )
    if camera_view == "xy":
        plotter.view_xy()
    elif camera_view == "xz":
        plotter.view_xz()
    else:
        plotter.view_isometric()
    plotter.camera.zoom(1.08)

    img_arr = plotter.screenshot(return_img=True)
    plotter.close()

    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(img_arr, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def export_pdf_report_3d(
    path: str | Path,
    project: Project3D,
    results: AnalysisResults3D,
) -> Path:
    """Generate a multi-section A4 engineering PDF report for a 3D slope stability analysis."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"SlipSurfaceIQ-3D Report — {project.summary.title}",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title3D",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub3D",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "H2_3D",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle("Body3D", parent=styles["Normal"], fontSize=9, leading=12)

    story: list = []
    story.append(Paragraph("SlipSurfaceIQ-3D — Engineering Analysis Report", title_style))
    story.append(
        Paragraph(
            f"<b>Project:</b> {project.summary.title} &nbsp;|&nbsp; "
            f"<b>Author:</b> {project.summary.author or '—'} &nbsp;|&nbsp; "
            f"<b>Version:</b> {__version__} &nbsp;|&nbsp; "
            f"<b>Date:</b> {project.summary.date_created}",
            sub_style,
        )
    )
    if project.summary.notes:
        story.append(Paragraph(project.summary.notes, body_style))
        story.append(Spacer(1, 4 * mm))

    # 1. 3D Factor of Safety Summary Table
    story.append(Paragraph("1. Three-Dimensional Limit Equilibrium Results", h2_style))
    fos_rows = [
        [
            "3D LEM Method",
            results.result_label,
            "Converged",
            "Admissible",
            "lambda",
            "m_alpha,min",
            "|Res| max",
        ]
    ]
    for key, res in results.method_results.items():
        label = METHODS_3D[key][0]
        fos_txt = f"{res.fos:.4f}" if np.isfinite(res.fos) else "N/A"
        lam_txt = f"{res.lam:.4f}" if np.isfinite(res.lam) else "—"
        ma_txt = f"{res.m_alpha_min:.3f}" if np.isfinite(res.m_alpha_min) else "—"
        rmax = np.nanmax(
            [
                abs(res.force_residual) if np.isfinite(res.force_residual) else 0.0,
                abs(res.moment_residual) if np.isfinite(res.moment_residual) else 0.0,
            ]
        )
        fos_rows.append(
            [
                label,
                fos_txt,
                "Yes" if res.converged else "No",
                "Yes" if res.admissible else "No",
                lam_txt,
                ma_txt,
                f"{rmax:.2e}",
            ]
        )

    t_fos = Table(
        fos_rows, colWidths=[48 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 24 * mm]
    )
    t_fos.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_fos)
    story.append(Spacer(1, 3 * mm))
    if results.design_summary:
        story.append(
            Paragraph(
                f"<b>Design standard:</b> {results.design_summary}. Reported values are "
                "over-design factors; ODF &ge; 1.0 satisfies the limit state.",
                body_style,
            )
        )
    if len(results.local_minima) > 1:
        modes = "; ".join(
            f"#{i} {results.result_label} {lm.fos:.3f} "
            f"(V = {lm.volume:.0f} m<sup>3</sup>, centroid {lm.centroid_xy[0]:.1f}, "
            f"{lm.centroid_xy[1]:.1f} m)"
            for i, lm in enumerate(results.local_minima, 1)
        )
        story.append(Paragraph(f"<b>Distinct critical surfaces:</b> {modes}", body_style))

    cols = results.columns
    story.append(
        Paragraph(
            f"<b>Active Columns:</b> {cols.n} ({cols.nx_grid} &times; {cols.ny_grid}, "
            f"&Delta;x = {cols.dx:.2f} m, &Delta;y = {cols.dy:.2f} m) &nbsp;|&nbsp; "
            f"<b>Volume:</b> {cols.total_volume:.1f} m<sup>3</sup> &nbsp;|&nbsp; "
            f"<b>Weight:</b> {cols.total_weight:.1f} kN &nbsp;|&nbsp; "
            f"<b>Max Depth:</b> {cols.max_depth:.2f} m &nbsp;|&nbsp; "
            f"<b>Sliding Azimuth:</b> {cols.azimuth_deg:.1f}&deg;",
            body_style,
        )
    )

    # 2. Offscreen 3D Visualization Figures (Isometric + Plan XY)
    story.append(Paragraph("2. Three-Dimensional Slip Mass & Effective Normal Stress", h2_style))
    png_iso = render_offscreen_scene_png(
        project, results, camera_view="iso", scalar_field="SigmaN_Eff_kPa"
    )
    img_iso = Image(io.BytesIO(png_iso), width=172 * mm, height=106 * mm)
    story.append(img_iso)
    story.append(Spacer(1, 3 * mm))

    png_xy = render_offscreen_scene_png(
        project, results, camera_view="xy", scalar_field="TauMob_kPa"
    )
    img_xy = Image(io.BytesIO(png_xy), width=172 * mm, height=106 * mm)
    story.append(Paragraph("Plan View (XY) — Mobilized Basal Shear Stress [kPa]", body_style))
    story.append(img_xy)

    # 3. Material Properties Table
    story.append(Paragraph("3. Geotechnical Material Properties", h2_style))
    mat_rows = [
        ["Material", "Constitutive Model", "gamma [kN/m3]", "gamma_sat", "Key Strength Parameters"]
    ]
    for m in project.materials:
        if m.model == "MC":
            p_txt = f"c' = {m.c:g} kPa, phi' = {m.phi:g} deg"
        elif m.model == "UD":
            p_txt = f"su = {m.su:g} kPa (phi = 0)"
        elif m.model == "UDG":
            p_txt = f"su = {m.su_top:g}..{m.su_bottom:g} kPa"
        elif m.model == "HB":
            p_txt = f"sigma_ci = {m.sigma_ci:g} kPa, GSI = {m.gsi:g}, mi = {m.mi:g}, D = {m.d:g}"
        elif m.model == "ANISO3D":
            p_txt = (
                f"Mass({m.c_mass:g}, {m.phi_mass:g} deg) / Bed({m.c_bedding:g}, {m.phi_bedding:g} deg) "
                f"@ {m.dip_deg:g}/{m.dip_dir_deg:g} deg"
            )
        else:
            p_txt = MODEL_LABELS.get(m.model, m.model)
        mat_rows.append(
            [m.name, MODEL_LABELS.get(m.model, m.model), f"{m.gamma:g}", f"{m.gamma_sat:g}", p_txt]
        )

    t_mat = Table(mat_rows, colWidths=[34 * mm, 46 * mm, 24 * mm, 22 * mm, 50 * mm])
    t_mat.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t_mat)

    doc.build(story)
    return target
