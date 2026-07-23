"""Premium clinical PDF report for CT Workbench cases (ReportLab).

Doctor-facing Russian layout: summary cards, modern charts, anatomical scheme,
service table. Public entry: ``build_case_report_pdf(report, output_path)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.visualization.displacement_plots import (
    estimate_kidney_centers,
    predictions_vector,
    quality_check_messages,
    quality_checks,
    vector_norm,
)

# ---------------------------------------------------------------------------
# Palette & typography
# ---------------------------------------------------------------------------

PRIMARY = HexColor("#1B3A5C")  # dark blue
PRIMARY_SOFT = HexColor("#2C5282")
SECONDARY = HexColor("#38BDF8")  # cyan accent
POSITIVE = HexColor("#16A34A")
NEGATIVE = HexColor("#DC2626")
WARNING = HexColor("#EA580C")
GRAY = HexColor("#64748B")
GRAY_LIGHT = HexColor("#94A3B8")
GRAY_LINE = HexColor("#E2E8F0")
GRAY_BG = HexColor("#F1F5F9")
CARD_BG = HexColor("#F8FAFC")
NEAR_ZERO = HexColor("#94A3B8")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
GAP = 8 * mm
COL_GAP = 6 * mm

_FONT_REG = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONTS_READY = False

FONT_REG = _FONT_REG
FONT_BOLD = _FONT_BOLD


def _register_fonts() -> None:
    global _FONTS_READY, _FONT_REG, _FONT_BOLD, FONT_REG, FONT_BOLD
    if _FONTS_READY:
        return
    candidates = [
        (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf"),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ),
    ]
    for regular, bold in candidates:
        try:
            pdfmetrics.registerFont(TTFont("ReportBody", regular))
            pdfmetrics.registerFont(TTFont("ReportBodyBold", bold))
            _FONT_REG = "ReportBody"
            _FONT_BOLD = "ReportBodyBold"
            FONT_REG = _FONT_REG
            FONT_BOLD = _FONT_BOLD
            _FONTS_READY = True
            return
        except Exception:
            continue
    _FONT_REG = "Helvetica"
    _FONT_BOLD = "Helvetica-Bold"
    FONT_REG = _FONT_REG
    FONT_BOLD = _FONT_BOLD
    _FONTS_READY = True


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _sign_color(value: float, *, eps: float = 0.5) -> Color:
    if abs(value) < eps:
        return NEAR_ZERO
    return POSITIVE if value > 0 else NEGATIVE


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _wrap(text: str, font: str, size: float, max_width: float) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if pdfmetrics.stringWidth(trial, font, size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


# ---------------------------------------------------------------------------
# Icons (vector paths)
# ---------------------------------------------------------------------------


def draw_icon(
    c: canvas.Canvas,
    name: str,
    x: float,
    y: float,
    size: float = 14,
    color: Color = PRIMARY,
) -> None:
    """Draw a small SVG-style icon at (x, y) bottom-left of the icon box."""
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.2)
    c.setLineCap(1)
    c.setLineJoin(1)
    s = size

    if name == "kidney":
        # Bean / kidney silhouette
        path = c.beginPath()
        path.moveTo(x + 0.22 * s, y + 0.15 * s)
        path.curveTo(x + 0.05 * s, y + 0.35 * s, x + 0.05 * s, y + 0.70 * s, x + 0.28 * s, y + 0.88 * s)
        path.curveTo(x + 0.45 * s, y + 1.00 * s, x + 0.62 * s, y + 0.92 * s, x + 0.72 * s, y + 0.72 * s)
        path.curveTo(x + 0.82 * s, y + 0.50 * s, x + 0.78 * s, y + 0.28 * s, x + 0.58 * s, y + 0.18 * s)
        path.curveTo(x + 0.42 * s, y + 0.10 * s, x + 0.32 * s, y + 0.08 * s, x + 0.22 * s, y + 0.15 * s)
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.15))
        c.drawPath(path, stroke=1, fill=1)
        c.setStrokeColor(color)
        c.line(x + 0.55 * s, y + 0.35 * s, x + 0.78 * s, y + 0.55 * s)

    elif name == "spine":
        for i in range(4):
            cy = y + 0.12 * s + i * 0.22 * s
            c.roundRect(x + 0.30 * s, cy, 0.40 * s, 0.16 * s, 1.5, stroke=1, fill=0)
        c.line(x + 0.50 * s, y + 0.08 * s, x + 0.50 * s, y + 0.92 * s)

    elif name == "patient":
        c.circle(x + 0.50 * s, y + 0.72 * s, 0.16 * s, stroke=1, fill=0)
        path = c.beginPath()
        path.moveTo(x + 0.22 * s, y + 0.18 * s)
        path.curveTo(x + 0.22 * s, y + 0.48 * s, x + 0.78 * s, y + 0.48 * s, x + 0.78 * s, y + 0.18 * s)
        c.drawPath(path, stroke=1, fill=0)

    elif name == "ct":
        c.circle(x + 0.50 * s, y + 0.50 * s, 0.38 * s, stroke=1, fill=0)
        c.circle(x + 0.50 * s, y + 0.50 * s, 0.18 * s, stroke=1, fill=0)
        c.setLineWidth(2.0)
        c.line(x + 0.12 * s, y + 0.50 * s, x + 0.28 * s, y + 0.50 * s)
        c.line(x + 0.72 * s, y + 0.50 * s, x + 0.88 * s, y + 0.50 * s)

    elif name == "prediction":
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.12))
        c.roundRect(x + 0.10 * s, y + 0.15 * s, 0.80 * s, 0.70 * s, 2, stroke=1, fill=1)
        c.setStrokeColor(color)
        c.setFillColor(color)
        # Mini chart line
        path = c.beginPath()
        path.moveTo(x + 0.22 * s, y + 0.35 * s)
        path.lineTo(x + 0.40 * s, y + 0.55 * s)
        path.lineTo(x + 0.55 * s, y + 0.42 * s)
        path.lineTo(x + 0.78 * s, y + 0.68 * s)
        c.drawPath(path, stroke=1, fill=0)

    elif name == "check":
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.12))
        c.circle(x + 0.50 * s, y + 0.50 * s, 0.40 * s, stroke=1, fill=1)
        c.setStrokeColor(color)
        c.setLineWidth(1.8)
        path = c.beginPath()
        path.moveTo(x + 0.28 * s, y + 0.50 * s)
        path.lineTo(x + 0.44 * s, y + 0.34 * s)
        path.lineTo(x + 0.74 * s, y + 0.68 * s)
        c.drawPath(path, stroke=1, fill=0)

    elif name == "info":
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.12))
        c.circle(x + 0.50 * s, y + 0.50 * s, 0.40 * s, stroke=1, fill=1)
        c.setFillColor(color)
        c.circle(x + 0.50 * s, y + 0.72 * s, 0.06 * s, stroke=0, fill=1)
        c.setStrokeColor(color)
        c.setLineWidth(1.6)
        c.line(x + 0.50 * s, y + 0.58 * s, x + 0.50 * s, y + 0.28 * s)
        c.line(x + 0.50 * s, y + 0.28 * s, x + 0.62 * s, y + 0.28 * s)

    else:
        c.circle(x + 0.50 * s, y + 0.50 * s, 0.35 * s, stroke=1, fill=0)

    c.restoreState()


# ---------------------------------------------------------------------------
# Layout primitives
# ---------------------------------------------------------------------------


def draw_header(
    c: canvas.Canvas,
    title: str,
    *,
    subtitle: str = "",
    page_num: int = 1,
    page_count: int = 1,
) -> float:
    """Top brand bar + page title. Returns y below the header block."""
    y_top = PAGE_H - MARGIN
    # Accent strip
    c.setFillColor(PRIMARY)
    c.rect(0, PAGE_H - 4.5 * mm, PAGE_W, 4.5 * mm, stroke=0, fill=1)
    c.setFillColor(SECONDARY)
    c.rect(0, PAGE_H - 5.2 * mm, 28 * mm, 0.7 * mm, stroke=0, fill=1)

    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 9)
    c.drawString(MARGIN, y_top - 10 * mm, "CT Workbench")
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 8)
    c.drawRightString(PAGE_W - MARGIN, y_top - 10 * mm, f"стр. {page_num} / {page_count}")

    title_y = y_top - 18 * mm
    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 20)
    c.drawString(MARGIN, title_y, title)
    next_y = title_y - 4 * mm
    if subtitle:
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 9)
        for line in _wrap(subtitle, FONT_REG, 9, CONTENT_W):
            next_y -= 4 * mm
            c.drawString(MARGIN, next_y, line)
        next_y -= 3 * mm
    else:
        next_y -= 5 * mm
    # Divider
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, next_y, PAGE_W - MARGIN, next_y)
    return next_y - GAP


def draw_footer(c: canvas.Canvas, disclaimer: str = "") -> None:
    """Bottom bar with compact disclaimer."""
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.5)
    c.line(MARGIN, MARGIN - 2 * mm, PAGE_W - MARGIN, MARGIN - 2 * mm)
    c.setFillColor(GRAY_LIGHT)
    c.setFont(FONT_REG, 7)
    text = disclaimer or "Исследовательский инструмент. Не заменяет клиническое решение врача."
    lines = _wrap(text, FONT_REG, 7, CONTENT_W)
    y = MARGIN - 5.5 * mm
    for line in lines[:2]:
        c.drawString(MARGIN, y, line)
        y -= 3 * mm


def draw_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str = "",
    value: str = "",
    unit: str = "",
    caption: str = "",
    accent: Color = PRIMARY,
    icon: str = "",
    fill: Color = CARD_BG,
) -> None:
    """Rounded info card: title, large value, unit, caption, color indicator."""
    # Soft shadow
    c.setFillColor(HexColor("#E8EEF4"))
    c.roundRect(x + 1.2, y - 1.2, w, h, 6, stroke=0, fill=1)
    c.setFillColor(fill)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y, w, h, 6, stroke=1, fill=1)

    # Left accent bar
    c.setFillColor(accent)
    c.roundRect(x, y + 2, 2.8, h - 4, 1.2, stroke=0, fill=1)

    pad = 8
    tx = x + pad + 4
    top = y + h - pad - 2

    if icon:
        draw_icon(c, icon, x + w - pad - 16, top - 14, 14, accent)

    if title:
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 8)
        c.drawString(tx, top - 2, title)

    if value:
        c.setFillColor(PRIMARY)
        c.setFont(FONT_BOLD, 18)
        c.drawString(tx, y + h * 0.38, value)
        if unit:
            vw = pdfmetrics.stringWidth(value, FONT_BOLD, 18)
            c.setFillColor(GRAY)
            c.setFont(FONT_REG, 9)
            c.drawString(tx + vw + 4, y + h * 0.40, unit)

    if caption:
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 7.5)
        for i, line in enumerate(_wrap(caption, FONT_REG, 7.5, w - pad * 2 - 8)[:2]):
            c.drawString(tx, y + 8 + (1 - i) * 9, line)


def draw_patient_info(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    meta: Mapping[str, Any],
) -> float:
    """Patient info card under the title. Returns y below the card."""
    h = 28 * mm
    c.setFillColor(white)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y - h, w, h, 7, stroke=1, fill=1)
    c.setFillColor(HexColor("#EFF6FF"))
    c.roundRect(x, y - h, 22 * mm, h, 7, stroke=0, fill=1)
    # Fix right edge of blue panel (clip visually with white overlay strip)
    c.setFillColor(HexColor("#EFF6FF"))
    c.rect(x + 14 * mm, y - h, 8 * mm, h, stroke=0, fill=1)

    draw_icon(c, "patient", x + 5 * mm, y - h / 2 - 8, 16, PRIMARY)

    rows = [
        ("Пациент / метка", str(meta.get("patient_label") or "—")),
        ("Идентификатор кейса", str(meta.get("case_id") or "—")),
        ("Дата отчёта", str(meta.get("updated_at") or "—")),
        ("Статус", str(meta.get("status") or "—")),
    ]
    col_w = (w - 28 * mm) / 2
    for i, (label, val) in enumerate(rows):
        col = i % 2
        row = i // 2
        lx = x + 26 * mm + col * col_w
        ly = y - 9 * mm - row * 11 * mm
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 7.5)
        c.drawString(lx, ly + 4 * mm, label)
        c.setFillColor(PRIMARY)
        c.setFont(FONT_BOLD, 10)
        c.drawString(lx, ly, val[:42])

    return y - h - GAP


def draw_summary(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    left: Any,
    right: Any,
) -> float:
    """«Клиническое резюме» compact block. Returns y below."""
    h = 26 * mm
    c.setFillColor(CARD_BG)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y - h, w, h, 6, stroke=1, fill=1)

    draw_icon(c, "info", x + 4 * mm, y - 9 * mm, 12, PRIMARY_SOFT)
    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 11)
    c.drawString(x + 12 * mm, y - 8 * mm, "Клиническое резюме")

    ln = vector_norm(left)
    rn = vector_norm(right)
    lines = [
        f"Ожидаемое смещение при переходе со спины на бок: "
        f"левая почка {ln:.1f} мм, правая {rn:.1f} мм (суммарный вектор).",
        "Направления: вправо–влево (+ вправо), вперёд–назад (+ кпереди), вверх–вниз (+ краниально).",
    ]
    ty = y - 14 * mm
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 8.5)
    for paragraph in lines:
        for line in _wrap(paragraph, FONT_REG, 8.5, w - 10 * mm):
            c.drawString(x + 5 * mm, ty, line)
            ty -= 3.8 * mm
    return y - h - GAP


def draw_quality_block(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    *,
    coverage_pct: Optional[float],
    checks: Mapping[str, bool],
    model_id: str = "—",
) -> float:
    """Card: completeness, prediction quality, model check. Returns y below."""
    h = 32 * mm
    all_ok = bool(checks.get("all_passed"))
    accent = POSITIVE if all_ok else WARNING
    c.setFillColor(white)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y - h, w, h, 6, stroke=1, fill=1)
    c.setFillColor(accent)
    c.roundRect(x, y - h + 2, 2.8, h - 4, 1.2, stroke=0, fill=1)

    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 10)
    c.drawString(x + 6 * mm, y - 7 * mm, "Качество данных и прогноза")
    draw_icon(c, "check" if all_ok else "info", x + w - 12 * mm, y - 10 * mm, 12, accent)

    cov = f"{float(coverage_pct):.0f}%" if coverage_pct is not None else "н/д"
    quality_label = "Проверки пройдены" if all_ok else "Есть замечания"
    items = [
        ("Полнота данных КТ", cov, SECONDARY, "ct"),
        ("Качество прогноза", quality_label, accent, "prediction"),
        ("Модель", str(model_id)[:28], PRIMARY, "check"),
    ]
    card_w = (w - 14 * mm) / 3
    for i, (title, val, col, icon) in enumerate(items):
        cx = x + 5 * mm + i * (card_w + 2 * mm)
        cy = y - h + 4 * mm
        ch = 18 * mm
        c.setFillColor(GRAY_BG)
        c.roundRect(cx, cy, card_w, ch, 4, stroke=0, fill=1)
        draw_icon(c, icon, cx + 3 * mm, cy + ch - 12, 10, col)
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 7)
        c.drawString(cx + 10 * mm, cy + ch - 8, title)
        c.setFillColor(PRIMARY)
        c.setFont(FONT_BOLD, 9)
        c.drawString(cx + 3 * mm, cy + 5, val)

    return y - h - GAP


def draw_table(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    rows: Sequence[Tuple[str, str]],
    *,
    col1_ratio: float = 0.42,
) -> float:
    """Compact 2-column parameter|value table. Returns y below."""
    row_h = 7.2 * mm
    header_h = 8 * mm
    c.setFillColor(PRIMARY)
    c.roundRect(x, y - header_h, w, header_h, 4, stroke=0, fill=1)
    # square bottom of header so body joins cleanly
    c.rect(x, y - header_h, w, 4, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont(FONT_BOLD, 8)
    c.drawString(x + 4 * mm, y - 5.5 * mm, "Параметр")
    c.drawString(x + w * col1_ratio + 3 * mm, y - 5.5 * mm, "Значение")

    yy = y - header_h
    for i, (param, value) in enumerate(rows):
        yy -= row_h
        bg = GRAY_BG if i % 2 == 0 else white
        c.setFillColor(bg)
        c.rect(x, yy, w, row_h, stroke=0, fill=1)
        c.setStrokeColor(GRAY_LINE)
        c.setLineWidth(0.4)
        c.line(x, yy, x + w, yy)
        c.setFillColor(PRIMARY)
        c.setFont(FONT_REG, 8)
        c.drawString(x + 4 * mm, yy + 2.2 * mm, str(param)[:48])
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 8)
        c.drawString(x + w * col1_ratio + 3 * mm, yy + 2.2 * mm, str(value)[:56])

    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, yy, w, y - yy, 4, stroke=1, fill=0)
    return yy - GAP


# ---------------------------------------------------------------------------
# Charts (ReportLab — no heavy matplotlib bitmaps)
# ---------------------------------------------------------------------------


def draw_bar_chart(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    description: str,
    labels: Sequence[str],
    values: Sequence[float],
    unit: str = "мм",
    style: str = "lollipop",
) -> None:
    """Modern chart: lollipop / horizontal bullet. ``style``: lollipop|bullet|dumbbell."""
    # Card frame
    c.setFillColor(white)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y - h, w, h, 6, stroke=1, fill=1)

    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 11)
    c.drawString(x + 5 * mm, y - 7 * mm, title)
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 8)
    desc_y = y - 11 * mm
    for line in _wrap(description, FONT_REG, 8, w - 10 * mm)[:2]:
        c.drawString(x + 5 * mm, desc_y, line)
        desc_y -= 3.5 * mm

    n = max(len(values), 1)
    plot_top = desc_y - 2 * mm
    plot_bottom = y - h + 8 * mm
    plot_left = x + 42 * mm
    plot_right = x + w - 18 * mm
    plot_w = plot_right - plot_left
    plot_h = plot_top - plot_bottom

    abs_max = max((abs(v) for v in values), default=1.0)
    abs_max = max(abs_max, 5.0)
    zero_x = plot_left + plot_w * 0.5 if style == "lollipop" else plot_left

    # Axis
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.6)
    if style == "lollipop":
        c.line(zero_x, plot_bottom, zero_x, plot_top)
        for tick in (-abs_max, -abs_max / 2, abs_max / 2, abs_max):
            tx = zero_x + (tick / abs_max) * (plot_w * 0.48)
            c.setStrokeColor(GRAY_LINE)
            c.line(tx, plot_bottom, tx, plot_top)
            c.setFillColor(GRAY_LIGHT)
            c.setFont(FONT_REG, 6.5)
            c.drawCentredString(tx, plot_bottom - 3.5 * mm, f"{tick:.0f}")
    else:
        c.line(plot_left, plot_bottom, plot_right, plot_bottom)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            tx = plot_left + frac * plot_w
            c.setStrokeColor(GRAY_LINE)
            c.line(tx, plot_bottom, tx, plot_top)
            c.setFillColor(GRAY_LIGHT)
            c.setFont(FONT_REG, 6.5)
            c.drawCentredString(tx, plot_bottom - 3.5 * mm, f"{frac * abs_max:.0f}")

    row_h = plot_h / n
    for i, (lab, val) in enumerate(zip(labels, values)):
        cy = plot_top - (i + 0.5) * row_h
        col = _sign_color(val) if style == "lollipop" else PRIMARY
        c.setFillColor(PRIMARY)
        c.setFont(FONT_REG, 7.5)
        c.drawRightString(plot_left - 3 * mm, cy - 2, lab)

        if style == "lollipop":
            end_x = zero_x + (val / abs_max) * (plot_w * 0.48)
            c.setStrokeColor(col)
            c.setLineWidth(1.4)
            c.line(zero_x, cy, end_x, cy)
            c.setFillColor(col)
            c.circle(end_x, cy, 3.2, stroke=0, fill=1)
            c.setFillColor(PRIMARY)
            c.setFont(FONT_BOLD, 8)
            offset = 5 if val >= 0 else -5
            anchor = end_x + offset
            if val >= 0:
                c.drawString(anchor, cy - 2.5, f"{_fmt(val)} {unit}")
            else:
                c.drawRightString(anchor, cy - 2.5, f"{_fmt(val)} {unit}")
        elif style == "bullet":
            # Track
            c.setFillColor(GRAY_BG)
            c.roundRect(plot_left, cy - 3.5, plot_w, 7, 2, stroke=0, fill=1)
            # Reference bands
            c.setFillColor(HexColor("#BBF7D0"))
            c.roundRect(plot_left, cy - 3.5, plot_w * min(5.0 / abs_max, 1.0), 7, 2, stroke=0, fill=1)
            bar_w = min(abs(val) / abs_max, 1.0) * plot_w
            c.setFillColor(col if style == "lollipop" else PRIMARY_SOFT)
            c.roundRect(plot_left, cy - 2.2, bar_w, 4.4, 1.5, stroke=0, fill=1)
            c.setFillColor(PRIMARY)
            c.setFont(FONT_BOLD, 8)
            c.drawString(plot_left + bar_w + 3, cy - 2.5, f"{_fmt(abs(val))} {unit}")
        else:  # dumbbell fallback = thin progress
            c.setStrokeColor(GRAY_LINE)
            c.setLineWidth(1.0)
            c.line(plot_left, cy, plot_right, cy)
            end_x = plot_left + min(abs(val) / abs_max, 1.0) * plot_w
            c.setFillColor(PRIMARY)
            c.circle(end_x, cy, 3.5, stroke=0, fill=1)
            c.setFont(FONT_BOLD, 8)
            c.drawString(end_x + 5, cy - 2.5, f"{_fmt(abs(val))} {unit}")


def draw_kidney_scheme(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    left: Any,
    right: Any,
    left_sup: Any,
    right_sup: Any,
    spine: Any,
    estimated: bool = False,
) -> None:
    """Anatomical schematic: body silhouette, spine, kidneys, displacement arrows."""
    c.setFillColor(white)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y - h, w, h, 6, stroke=1, fill=1)

    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 12)
    c.drawString(x + 5 * mm, y - 7 * mm, "Анатомическая схема смещения")
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 8)
    sub = "Схема: положение на спине → прогноз на боку. Стрелки — вектор смещения (мм)."
    if estimated:
        sub += " Центры почек оценены по размерам тела."
    c.drawString(x + 5 * mm, y - 11.5 * mm, sub)

    # Drawing area (frontal / coronal-like schematic)
    ax = x + 8 * mm
    ay = y - h + 10 * mm
    aw = w - 16 * mm
    ah = h - 28 * mm

    cx = ax + aw * 0.42
    cy = ay + ah * 0.48

    # Body silhouette
    c.setStrokeColor(HexColor("#CBD5E1"))
    c.setFillColor(HexColor("#F8FAFC"))
    c.setLineWidth(1.4)
    body = c.beginPath()
    body.moveTo(cx, cy + ah * 0.42)
    body.curveTo(cx - aw * 0.22, cy + ah * 0.38, cx - aw * 0.28, cy + ah * 0.10, cx - aw * 0.26, cy - ah * 0.05)
    body.curveTo(cx - aw * 0.24, cy - ah * 0.28, cx - aw * 0.18, cy - ah * 0.40, cx, cy - ah * 0.42)
    body.curveTo(cx + aw * 0.18, cy - ah * 0.40, cx + aw * 0.24, cy - ah * 0.28, cx + aw * 0.26, cy - ah * 0.05)
    body.curveTo(cx + aw * 0.28, cy + ah * 0.10, cx + aw * 0.22, cy + ah * 0.38, cx, cy + ah * 0.42)
    c.drawPath(body, stroke=1, fill=1)

    # Spine
    c.setStrokeColor(PRIMARY)
    c.setFillColor(PRIMARY)
    c.setLineWidth(2.2)
    c.line(cx, cy - ah * 0.32, cx, cy + ah * 0.28)
    for i in range(6):
        zy = cy - ah * 0.28 + i * (ah * 0.10)
        c.roundRect(cx - 4, zy, 8, 6, 1.5, stroke=0, fill=1)

    # Scale mm → schematic pixels (clamp for readability)
    def _arrow_delta(vec: Any, scale: float = 1.8) -> Tuple[float, float]:
        dx = float(vec[0]) * scale
        dz = float(vec[2]) * scale  # up-down on page vertical
        # Cap arrow length
        mag = math.hypot(dx, dz)
        max_len = min(aw, ah) * 0.18
        if mag > max_len and mag > 0:
            dx *= max_len / mag
            dz *= max_len / mag
        return dx, dz

    # Kidney positions (patient left = viewer's right in anatomical? 
    # Clinical frontal: patient's right on viewer's left.
    # Our "left kidney" sits on patient's left = viewer's right.
    left_x = cx + aw * 0.12
    right_x = cx - aw * 0.12
    kidney_y = cy + ah * 0.02

    def _kidney(kx: float, ky: float, color: Color, label: str) -> None:
        c.setStrokeColor(color)
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.18))
        c.setLineWidth(1.5)
        path = c.beginPath()
        path.moveTo(kx, ky + 18)
        path.curveTo(kx - 14, ky + 10, kx - 16, ky - 8, kx - 6, ky - 18)
        path.curveTo(kx + 2, ky - 22, kx + 12, ky - 12, kx + 14, ky + 2)
        path.curveTo(kx + 14, ky + 14, kx + 8, ky + 20, kx, ky + 18)
        c.drawPath(path, stroke=1, fill=1)
        c.setFillColor(color)
        c.setFont(FONT_BOLD, 8)
        c.drawCentredString(kx, ky - 28, label)

    _kidney(left_x, kidney_y, HexColor("#B91C1C"), "Левая")
    _kidney(right_x, kidney_y, HexColor("#1D4ED8"), "Правая")

    def _draw_arrow(x0: float, y0: float, dx: float, dy: float, color: Color, text: str) -> None:
        x1, y1 = x0 + dx, y0 + dy
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(2.0)
        c.line(x0, y0, x1, y1)
        # Arrowhead
        ang = math.atan2(dy, dx)
        ah_len = 7
        c.line(x1, y1, x1 - ah_len * math.cos(ang - 0.4), y1 - ah_len * math.sin(ang - 0.4))
        c.line(x1, y1, x1 - ah_len * math.cos(ang + 0.4), y1 - ah_len * math.sin(ang + 0.4))
        # Ghost target kidney outline
        c.setStrokeColor(Color(color.red, color.green, color.blue, alpha=0.45))
        c.setDash(2, 2)
        c.setLineWidth(1.0)
        c.ellipse(x1 - 10, y1 - 14, x1 + 10, y1 + 14, stroke=1, fill=0)
        c.setDash()
        c.setFillColor(color)
        c.setFont(FONT_BOLD, 8)
        c.drawString(x1 + 8, y1 + 4, text)

    ldx, ldz = _arrow_delta(left)
    rdx, rdz = _arrow_delta(right)
    _draw_arrow(left_x, kidney_y, ldx, ldz, HexColor("#B91C1C"), f"{vector_norm(left):.1f} мм")
    _draw_arrow(right_x, kidney_y, rdx, rdz, HexColor("#1D4ED8"), f"{vector_norm(right):.1f} мм")

    # Component callouts (right panel)
    panel_x = ax + aw * 0.68
    panel_y = ay + ah * 0.75
    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 9)
    c.drawString(panel_x, panel_y, "Компоненты")

    def _comp_block(side: str, vec: Any, py: float, accent: Color) -> float:
        c.setFillColor(accent)
        c.setFont(FONT_BOLD, 8)
        c.drawString(panel_x, py, side)
        py -= 4.2 * mm
        for lab, val in (
            ("вправо–влево", float(vec[0])),
            ("вперёд–назад", float(vec[1])),
            ("вверх–вниз", float(vec[2])),
        ):
            c.setFillColor(_sign_color(val))
            c.circle(panel_x + 2, py + 2, 2.2, stroke=0, fill=1)
            c.setFillColor(GRAY)
            c.setFont(FONT_REG, 7.5)
            c.drawString(panel_x + 6, py, f"{lab}: {_fmt(val)} мм")
            py -= 3.8 * mm
        return py - 2 * mm

    py = panel_y - 6 * mm
    py = _comp_block("Левая почка", left, py, HexColor("#B91C1C"))
    _comp_block("Правая почка", right, py, HexColor("#1D4ED8"))

    # Legend
    leg_y = ay + 2 * mm
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 7)
    c.drawString(ax, leg_y + 8, "Легенда:")
    c.setStrokeColor(PRIMARY)
    c.setLineWidth(2)
    c.line(ax + 28, leg_y + 10, ax + 42, leg_y + 10)
    c.setFillColor(GRAY)
    c.drawString(ax + 46, leg_y + 8, "позвоночник")
    c.setFillColor(HexColor("#B91C1C"))
    c.circle(ax + 95, leg_y + 10, 3, stroke=0, fill=1)
    c.setFillColor(GRAY)
    c.drawString(ax + 102, leg_y + 8, "левая")
    c.setFillColor(HexColor("#1D4ED8"))
    c.circle(ax + 130, leg_y + 10, 3, stroke=0, fill=1)
    c.setFillColor(GRAY)
    c.drawString(ax + 137, leg_y + 8, "правая")
    c.setDash(2, 2)
    c.setStrokeColor(GRAY)
    c.ellipse(ax + 168, leg_y + 4, ax + 182, leg_y + 16, stroke=1, fill=0)
    c.setDash()
    c.drawString(ax + 186, leg_y + 8, "прогноз на боку")

    # Suppress unused warnings for centers (kept for API / future absolute placement)
    _ = (left_sup, right_sup, spine)


# ---------------------------------------------------------------------------
# Page composers
# ---------------------------------------------------------------------------


def _kidney_side_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    side_label: str,
    vec: Any,
    icon_color: Color,
) -> None:
    total = vector_norm(vec)
    # Outer card
    c.setFillColor(HexColor("#E8EEF4"))
    c.roundRect(x + 1.2, y - 1.2, w, h, 7, stroke=0, fill=1)
    c.setFillColor(white)
    c.setStrokeColor(GRAY_LINE)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 7, stroke=1, fill=1)
    c.setFillColor(icon_color)
    c.roundRect(x, y + 2, 3.2, h - 4, 1.4, stroke=0, fill=1)

    draw_icon(c, "kidney", x + 8, y + h - 22, 16, icon_color)
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 8)
    c.drawString(x + 28, y + h - 14, side_label)
    c.setFillColor(PRIMARY)
    c.setFont(FONT_BOLD, 22)
    c.drawString(x + 10, y + h - 36, _fmt(total))
    tw = pdfmetrics.stringWidth(_fmt(total), FONT_BOLD, 22)
    c.setFillColor(GRAY)
    c.setFont(FONT_REG, 10)
    c.drawString(x + 10 + tw + 4, y + h - 32, "мм суммарно")

    comps = [
        ("вправо–влево", float(vec[0])),
        ("вперёд–назад", float(vec[1])),
        ("вверх–вниз", float(vec[2])),
    ]
    base = y + 8
    row_h = (h - 48) / 3
    for i, (lab, val) in enumerate(comps):
        ry = base + (2 - i) * row_h
        col = _sign_color(val)
        c.setFillColor(GRAY_BG)
        c.roundRect(x + 8, ry, w - 16, row_h - 3, 3, stroke=0, fill=1)
        c.setFillColor(col)
        c.circle(x + 16, ry + (row_h - 3) / 2, 3, stroke=0, fill=1)
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 7.5)
        c.drawString(x + 24, ry + (row_h - 3) / 2 - 2, lab)
        c.setFillColor(PRIMARY)
        c.setFont(FONT_BOLD, 10)
        c.drawRightString(x + w - 12, ry + (row_h - 3) / 2 - 2, f"{_fmt(val)} мм")


def _draw_page1(
    c: canvas.Canvas,
    report: Mapping[str, Any],
    left: Any,
    right: Any,
    checks: Mapping[str, bool],
    page_num: int,
    page_count: int,
) -> None:
    meta = report.get("meta") or {}
    features_block = report.get("features") or {}
    prediction_block = report.get("prediction") or {}
    disclaimer = str(report.get("disclaimer") or "")

    coverage = features_block.get("coverage_pct")
    if coverage is None:
        coverage = (report.get("meta") or {}).get("coverage_pct")

    y = draw_header(
        c,
        "Отчёт о смещении почек",
        subtitle="Прогноз при переводе пациента со спины на бок (планирование доступа)",
        page_num=page_num,
        page_count=page_count,
    )
    y = draw_patient_info(c, MARGIN, y, CONTENT_W, meta)
    y = draw_summary(c, MARGIN, y, CONTENT_W, left, right)

    card_h = 52 * mm
    card_w = (CONTENT_W - COL_GAP) / 2
    _kidney_side_card(
        c, MARGIN, y - card_h, card_w, card_h,
        side_label="Левая почка", vec=left, icon_color=HexColor("#B91C1C"),
    )
    _kidney_side_card(
        c, MARGIN + card_w + COL_GAP, y - card_h, card_w, card_h,
        side_label="Правая почка", vec=right, icon_color=HexColor("#1D4ED8"),
    )
    y = y - card_h - GAP

    y = draw_quality_block(
        c,
        MARGIN,
        y,
        CONTENT_W,
        coverage_pct=_as_float(coverage),
        checks=checks,
        model_id=str(prediction_block.get("model_id") or "—"),
    )

    # Compact notes (geometry checks + API sanity warnings)
    notes = quality_check_messages(checks)
    for warn in list(report.get("prediction_warnings") or [])[:3]:
        text = str(warn).strip()
        if text and text not in notes:
            notes.append(text)
    if notes:
        box_h = 8 * mm + 4.2 * mm * min(len(notes), 3)
        c.setFillColor(CARD_BG)
        c.setStrokeColor(GRAY_LINE)
        c.roundRect(MARGIN, y - box_h, CONTENT_W, box_h, 5, stroke=1, fill=1)
        c.setFillColor(PRIMARY)
        c.setFont(FONT_BOLD, 9)
        c.drawString(MARGIN + 4 * mm, y - 5.5 * mm, "Замечания")
        ty = y - 10 * mm
        c.setFillColor(GRAY)
        c.setFont(FONT_REG, 8)
        for note in notes[:3]:
            for line in _wrap("• " + note, FONT_REG, 8, CONTENT_W - 10 * mm)[:2]:
                c.drawString(MARGIN + 4 * mm, ty, line)
                ty -= 3.8 * mm
    draw_footer(c, disclaimer)


def _draw_page_charts(
    c: canvas.Canvas,
    left: Any,
    right: Any,
    page_num: int,
    page_count: int,
    disclaimer: str,
) -> None:
    y = draw_header(
        c,
        "Компоненты смещения",
        subtitle="Горизонтальные диаграммы (мм). Зелёный — положительное направление, красный — отрицательное.",
        page_num=page_num,
        page_count=page_count,
    )
    labels = [
        "Лев. вправо–влево",
        "Лев. вперёд–назад",
        "Лев. вверх–вниз",
        "Прав. вправо–влево",
        "Прав. вперёд–назад",
        "Прав. вверх–вниз",
    ]
    values = [float(left[0]), float(left[1]), float(left[2]), float(right[0]), float(right[1]), float(right[2])]
    chart1_h = 95 * mm
    draw_bar_chart(
        c,
        MARGIN,
        y,
        CONTENT_W,
        chart1_h,
        title="Смещение по осям",
        description="Lollipop-диаграмма: точка — величина и знак смещения относительно нуля.",
        labels=labels,
        values=values,
        unit="мм",
        style="lollipop",
    )
    y = y - chart1_h - GAP

    norms = [vector_norm(left), vector_norm(right), (vector_norm(left) + vector_norm(right)) / 2.0]
    chart2_h = 58 * mm
    draw_bar_chart(
        c,
        MARGIN,
        y,
        CONTENT_W,
        chart2_h,
        title="Суммарное смещение",
        description="Bullet-диаграмма: длина полосы — модуль вектора. Зелёная зона — ориентир до 5 мм.",
        labels=["Левая почка", "Правая почка", "Среднее"],
        values=norms,
        unit="мм",
        style="bullet",
    )
    draw_footer(c, disclaimer)


def _draw_page_scheme(
    c: canvas.Canvas,
    left: Any,
    right: Any,
    left_sup: Any,
    right_sup: Any,
    spine: Any,
    estimated: bool,
    page_num: int,
    page_count: int,
    disclaimer: str,
) -> None:
    y = draw_header(
        c,
        "Анатомическая схема",
        subtitle="Иллюстративная схема (не масштаб 1:1). Для ориентации векторов смещения.",
        page_num=page_num,
        page_count=page_count,
    )
    draw_kidney_scheme(
        c,
        MARGIN,
        y,
        CONTENT_W,
        y - MARGIN - 4 * mm,
        left=left,
        right=right,
        left_sup=left_sup,
        right_sup=right_sup,
        spine=spine,
        estimated=estimated,
    )
    draw_footer(c, disclaimer)


def _draw_page_service(
    c: canvas.Canvas,
    report: Mapping[str, Any],
    page_num: int,
    page_count: int,
    disclaimer: str,
) -> None:
    meta = report.get("meta") or {}
    prediction_block = report.get("prediction") or {}
    features_block = report.get("features") or {}
    extraction = report.get("extraction") or {}

    y = draw_header(
        c,
        "Служебные сведения",
        subtitle="Технические параметры кейса и модели (для сопровождения).",
        page_num=page_num,
        page_count=page_count,
    )

    missing = features_block.get("missing_features") or []
    missing_txt = ", ".join(str(m) for m in missing[:8]) if missing else "—"
    if len(missing) > 8:
        missing_txt += f" … +{len(missing) - 8}"

    rows: List[Tuple[str, str]] = [
        ("Идентификатор кейса", str(meta.get("case_id") or "—")),
        ("Пациент / метка", str(meta.get("patient_label") or "—")),
        ("Дата обновления", str(meta.get("updated_at") or "—")),
        ("Статус кейса", str(meta.get("status") or "—")),
        ("Файл модели", str(prediction_block.get("model_id") or "—")),
        ("Число признаков модели", str(prediction_block.get("feature_count") or "—")),
        ("Режим обогащения", str(prediction_block.get("enrichment_mode") or "—")),
        ("Полнота признаков, %", str(features_block.get("coverage_pct", meta.get("coverage_pct", "—")))),
        ("Серия DICOM", str(extraction.get("series_description") or "—")),
        ("Число срезов", str(extraction.get("series_slices") or "—")),
        ("Статус извлечения", str(extraction.get("status") or meta.get("status") or "—")),
        ("TotalSegmentator", str(extraction.get("totalsegmentator_status") or "—")),
        ("Недостающие измерения", missing_txt),
        ("Версия схемы отчёта", str(report.get("schema_version") or "ct_workbench_report_v1")),
    ]
    draw_table(c, MARGIN, y, CONTENT_W, rows)
    draw_footer(c, disclaimer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_case_report_pdf(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write multi-page clinical PDF report (ReportLab)."""
    _register_fonts()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_block = report.get("prediction") or {}
    predictions = prediction_block.get("predictions") or {}
    base_features = report.get("base_features") or {}
    disclaimer = str(report.get("disclaimer") or "")

    left, right = predictions_vector(predictions)
    checks = quality_checks(left, right)
    left_sup, right_sup, spine, estimated = estimate_kidney_centers(base_features)

    page_count = 4
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("Отчёт о смещении почек — CT Workbench")
    c.setAuthor("CT Workbench")

    _draw_page1(c, report, left, right, checks, 1, page_count)
    c.showPage()
    _draw_page_charts(c, left, right, 2, page_count, disclaimer)
    c.showPage()
    _draw_page_scheme(
        c, left, right, left_sup, right_sup, spine, estimated, 3, page_count, disclaimer
    )
    c.showPage()
    _draw_page_service(c, report, 4, page_count, disclaimer)
    c.save()
    return output_path


def generate_sample_report(output_path: str | Path) -> Path:
    """Write a realistic demo PDF to ``output_path``."""
    from src.api.cases.report_service import DISCLAIMER

    report: Dict[str, Any] = {
        "schema_version": "ct_workbench_report_v1",
        "disclaimer": DISCLAIMER,
        "meta": {
            "case_id": "demo-clinical-2026",
            "patient_label": "Пациент А., 54 года",
            "updated_at": "2026-07-23T14:30:00+03:00",
            "status": "predicted",
            "coverage_pct": 78.0,
        },
        "extraction": {
            "status": "extracted",
            "totalsegmentator_status": "ok",
            "series_description": "Abdomen CT native",
            "series_slices": 412,
        },
        "base_features": {
            "spine_center_x": 0.0,
            "spine_center_y": 12.0,
            "spine_center_z": 110.0,
            "body_width_mm": 290.0,
            "kidney_left_center_x_rel": -48.0,
            "kidney_left_center_y_rel": 8.0,
            "kidney_left_center_z_rel": 95.0,
            "kidney_right_center_x_rel": 52.0,
            "kidney_right_center_y_rel": 6.0,
            "kidney_right_center_z_rel": 92.0,
        },
        "features": {
            "coverage_pct": 78.0,
            "missing_features": ["kidney_left_volume_cm3", "perinephric_fat_right_mm"],
        },
        "prediction": {
            "predictions": {
                "kidney_left_delta_x": 11.4,
                "kidney_left_delta_y": -3.2,
                "kidney_left_delta_z": 2.8,
                "kidney_right_delta_x": -9.7,
                "kidney_right_delta_y": 2.1,
                "kidney_right_delta_z": -1.6,
            },
            "model_id": "adaptive_ensemble_clinical_honest_reworked.pkl",
            "enrichment_mode": "na_trends",
            "feature_count": 111,
        },
        "manual_overrides": {},
    }
    return build_case_report_pdf(report, output_path)
