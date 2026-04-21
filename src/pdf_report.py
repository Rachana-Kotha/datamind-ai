"""
DataMind AI — PDF Report Generator
Creates a professional, branded PDF intelligence report using ReportLab.
"""

import io
import os
from datetime import datetime
from typing import List, Dict, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate


# ── Brand colors ──────────────────────────────────────────────────────────────
PURPLE      = HexColor("#4F46E5")
PURPLE_DARK = HexColor("#3730A3")
PURPLE_LITE = HexColor("#EEF2FF")
TEAL        = HexColor("#0F6E56")
AMBER       = HexColor("#B45309")
RED         = HexColor("#B91C1C")
GREEN       = HexColor("#15803D")
GRAY_DARK   = HexColor("#1F2937")
GRAY_MID    = HexColor("#6B7280")
GRAY_LITE   = HexColor("#F9FAFB")
BORDER      = HexColor("#E5E7EB")


def _styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": s("ct", fontName="Helvetica-Bold", fontSize=36,
                         textColor=PURPLE, spaceAfter=8, alignment=TA_CENTER),
        "cover_sub":   s("cs", fontName="Helvetica", fontSize=16,
                         textColor=GRAY_MID, spaceAfter=4, alignment=TA_CENTER),
        "cover_meta":  s("cm", fontName="Helvetica", fontSize=10,
                         textColor=GRAY_MID, spaceAfter=2, alignment=TA_CENTER),
        "section":     s("sec", fontName="Helvetica-Bold", fontSize=18,
                         textColor=PURPLE_DARK, spaceBefore=24, spaceAfter=8),
        "subsection":  s("sub", fontName="Helvetica-Bold", fontSize=13,
                         textColor=GRAY_DARK, spaceBefore=14, spaceAfter=4),
        "body":        s("body", fontName="Helvetica", fontSize=10,
                         textColor=GRAY_DARK, leading=16, spaceAfter=8,
                         alignment=TA_JUSTIFY),
        "kpi_title":   s("kt", fontName="Helvetica-Bold", fontSize=14,
                         textColor=PURPLE_DARK, spaceAfter=3),
        "kpi_story":   s("ks", fontName="Helvetica", fontSize=10,
                         textColor=GRAY_DARK, leading=15, spaceAfter=6,
                         alignment=TA_JUSTIFY),
        "kpi_val":     s("kv", fontName="Helvetica-Bold", fontSize=22,
                         textColor=PURPLE, spaceAfter=2, alignment=TA_CENTER),
        "kpi_lbl":     s("kl", fontName="Helvetica", fontSize=8,
                         textColor=GRAY_MID, alignment=TA_CENTER),
        "tag":         s("tag", fontName="Helvetica-Bold", fontSize=8,
                         textColor=white, backColor=PURPLE),
        "footer":      s("foot", fontName="Helvetica", fontSize=8,
                         textColor=GRAY_MID, alignment=TA_CENTER),
        "caption":     s("cap", fontName="Helvetica-Oblique", fontSize=9,
                         textColor=GRAY_MID, alignment=TA_CENTER, spaceAfter=8),
        "bold_body":   s("bb", fontName="Helvetica-Bold", fontSize=10,
                         textColor=GRAY_DARK, spaceAfter=4),
        "insight":     s("ins", fontName="Helvetica", fontSize=10,
                         textColor=TEAL, leading=15, spaceAfter=6),
        "warning":     s("warn", fontName="Helvetica", fontSize=10,
                         textColor=AMBER, leading=15, spaceAfter=6),
        "risk":        s("risk", fontName="Helvetica", fontSize=10,
                         textColor=RED, leading=15, spaceAfter=6),
    }


def _header_footer(canvas, doc, dataset_name: str, generated_at: str):
    canvas.saveState()
    w, h = letter

    # Header bar
    canvas.setFillColor(PURPLE)
    canvas.rect(0, h - 0.55 * inch, w, 0.55 * inch, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(0.5 * inch, h - 0.35 * inch, "DataMind AI")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - 0.5 * inch, h - 0.35 * inch, f"{dataset_name}  ·  {generated_at}")

    # Footer bar
    canvas.setFillColor(GRAY_LITE)
    canvas.rect(0, 0, w, 0.45 * inch, fill=1, stroke=0)
    canvas.setFillColor(GRAY_MID)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(w / 2, 0.16 * inch,
        f"DataMind AI Intelligence Report  ·  Page {doc.page}  ·  Confidential")
    canvas.restoreState()


def _kpi_card_table(kpi: Dict, styles: Dict) -> Table:
    """Build a single KPI card as a ReportLab Table."""
    col = kpi.get("title", kpi.get("column", "KPI"))
    mean_val = kpi.get("mean", 0)
    trend = kpi.get("trend_direction", "flat")
    trend_pct = kpi.get("trend_pct", 0)
    trend_emoji = "▲" if trend == "up" else "▼" if trend == "down" else "●"
    trend_color = GREEN if trend == "up" else RED if trend == "down" else GRAY_MID

    # Format value
    if abs(mean_val) >= 1_000_000:
        val_str = f"{mean_val/1_000_000:.2f}M"
    elif abs(mean_val) >= 1_000:
        val_str = f"{mean_val/1_000:.2f}K"
    else:
        val_str = f"{mean_val:.3f}"

    corr = kpi.get("target_correlation")
    corr_str = f"r={corr:.2f}" if corr is not None else ""

    inner = [
        [Paragraph(col, styles["kpi_title"])],
        [Paragraph(val_str, styles["kpi_val"])],
        [Paragraph("avg value", styles["kpi_lbl"])],
        [Table([
            [
                Paragraph(f"{trend_emoji} {trend_pct:+.1f}%", ParagraphStyle("tp", fontName="Helvetica-Bold", fontSize=9, textColor=trend_color, alignment=TA_CENTER)),
                Paragraph(corr_str, ParagraphStyle("cp", fontName="Helvetica", fontSize=9, textColor=GRAY_MID, alignment=TA_CENTER)),
                Paragraph(f"n={kpi.get('count', 0):,}", ParagraphStyle("np", fontName="Helvetica", fontSize=9, textColor=GRAY_MID, alignment=TA_CENTER)),
            ]
        ], colWidths=[1.6*inch, 1.6*inch, 1.5*inch],
        style=TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))],
    ]

    card = Table(inner, colWidths=[4.8*inch])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LITE),
        ("ROUNDEDCORNERS", [8]),
        ("BOX", (0, 0), (-1, -1), 0.5, PURPLE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return card


def _leaderboard_table(leaderboard: List[Dict], task_type: str, styles: Dict) -> Table:
    if task_type == "classification":
        headers = ["Rank", "Model", "Accuracy", "F1 Score", "CV Mean", "CV Std"]
        rows = [[
            str(i + 1),
            r.get("model", ""),
            f"{r.get('accuracy', 0):.4f}",
            f"{r.get('f1', 0):.4f}",
            f"{r.get('cv_mean', 0):.4f}",
            f"±{r.get('cv_std', 0):.4f}",
        ] for i, r in enumerate(leaderboard)]
    else:
        headers = ["Rank", "Model", "R²", "RMSE", "CV Mean", "CV Std"]
        rows = [[
            str(i + 1),
            r.get("model", ""),
            f"{r.get('r2', 0):.4f}",
            f"{r.get('rmse', 0):.4f}",
            f"{r.get('cv_mean', 0):.4f}",
            f"±{r.get('cv_std', 0):.4f}",
        ] for i, r in enumerate(leaderboard)]

    data = [headers] + rows
    col_w = [0.5*inch, 2.3*inch, 1*inch, 1*inch, 1*inch, 1*inch]
    t = Table(data, colWidths=col_w)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, GRAY_LITE]),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if rows:
        style.append(("BACKGROUND", (0, 1), (-1, 1), HexColor("#EEF2FF")))
        style.append(("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"))
        style.append(("TEXTCOLOR", (0, 1), (-1, 1), PURPLE_DARK))
    t.setStyle(TableStyle(style))
    return t


def generate_pdf(
    dataset_name: str,
    target_col: str,
    task_type: str,
    narrative: str,
    kpis: List[Dict],
    ml_findings: Dict,
    eda_findings: Dict,
    insights: List[Dict],
    risks: List[Dict],
    debate: List[Dict],
    share_url: Optional[str] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Build the full PDF report and return as bytes (also saves to output_path if given).
    """
    buf = io.BytesIO()
    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    short_date = datetime.now().strftime("%Y-%m-%d")
    st = _styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.65 * inch,
        title=f"DataMind AI — {dataset_name}",
        author="DataMind AI",
        subject=f"Intelligence Report: {dataset_name}",
    )

    story = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("🧠 DataMind AI", st["cover_title"]))
    story.append(Paragraph("Data Intelligence Report", st["cover_sub"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=PURPLE, spaceAfter=20))

    meta = [
        ["Dataset", dataset_name],
        ["Target variable", target_col],
        ["Task type", task_type.title()],
        ["Generated", now],
        ["Best model", ml_findings.get("best_model", "N/A")],
        ["Best score", f"{ml_findings.get('best_score', 0):.4f} ({ml_findings.get('metric_name', 'score')})"],
    ]
    if share_url:
        meta.append(["Report URL", share_url])

    meta_table = Table(
        meta,
        colWidths=[1.8 * inch, 5 * inch],
        style=TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("TEXTCOLOR", (0, 0), (0, -1), PURPLE_DARK),
            ("TEXTCOLOR", (1, 0), (1, -1), GRAY_DARK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, BORDER),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * inch))

    shape = eda_findings.get("shape", {})
    summary_data = [[
        Paragraph(f"{shape.get('rows', 0):,}\nRows", ParagraphStyle("sv", fontName="Helvetica-Bold", fontSize=18, textColor=PURPLE, alignment=TA_CENTER, leading=22)),
        Paragraph(f"{shape.get('cols', 0)}\nColumns", ParagraphStyle("sv2", fontName="Helvetica-Bold", fontSize=18, textColor=PURPLE, alignment=TA_CENTER, leading=22)),
        Paragraph(f"{len(kpis)}\nTop KPIs", ParagraphStyle("sv3", fontName="Helvetica-Bold", fontSize=18, textColor=PURPLE, alignment=TA_CENTER, leading=22)),
        Paragraph(f"{len(insights)}\nInsights", ParagraphStyle("sv4", fontName="Helvetica-Bold", fontSize=18, textColor=PURPLE, alignment=TA_CENTER, leading=22)),
        Paragraph(f"{len(risks)}\nRisks", ParagraphStyle("sv5", fontName="Helvetica-Bold", fontSize=18, textColor=RED if risks else GREEN, alignment=TA_CENTER, leading=22)),
    ]]
    st_table = Table(summary_data, colWidths=[1.35 * inch] * 5)
    st_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PURPLE_LITE),
        ("BOX", (0, 0), (-1, -1), 1, PURPLE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(st_table)
    story.append(PageBreak())

    # ── Data narrative ────────────────────────────────────────────────────────
    story.append(Paragraph("The Data Story", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=12))

    for para in narrative.split("\n\n"):
        clean = para.strip().replace("**", "<b>").replace("**", "</b>")
        clean = _md_to_rl(para.strip())
        if clean:
            story.append(Paragraph(clean, st["body"]))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 0.2 * inch))

    # ── KPIs ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Top 5 KPIs", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=8))
    story.append(Paragraph(
        "These five KPIs were selected by the AI council based on predictive power, business relevance, and data quality.",
        st["caption"]
    ))

    # Two-column KPI cards
    for i in range(0, len(kpis), 2):
        row_cards = kpis[i:i+2]
        cells = []
        for kpi in row_cards:
            card = _kpi_card_table(kpi, st)
            cells.append(card)
        if len(cells) == 1:
            cells.append("")  # padding

        pair_table = Table([cells], colWidths=[3.6 * inch, 3.6 * inch])
        pair_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(pair_table)

    # KPI stories
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("KPI Narratives", st["subsection"]))
    for kpi in kpis:
        story.append(Paragraph(f"● {kpi.get('title', kpi.get('column', 'KPI'))}", st["kpi_title"]))
        story.append(Paragraph(_md_to_rl(kpi.get("story", "")), st["kpi_story"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ── Model leaderboard ────────────────────────────────────────────────────
    story.append(Paragraph("Model Leaderboard", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=8))

    leaderboard = ml_findings.get("leaderboard", [])
    if leaderboard:
        story.append(_leaderboard_table(leaderboard, task_type, st))
        story.append(Spacer(1, 0.1 * inch))
        best = leaderboard[0]
        metric = ml_findings.get("metric_name", "score")
        story.append(Paragraph(
            f"<b>{best.get('model', 'Best model')}</b> achieved the highest {metric} of "
            f"<b>{best.get(metric, best.get('r2', 0)):.4f}</b> with cross-validation "
            f"mean of {best.get('cv_mean', 0):.4f} ± {best.get('cv_std', 0):.4f}.",
            st["body"]
        ))

    # Feature importance
    fi = ml_findings.get("feature_importance")
    if fi and fi.get("features"):
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Feature Importances", st["subsection"]))
        fi_data = [["Feature", "Importance", "Relative"]]
        max_imp = max(fi["importances"]) if fi["importances"] else 1
        for feat, imp in zip(fi["features"][:10], fi["importances"][:10]):
            bar = "█" * max(1, int(imp / max_imp * 20))
            fi_data.append([feat, f"{imp:.4f}", bar])
        fi_table = Table(fi_data, colWidths=[3 * inch, 1.2 * inch, 3 * inch])
        fi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, GRAY_LITE]),
            ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TEXTCOLOR", (2, 1), (2, -1), PURPLE),
        ]))
        story.append(fi_table)

    story.append(PageBreak())

    # ── Insights ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Insights", st["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=8))

    for ins in insights:
        t = ins.get("type", "insight")
        color = {"warning": AMBER, "success": GREEN, "insight": TEAL}.get(t, TEAL)
        icon = {"warning": "⚠", "success": "✓", "insight": "💡"}.get(t, "•")
        story.append(Paragraph(
            f'<font color="{color.hexval()}">{icon} <b>{ins.get("title", "")}</b></font>',
            ParagraphStyle("ihead", fontName="Helvetica-Bold", fontSize=11,
                           textColor=color, spaceAfter=3)
        ))
        story.append(Paragraph(ins.get("detail", ""), st["kpi_story"]))
        story.append(Spacer(1, 6))

    # ── Risks ─────────────────────────────────────────────────────────────────
    if risks:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Risk Audit", st["subsection"]))
        for risk in risks:
            sev = risk.get("severity", "low")
            color = RED if sev in ["high", "critical"] else AMBER if sev == "medium" else GREEN
            story.append(Paragraph(
                f'<font color="{color.hexval()}">[{sev.upper()}] <b>{risk.get("risk", "")}</b></font>',
                ParagraphStyle("rhead", fontName="Helvetica-Bold", fontSize=10,
                               textColor=color, spaceAfter=3)
            ))
            story.append(Paragraph(risk.get("detail", ""), st["kpi_story"]))
            story.append(Spacer(1, 6))

    # ── Agent debate ──────────────────────────────────────────────────────────
    if debate:
        story.append(PageBreak())
        story.append(Paragraph("Agent Council Debate", st["section"]))
        story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=8))
        story.append(Paragraph(
            "The following is a transcript of the AI agents challenging each other's findings.",
            st["caption"]
        ))
        for entry in debate:
            agent = entry.get("agent", "Agent")
            emoji = entry.get("emoji", "🤖")
            message = entry.get("message", "")
            story.append(Paragraph(
                f"<b>{emoji} {agent}:</b>  {message}",
                ParagraphStyle("deb", fontName="Helvetica", fontSize=10,
                               textColor=GRAY_DARK, leading=15, spaceAfter=8,
                               leftIndent=12, borderPadding=(6, 8, 6, 8))
            ))

    # ── Share URL ─────────────────────────────────────────────────────────────
    if share_url:
        story.append(Spacer(1, 0.3 * inch))
        story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=10))
        story.append(Paragraph("Share This Report", st["subsection"]))
        story.append(Paragraph(
            f"Access this report online at: <b>{share_url}</b>",
            ParagraphStyle("url", fontName="Helvetica", fontSize=10,
                           textColor=PURPLE, spaceAfter=4)
        ))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage=lambda c, d: _header_footer(c, d, dataset_name, now),
        onLaterPages=lambda c, d: _header_footer(c, d, dataset_name, now),
    )

    pdf_bytes = buf.getvalue()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    return pdf_bytes


def _md_to_rl(text: str) -> str:
    """Convert basic markdown bold (**text**) to ReportLab XML."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text
