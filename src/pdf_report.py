"""
PDF report generator for lending assessment memos.

Uses ``reportlab`` to produce a professionally formatted, multi-section
lending assessment document that can be downloaded directly from the
Streamlit UI via ``st.download_button``.
"""

import io
from datetime import datetime
from typing import Dict

# Decision → (R, G, B) in [0, 1] range
_DECISION_COLORS = {
    "APPROVE":             (0.04, 0.60, 0.25),
    "CONDITIONAL APPROVE": (0.78, 0.48, 0.00),
    "DECLINE":             (0.80, 0.10, 0.10),
}


def generate_pdf_report(
    report:           Dict[str, str],
    borrower_profile: Dict,
    risk_score:       float,
    risk_label:       str,
) -> bytes:
    """Generate a professional PDF lending assessment memo.

    Parameters
    ----------
    report :
        Structured report dict from :func:`src.agent.run_lending_agent`.
    borrower_profile :
        Raw borrower attribute dict for the appendix table.
    risk_score :
        Model default probability [0, 1].
    risk_label :
        Risk classification string.

    Returns
    -------
    bytes
        Raw PDF bytes (pass directly to ``st.download_button``).
    """
    from reportlab.lib              import colors
    from reportlab.lib.pagesizes   import letter
    from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units       import inch
    from reportlab.platypus        import (
        HRFlowable, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=inch,
        rightMargin=inch,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def _add(name, **kw):
        styles.add(ParagraphStyle(name, parent=styles["Normal"], **kw))

    _add("ReportTitle",   fontSize=18, spaceAfter=4,  textColor=colors.HexColor("#1e3a5f"),
         fontName="Helvetica-Bold")
    _add("SubTitle",      fontSize=10, spaceAfter=12, textColor=colors.grey)
    _add("SectionHead",   fontSize=12, spaceBefore=14, spaceAfter=5,
         textColor=colors.HexColor("#1e3a5f"), fontName="Helvetica-Bold")
    _add("BodyText",      fontSize=10, leading=14, spaceAfter=8)
    _add("DisclaimerTxt", fontSize=8,  leading=12, spaceAfter=6,
         textColor=colors.grey, fontName="Helvetica-Oblique")

    # ── Decision colour ────────────────────────────────────────────────────────
    decision = report.get("decision", "DECLINE").strip().upper()
    rgb      = _DECISION_COLORS.get(decision, (0.5, 0.5, 0.5))
    dec_col  = colors.Color(*rgb)

    # ── Story ─────────────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("LENDING ASSESSMENT REPORT", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')} | CONFIDENTIAL",
        styles["SubTitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a5f")))
    story.append(Spacer(1, 0.15 * inch))

    # Summary banner table
    banner_data = [
        ["Risk Classification", "Default Probability", "Recommended Decision"],
        [risk_label, f"{risk_score * 100:.1f}%", decision],
    ]
    banner_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f4f8")]),
        ("FONTNAME",      (2, 1), (2, 1),   "Helvetica-Bold"),
        ("TEXTCOLOR",     (2, 1), (2, 1),   dec_col),
        ("FONTSIZE",      (2, 1), (2, 1),   13),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#c0cfe0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])
    story.append(Table(
        banner_data,
        colWidths=[2.1 * inch, 2.1 * inch, 2.3 * inch],
        style=banner_style,
    ))
    story.append(Spacer(1, 0.2 * inch))

    # Narrative sections
    sections = [
        ("1. Borrower Profile Summary",   "profile_summary"),
        ("2. Credit Risk Analysis",        "risk_analysis"),
        ("3. Decision Rationale",          "decision_rationale"),
        ("4. Risk Mitigation Suggestions", "risk_mitigation"),
        ("5. References & Sources",        "references"),
    ]
    for title, key in sections:
        story.append(Paragraph(title, styles["SectionHead"]))
        story.append(Paragraph(report.get(key, "N/A"), styles["BodyText"]))

    # Appendix: raw borrower data
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("6. Appendix — Raw Borrower Inputs", styles["SectionHead"]))
    bp_rows = [["Field", "Value"]] + [
        [k.replace("_", " ").title(), str(v)]
        for k, v in borrower_profile.items()
    ]
    bp_style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#e8eef4")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d8e0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    story.append(Table(bp_rows, colWidths=[2.5 * inch, 4 * inch], style=bp_style))

    # Disclaimer footer
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "⚠  DISCLAIMER: " + report.get("disclaimer", ""),
        styles["DisclaimerTxt"],
    ))

    doc.build(story)
    return buf.getvalue()
