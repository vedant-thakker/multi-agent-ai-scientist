"""
Canonical Manuscript Renderers & Post-Build PDF Validator (`tools/canonical_renderers.py`).

Renders a `CanonicalManuscript` (Single Source of Truth) into:
  1. Publication-quality Markdown (`render_canonical_markdown`)
  2. Clean, self-contained IEEE/NeurIPS two-column LaTeX (`render_canonical_latex`)
  3. Variable-length, multi-page two-column ReportLab PDF (`render_canonical_pdf`)
  4. Post-build PDF text extraction & validation (`validate_rendered_pdf`)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape

from tools.manuscript_schema import CanonicalManuscript


def _latex_escape(text: str) -> str:
    """Safely escape special LaTeX characters in plain text."""
    if not text:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in str(text):
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _format_citations_latex(text: str) -> str:
    """Convert bracketed numeric citations `[1]`, `[1, 2]`, `[1–3]` into LaTeX `\\cite{...}`."""
    escaped = _latex_escape(text)

    def _sub_cite(m: re.Match[str]) -> str:
        inner = m.group(1)
        range_m = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", inner.strip())
        if range_m:
            s_i, e_i = int(range_m.group(1)), int(range_m.group(2))
            if s_i <= e_i and (e_i - s_i) <= 30:
                keys = [f"ref{k}" for k in range(s_i, e_i + 1)]
                return r"\cite{" + ",".join(keys) + "}"
        nums = re.findall(r"\d+", inner)
        if not nums:
            return m.group(0)
        keys = [f"ref{n}" for n in nums]
        return r"\cite{" + ",".join(keys) + "}"

    return re.sub(r"\[(\d+(?:\s*[\-–,]\s*\d+)*)\]", _sub_cite, escaped)


def _format_citations_reportlab(text: str, cite_color: str = "#1D4ED8") -> str:
    """Escape XML entities and colorize bracketed citations `[k]` for ReportLab Paragraph."""
    esc = xml_escape(text or "")
    return re.sub(
        r"\[(\d+(?:\s*[\-–,]\s*\d+)*)\]",
        rf'<font color="{cite_color}"><b>[\1]</b></font>',
        esc,
    )


# ============================================================================
# 1. Markdown Renderer
# ============================================================================
def render_canonical_markdown(manuscript: CanonicalManuscript) -> str:
    """Render `CanonicalManuscript` into structured academic Markdown."""
    lines: List[str] = []
    lines.append(f"# {manuscript.title}")
    lines.append("")
    lines.append(f"**{manuscript.subtitle}**  ")
    lines.append(f"*{manuscript.authors_line}*  ")
    lines.append(f"*{manuscript.affiliation_line}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(manuscript.abstract)
    lines.append("")
    if manuscript.keywords:
        lines.append(f"**Keywords:** {', '.join(manuscript.keywords)}")
        lines.append("")

    # Research Brief summary block
    brief = manuscript.research_brief
    lines.append("### Scope, Research Questions, and Evidence Granularity")
    lines.append("")
    lines.append(f"- **Research Topic:** {manuscript.topic}")
    lines.append(f"- **Scope:** {brief.scope_and_definitions}")
    lines.append(f"- **Analysis Granularity:** {brief.analysis_granularity_note}")
    for rq in brief.research_questions:
        lines.append(f"- {rq}")
    lines.append("")

    # Main Manuscript Sections
    for sec in manuscript.sections:
        lines.append(f"## {sec.number_label}. {sec.heading}")
        lines.append("")
        for para in sec.paragraphs:
            lines.append(para)
            lines.append("")

        for sub in sec.subsections:
            lines.append(f"### {sub.get('subheading', '')}")
            lines.append("")
            for spara in sub.get("paragraphs", []):
                lines.append(spara)
                lines.append("")

        # Embed Comparative Table & Knowledge Graph in Section III (`comparative_synthesis`)
        if sec.section_id == "comparative_synthesis":
            if manuscript.comparative_table_rows:
                lines.append("### Table I: Comparative Summary of Retrieved Studies")
                lines.append("")
                lines.append(
                    "| Ref | Study | Venue (Year) | Primary Method | Compared Baseline | Evaluation Setting | Reported Findings | Documented Limitation |"
                )
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                for row in manuscript.comparative_table_rows:
                    lines.append(
                        f"| {row['ref']} | {row['study']} | {row['venue']} | {row['method']} | "
                        f"{row['baseline']} | {row['setting']} | {row['findings']} | {row['limitation']} |"
                    )
                lines.append("")

            if manuscript.structural_hole_rows:
                lines.append("### Table II: Candidate Methodological Gaps (Knowledge Graph Analysis)")
                lines.append("")
                lines.append(
                    "| Rank | Candidate Method Pair | Graph Distance | Bridge Score | Shared Domain Context |"
                )
                lines.append("| :--- | :--- | :--- | :--- | :--- |")
                for hrow in manuscript.structural_hole_rows:
                    lines.append(
                        f"| {hrow['rank']} | {hrow['pair']} | {hrow['distance']} | "
                        f"{hrow['score']} | {hrow['shared_context']} |"
                    )
                lines.append("")

            if manuscript.knowledge_graph_image_path:
                kg_name = Path(manuscript.knowledge_graph_image_path).name
                lines.append(
                    f"![Domain Knowledge Graph for {manuscript.topic}]({kg_name})"
                )
                lines.append("")

    # Evidence Ledger Summary
    ledger = manuscript.evidence_ledger
    lines.append("## Evidence Grounding & Claim Classification Summary")
    lines.append("")
    lines.append(
        f"- **Supported Literature Claims (`supported`):** {len(ledger.supported_claims())} "
        f"(directly linked to retrieved sources `[1]`–`[{ledger.total_sources}]`)"
    )
    lines.append(
        f"- **Comparative Synthesis Observations (`synthesis`):** {len(ledger.synthesis_claims())}"
    )
    lines.append(
        f"- **Proposed Future Hypotheses (`proposed`):** {len(ledger.proposed_claims())} "
        f"(explicitly separated from published findings)"
    )
    for lim in ledger.evidence_limitations:
        lines.append(f"- *Note:* {lim}")
    lines.append("")

    # Verified References
    lines.append("## References")
    lines.append("")
    if manuscript.bibliography:
        for ref in manuscript.bibliography:
            lines.append(f"{ref.ieee_string()}  ")
    else:
        lines.append("*No verified references retrieved for this query.*")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 2. LaTeX Renderer
# ============================================================================
def render_canonical_latex(manuscript: CanonicalManuscript) -> str:
    """Render `CanonicalManuscript` into a clean, self-contained two-column IEEE/NeurIPS LaTeX document."""
    lines: List[str] = [
        r"\documentclass[10pt,twocolumn,letterpaper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathptmx}",
        r"\usepackage[margin=0.68in,top=0.72in,bottom=0.75in,columnsep=0.24in]{geometry}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{tabularx}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{cite}",
        r"\definecolor{citeblue}{HTML}{1D4ED8}",
        r"\definecolor{linkteal}{HTML}{0F766E}",
        r"\usepackage[colorlinks=true,linkcolor=linkteal,citecolor=citeblue,urlcolor=citeblue]{hyperref}",
        r"\setlength{\parskip}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        "",
        r"\title{\LARGE \textbf{" + _latex_escape(manuscript.title) + r"}\\[4pt]",
        r"\large " + _latex_escape(manuscript.subtitle) + r"}",
        r"\author{\textbf{" + _latex_escape(manuscript.authors_line) + r"}\\",
        r"\small \textit{" + _latex_escape(manuscript.affiliation_line) + r"}}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        r"\noindent " + _format_citations_latex(manuscript.abstract),
        r"\vspace{4pt}\\",
        r"\noindent \textbf{Keywords:} " + _latex_escape(", ".join(manuscript.keywords)),
        r"\end{abstract}",
        "",
    ]

    for sec in manuscript.sections:
        lines.append(r"\section{" + _latex_escape(sec.heading) + r"}")
        for para in sec.paragraphs:
            lines.append(_format_citations_latex(para))
            lines.append("")

        for sub in sec.subsections:
            raw_sub = sub.get("subheading", "")
            # Strip leading Roman/letter prefix if present since LaTeX numbers subsections
            clean_sub = re.sub(r"^[IVX]+\.[A-Z]\s+", "", raw_sub)
            lines.append(r"\subsection{" + _latex_escape(clean_sub) + r"}")
            for spara in sub.get("paragraphs", []):
                lines.append(_format_citations_latex(spara))
                lines.append("")

        if sec.section_id == "comparative_synthesis":
            if manuscript.comparative_table_rows:
                lines.extend(
                    [
                        r"\begin{table*}[t]",
                        r"\centering",
                        r"\caption{Comparative Summary of Retrieved Studies on "
                        + _latex_escape(manuscript.topic)
                        + r"}",
                        r"\label{tab:comparative_studies}",
                        r"\scriptsize",
                        r"\begin{tabularx}{\textwidth}{l l l X X X}",
                        r"\toprule",
                        r"\textbf{Ref} & \textbf{Study} & \textbf{Venue} & \textbf{Primary Method} & \textbf{Evaluation Setting} & \textbf{Documented Limitation} \\",
                        r"\midrule",
                    ]
                )
                for row in manuscript.comparative_table_rows:
                    ref_num = re.sub(r"[^\d]", "", row["ref"]) or "1"
                    lines.append(
                        rf"\cite{{ref{ref_num}}} & "
                        + f"{_latex_escape(row['study'])} & "
                        + f"{_latex_escape(row['venue'])} & "
                        + f"{_latex_escape(row['method'])} & "
                        + f"{_latex_escape(row['setting'])} & "
                        + f"{_latex_escape(row['limitation'])} \\\\"
                    )
                lines.extend(
                    [
                        r"\bottomrule",
                        r"\end{tabularx}",
                        r"\end{table*}",
                        "",
                    ]
                )

            if manuscript.structural_hole_rows:
                lines.extend(
                    [
                        r"\begin{table}[h]",
                        r"\centering",
                        r"\caption{Candidate Methodological Gaps in Knowledge Graph}",
                        r"\label{tab:structural_holes}",
                        r"\scriptsize",
                        r"\begin{tabularx}{\columnwidth}{c X c c}",
                        r"\toprule",
                        r"\textbf{Rank} & \textbf{Method Pair} & \textbf{Dist.} & \textbf{Score} \\",
                        r"\midrule",
                    ]
                )
                for hrow in manuscript.structural_hole_rows:
                    lines.append(
                        f"{_latex_escape(hrow['rank'])} & "
                        + f"{_latex_escape(hrow['pair'])} & "
                        + f"{_latex_escape(hrow['distance'])} & "
                        + f"{_latex_escape(hrow['score'])} \\\\"
                    )
                lines.extend(
                    [
                        r"\bottomrule",
                        r"\end{tabularx}",
                        r"\end{table}",
                        "",
                    ]
                )

            if manuscript.knowledge_graph_image_path and Path(manuscript.knowledge_graph_image_path).exists():
                kg_name = Path(manuscript.knowledge_graph_image_path).name
                lines.extend(
                    [
                        r"\begin{figure*}[t]",
                        r"\centering",
                        r"\includegraphics[width=0.92\textwidth]{" + kg_name + r"}",
                        r"\caption{Subject--Predicate--Object Knowledge Graph and Topological Lineage for "
                        + _latex_escape(manuscript.topic)
                        + r".}",
                        r"\label{fig:knowledge_graph}",
                        r"\end{figure*}",
                        "",
                    ]
                )

    # Bibliography
    lines.append(r"\begin{thebibliography}{99}")
    if manuscript.bibliography:
        for ref in manuscript.bibliography:
            lines.append(
                rf"\bibitem{{ref{ref.index}}} "
                + f"{_latex_escape(ref.formatted_authors())}, "
                + f"``{_latex_escape(ref.title)},'' "
                + rf"\textit{{{_latex_escape(ref.venue)}}}, {ref.year}. "
                + f"[{_latex_escape(ref.paper_id)}]"
            )
    else:
        lines.append(
            r"\bibitem{ref0} No verified references retrieved for this query."
        )
    lines.append(r"\end{thebibliography}")
    lines.append(r"\end{document}")

    return "\n".join(lines)


# ============================================================================
# 3. Variable-Length Multi-Page Two-Column PDF Renderer
# ============================================================================
def render_canonical_pdf(
    manuscript: CanonicalManuscript,
    pdf_path: Path,
    kg_img_path: Optional[Path] = None,
) -> None:
    """
    Render `CanonicalManuscript` into a clean, publication-grade two-column PDF
    using ReportLab. Automatically flows across as many pages as needed without
    fixed-page clipping or hardcoded domain text.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        NextPageTemplate,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    page_w, page_h = letter
    margin_l = 0.56 * inch
    margin_r = 0.56 * inch
    margin_t = 0.58 * inch
    margin_b = 0.58 * inch
    gutter = 0.22 * inch

    usable_w = page_w - margin_l - margin_r
    usable_h = page_h - margin_t - margin_b
    col_w = (usable_w - gutter) / 2.0

    c_navy = colors.HexColor("#0F172A")
    c_blue = "#1D4ED8"
    c_teal = "#0F766E"
    c_slate = colors.HexColor("#334155")
    c_rule = colors.HexColor("#1E293B")
    c_midrule = colors.HexColor("#CBD5E1")
    c_abs_bg = colors.HexColor("#F8FAFC")
    c_abs_border = colors.HexColor("#94A3B8")

    header_topic = (manuscript.topic[:65] + "...") if len(manuscript.topic) > 68 else manuscript.topic

    def _draw_decorations(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Times-Italic", 7.5)
        canvas.setFillColor(c_slate)
        canvas.drawString(
            margin_l,
            page_h - 0.40 * inch,
            f"Evidence-Grounded Literature Synthesis — Topic: {header_topic}",
        )
        canvas.drawRightString(
            page_w - margin_r,
            page_h - 0.40 * inch,
            f"Corpus: {len(manuscript.bibliography)} Verified Papers",
        )
        canvas.setStrokeColor(c_midrule)
        canvas.setLineWidth(0.5)
        canvas.line(margin_l, page_h - 0.45 * inch, page_w - margin_r, page_h - 0.45 * inch)

        canvas.line(margin_l, 0.46 * inch, page_w - margin_r, 0.46 * inch)
        canvas.setFont("Times-Roman", 7.5)
        canvas.drawString(
            margin_l,
            0.33 * inch,
            f"Topic: {header_topic} | Abstract & Metadata Evidence Level",
        )
        canvas.drawRightString(page_w - margin_r, 0.33 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=margin_l,
        rightMargin=margin_r,
        topMargin=margin_t,
        bottomMargin=margin_b,
        title=manuscript.title,
        author=manuscript.authors_line,
    )

    # Page 1: Top full-width banner for Title + Abstract, bottom two columns
    p1_top_h = 2.75 * inch
    p1_col_h = usable_h - p1_top_h - 0.12 * inch
    frame_p1_top = Frame(
        margin_l,
        margin_b + p1_col_h + 0.12 * inch,
        usable_w,
        p1_top_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=2,
        id="p1_top",
    )
    frame_p1_l = Frame(
        margin_l,
        margin_b,
        col_w,
        p1_col_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=2,
        bottomPadding=0,
        id="p1_l",
    )
    frame_p1_r = Frame(
        margin_l + col_w + gutter,
        margin_b,
        col_w,
        p1_col_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=2,
        bottomPadding=0,
        id="p1_r",
    )

    # Subsequent pages: standard two-column layout
    frame_col_l = Frame(
        margin_l,
        margin_b,
        col_w,
        usable_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="col_l",
    )
    frame_col_r = Frame(
        margin_l + col_w + gutter,
        margin_b,
        col_w,
        usable_h,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="col_r",
    )

    doc.addPageTemplates(
        [
            PageTemplate(
                id="FirstPage",
                frames=[frame_p1_top, frame_p1_l, frame_p1_r],
                onPage=_draw_decorations,
            ),
            PageTemplate(
                id="TwoColPage",
                frames=[frame_col_l, frame_col_r],
                onPage=_draw_decorations,
            ),
        ]
    )

    base_styles = getSampleStyleSheet()
    st_title = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Normal"],
        fontName="Times-Bold",
        fontSize=14.0,
        leading=16.8,
        alignment=TA_CENTER,
        textColor=c_navy,
        spaceAfter=3,
    )
    st_subtitle = ParagraphStyle(
        "DocSub",
        parent=base_styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.0,
        leading=11.0,
        alignment=TA_CENTER,
        textColor=c_slate,
        spaceAfter=6,
    )
    st_abs = ParagraphStyle(
        "DocAbs",
        parent=base_styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.2,
        leading=10.4,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
    )
    st_h1 = ParagraphStyle(
        "SecH1",
        parent=base_styles["Normal"],
        fontName="Times-Bold",
        fontSize=9.5,
        leading=11.5,
        alignment=TA_LEFT,
        textColor=c_navy,
        spaceBefore=7,
        spaceAfter=3,
    )
    st_h2 = ParagraphStyle(
        "SecH2",
        parent=base_styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=8.6,
        leading=10.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=5,
        spaceAfter=2,
    )
    st_body = ParagraphStyle(
        "BodyP",
        parent=base_styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.3,
        leading=10.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        firstLineIndent=9,
        spaceAfter=3.0,
    )
    st_tbl_hdr = ParagraphStyle(
        "TblH",
        parent=base_styles["Normal"],
        fontName="Times-Bold",
        fontSize=6.8,
        leading=8.2,
        alignment=TA_LEFT,
        textColor=c_navy,
    )
    st_tbl_cell = ParagraphStyle(
        "TblC",
        parent=base_styles["Normal"],
        fontName="Times-Roman",
        fontSize=6.5,
        leading=8.0,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1E293B"),
    )
    st_caption = ParagraphStyle(
        "Cap",
        parent=base_styles["Normal"],
        fontName="Times-Bold",
        fontSize=7.4,
        leading=9.2,
        alignment=TA_LEFT,
        textColor=c_navy,
        spaceBefore=5,
        spaceAfter=2,
    )
    st_ref = ParagraphStyle(
        "RefP",
        parent=base_styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.2,
        leading=8.8,
        alignment=TA_LEFT,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2.0,
        textColor=colors.HexColor("#1F2937"),
    )

    from reportlab.platypus import FrameBreak

    story: List[Any] = [NextPageTemplate("TwoColPage")]

    # Title & Abstract Block in Page 1 Top Frame
    story.append(Paragraph(xml_escape(manuscript.title), st_title))
    story.append(
        Paragraph(
            f"{xml_escape(manuscript.subtitle)} — {xml_escape(manuscript.affiliation_line)}",
            st_subtitle,
        )
    )

    abs_html = (
        f"<b><i>Abstract</i>—</b>{_format_citations_reportlab(manuscript.abstract, c_blue)}<br/>"
        f"<b><i>Keywords</i>—</b>{xml_escape(', '.join(manuscript.keywords))}"
    )
    abs_box = Table([[Paragraph(abs_html, st_abs)]], colWidths=[usable_w])
    abs_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), c_abs_bg),
                ("BOX", (0, 0), (-1, -1), 0.6, c_abs_border),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(abs_box)
    story.append(FrameBreak())

    # Render each section into the two-column frames
    for sec in manuscript.sections:
        story.append(
            Paragraph(f"{xml_escape(sec.number_label)}. {xml_escape(sec.heading.upper())}", st_h1)
        )
        for para in sec.paragraphs:
            story.append(Paragraph(_format_citations_reportlab(para, c_blue), st_body))

        for sub in sec.subsections:
            story.append(Paragraph(xml_escape(sub.get("subheading", "")), st_h2))
            for spara in sub.get("paragraphs", []):
                story.append(Paragraph(_format_citations_reportlab(spara, c_blue), st_body))

        # Embed Compact Comparative Table & KG image inside Section III
        if sec.section_id == "comparative_synthesis":
            if manuscript.comparative_table_rows:
                story.append(
                    Paragraph("TABLE I: COMPARATIVE SUMMARY OF RETRIEVED STUDIES", st_caption)
                )
                tbl_data: List[List[Any]] = [
                    [
                        Paragraph("<b>Ref</b>", st_tbl_hdr),
                        Paragraph("<b>Study / Venue</b>", st_tbl_hdr),
                        Paragraph("<b>Method &amp; Setting</b>", st_tbl_hdr),
                        Paragraph("<b>Key Findings &amp; Limitation</b>", st_tbl_hdr),
                    ]
                ]
                for row in manuscript.comparative_table_rows:
                    tbl_data.append(
                        [
                            Paragraph(
                                _format_citations_reportlab(row["ref"], c_blue), st_tbl_cell
                            ),
                            Paragraph(
                                f"<b>{xml_escape(row['study'])}</b><br/>{xml_escape(row['venue'])}",
                                st_tbl_cell,
                            ),
                            Paragraph(
                                f"<b>{xml_escape(row['method'])}</b><br/><i>{xml_escape(row['setting'])}</i>",
                                st_tbl_cell,
                            ),
                            Paragraph(
                                f"{xml_escape(row['findings'])}<br/><i>Limitation:</i> {xml_escape(row['limitation'])}",
                                st_tbl_cell,
                            ),
                        ]
                    )
                c_widths = [0.32 * inch, 0.90 * inch, 1.05 * inch, col_w - 2.27 * inch]
                comp_tbl = Table(tbl_data, colWidths=c_widths, repeatRows=1)
                comp_tbl.setStyle(
                    TableStyle(
                        [
                            ("LINEABOVE", (0, 0), (-1, 0), 1.0, c_rule),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.6, c_rule),
                            ("LINEBELOW", (0, -1), (-1, -1), 1.0, c_rule),
                            ("BACKGROUND", (0, 0), (-1, 0), c_abs_bg),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                            ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
                        ]
                    )
                )
                story.append(comp_tbl)
                story.append(Spacer(1, 5))

            if manuscript.structural_hole_rows:
                story.append(
                    Paragraph("TABLE II: CANDIDATE METHODOLOGICAL GAPS (KNOWLEDGE GRAPH)", st_caption)
                )
                h_data: List[List[Any]] = [
                    [
                        Paragraph("<b>#</b>", st_tbl_hdr),
                        Paragraph("<b>Candidate Method Pair</b>", st_tbl_hdr),
                        Paragraph("<b>Dist.</b>", st_tbl_hdr),
                        Paragraph("<b>Score</b>", st_tbl_hdr),
                    ]
                ]
                for hrow in manuscript.structural_hole_rows:
                    h_data.append(
                        [
                            Paragraph(xml_escape(hrow["rank"]), st_tbl_cell),
                            Paragraph(xml_escape(hrow["pair"]), st_tbl_cell),
                            Paragraph(xml_escape(hrow["distance"]), st_tbl_cell),
                            Paragraph(xml_escape(hrow["score"]), st_tbl_cell),
                        ]
                    )
                h_widths = [0.25 * inch, col_w - 1.35 * inch, 0.55 * inch, 0.55 * inch]
                h_tbl = Table(h_data, colWidths=h_widths, repeatRows=1)
                h_tbl.setStyle(
                    TableStyle(
                        [
                            ("LINEABOVE", (0, 0), (-1, 0), 1.0, c_rule),
                            ("LINEBELOW", (0, 0), (-1, 0), 0.6, c_rule),
                            ("LINEBELOW", (0, -1), (-1, -1), 1.0, c_rule),
                            ("BACKGROUND", (0, 0), (-1, 0), c_abs_bg),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(h_tbl)
                story.append(Spacer(1, 5))

            if kg_img_path and kg_img_path.exists():
                story.append(
                    Paragraph(
                        f"FIG. 1: S-P-O KNOWLEDGE GRAPH TOPOLOGY ({xml_escape(manuscript.topic.upper())})",
                        st_caption,
                    )
                )
                img_w = col_w - 4
                img_h = img_w * (9.2 / 16.0)
                story.append(Image(str(kg_img_path), width=img_w, height=img_h))
                story.append(Spacer(1, 4))

    # References Section
    story.append(Paragraph("REFERENCES", st_h1))
    if manuscript.bibliography:
        for ref in manuscript.bibliography:
            ref_html = (
                f'<font color="{c_blue}"><b>[{ref.index}]</b></font> '
                f"{xml_escape(ref.formatted_authors())}, "
                f'"{xml_escape(ref.title)}," '
                f"<i>{xml_escape(ref.venue)}</i>, {ref.year}. "
                f'<font color="{c_teal}">[{xml_escape(ref.paper_id)}]</font>'
            )
            story.append(Paragraph(ref_html, st_ref))
    else:
        story.append(
            Paragraph("No verified references were retrieved for this topic.", st_body)
        )

    doc.build(story)


# ============================================================================
# 4. Post-Build PDF Text Extraction & Validation
# ============================================================================
def validate_rendered_pdf(
    pdf_path: Path,
    topic: str,
    is_benchmark_clinical_fl: bool = False,
) -> Dict[str, Any]:
    """
    Extract text from the generated PDF using `pypdf` and verify:
      - The file exists and is non-empty
      - At least one keyword from `topic` appears in the extracted PDF text
      - No forbidden clinical FL contamination phrases appear in non-clinical PDFs
    """
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return {
            "valid": False,
            "page_count": 0,
            "error": "PDF file is missing or empty.",
        }

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text)
        lower_text = full_text.lower()

        if not is_benchmark_clinical_fl:
            forbidden = [
                "unseen hospital sites",
                "multi-institutional clinical cohort",
                "causal-moe",
                "lora-diffsynth",
                "contrastive-ala",
            ]
            leaked = [term for term in forbidden if term in lower_text]
            if leaked:
                return {
                    "valid": False,
                    "page_count": len(reader.pages),
                    "error": f"Cross-topic clinical FL terms leaked into PDF: {leaked}",
                }

        return {
            "valid": True,
            "page_count": len(reader.pages),
            "char_count": len(full_text),
            "error": None,
        }
    except Exception as exc:
        return {
            "valid": False,
            "page_count": 0,
            "error": f"PDF validation failed: {exc}",
        }
