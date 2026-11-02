"""
6-Stage Grounded Manuscript Workflow (`tools/manuscript_workflow.py`).

Implements the end-to-end evidence-grounded synthesis pipeline:
  - Stage A: Research Brief (`build_research_brief`)
  - Stage B: Evidence Ledger (`build_evidence_ledger`)
  - Stage C: Adaptive Paper Outline (`build_paper_outline`)
  - Stage D: Grounded Section-by-Section Writing (`generate_manuscript_sections`)
  - Stage E: Citation Verification (`verify_manuscript_citations`)
  - Stage F: Academic Quality Review & Repair (`run_quality_review_and_repair`)

Produces a `CanonicalManuscript` object that serves as the single source of truth
for Markdown, LaTeX, and PDF rendering.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agents.critic import CriticAgent
from agents.investigator import ResearchHypothesis
from tools.arxiv_search import AcademicPaper
from tools.llm_client import (
    LLMGenerationError,
    OpenRouterClient,
    synthesize_scholarly_prose,
    verify_and_sanitize_citations_in_text,
)
from tools.manuscript_schema import (
    CanonicalManuscript,
    CitationEntry,
    EvidenceClaim,
    EvidenceLedger,
    ManuscriptSection,
    OutlineSection,
    PaperOutline,
    QualityReviewReport,
    ResearchBrief,
    ResearchTopicContext,
)


def _extract_citation_numbers(text: str, max_idx: int) -> List[int]:
    """Extract sorted unique valid citation indices `[k]` from text."""
    found: set[int] = set()
    for match in re.finditer(r"\[(\d+(?:\s*[\-–,]\s*\d+)*)\]", text or ""):
        inner = match.group(1)
        range_m = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", inner.strip())
        if range_m:
            s_i, e_i = int(range_m.group(1)), int(range_m.group(2))
            for k in range(max(1, s_i), min(max_idx, e_i) + 1):
                found.add(k)
        else:
            for num_str in re.findall(r"\d+", inner):
                k = int(num_str)
                if 1 <= k <= max_idx:
                    found.add(k)
    return sorted(found)


# ============================================================================
# STAGE A: Research Brief
# ============================================================================
def build_research_brief(
    context: ResearchTopicContext,
    papers: List[AcademicPaper],
) -> ResearchBrief:
    """
    Construct Stage A `ResearchBrief` defining the exact scope, key research questions,
    subtopics, terminology, and inclusion/exclusion criteria grounded in `context` and `papers`.
    """
    topic = context.topic
    subtopics: List[str] = []
    methods_seen: List[str] = []
    for p in papers:
        if p.category and p.category not in subtopics:
            subtopics.append(p.category)
        if p.primary_method and p.primary_method not in methods_seen:
            methods_seen.append(p.primary_method)

    if not subtopics:
        subtopics = [f"Core Methodologies in {topic}", f"Evaluation & Open Challenges in {topic}"]

    terminology = list(dict.fromkeys(context.keywords + [m for m in methods_seen[:8]]))

    rqs = [
        f"RQ1: What are the primary algorithmic and methodological paradigms established across retrieved literature in {topic}?",
        f"RQ2: How do existing approaches in {topic} compare with respect to evaluation settings, baselines, and documented limitations?",
        f"RQ3: Which under-explored methodological intersections and future research directions emerge from a structured synthesis of {topic}?",
    ]

    return ResearchBrief(
        topic=topic,
        scope_and_definitions=(
            f"Systematic literature review and methodological synthesis of {topic}, "
            f"examining {len(papers)} retrieved peer-reviewed and preprint studies."
        ),
        research_questions=rqs,
        subtopics=subtopics,
        intended_paper_type=context.paper_type,
        disciplinary_terminology=terminology[:12],
        inclusion_criteria=[
            f"Primary research or authoritative survey directly addressing {topic}.",
            "Verifiable bibliographic metadata (title, authors, year, abstract, identifier).",
        ],
        exclusion_criteria=[
            f"Studies from unrelated domains without explicit relevance to {topic}.",
            "Unverified citations or claims lacking support in retrieved source abstracts.",
        ],
        evidence_requirements=(
            "Every factual claim regarding prior work must map to a retrieved source [1]–[N]. "
            "Comparative insights must be labeled as synthesis, and novel research directions "
            "must be explicitly framed as proposed hypotheses."
        ),
        analysis_granularity_note=(
            "Evidence extraction and comparative synthesis are grounded in retrieved paper "
            "abstracts and verified bibliographic metadata; full-text experimental appendices "
            "were not directly inspected unless stated in the abstract."
        ),
    )


# ============================================================================
# STAGE B: Evidence Ledger
# ============================================================================
def build_evidence_ledger(
    context: ResearchTopicContext,
    papers: List[AcademicPaper],
    miner_report: Dict[str, Any],
    hypotheses: List[ResearchHypothesis],
) -> EvidenceLedger:
    """
    Construct Stage B `EvidenceLedger` linking every claim to retrieved paper IDs
    and classifying each item as:
      - `supported`: directly supported by retrieved papers
      - `synthesis`: reasonable comparative synthesis across retrieved papers
      - `proposed`: novel hypothesis or future work proposed by this paper
      - `unverified`: excluded from factual presentation
    """
    claims: List[EvidenceClaim] = []

    # 1. Supported claims (1 per retrieved paper, directly grounded in its metadata/abstract)
    for idx, p in enumerate(papers, start=1):
        metrics_summary = "; ".join(f"{k}: {v}" for k, v in list(p.metrics.items())[:2])
        ds_summary = ", ".join(p.datasets[:2]) if p.datasets else "domain benchmarks"
        stmt = (
            f"{p.first_author_citation()} [{idx}] introduced {p.primary_method} "
            f"(comparing against {p.improves_upon}) in the context of {ds_summary}, "
            f"reporting {metrics_summary} while noting limitations regarding {p.limitations}."
        )
        claims.append(
            EvidenceClaim(
                claim_id=f"SUP-{idx:02d}",
                statement=stmt,
                classification="supported",
                source_paper_ids=[p.paper_id],
                citation_indices=[idx],
                evidence_excerpt=(
                    p.full_text_excerpt[:220] if p.full_text_excerpt else p.abstract[:220]
                ),
                section_target="thematic_literature_analysis",
            )
        )

    # 2. Synthesis claims (cross-paper thematic and comparative observations)
    cat_groups: Dict[str, List[Tuple[int, AcademicPaper]]] = {}
    for idx, p in enumerate(papers, start=1):
        cat_groups.setdefault(p.category or f"Core Approaches in {context.topic}", []).append((idx, p))

    for c_idx, (cat_name, group) in enumerate(cat_groups.items(), start=1):
        cites = [i for i, _ in group]
        p_ids = [p.paper_id for _, p in group]
        methods_list = ", ".join(p.primary_method for _, p in group[:4])
        cite_str = ", ".join(f"[{i}]" for i in cites)
        claims.append(
            EvidenceClaim(
                claim_id=f"SYN-{c_idx:02d}",
                statement=(
                    f"Comparative synthesis across {cat_name} ({cite_str}) indicates that "
                    f"methods such as {methods_list} address core objectives in {context.topic} "
                    f"but face complementary trade-offs in scalability, generalization, and assumptions."
                ),
                classification="synthesis",
                source_paper_ids=p_ids,
                citation_indices=cites,
                evidence_excerpt=f"Synthesized across {len(group)} retrieved studies in theme '{cat_name}'.",
                section_target="comparative_synthesis",
            )
        )

    # 3. Proposed claims (from Investigator hypotheses, strictly labeled as proposed)
    for h_idx, hyp in enumerate(hypotheses, start=1):
        claims.append(
            EvidenceClaim(
                claim_id=f"PROP-{h_idx:02d}",
                statement=(
                    f"Proposed Research Direction ({hyp.short_name}): We hypothesize that integrating "
                    f"{hyp.concept_a} with {hyp.concept_b} can address complementary limitations "
                    f"identified in the literature. Proposed evaluation protocol: {hyp.experimental_setup}"
                ),
                classification="proposed",
                source_paper_ids=[],
                citation_indices=[],
                evidence_excerpt="Formulated as a falsifiable future research direction requiring empirical validation.",
                section_target="proposed_research_directions",
            )
        )

    limitations = [
        "All literature claims are extracted from retrieved paper abstracts and bibliographic metadata (ArXiv / Semantic Scholar).",
        "Quantitative figures reported in individual abstracts reflect the original authors' experimental protocols and are not directly re-benchmarked in this review.",
    ]
    if len(papers) < 5:
        limitations.append(
            f"The retrieved corpus contains {len(papers)} verified studies; conclusions are scoped strictly to this evidence base."
        )

    return EvidenceLedger(
        topic=context.topic,
        total_sources=len(papers),
        claims=claims,
        evidence_limitations=limitations,
    )


# ============================================================================
# STAGE C: Adaptive Paper Outline
# ============================================================================
def build_paper_outline(
    context: ResearchTopicContext,
    brief: ResearchBrief,
    papers: List[AcademicPaper],
    miner_report: Dict[str, Any],
    hypotheses: List[ResearchHypothesis],
    scholarly_title: str,
) -> PaperOutline:
    """
    Construct Stage C `PaperOutline` adapted to the topic, retrieved themes, and
    whether structural-hole graph analysis is applicable.
    """
    all_indices = list(range(1, len(papers) + 1))
    g_sum = miner_report.get("graph_summary", {})
    hole_applicable = bool(
        g_sum.get("structural_hole_applicable", True) and miner_report.get("structural_holes")
    )

    cat_names: List[str] = []
    for p in papers:
        c = p.category or f"Methodological Foundations in {context.topic}"
        if c not in cat_names:
            cat_names.append(c)

    sections: List[OutlineSection] = [
        OutlineSection(
            section_id="introduction",
            heading=f"Introduction and Scope: {context.topic}",
            purpose=f"Define the scope, motivation, research questions, and review methodology for {context.topic}.",
            target_citation_indices=all_indices[: min(6, len(all_indices))],
        ),
        OutlineSection(
            section_id="thematic_literature_analysis",
            heading=f"Thematic Literature Review of {context.topic}",
            purpose="Analyze retrieved studies grouped by natural methodological themes emerging from the corpus.",
            target_citation_indices=all_indices,
            subsections=cat_names,
        ),
        OutlineSection(
            section_id="comparative_synthesis",
            heading="Comparative Methodology Synthesis and Knowledge Graph Topology",
            purpose=(
                "Compare retrieved approaches across baselines, evaluation settings, and documented limitations"
                + (", including knowledge-graph structural gap analysis." if hole_applicable else ".")
            ),
            target_citation_indices=all_indices,
        ),
    ]

    if hypotheses:
        sections.append(
            OutlineSection(
                section_id="proposed_research_directions",
                heading=f"Proposed Future Research Directions in {context.topic}",
                purpose=(
                    "Present clearly labeled, falsifiable research hypotheses and methodological "
                    "quality-control criteria without claiming prior experimental validation."
                ),
                target_citation_indices=all_indices[: min(4, len(all_indices))],
                subsections=[h.title for h in hypotheses],
            )
        )

    sections.append(
        OutlineSection(
            section_id="conclusion",
            heading="Limitations, Open Challenges, and Conclusion",
            purpose="Summarize supported findings, acknowledge abstract-level scope constraints, and conclude.",
            target_citation_indices=all_indices,
        )
    )

    return PaperOutline(
        topic=context.topic,
        paper_type=context.paper_type,
        tentative_title=scholarly_title,
        sections=sections,
        structural_hole_applicable=hole_applicable,
    )


# ============================================================================
# STAGE D: Grounded Section-by-Section Writing
# ============================================================================
def generate_manuscript_sections(
    context: ResearchTopicContext,
    brief: ResearchBrief,
    ledger: EvidenceLedger,
    outline: PaperOutline,
    papers: List[AcademicPaper],
    miner_report: Dict[str, Any],
    hypotheses: List[ResearchHypothesis],
    adversarial_history: List[Dict[str, Any]],
    llm_prose: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[ManuscriptSection]]:
    """
    Generate Stage D manuscript abstract and sections grounded strictly in the
    retrieved `papers` and `ledger`. Uses `llm_prose` where available and supplements
    with deterministic, citation-verified analysis of every retrieved paper.
    """
    topic = context.topic
    n_papers = len(papers)
    max_idx = n_papers

    # Group papers by their natural category
    cat_groups: Dict[str, List[Tuple[int, AcademicPaper]]] = {}
    for idx, p in enumerate(papers, start=1):
        cat = p.category or f"Core Approaches in {topic}"
        cat_groups.setdefault(cat, []).append((idx, p))

    # 1. Abstract
    if llm_prose and llm_prose.get("abstract"):
        abstract_text = verify_and_sanitize_citations_in_text(llm_prose["abstract"], max_idx)
    else:
        cite_span = f"[1–{n_papers}]" if n_papers > 1 else ("[1]" if n_papers == 1 else "")
        methods_sample = ", ".join(p.primary_method for p in papers[:4])
        themes_sample = ", ".join(list(cat_groups.keys())[:3])
        abstract_text = (
            f"Research in {topic} has progressed rapidly across diverse methodological paradigms, "
            f"yet systematic comparisons that separate established empirical findings from open "
            f"architectural hypotheses remain essential. In this literature review, we examine "
            f"{n_papers} retrieved studies {cite_span} spanning key themes including {themes_sample}. "
            f"We analyze representative methodologies—such as {methods_sample}—evaluating their "
            f"underlying assumptions, comparative baselines, evaluation benchmarks, and documented "
            f"limitations based on retrieved abstracts and bibliographic metadata. Furthermore, we "
            f"construct a domain knowledge graph to examine methodological lineage and identify "
            f"under-explored intersections across the corpus. Finally, we formulate falsifiable future "
            f"research hypotheses and experimental evaluation protocols designed to address open "
            f"challenges in {topic}."
        )

    sections: List[ManuscriptSection] = []

    # ------------------------------------------------------------------------
    # Section I: Introduction and Scope
    # ------------------------------------------------------------------------
    if llm_prose and llm_prose.get("intro_p1") and llm_prose.get("intro_p2"):
        p1 = verify_and_sanitize_citations_in_text(llm_prose["intro_p1"], max_idx)
        p2 = verify_and_sanitize_citations_in_text(llm_prose["intro_p2"], max_idx)
        p3 = verify_and_sanitize_citations_in_text(
            llm_prose.get("intro_p3")
            or (
                f"To provide a structured and verifiable synthesis of {topic}, this paper "
                f"analyzes {n_papers} retrieved studies [1–{n_papers}] using an evidence ledger "
                f"that explicitly distinguishes published findings from comparative synthesis and "
                f"proposed future research directions. Note that our analysis is grounded in "
                f"retrieved abstracts and verified bibliographic metadata."
            ),
            max_idx,
        )
        intro_paragraphs = [p1, p2, p3]
    else:
        first_cites = ", ".join(f"[{i}]" for i in range(1, min(4, n_papers + 1)))
        later_cites = ", ".join(f"[{i}]" for i in range(min(4, n_papers), n_papers + 1))
        intro_paragraphs = [
            (
                f"The study of {topic} encompasses critical theoretical and engineering challenges "
                f"that have motivated a growing body of primary literature ({first_cites}). "
                f"Foundational and contemporary investigations have introduced specialized methods "
                f"such as {', '.join(p.primary_method for p in papers[:3])}, advancing performance "
                f"across representative benchmarks while revealing trade-offs in scalability, "
                f"robustness, and generalization."
            ),
            (
                f"Across the broader literature ({later_cites}), researchers have investigated "
                f"complementary aspects of {topic}, including {', '.join(list(cat_groups.keys())[:3])}. "
                f"However, individual contributions frequently evaluate distinct baselines and operating "
                f"assumptions, creating a need for a unified comparative synthesis that maps "
                f"methodological lineage and clarifies open technical bottlenecks."
            ),
            (
                f"Scope and Evidence Granularity: This review synthesizes {n_papers} retrieved studies "
                f"addressing {topic}. Each source is indexed in an explicit Evidence Ledger separating "
                f"published findings from comparative synthesis and proposed future research hypotheses. "
                f"Where full-text PDFs are external, analysis is conducted on verified abstracts and "
                f"bibliographic metadata retrieved from ArXiv and Semantic Scholar."
            ),
        ]

    intro_cites = _extract_citation_numbers(" ".join(intro_paragraphs), max_idx)
    sections.append(
        ManuscriptSection(
            section_id="introduction",
            number_label="I",
            heading=f"Introduction and Scope: {topic}",
            paragraphs=intro_paragraphs,
            subsections=[],
            citations_used=intro_cites,
        )
    )

    # ------------------------------------------------------------------------
    # Section II: Thematic Literature Review
    # ------------------------------------------------------------------------
    thematic_intro_paras: List[str] = []
    if llm_prose and llm_prose.get("thematic_synthesis"):
        thematic_intro_paras.append(
            verify_and_sanitize_citations_in_text(llm_prose["thematic_synthesis"], max_idx)
        )
    else:
        thematic_intro_paras.append(
            f"We organize the retrieved literature on {topic} into {len(cat_groups)} thematic "
            f"clusters based on primary methodological focus and evaluation paradigm. Every "
            f"study summarized below corresponds directly to a verified entry in the retrieved "
            f"evidence corpus [1–{n_papers}]."
        )

    thematic_subsections: List[Dict[str, Any]] = []
    for sub_idx, (cat_name, group) in enumerate(cat_groups.items(), start=1):
        sub_paras: List[str] = []
        for cite_i, p in group:
            m_items = "; ".join(f"{k}: {v}" for k, v in list(p.metrics.items())[:2])
            ds_items = ", ".join(p.datasets[:2]) if p.datasets else "standard domain benchmarks"
            excerpt_clean = (p.abstract or "").strip()
            if len(excerpt_clean) > 260:
                excerpt_clean = excerpt_clean[:257].rsplit(" ", 1)[0] + "..."
            sub_paras.append(
                f"{p.first_author_citation()} [{cite_i}] (\"{p.title},\" {p.venue}) examine "
                f"{p.primary_method} in comparison with {p.improves_upon} across {ds_items}. "
                f"Based on the retrieved abstract ({m_items}), the study highlights key findings: "
                f"\"{excerpt_clean}\" Documented technical considerations include {p.limitations}."
            )
        thematic_subsections.append(
            {
                "subheading": f"II.{chr(64 + sub_idx)} {cat_name}",
                "paragraphs": sub_paras,
            }
        )

    all_thematic_text = " ".join(thematic_intro_paras) + " " + " ".join(
        " ".join(s["paragraphs"]) for s in thematic_subsections
    )
    sections.append(
        ManuscriptSection(
            section_id="thematic_literature_analysis",
            number_label="II",
            heading=f"Thematic Literature Review of {topic}",
            paragraphs=thematic_intro_paras,
            subsections=thematic_subsections,
            citations_used=_extract_citation_numbers(all_thematic_text, max_idx),
        )
    )

    # ------------------------------------------------------------------------
    # Section III: Comparative Synthesis & Knowledge Graph Analysis
    # ------------------------------------------------------------------------
    g_sum = miner_report.get("graph_summary", {})
    holes = miner_report.get("structural_holes", [])
    comp_paras: List[str] = []

    if llm_prose and llm_prose.get("cross_pillar_synthesis"):
        comp_paras.append(
            verify_and_sanitize_citations_in_text(llm_prose["cross_pillar_synthesis"], max_idx)
        )
    else:
        comp_paras.append(
            f"Comparing methodologies across the {n_papers} retrieved studies in {topic} reveals "
            f"systematic trade-offs between model expressiveness, computational efficiency, and "
            f"robustness under real-world operating constraints [1–{n_papers}]. While individual "
            f"approaches demonstrate strong performance within their target evaluation settings, "
            f"cross-study comparison highlights complementary strengths across themes."
        )

    top_pr = g_sum.get("top_pagerank", [])
    pr_str = ", ".join(f"{ item['node'] } ({ item['pagerank']:.4f})" for item in top_pr[:4])
    comp_paras.append(
        f"Knowledge Graph Structure: To map explicit relationships across the retrieved corpus, "
        f"we constructed a directed Subject–Predicate–Object (S-P-O) graph comprising "
        f"{g_sum.get('total_nodes', 0)} concept nodes, {g_sum.get('total_edges', 0)} directed edges, "
        f"and {g_sum.get('total_triads', 0)} verified provenance triads. Central concepts by "
        f"PageRank centrality include {pr_str or 'core domain methodologies'}."
    )

    if outline.structural_hole_applicable and holes:
        hole_descriptions = []
        for idx, h in enumerate(holes[:3], start=1):
            hole_descriptions.append(
                f"({idx}) {h['concept_a']} and {h['concept_b']} "
                f"(undirected method distance: {h.get('method_graph_distance', 4)} hops; "
                f"bridge score: {h.get('hole_bridge_score', 0.0):.4f}), which address related "
                f"domain challenges ({', '.join(h.get('shared_datasets', [])[:2])}) without direct "
                f"methodological integration in the retrieved corpus"
            )
        comp_paras.append(
            f"Candidate Research Gaps (Structural Holes): Topological analysis identified "
            f"{len(holes)} candidate methodological gaps: {'; '.join(hole_descriptions)}. "
            f"These structural gaps represent synthesized opportunities for cross-paradigm "
            f"investigation rather than established empirical results."
        )
    else:
        note = g_sum.get("structural_hole_note") or (
            "The retrieved corpus is highly cohesive or compact; we therefore refrain from "
            "forcing an artificial structural-hole narrative."
        )
        comp_paras.append(f"Graph Topology Note: {note}")

    sections.append(
        ManuscriptSection(
            section_id="comparative_synthesis",
            number_label="III",
            heading="Comparative Methodology Synthesis and Knowledge Graph Analysis",
            paragraphs=comp_paras,
            subsections=[],
            citations_used=_extract_citation_numbers(" ".join(comp_paras), max_idx),
        )
    )

    # ------------------------------------------------------------------------
    # Section IV: Proposed Future Research Directions (Strictly Labeled as Proposed)
    # ------------------------------------------------------------------------
    if hypotheses:
        prop_intro = [
            (
                f"Disclaimer on Proposed Research Directions: The directions below are "
                f"formulated by our synthesis pipeline as falsifiable future research hypotheses "
                f"for {topic}. Unlike the published findings reviewed in Section II [1–{n_papers}], "
                f"these proposed integrations have not yet been empirically benchmarked in this study "
                f"and are presented as structured protocols for future investigation."
            )
        ]
        prop_subs: List[Dict[str, Any]] = []
        for h_idx, hyp in enumerate(hypotheses, start=1):
            sub_p = [
                f"Motivation and Gap: {hyp.structural_hole_summary}",
                f"Hypothesis Statement ({hyp.short_name}): {hyp.core_claim}",
                f"Conceptual Formulation: {hyp.mathematical_formulation}",
                f"Proposed Evaluation Protocol and Falsification Criteria: {hyp.experimental_setup}",
                f"Target Outcomes to Test: {hyp.expected_metrics_improvement}",
            ]
            if hyp.refinement_notes:
                sub_p.append(
                    "Methodological Scope Refinements: " + " | ".join(hyp.refinement_notes)
                )
            prop_subs.append(
                {
                    "subheading": f"IV.{chr(64 + h_idx)} {hyp.title}",
                    "paragraphs": sub_p,
                }
            )

        sections.append(
            ManuscriptSection(
                section_id="proposed_research_directions",
                number_label="IV",
                heading=f"Proposed Future Research Directions in {topic}",
                paragraphs=prop_intro,
                subsections=prop_subs,
                citations_used=_extract_citation_numbers(" ".join(prop_intro), max_idx),
            )
        )

    # ------------------------------------------------------------------------
    # Section V (or IV): Limitations and Conclusion
    # ------------------------------------------------------------------------
    concl_num = "V" if hypotheses else "IV"
    concl_paras: List[str] = []
    if llm_prose and llm_prose.get("conclusion"):
        concl_paras.append(
            verify_and_sanitize_citations_in_text(llm_prose["conclusion"], max_idx)
        )
    else:
        concl_paras.append(
            f"This review synthesized {n_papers} retrieved studies on {topic} [1–{n_papers}], "
            f"organizing primary contributions into coherent thematic paradigms and mapping "
            f"their comparative baselines, evaluation settings, and technical limitations."
        )

    concl_paras.append(
        "Methodological and Evidence Scope Limitations: In accordance with strict evidence "
        "grounding, all summaries of prior work in this manuscript are derived from retrieved "
        "abstracts and verified bibliographic records from ArXiv and Semantic Scholar. "
        "Comparative observations and knowledge-graph gap analyses represent cross-study "
        "synthesis, while the research directions in the preceding section are explicitly "
        "framed as falsifiable proposals for future empirical validation."
    )

    sections.append(
        ManuscriptSection(
            section_id="conclusion",
            number_label=concl_num,
            heading="Limitations, Open Challenges, and Conclusion",
            paragraphs=concl_paras,
            subsections=[],
            citations_used=_extract_citation_numbers(" ".join(concl_paras), max_idx),
        )
    )

    return abstract_text, sections


# ============================================================================
# STAGE E: Citation Verification
# ============================================================================
def verify_manuscript_citations(
    abstract: str,
    sections: List[ManuscriptSection],
    papers: List[AcademicPaper],
) -> Tuple[str, List[ManuscriptSection], List[CitationEntry]]:
    """
    Execute Stage E Citation Verification:
      1. Build the authoritative `CitationEntry` manifest `[1]..[N]` from retrieved `papers`.
      2. Sanitize every paragraph and subsection in `abstract` and `sections` so that
         any out-of-range or fabricated citation `[k]` (k < 1 or k > N) is stripped.
      3. Recompute `citations_used` per section.
    """
    max_idx = len(papers)
    bibliography: List[CitationEntry] = []
    for idx, p in enumerate(papers, start=1):
        bibliography.append(
            CitationEntry(
                index=idx,
                paper_id=p.paper_id,
                title=p.title,
                authors=list(p.authors),
                year=int(p.year),
                venue=p.venue or "Preprint",
                url=p.url,
                source=p.source,
                analysis_scope=getattr(p, "analysis_scope", "abstract_only"),
            )
        )

    clean_abstract = verify_and_sanitize_citations_in_text(abstract, max_idx)
    for sec in sections:
        sec.paragraphs = [
            verify_and_sanitize_citations_in_text(p, max_idx) for p in sec.paragraphs
        ]
        for sub in sec.subsections:
            sub["paragraphs"] = [
                verify_and_sanitize_citations_in_text(p, max_idx)
                for p in sub.get("paragraphs", [])
            ]
        combined = " ".join(sec.paragraphs) + " " + " ".join(
            " ".join(sub.get("paragraphs", [])) for sub in sec.subsections
        )
        sec.citations_used = _extract_citation_numbers(combined, max_idx)

    return clean_abstract, sections, bibliography


# ============================================================================
# STAGE F: Academic Quality Review &Targeted Repair
# ============================================================================
def run_quality_review_and_repair(
    context: ResearchTopicContext,
    abstract: str,
    sections: List[ManuscriptSection],
    papers: List[AcademicPaper],
    critic: Optional[CriticAgent] = None,
) -> Tuple[str, List[ManuscriptSection], QualityReviewReport]:
    """
    Execute Stage F Quality Review using `CriticAgent.audit_manuscript_sections`.
    If any cross-topic contamination, out-of-range citation, or unsupported validation
    claim is detected, automatically repair the offending section and re-verify.
    """
    reviewer = critic or CriticAgent()
    sec_dicts = [s.to_dict() for s in sections]
    report = reviewer.audit_manuscript_sections(
        topic=context.topic,
        sections=sec_dicts,
        papers=papers,
        iteration=1,
    )

    if report.approved:
        return abstract, sections, report

    # Perform targeted deterministic repair on any flagged sections
    max_idx = len(papers)
    forbidden_clinical = [
        "unseen hospital sites",
        "multi-institutional clinical cohort",
        "hipaa/gdpr",
        "clinical silos",
        "electronic health records",
        "patient privacy",
    ]
    forbidden_validation = [
        "we experimentally proved",
        "our experiments confirm that",
        "we achieved state-of-the-art accuracy on",
    ]

    corrected_ids: List[str] = []
    for sec in sections:
        modified = False
        new_paras: List[str] = []
        for p in sec.paragraphs:
            p_clean = verify_and_sanitize_citations_in_text(p, max_idx)
            if not context.is_benchmark_clinical_fl:
                for term in forbidden_clinical:
                    if term in p_clean.lower():
                        p_clean = re.sub(re.escape(term), context.topic, p_clean, flags=re.IGNORECASE)
                        modified = True
            for phrase in forbidden_validation:
                if phrase in p_clean.lower():
                    p_clean = re.sub(
                        re.escape(phrase),
                        "proposed future evaluation aims to test whether",
                        p_clean,
                        flags=re.IGNORECASE,
                    )
                    modified = True
            new_paras.append(p_clean)
        sec.paragraphs = new_paras

        for sub in sec.subsections:
            sub_paras: List[str] = []
            for p in sub.get("paragraphs", []):
                p_clean = verify_and_sanitize_citations_in_text(p, max_idx)
                if not context.is_benchmark_clinical_fl:
                    for term in forbidden_clinical:
                        if term in p_clean.lower():
                            p_clean = re.sub(re.escape(term), context.topic, p_clean, flags=re.IGNORECASE)
                            modified = True
                for phrase in forbidden_validation:
                    if phrase in p_clean.lower():
                        p_clean = re.sub(
                            re.escape(phrase),
                            "proposed future evaluation aims to test whether",
                            p_clean,
                            flags=re.IGNORECASE,
                        )
                        modified = True
                sub_paras.append(p_clean)
            sub["paragraphs"] = sub_paras

        if modified:
            corrected_ids.append(sec.section_id)

    # Re-audit after repair
    report_2 = reviewer.audit_manuscript_sections(
        topic=context.topic,
        sections=[s.to_dict() for s in sections],
        papers=papers,
        iteration=2,
    )
    report_2.corrected_sections = corrected_ids
    return abstract, sections, report_2


# ============================================================================
# Top-Level Orchestrator: Assemble CanonicalManuscript
# ============================================================================
def assemble_canonical_manuscript(
    context: ResearchTopicContext,
    papers: List[AcademicPaper],
    miner_report: Dict[str, Any],
    hypotheses: List[ResearchHypothesis],
    adversarial_history: List[Dict[str, Any]],
    kg_image_path: Optional[str] = None,
    llm_client: Optional[OpenRouterClient] = None,
    raise_on_llm_error: bool = False,
) -> CanonicalManuscript:
    """
    Execute Stages A through F to produce a verified `CanonicalManuscript`.
    Handles insufficient evidence cleanly when `papers` is empty.
    """
    topic = context.topic

    # Handle insufficient evidence transparently without fabricating papers or prose
    if not papers:
        brief = build_research_brief(context, [])
        ledger = EvidenceLedger(
            topic=topic,
            total_sources=0,
            claims=[],
            evidence_limitations=[
                f"No relevant academic papers passed relevance verification for topic '{topic}'.",
                "In accordance with anti-hallucination policy, no synthetic references or speculative claims were generated.",
            ],
        )
        insuff_section = ManuscriptSection(
            section_id="insufficient_evidence",
            number_label="I",
            heading=f"Insufficient Retrieved Evidence Report: {topic}",
            paragraphs=[
                (
                    f"An automated literature search across ArXiv and Semantic Scholar for the "
                    f"research topic \"{topic}\" returned zero verified studies meeting the minimum "
                    f"topical relevance threshold ({context.min_relevance_score:.2f})."
                ),
                (
                    "In strict compliance with scientific integrity and evidence-grounding rules, "
                    "this pipeline refuses to manufacture substitute papers, synthetic citations, "
                    "or unverified domain claims when primary literature cannot be retrieved. "
                    "Please refine the query terminology, broaden the search keywords, or run with "
                    "live network access enabled."
                ),
            ],
            subsections=[],
            citations_used=[],
        )
        q_report = QualityReviewReport(
            approved=True,
            iteration=1,
            topic_relevance_passed=True,
            citation_verification_passed=True,
            factual_grounding_passed=True,
            issues=[],
            corrected_sections=[],
            summary_notes=["Insufficient evidence reported transparently without fabrication."],
        )
        return CanonicalManuscript(
            topic=topic,
            title=f"Insufficient Literature Evidence Report: {topic}",
            subtitle="Automated Retrieval & Grounding Diagnostic",
            authors_line="Multi-Agent AI Research Synthesis Pipeline",
            affiliation_line="Evidence-Grounded Automated Literature Analysis",
            paper_type="insufficient_evidence_report",
            generation_mode="insufficient_evidence",
            llm_model_used=None,
            abstract=(
                f"No verified academic papers matching the topic \"{topic}\" were retrieved "
                f"from configured scholarly databases. To prevent hallucination or cross-topic "
                f"contamination, manuscript generation was halted and this diagnostic report was produced."
            ),
            keywords=context.keywords or [topic],
            research_brief=brief,
            evidence_ledger=ledger,
            sections=[insuff_section],
            comparative_table_rows=[],
            structural_hole_rows=[],
            proposed_directions=[],
            bibliography=[],
            quality_review=q_report,
            knowledge_graph_image_path=None,
            llm_diagnostics=[],
        )

    # Attempt OpenRouter synthesis when allowed
    llm_prose: Optional[Dict[str, Any]] = None
    llm_diagnostics: List[Dict[str, Any]] = []
    llm_model_used: Optional[str] = None
    generation_mode = "evidence_extractive_fallback"

    if context.allow_llm and not context.is_benchmark_clinical_fl:
        client = llm_client or OpenRouterClient()
        if client.is_configured():
            try:
                llm_prose = synthesize_scholarly_prose(
                    topic=topic,
                    papers=papers,
                    miner_report=miner_report,
                    hypotheses=hypotheses,
                    adversarial_history=adversarial_history,
                    raise_on_error=raise_on_llm_error,
                    client=client,
                )
                llm_diagnostics = list(client.diagnostics)
                if llm_prose:
                    generation_mode = "openrouter_llm"
                    llm_model_used = client.model
            except LLMGenerationError:
                llm_diagnostics = list(client.diagnostics)
                if raise_on_llm_error:
                    raise
                llm_prose = None

    # Resolve scholarly title
    if llm_prose and llm_prose.get("scholarly_title"):
        candidate_title = llm_prose["scholarly_title"].strip()
        # Ensure user topic keywords appear in the title
        if any(kw in candidate_title.lower() for kw in context.keywords[:2]) or topic.lower() in candidate_title.lower():
            scholarly_title = candidate_title
        else:
            scholarly_title = f"{topic}: {candidate_title}"
    else:
        if ":" in topic or len(topic) >= 50:
            scholarly_title = topic
        else:
            scholarly_title = (
                f"{topic}: A Systematic Literature Review, Comparative Methodology Synthesis, "
                f"and Future Research Directions"
            )

    # Stage A: Research Brief
    brief = build_research_brief(context, papers)

    # Stage B: Evidence Ledger
    ledger = build_evidence_ledger(context, papers, miner_report, hypotheses)

    # Stage C: Adaptive Paper Outline
    outline = build_paper_outline(
        context=context,
        brief=brief,
        papers=papers,
        miner_report=miner_report,
        hypotheses=hypotheses,
        scholarly_title=scholarly_title,
    )

    # Stage D: Grounded Section Writing
    abstract_raw, sections_raw = generate_manuscript_sections(
        context=context,
        brief=brief,
        ledger=ledger,
        outline=outline,
        papers=papers,
        miner_report=miner_report,
        hypotheses=hypotheses,
        adversarial_history=adversarial_history,
        llm_prose=llm_prose,
    )

    # Stage E: Citation Verification
    abstract_verified, sections_verified, bibliography = verify_manuscript_citations(
        abstract=abstract_raw,
        sections=sections_raw,
        papers=papers,
    )

    # Stage F: Academic Quality Review & Repair
    abstract_final, sections_final, quality_report = run_quality_review_and_repair(
        context=context,
        abstract=abstract_verified,
        sections=sections_verified,
        papers=papers,
    )

    # Build comparative table rows directly from retrieved papers
    comp_rows: List[Dict[str, str]] = []
    for idx, p in enumerate(papers, start=1):
        m_str = "; ".join(f"{k}: {v}" for k, v in list(p.metrics.items())[:2])
        ds_str = ", ".join(p.datasets[:2]) if p.datasets else "Domain Benchmark"
        comp_rows.append(
            {
                "ref": f"[{idx}]",
                "study": p.first_author_citation(),
                "venue": f"{p.venue} ({p.year})",
                "method": p.primary_method,
                "baseline": p.improves_upon,
                "setting": ds_str,
                "findings": m_str,
                "limitation": p.limitations,
            }
        )

    # Build structural hole rows
    hole_rows: List[Dict[str, str]] = []
    for idx, h in enumerate(miner_report.get("structural_holes", []), start=1):
        hole_rows.append(
            {
                "rank": str(idx),
                "pair": f"{h['concept_a']} <-> {h['concept_b']}",
                "distance": f"{h.get('method_graph_distance', 4)} hops",
                "score": f"{h.get('hole_bridge_score', 0.0):.4f}",
                "shared_context": ", ".join(h.get("shared_datasets", [])[:2]),
            }
        )

    # Build proposed direction summaries
    prop_dirs = [h.to_dict() for h in hypotheses]

    kw_list = list(brief.disciplinary_terminology[:6])
    if llm_prose and llm_prose.get("index_terms"):
        parsed_kws = [k.strip() for k in str(llm_prose["index_terms"]).split(",") if k.strip()]
        if parsed_kws:
            kw_list = parsed_kws[:6]

    return CanonicalManuscript(
        topic=topic,
        title=scholarly_title,
        subtitle=f"Evidence-Grounded Literature Review and Synthesis ({len(papers)} Verified Studies)",
        authors_line="Automated Scientific Literature Review & Synthesis Pipeline",
        affiliation_line=f"Topic Scope: {topic} | Analysis Level: Abstract & Bibliographic Metadata",
        paper_type=context.paper_type,
        generation_mode=generation_mode,
        llm_model_used=llm_model_used,
        abstract=abstract_final,
        keywords=kw_list,
        research_brief=brief,
        evidence_ledger=ledger,
        sections=sections_final,
        comparative_table_rows=comp_rows,
        structural_hole_rows=hole_rows,
        proposed_directions=prop_dirs,
        bibliography=bibliography,
        quality_review=quality_report,
        knowledge_graph_image_path=kg_image_path,
        llm_diagnostics=llm_diagnostics,
    )
