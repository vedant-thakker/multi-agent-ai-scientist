"""
Canonical Data Structures for Evidence-Grounded, Topic-Adaptive Manuscript Generation.

Defines the shared topic context and the 6-stage grounded workflow models:
  - ResearchTopicContext (Strict Topic Isolation & Configuration)
  - ResearchBrief (Stage A)
  - EvidenceClaim & EvidenceLedger (Stage B)
  - OutlineSection & PaperOutline (Stage C)
  - ManuscriptSection & CitationEntry (Stage D & E)
  - ReviewIssue & QualityReviewReport (Stage F)
  - CanonicalManuscript (Single source of truth for Markdown, LaTeX, and PDF)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "via", "using", "based", "towards", "toward", "into", "over",
    "under", "across", "through", "study", "analysis", "survey", "review", "paper",
}


def is_canonical_clinical_fl_topic(topic: str) -> bool:
    """
    Return True ONLY when the topic explicitly asks for Federated Learning
    in Healthcare / Clinical / Medical settings.
    """
    t = (topic or "").lower().strip()
    has_fl = "federated learning" in t or ("federated" in t and "learning" in t)
    has_clinical = any(
        w in t for w in ("health", "healthcare", "medical", "clinical", "hospital", "patient", "medicine")
    )
    return bool(has_fl and has_clinical)


def extract_topic_keywords(topic: str) -> List[str]:
    """Extract normalized, domain-meaningful keywords from the user topic."""
    raw_words = re.findall(r"[a-zA-Z0-9\-]+", (topic or "").lower())
    keywords: List[str] = []
    for w in raw_words:
        w_clean = w.strip("-")
        if len(w_clean) >= 2 and w_clean not in STOPWORDS and w_clean not in keywords:
            keywords.append(w_clean)
    return keywords


def make_topic_slug(topic: str) -> str:
    """Create a collision-safe filesystem/collection slug for the topic."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", (topic or "").strip().lower()).strip("_")
    if not clean:
        clean = "research_topic"
    short_hash = hashlib.md5((topic or "").strip().lower().encode("utf-8")).hexdigest()[:6]
    return f"{clean[:42]}_{short_hash}"


@dataclass
class ResearchTopicContext:
    """
    Single authoritative topic and configuration object propagated through
    retrieval, extraction, knowledge graph construction, ideation, review, and writing.
    """

    topic: str
    paper_type: str = "literature_review"  # "literature_review" | "survey_and_synthesis"
    max_papers: int = 20
    max_iterations: int = 3
    prefer_live_api: bool = True
    offline_mode: bool = False
    allow_llm: bool = True
    min_relevance_score: float = 0.15
    topic_slug: str = ""
    keywords: List[str] = field(default_factory=list)
    is_benchmark_clinical_fl: bool = False

    def __post_init__(self) -> None:
        self.topic = (self.topic or "").strip()
        if not self.topic:
            raise ValueError("Research topic must not be empty.")
        self.topic_slug = self.topic_slug or make_topic_slug(self.topic)
        self.keywords = self.keywords or extract_topic_keywords(self.topic)
        self.is_benchmark_clinical_fl = is_canonical_clinical_fl_topic(self.topic)
        if self.offline_mode:
            self.prefer_live_api = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchBrief:
    """Stage A: Structured research brief derived from the user's topic and retrieved corpus."""

    topic: str
    scope_and_definitions: str
    research_questions: List[str]
    subtopics: List[str]
    intended_paper_type: str
    disciplinary_terminology: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    evidence_requirements: str
    analysis_granularity_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceClaim:
    """A single claim in the Stage B Evidence Ledger with explicit provenance and classification."""

    claim_id: str
    statement: str
    classification: str  # "supported" | "synthesis" | "proposed" | "unverified"
    source_paper_ids: List[str] = field(default_factory=list)
    citation_indices: List[int] = field(default_factory=list)
    evidence_excerpt: str = ""
    section_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceLedger:
    """Stage B: Structured mapping of factual claims, syntheses, and proposals to retrieved sources."""

    topic: str
    total_sources: int
    claims: List[EvidenceClaim] = field(default_factory=list)
    evidence_limitations: List[str] = field(default_factory=list)

    def supported_claims(self) -> List[EvidenceClaim]:
        return [c for c in self.claims if c.classification == "supported"]

    def synthesis_claims(self) -> List[EvidenceClaim]:
        return [c for c in self.claims if c.classification == "synthesis"]

    def proposed_claims(self) -> List[EvidenceClaim]:
        return [c for c in self.claims if c.classification == "proposed"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "total_sources": self.total_sources,
            "claims": [c.to_dict() for c in self.claims],
            "evidence_limitations": self.evidence_limitations,
        }


@dataclass
class OutlineSection:
    """A section entry in the Stage C adaptive paper outline."""

    section_id: str
    heading: str
    purpose: str
    target_citation_indices: List[int] = field(default_factory=list)
    subsections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperOutline:
    """Stage C: Topic-adaptive manuscript outline."""

    topic: str
    paper_type: str
    tentative_title: str
    sections: List[OutlineSection] = field(default_factory=list)
    structural_hole_applicable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "paper_type": self.paper_type,
            "tentative_title": self.tentative_title,
            "sections": [s.to_dict() for s in self.sections],
            "structural_hole_applicable": self.structural_hole_applicable,
        }


@dataclass
class CitationEntry:
    """Stage E: Verified bibliographic reference entry tied to a retrieved paper."""

    index: int
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    url: str
    source: str
    analysis_scope: str = "abstract_only"

    def formatted_authors(self, max_authors: int = 4) -> str:
        if not self.authors:
            return "Unknown Author"
        if len(self.authors) > max_authors:
            return ", ".join(self.authors[:3]) + ", et al."
        return ", ".join(self.authors)

    def ieee_string(self) -> str:
        return (
            f"[{self.index}] {self.formatted_authors()}, \"{self.title},\" "
            f"{self.venue}, {self.year}. [Online]. Available: {self.url} ({self.paper_id})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ManuscriptSection:
    """A validated section in the canonical manuscript."""

    section_id: str
    number_label: str
    heading: str
    paragraphs: List[str] = field(default_factory=list)
    subsections: List[Dict[str, Any]] = field(default_factory=list)
    citations_used: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewIssue:
    """A specific quality or grounding issue identified during Stage F review."""

    section_id: str
    category: str  # "topic_relevance" | "factual_grounding" | "citation_accuracy" | "unsupported_claim" | "tone_or_redundancy"
    severity: str  # "error" | "warning"
    description: str
    correction_directive: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReviewReport:
    """Stage F: Quality-control audit result."""

    approved: bool
    iteration: int
    topic_relevance_passed: bool
    citation_verification_passed: bool
    factual_grounding_passed: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    corrected_sections: List[str] = field(default_factory=list)
    summary_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "iteration": self.iteration,
            "topic_relevance_passed": self.topic_relevance_passed,
            "citation_verification_passed": self.citation_verification_passed,
            "factual_grounding_passed": self.factual_grounding_passed,
            "issues": [i.to_dict() for i in self.issues],
            "corrected_sections": self.corrected_sections,
            "summary_notes": self.summary_notes,
        }


@dataclass
class CanonicalManuscript:
    """
    Single canonical manuscript representation shared across Markdown, LaTeX, and PDF renderers.
    Guarantees semantic and bibliographic consistency across all three output formats.
    """

    topic: str
    title: str
    subtitle: str
    authors_line: str
    affiliation_line: str
    paper_type: str
    generation_mode: str  # "openrouter_llm" | "evidence_extractive_fallback" | "canonical_benchmark" | "insufficient_evidence"
    llm_model_used: Optional[str]
    abstract: str
    keywords: List[str]
    research_brief: ResearchBrief
    evidence_ledger: EvidenceLedger
    sections: List[ManuscriptSection]
    comparative_table_rows: List[Dict[str, str]]
    structural_hole_rows: List[Dict[str, str]]
    proposed_directions: List[Dict[str, Any]]
    bibliography: List[CitationEntry] = field(default_factory=list)
    quality_review: Optional[QualityReviewReport] = None
    knowledge_graph_image_path: Optional[str] = None
    limitations_and_scope_notice: str = ""
    kg_summary: Dict[str, Any] = field(default_factory=dict)
    llm_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def citations(self) -> List[CitationEntry]:
        return self.bibliography

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
