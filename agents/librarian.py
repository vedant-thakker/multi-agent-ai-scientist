"""
Agent 1: The Retrieval & Ingestion Agent ("The Librarian").

Responsibilities:
- Accepts a user research topic / `ResearchTopicContext`.
- Queries ArXiv API and Semantic Scholar API (or verified domain fixtures in offline mode).
- Validates paper relevance against the topic and rejects unrelated results.
- Never fabricates papers, authors, venues, or clinical FL filler for arbitrary topics.
- Clears previous topic chunks in ChromaDB and embeds the verified corpus with full provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.arxiv_search import (
    AcademicPaper,
    fetch_academic_literature,
    verify_corpus_relevance,
)
from tools.manuscript_schema import ResearchTopicContext, is_canonical_clinical_fl_topic
from tools.rag_engine import ChromaRAGEngine


def is_clinical_fl_topic(topic: str) -> bool:
    """Check if the given topic refers to the benchmark clinical Federated Learning domain."""
    return is_canonical_clinical_fl_topic(topic)


class LibrarianAgent:
    """The Retrieval & Ingestion Agent (The Librarian)."""

    ROLE = "Senior Scientific Literature Librarian & Corpus Curator"
    GOAL = (
        "Discover, retrieve, deduplicate, relevance-filter, chunk, and index peer-reviewed "
        "and preprint papers from ArXiv and Semantic Scholar into the Vector Database."
    )
    BACKSTORY = (
        "You are an expert computational bibliometrician who curates foundational and "
        "state-of-the-art literature across scientific and engineering disciplines. "
        "You never hallucinate citations, never substitute unrelated domain literature, "
        "and ensure every chunk carries complete paper provenance."
    )
    SYSTEM_PROMPT = (
        "You are THE LIBRARIAN (Retrieval & Ingestion Agent). Given a research topic, "
        "query ArXiv and Semantic Scholar, validate relevance to the exact topic, "
        "preserve verifiable metadata, split documents using RecursiveCharacterTextSplitter, "
        "and index them into a topic-isolated ChromaDB collection."
    )
    DELEGATION_RULE = "Delegates indexed Vector DB chunks and verified paper manifest to The Data Miner (MinerAgent)."

    def __init__(self, rag_engine: ChromaRAGEngine) -> None:
        self.rag_engine = rag_engine

    def build_crewai_agent(self, llm: Optional[Any] = None) -> Any:
        """Return a configured CrewAI Agent instance when `crewai` is installed."""
        try:
            from crewai import Agent

            kwargs: Dict[str, Any] = {
                "role": self.ROLE,
                "goal": self.GOAL,
                "backstory": self.BACKSTORY,
                "allow_delegation": False,
                "verbose": True,
            }
            if llm is not None:
                kwargs["llm"] = llm
            return Agent(**kwargs)
        except Exception:
            return None

    def run(
        self,
        topic: str,
        max_papers: int = 20,
        prefer_live_api: bool = True,
        topic_context: Optional[ResearchTopicContext] = None,
        context: Optional[ResearchTopicContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase 1 (Knowledge Acquisition):
        1. Fetch relevant papers from ArXiv / Semantic Scholar (or verified domain fixtures).
        2. Verify topic relevance and reject off-topic candidates.
        3. Clear stale vector store state and ingest verified documents into `ChromaRAGEngine`.
        """
        ctx = context or topic_context or ResearchTopicContext(
            topic=topic,
            max_papers=max_papers,
            prefer_live_api=prefer_live_api,
            offline_mode=not prefer_live_api,
        )

        raw_papers: List[AcademicPaper] = fetch_academic_literature(
            query=ctx.topic,
            max_papers=max_papers,
            prefer_live_api=prefer_live_api,
            min_relevance_score=ctx.min_relevance_score,
        )

        relevance_audit = verify_corpus_relevance(
            topic=ctx.topic,
            papers=raw_papers,
            min_relevance=ctx.min_relevance_score,
            min_papers=1,
        )
        papers: List[AcademicPaper] = relevance_audit["relevant_papers"][:max_papers]

        # Always clear previous topic's chunks to prevent cross-run contamination
        ingestion_stats = self.rag_engine.ingest_papers(
            papers,
            clear_existing=True,
            topic_slug=ctx.topic_slug,
        )

        status = "ok" if papers else "insufficient_evidence"
        analysis_notice = (
            "All retrieved records were analyzed using accessible title, abstract, and bibliographic metadata "
            "(full-text PDFs were not claimed as read)."
        )

        return {
            "agent": "The Librarian (Retrieval & Ingestion Agent)",
            "topic": ctx.topic,
            "topic_slug": ctx.topic_slug,
            "status": status,
            "is_sufficient": bool(papers),
            "papers_fetched": len(papers),
            "papers": papers,
            "relevance_audit": {
                "is_sufficient": relevance_audit["is_sufficient"],
                "relevant_count": relevance_audit["relevant_count"],
                "rejected_count": relevance_audit["rejected_count"],
                "mean_relevance_score": relevance_audit["mean_relevance_score"],
            },
            "analysis_scope_notice": analysis_notice,
            "ingestion_stats": ingestion_stats,
            "delegated_to": "The Data Miner (Extraction & Graphing Agent)",
        }
