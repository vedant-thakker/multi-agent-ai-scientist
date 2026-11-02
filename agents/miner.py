"""
Agent 2: The Extraction & Graphing Agent ("The Data Miner").

Responsibilities:
- Reads retrieved document chunks from ChromaDB.
- Extracts key methodologies, datasets, baseline metrics, and limitations.
- Identifies Subject-Predicate-Object (S-P-O) relationships
  (e.g., "Model X -> improves_upon -> Model Y") and populates the NetworkX Knowledge Graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.arxiv_search import AcademicPaper
from tools.kg_builder import KnowledgeGraphBuilder
from tools.rag_engine import ChromaRAGEngine


class MinerAgent:
    """The Extraction & Graphing Agent (The Data Miner)."""

    ROLE = "Knowledge Graph Architect & Scientific Information Extractor"
    GOAL = (
        "Mine retrieved literature chunks for methodologies, datasets, evaluation metrics, "
        "and comparative lineage relations to construct a directed NetworkX Knowledge Graph."
    )
    BACKSTORY = (
        "You specialize in scientific NLP and ontology engineering. You transform unstructured "
        "academic text into structured Subject-Predicate-Object (S-P-O) knowledge triads that "
        "trace how modern architectures evolve from and improve upon prior baselines."
    )
    SYSTEM_PROMPT = (
        "You are THE DATA MINER (Extraction & Graphing Agent). Read the chunks stored in "
        "ChromaDB, extract key methodologies, datasets, and baseline metrics, extract "
        "Subject-Predicate-Object triads (e.g., 'Model X -> improves_upon -> Model Y'), "
        "and populate the NetworkX Knowledge Graph. "
        "Delegation Rule: Hand off the populated Knowledge Graph to The Principal Investigator."
    )
    DELEGATION_RULE = (
        "Delegates the populated NetworkX Knowledge Graph to The Principal Investigator (InvestigatorAgent)."
    )

    def __init__(
        self,
        rag_engine: ChromaRAGEngine,
        kg_builder: KnowledgeGraphBuilder,
    ) -> None:
        self.rag_engine = rag_engine
        self.kg_builder = kg_builder

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
        papers: List[AcademicPaper],
        output_dir: Optional[Path] = None,
        topic: str = "",
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase 2 (Graph Construction):
        1. Pull all embedded chunks from ChromaDB.
        2. Run triad and methodology extraction into `KnowledgeGraphBuilder`.
        3. Detect candidate structural holes (when supported by evidence) and export graph artifacts.
        """
        resolved_topic = topic or (getattr(context, "topic", "") if context else "")
        chunks = self.rag_engine.get_all_chunks()
        graph_summary = self.kg_builder.build_from_papers_and_chunks(
            papers=papers,
            chunks=chunks,
            topic=resolved_topic,
        )
        structural_holes = self.kg_builder.find_structural_holes(top_k=3)

        exported_paths: Dict[str, str] = {}
        if output_dir is not None:
            exported_paths = self.kg_builder.export_graph(
                output_dir=output_dir,
                structural_holes=structural_holes,
                topic=resolved_topic,
            )

        return {
            "agent": "The Data Miner (Extraction & Graphing Agent)",
            "topic": topic,
            "graph_summary": graph_summary,
            "triads_sample": [t.to_dict() for t in self.kg_builder.triads[:15]],
            "methodology_table": self.kg_builder.methodology_records,
            "structural_holes": structural_holes,
            "structural_hole_applicable": len(structural_holes) > 0,
            "exported_paths": exported_paths,
            "delegated_to": "The Principal Investigator (Ideation Agent)",
        }
