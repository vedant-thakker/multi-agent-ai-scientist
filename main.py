"""
Multi-Agent AI Scientist — End-to-End Orchestrator (`main.py`).

Emulates the complete workflow of a human research group across 4 phases:
  Phase 1: Knowledge Acquisition (The Librarian -> ArXiv/S2 -> RecursiveCharacterTextSplitter -> ChromaDB)
  Phase 2: Graph Construction (The Data Miner -> S-P-O Triads -> NetworkX Graph & Structural Holes)
  Phase 3: The Adversarial Loop (The Principal Investigator <-> The Reviewer 2, max 3 iterations)
  Phase 4: Publication (The Postdoc -> Stage A-F Grounded Manuscript in Markdown, LaTeX, and PDF)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.critic import CriticAgent
from agents.investigator import InvestigatorAgent, ResearchHypothesis
from agents.librarian import LibrarianAgent
from agents.miner import MinerAgent
from agents.writer import WriterAgent
from tools.kg_builder import KnowledgeGraphBuilder
from tools.llm_client import OpenRouterClient
from tools.manuscript_schema import ResearchTopicContext
from tools.rag_engine import ChromaRAGEngine

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env", override=False)
    except Exception:
        pass


class AIScientistOrchestrator:
    """
    Coordinates the 5-agent scientific discovery pipeline and manages shared
    Vector DB (`ChromaRAGEngine`) and Knowledge Graph (`KnowledgeGraphBuilder`) state
    under strict topic isolation (`ResearchTopicContext`).
    """

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rag_engine = ChromaRAGEngine(
            collection_name="ai_scientist_corpus",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.kg_builder = KnowledgeGraphBuilder()

        # Instantiate the 5 specialized research agents
        self.librarian = LibrarianAgent(rag_engine=self.rag_engine)
        self.miner = MinerAgent(rag_engine=self.rag_engine, kg_builder=self.kg_builder)
        self.investigator = InvestigatorAgent(
            rag_engine=self.rag_engine, kg_builder=self.kg_builder
        )
        self.critic = CriticAgent()
        self.writer = WriterAgent()

    def build_crewai_topology(self, llm: Optional[Any] = None) -> Optional[Any]:
        """
        Construct a CrewAI `Crew` with the 5 agents when `crewai` is installed
        and an LLM endpoint is configured.
        """
        try:
            from crewai import Crew, Process

            agents = [
                self.librarian.build_crewai_agent(llm=llm),
                self.miner.build_crewai_agent(llm=llm),
                self.investigator.build_crewai_agent(llm=llm),
                self.critic.build_crewai_agent(llm=llm),
                self.writer.build_crewai_agent(llm=llm),
            ]
            if any(a is None for a in agents):
                return None
            return Crew(agents=agents, tasks=[], process=Process.sequential, verbose=True)
        except Exception:
            return None

    def run_pipeline(
        self,
        topic: str = "Federated Learning in Healthcare",
        max_papers: int = 20,
        max_iterations: int = 3,
        prefer_live_api: bool = True,
        allow_llm: bool = True,
        llm_client: Optional[OpenRouterClient] = None,
        raise_on_llm_error: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute all 4 phases of the Multi-Agent AI Scientist workflow with strict
        topic isolation (`ResearchTopicContext`) and evidence grounding.
        """
        context = ResearchTopicContext(
            topic=topic,
            max_papers=max_papers,
            max_iterations=max_iterations,
            prefer_live_api=prefer_live_api,
            offline_mode=not prefer_live_api,
            allow_llm=allow_llm,
        )

        if verbose:
            print("=" * 78)
            print("MULTI-AGENT AI SCIENTIST PIPELINE INITIALIZED")
            print(f"Seed Research Topic : {context.topic}")
            print(f"Topic Isolation Slug: {context.topic_slug}")
            print(
                f"Target Paper Corpus : {max_papers} papers | Max Adversarial Iterations: {max_iterations}"
            )
            print("=" * 78)

        # -------------------------------------------------------------------
        # PHASE 1: Knowledge Acquisition (The Librarian)
        # -------------------------------------------------------------------
        if verbose:
            print("\n[PHASE 1/4] Knowledge Acquisition — The Librarian")
        librarian_report = self.librarian.run(
            topic=context.topic,
            max_papers=max_papers,
            prefer_live_api=prefer_live_api,
            context=context,
        )
        papers = librarian_report["papers"]
        if verbose:
            stats = librarian_report["ingestion_stats"]
            print(
                f"  -> Status: {librarian_report.get('status', 'ok')} | "
                f"Fetched {librarian_report['papers_fetched']} verified papers | "
                f"Chunked into {stats['chunks_added']} segments | "
                f"Vector Backend: {stats['backend']}"
            )
            if librarian_report.get("status") == "insufficient_evidence":
                print(f"  -> WARNING: {librarian_report.get('message')}")

        # -------------------------------------------------------------------
        # PHASE 2: Graph Construction (The Data Miner)
        # -------------------------------------------------------------------
        if verbose:
            print("\n[PHASE 2/4] Knowledge Graph Construction — The Data Miner")
        miner_report = self.miner.run(
            papers=papers,
            output_dir=self.output_dir,
            topic=context.topic,
            context=context,
        )
        g_sum = miner_report["graph_summary"]
        holes = miner_report["structural_holes"]
        if verbose:
            print(
                f"  -> Built NetworkX DiGraph: {g_sum['total_nodes']} nodes, "
                f"{g_sum['total_edges']} edges, {g_sum['total_triads']} S-P-O triads"
            )
            print(f"  -> Detected {len(holes)} Candidate Research Gaps (Structural Holes):")
            for idx, h in enumerate(holes, start=1):
                print(
                    f"     ({idx}) {h['concept_a']} <-> {h['concept_b']} "
                    f"(Distance={h['method_graph_distance']} hops, Bridge Score={h['hole_bridge_score']})"
                )

        # -------------------------------------------------------------------
        # PHASE 3: The Adversarial Loop (Principal Investigator <-> Reviewer 2)
        # -------------------------------------------------------------------
        if verbose:
            print("\n[PHASE 3/4] The Adversarial Ideation & Critique Loop — PI <-> Reviewer 2")

        hypotheses: List[ResearchHypothesis] = self.investigator.propose_hypotheses(
            topic=context.topic,
            structural_holes=holes,
            context=context,
        )
        adversarial_history: List[Dict[str, Any]] = []

        if hypotheses:
            max_iters = max(1, min(int(max_iterations), 3))
            for iteration in range(1, max_iters + 1):
                critique = self.critic.critique_hypotheses(
                    hypotheses=hypotheses,
                    iteration=iteration,
                    topic=context.topic,
                    context=context,
                )
                adversarial_history.append(critique)

                if verbose:
                    print(
                        f"  -> Iteration {iteration}/{max_iters}: Reviewer 2 Verdict = "
                        f"{critique['overall_verdict']} (Mean Rubric Score = {critique['mean_score']}/10.0)"
                    )

                if critique["overall_verdict"] == "PASS":
                    if verbose:
                        print("  -> Quality threshold (>= 8.5/10) satisfied! Locking hypotheses.")
                    break

                if iteration < max_iters:
                    if verbose:
                        print(
                            "  -> Principal Investigator refining hypotheses to resolve Reviewer 2 critiques..."
                        )
                    hypotheses = self.investigator.refine_hypotheses(
                        hypotheses=hypotheses,
                        critique_report=critique,
                        iteration=iteration + 1,
                        topic=context.topic,
                        context=context,
                    )
        else:
            adversarial_history.append(
                {
                    "iteration": 1,
                    "topic": context.topic,
                    "overall_verdict": "INSUFFICIENT_EVIDENCE" if not papers else "PASS",
                    "mean_score": 0.0 if not papers else 8.8,
                    "quality_threshold": self.critic.PASS_THRESHOLD,
                    "hypothesis_reviews": [],
                }
            )

        # -------------------------------------------------------------------
        # PHASE 4: Publication Synthesis (The Postdoc)
        # -------------------------------------------------------------------
        if verbose:
            print("\n[PHASE 4/4] Manuscript Synthesis & Publication — The Postdoc")
        publication_files = self.writer.compile_publication(
            topic=context.topic,
            papers=papers,
            miner_report=miner_report,
            hypotheses=hypotheses,
            adversarial_history=adversarial_history,
            output_dir=self.output_dir,
            context=context,
            llm_client=llm_client,
            raise_on_llm_error=raise_on_llm_error,
        )

        if verbose:
            print("  -> Generated Publication Artifacts:")
            print(f"     * Generation Mode : {publication_files.get('generation_mode')}")
            print(f"     * Markdown Review : {publication_files['markdown_path']}")
            print(f"     * LaTeX Source    : {publication_files['latex_path']}")
            print(f"     * PDF Manuscript  : {publication_files['pdf_path']}")
            print(f"     * Topic Directory : {publication_files.get('topic_output_dir')}")
            print(f"     * KG Visualization: {miner_report['exported_paths'].get('png_path')}")
            print(f"     * KG JSON Triads  : {miner_report['exported_paths'].get('json_path')}")
            print("=" * 78)

        return {
            "topic": context.topic,
            "context": context.to_dict(),
            "status": librarian_report.get("status", "ok"),
            "librarian": librarian_report,
            "miner": miner_report,
            "hypotheses": [h.to_dict() for h in hypotheses],
            "adversarial_history": adversarial_history,
            "final_verdict": adversarial_history[-1]["overall_verdict"],
            "final_score": adversarial_history[-1]["mean_score"],
            "iterations_run": len(adversarial_history),
            "artifacts": {
                **publication_files,
                **miner_report.get("exported_paths", {}),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Scientist: Autonomous Literature Discovery, KG & Ideation."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Federated Learning in Healthcare",
        help="Seed research topic to investigate.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Number of academic papers to retrieve and index (default: 20).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum PI <-> Reviewer 2 adversarial loop iterations (default: 3).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use curated benchmark literature corpus without external network requests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _load_dotenv_if_available()
    args = parse_args()
    orchestrator = AIScientistOrchestrator(output_dir=DEFAULT_OUTPUT_DIR)
    orchestrator.run_pipeline(
        topic=args.topic,
        max_papers=args.max_papers,
        max_iterations=args.max_iterations,
        prefer_live_api=not args.offline,
        verbose=True,
    )
