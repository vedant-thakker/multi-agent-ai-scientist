"""
Comprehensive Unit & Integration Tests for the Multi-Agent AI Scientist System.

Covers all 12 required verification scenarios:
 1. Non-clinical topic: "Autonomous Drone Navigation" (zero clinical FL leakage)
 2. Non-clinical topic: "Quantum Computing for Optimization"
 3. Non-clinical topic: "Deep Reinforcement Learning in Robotics"
 4. Canonical clinical FL benchmark: "Federated Learning in Healthcare"
 5. Empty or weak retrieval result (reports insufficient evidence instead of fabricating papers)
 6. OpenRouter timeout or HTTP error (raises explicit LLMGenerationError / surfaces failure)
 7. Malformed JSON response from LLM (detected, logged, and rejected)
 8. Model output containing a fabricated / out-of-range citation (sanitized / stripped)
 9. Switching from Topic A to Topic B in consecutive runs (no RAG or manuscript bleed)
10. Stale cache present from a previous unrelated topic (topic-isolated collection & directories)
11. Insufficient evidence to support a strong claim (explicitly classified & hedged)
12. Consistency across Markdown, LaTeX, and PDF outputs (single CanonicalManuscript model)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdf
import pytest
import requests

from agents.critic import CriticAgent
from agents.investigator import InvestigatorAgent
from agents.librarian import LibrarianAgent
from agents.miner import MinerAgent
from agents.writer import WriterAgent
from main import AIScientistOrchestrator
from tools.arxiv_search import fetch_academic_literature
from tools.kg_builder import KnowledgeGraphBuilder
from tools.llm_client import (
    LLMGenerationError,
    OpenRouterClient,
    verify_and_sanitize_citations_in_text,
)
from tools.manuscript_schema import ResearchTopicContext
from tools.manuscript_workflow import assemble_canonical_manuscript
from tools.rag_engine import ChromaRAGEngine, RecursiveCharacterTextSplitter


FORBIDDEN_CLINICAL_TERMS = [
    "unseen hospital sites",
    "multi-institutional clinical cohort",
    "causal-moe",
    "lora-diffsynth",
    "contrastive-ala",
    "hipaa/gdpr",
    "patient ehr",
    "clinical silos",
]


def _assert_no_clinical_leakage(text: str) -> None:
    lower = text.lower()
    for term in FORBIDDEN_CLINICAL_TERMS:
        assert term not in lower, f"Forbidden clinical FL term '{term}' leaked into non-clinical output!"


def test_recursive_character_text_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=30)
    long_text = "\n\n".join(
        [
            f"Paragraph {i}: Autonomous quadrotors use visual-inertial odometry and trajectory optimization to navigate cluttered environments."
            for i in range(6)
        ]
    )
    chunks = splitter.split_text(long_text)
    assert len(chunks) >= 3
    assert all(len(c) <= 220 for c in chunks)


# ============================================================================
# Scenario 1: Non-clinical topic "Autonomous Drone Navigation"
# ============================================================================
def test_scenario_1_autonomous_drone_navigation(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Autonomous Drone Navigation",
        max_papers=6,
        max_iterations=2,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    assert result["status"] == "ok"
    assert result["librarian"]["papers_fetched"] == 6

    md_text = Path(result["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    tex_text = Path(result["artifacts"]["latex_path"]).read_text(encoding="utf-8")
    pdf_path = Path(result["artifacts"]["pdf_path"])

    assert "Autonomous Drone Navigation" in md_text
    assert "Autonomous Drone Navigation" in tex_text
    assert "Loquercio" in md_text or "EGO-Planner" in md_text

    _assert_no_clinical_leakage(md_text)
    _assert_no_clinical_leakage(tex_text)

    reader = pypdf.PdfReader(str(pdf_path))
    assert len(reader.pages) >= 1
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Autonomous Drone Navigation" in pdf_text
    _assert_no_clinical_leakage(pdf_text)
    assert result["artifacts"]["pdf_validation"]["valid"] is True


# ============================================================================
# Scenario 2: Non-clinical topic "Quantum Computing for Optimization"
# ============================================================================
def test_scenario_2_quantum_computing_for_optimization(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Quantum Computing for Optimization",
        max_papers=6,
        max_iterations=2,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    assert result["status"] == "ok"
    assert result["librarian"]["papers_fetched"] == 6

    md_text = Path(result["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    tex_text = Path(result["artifacts"]["latex_path"]).read_text(encoding="utf-8")
    pdf_path = Path(result["artifacts"]["pdf_path"])

    assert "Quantum Computing for Optimization" in md_text
    assert "QAOA" in md_text or "Farhi" in md_text
    _assert_no_clinical_leakage(md_text)
    _assert_no_clinical_leakage(tex_text)

    reader = pypdf.PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Quantum" in pdf_text
    _assert_no_clinical_leakage(pdf_text)


# ============================================================================
# Scenario 3: Non-clinical topic "Deep Reinforcement Learning in Robotics"
# ============================================================================
def test_scenario_3_deep_rl_in_robotics(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Deep Reinforcement Learning in Robotics",
        max_papers=6,
        max_iterations=2,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    assert result["status"] == "ok"
    assert result["librarian"]["papers_fetched"] == 6

    md_text = Path(result["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    pdf_path = Path(result["artifacts"]["pdf_path"])

    assert "Deep Reinforcement Learning in Robotics" in md_text
    assert "Schulman" in md_text or "Haarnoja" in md_text or "Hwangbo" in md_text
    _assert_no_clinical_leakage(md_text)

    reader = pypdf.PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Robotics" in pdf_text
    _assert_no_clinical_leakage(pdf_text)


# ============================================================================
# Scenario 4: Canonical Clinical FL Benchmark
# ============================================================================
def test_scenario_4_canonical_clinical_fl_benchmark(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Federated Learning in Healthcare",
        max_papers=20,
        max_iterations=3,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    assert result["final_verdict"] == "PASS"
    assert result["iterations_run"] == 2
    assert len(result["hypotheses"]) == 3
    assert result["librarian"]["papers_fetched"] == 20

    md_path = Path(result["artifacts"]["markdown_path"])
    tex_path = Path(result["artifacts"]["latex_path"])
    pdf_path = Path(result["artifacts"]["pdf_path"])

    assert md_path.exists() and md_path.stat().st_size > 2000
    assert tex_path.exists() and tex_path.stat().st_size > 500
    assert pdf_path.exists() and pdf_path.stat().st_size > 100


# ============================================================================
# Scenario 5: Empty or Weak Retrieval Result -> Insufficient Evidence Report
# ============================================================================
def test_scenario_5_empty_retrieval_reports_insufficient_evidence(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Xenobiological Warp Drive Subspace Magnetohydrodynamics",
        max_papers=5,
        max_iterations=1,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    assert result["status"] == "insufficient_evidence"
    assert result["librarian"]["papers_fetched"] == 0
    assert len(result["hypotheses"]) == 0
    assert result["artifacts"]["generation_mode"] == "insufficient_evidence"

    md_text = Path(result["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    assert "Insufficient Literature Evidence Report" in md_text
    assert "No verified references retrieved" in md_text
    _assert_no_clinical_leakage(md_text)


# ============================================================================
# Scenario 6: OpenRouter Timeout or HTTP Error
# ============================================================================
def test_scenario_6_openrouter_timeout_and_http_error():
    client = OpenRouterClient(
        api_key="sk-or-test-dummy-key",
        model="meta-llama/llama-3.3-70b-instruct",
        timeout=2,
        max_retries=0,
    )

    # 6a. Timeout handling
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        with pytest.raises(LLMGenerationError) as exc_info:
            client.call_json(
                stage="test_timeout_stage",
                system_prompt="You are a test assistant.",
                user_prompt="Return JSON.",
            )
        assert exc_info.value.error_category == "timeout"
        assert exc_info.value.stage == "test_timeout_stage"

    # 6b. HTTP 500 server error handling
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(LLMGenerationError) as exc_info2:
            client.call_json(
                stage="test_http_502",
                system_prompt="You are a test assistant.",
                user_prompt="Return JSON.",
            )
        assert exc_info2.value.error_category == "server_error"
        assert exc_info2.value.status_code == 502


# ============================================================================
# Scenario 7: Malformed JSON Response from LLM
# ============================================================================
def test_scenario_7_malformed_json_response_rejected():
    client = OpenRouterClient(
        api_key="sk-or-test-dummy-key",
        model="meta-llama/llama-3.3-70b-instruct",
        max_retries=0,
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "This is definitely not a JSON object {broken: [1, 2"},
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(LLMGenerationError) as exc_info:
            client.call_json(
                stage="test_malformed_json",
                system_prompt="System",
                user_prompt="User",
                required_keys=["scholarly_title"],
            )
        assert exc_info.value.error_category == "malformed_json"
        assert client.diagnostics[-1]["error_category"] == "malformed_json"


# ============================================================================
# Scenario 8: Model Output Containing Fabricated / Out-of-Range Citations
# ============================================================================
def test_scenario_8_fabricated_citations_stripped_and_clamped():
    raw_text = (
        "Recent studies in drone navigation [1, 3] and trajectory optimization [99] "
        "demonstrate agile flight [1-4], whereas fabricated references [42, 105] "
        "and invalid spans [12–25] must never survive verification when only 6 papers exist."
    )
    cleaned = verify_and_sanitize_citations_in_text(raw_text, max_citation_index=6)

    assert "[1, 3]" in cleaned
    assert "[1–4]" in cleaned
    assert "[99]" not in cleaned
    assert "42" not in cleaned
    assert "105" not in cleaned
    assert "[12–25]" not in cleaned


# ============================================================================
# Scenario 9: Switching from Topic A to Topic B in Consecutive Runs
# ============================================================================
def test_scenario_9_consecutive_topic_switching_isolation(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)

    # Run 1: Clinical Federated Learning
    res_a = orchestrator.run_pipeline(
        topic="Federated Learning in Healthcare",
        max_papers=10,
        max_iterations=1,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )
    assert res_a["librarian"]["papers_fetched"] == 10

    # Run 2 on the SAME orchestrator instance: Quantum Computing for Optimization
    res_b = orchestrator.run_pipeline(
        topic="Quantum Computing for Optimization",
        max_papers=6,
        max_iterations=1,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )
    assert res_b["librarian"]["papers_fetched"] == 6

    # Verify RAG engine only holds chunks from Run 2 (Quantum)
    all_chunks = orchestrator.rag_engine.get_all_chunks()
    assert len(all_chunks) > 0
    for ch in all_chunks:
        title_str = ch.title if hasattr(ch, "title") else ch["title"]
        assert "Federated" not in title_str

    # Verify final output files are strictly about Quantum Computing
    md_text = Path(res_b["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    assert "Quantum Computing for Optimization" in md_text
    _assert_no_clinical_leakage(md_text)

    # Verify both topic-isolated directories exist independently
    assert Path(res_a["artifacts"]["topic_markdown_path"]).exists()
    assert Path(res_b["artifacts"]["topic_markdown_path"]).exists()
    assert res_a["artifacts"]["topic_output_dir"] != res_b["artifacts"]["topic_output_dir"]


# ============================================================================
# Scenario 10: Stale Cache / Unrelated Output Files Isolation
# ============================================================================
def test_scenario_10_stale_cache_and_graph_isolation(tmp_path: Path):
    # Pre-populate output directory with stale clinical FL files
    stale_md = tmp_path / "literature_review.md"
    stale_md.write_text("STALE CONTENT: unseen hospital sites and Causal-MoE", encoding="utf-8")

    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    res = orchestrator.run_pipeline(
        topic="Autonomous Drone Navigation",
        max_papers=6,
        max_iterations=1,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    fresh_md = Path(res["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    assert "STALE CONTENT" not in fresh_md
    _assert_no_clinical_leakage(fresh_md)
    assert "Autonomous Drone Navigation" in fresh_md


# ============================================================================
# Scenario 11: Insufficient Evidence for Strong Claim -> Hedged & Labeled
# ============================================================================
def test_scenario_11_single_paper_small_corpus_hedging_and_no_forced_holes(tmp_path: Path):
    papers = fetch_academic_literature(
        "Autonomous Drone Navigation", max_papers=1, prefer_live_api=False
    )
    assert len(papers) == 1

    rag = ChromaRAGEngine(collection_name="test_single_paper")
    rag.ingest_papers(papers, clear_existing=True, topic_slug="drone_single")
    kg = KnowledgeGraphBuilder()
    miner = MinerAgent(rag_engine=rag, kg_builder=kg)
    ctx = ResearchTopicContext(topic="Autonomous Drone Navigation", max_papers=1, allow_llm=False)
    miner_rep = miner.run(papers=papers, output_dir=tmp_path, topic=ctx.topic, context=ctx)

    # With only 1 paper / method node, structural hole analysis must honestly report not applicable
    assert miner_rep["graph_summary"]["structural_hole_applicable"] is False
    assert len(miner_rep["structural_holes"]) == 0

    pi = InvestigatorAgent(rag_engine=rag, kg_builder=kg)
    hyps = pi.propose_hypotheses(topic=ctx.topic, structural_holes=[], context=ctx)
    assert all(h.claim_status == "proposed" for h in hyps)

    canonical = assemble_canonical_manuscript(
        context=ctx,
        papers=papers,
        miner_report=miner_rep,
        hypotheses=hyps,
        adversarial_history=[],
    )
    assert len(canonical.evidence_ledger.supported_claims()) == 1
    assert any("1 verified studies" in lim for lim in canonical.evidence_ledger.evidence_limitations)


# ============================================================================
# Scenario 12: Semantic Consistency Across Markdown, LaTeX, and PDF
# ============================================================================
def test_scenario_12_markdown_latex_pdf_semantic_consistency(tmp_path: Path):
    orchestrator = AIScientistOrchestrator(output_dir=tmp_path)
    result = orchestrator.run_pipeline(
        topic="Autonomous Drone Navigation",
        max_papers=6,
        max_iterations=1,
        prefer_live_api=False,
        allow_llm=False,
        verbose=False,
    )

    canonical = result["artifacts"]["canonical_manuscript"]
    md_text = Path(result["artifacts"]["markdown_path"]).read_text(encoding="utf-8")
    tex_text = Path(result["artifacts"]["latex_path"]).read_text(encoding="utf-8")
    pdf_path = Path(result["artifacts"]["pdf_path"])

    reader = pypdf.PdfReader(str(pdf_path))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    pdf_text_norm = " ".join(pdf_text.split()).upper()

    # 1. Exact title consistency
    assert canonical.title in md_text
    assert canonical.title in tex_text

    # 2. Section headings consistency across MD, LaTeX, and PDF
    for sec in canonical.sections:
        assert sec.heading in md_text
        assert sec.heading in tex_text
        assert " ".join(sec.heading.split()).upper() in pdf_text_norm

    # 3. Bibliography consistency across MD, LaTeX, and PDF
    assert len(canonical.bibliography) == 6
    for ref in canonical.bibliography:
        assert ref.paper_id in md_text
        assert ref.paper_id in tex_text
        assert ref.paper_id in pdf_text
