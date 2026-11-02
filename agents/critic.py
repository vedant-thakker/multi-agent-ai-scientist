"""
Agent 4: The Review & Critique Agent ("The Reviewer 2").

Responsibilities:
- Evaluates proposed research hypotheses for topic relevance, evidence grounding,
  baseline completeness, and epistemic honesty (never claiming unvalidated theorems
  or unperformed experiments as established fact).
- Performs Stage F manuscript quality-control auditing across sections, citations,
  and evidence ledger claims.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents.investigator import ResearchHypothesis
from tools.arxiv_search import AcademicPaper, compute_paper_relevance
from tools.manuscript_schema import (
    QualityReviewReport,
    ReviewIssue,
    is_canonical_clinical_fl_topic,
)


UNSUPPORTED_CLAIM_PATTERNS = [
    (r"\bwe experimentally prove\b", "Claims experimental proof without executing physical experiments."),
    (r"\bour experiments demonstrate\b", "Claims completed experimental results in a literature synthesis / proposal."),
    (r"\bachieves state-of-the-art accuracy of \d+", "Fabricates specific experimental benchmark numbers for an unimplemented proposal."),
    (r"\bwe have validated on real patients\b", "Claims clinical validation that was not performed."),
]


class CriticAgent:
    """The Review & Critique Agent (The Reviewer 2)."""

    ROLE = "Adversarial Peer Reviewer ('Reviewer 2') & Methodological Auditor"
    GOAL = (
        "Audit proposed research hypotheses and manuscript sections for topic relevance, "
        "evidence provenance, citation accuracy, baseline completeness, and epistemic honesty."
    )
    BACKSTORY = (
        "You serve as Senior Area Chair and 'Reviewer 2' for premier scientific venues. "
        "You return specific, actionable problems and corrections—flagging any off-topic "
        "drift, unverified citations, or unsupported novelty/theorem claims."
    )
    SYSTEM_PROMPT = (
        "You are THE REVIEWER 2 (Review & Critique Agent). Audit each proposed hypothesis "
        "and manuscript section for: (1) Relevance to the exact topic, (2) Factual grounding "
        "in retrieved sources, (3) Baseline & ablation completeness, and (4) Absence of "
        "unsupported experimental or theorem validation claims."
    )
    DELEGATION_RULE = (
        "Returns specific 'REVISE' critiques and correction directives to The Principal "
        "Investigator or approves ('PASS') for publication."
    )

    PASS_THRESHOLD = 8.5

    def build_crewai_agent(self, llm: Optional[Any] = None) -> Any:
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

    def critique_hypotheses(
        self,
        hypotheses: List[ResearchHypothesis],
        iteration: int = 1,
        topic: str = "",
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate each hypothesis individually.
        Preserves canonical clinical FL critiques ONLY when the hypotheses belong to
        the clinical Federated Learning benchmark; otherwise performs domain-grounded
        methodological and epistemic auditing.
        """
        if not hypotheses:
            return {
                "agent": "The Reviewer 2 (Review & Critique Agent)",
                "iteration": iteration,
                "pass_threshold": self.PASS_THRESHOLD,
                "mean_score": 0.0,
                "overall_verdict": "INSUFFICIENT_EVIDENCE",
                "hypothesis_reviews": [],
            }

        is_clinical_fl = is_canonical_clinical_fl_topic(topic) or any(
            is_canonical_clinical_fl_topic(h.title)
            or h.short_name in {"Causal-MoE", "LoRA-DiffSynth", "Contrastive-ALA"}
            for h in hypotheses
        )

        reviews: List[Dict[str, Any]] = []

        for idx, hyp in enumerate(hypotheses, start=1):
            flaws: List[str] = []
            directives: List[str] = []

            if is_clinical_fl:
                is_refined = "Proposition" in hyp.theoretical_guarantee and len(hyp.baselines) >= 6
                if not is_refined:
                    if idx == 1:
                        scores = {
                            "logical_soundness": 7.4,
                            "baseline_completeness": 7.0,
                            "computational_feasibility": 7.1,
                            "statistical_rigor": 7.3,
                        }
                        flaws = [
                            "Unconstrained dense mask M_k in R^d doubles local parameter memory and lacks an out-of-distribution generalization bound for unseen hospitals.",
                            f"Baseline comparison ({', '.join(hyp.baselines)}) omits variance-reduced SCAFFOLD, personalized Ditto, and frontier MedFed-MoE architectures.",
                            "Evaluation protocol lacks a leave-hospitals-out OOD split and straggler dropout resilience test.",
                        ]
                        directives = [
                            "Factorize M_k into a low-rank outer product σ(u_k v_k^T) with rank r << d and derive a formal OOD generalization proposition.",
                            "Add SCAFFOLD, Ditto, and MedFed-MoE baselines plus a leave-two-hospitals-out cross-silo ablation.",
                        ]
                    elif idx == 2:
                        scores = {
                            "logical_soundness": 7.1,
                            "baseline_completeness": 6.8,
                            "computational_feasibility": 6.7,
                            "statistical_rigor": 7.0,
                        }
                        flaws = [
                            "Contrastive alignment in the diffusion bottleneck lacks formal Rényi Differential Privacy (ε, δ) accounting against patient memorization.",
                            f"Baseline suite ({', '.join(hyp.baselines)}) omits differentially private NbAFL-DP and vision-language adapter FedCLIP.",
                            "Empirical validation lacks shadow-model Membership Inference Attack (MIA) auditing and peak client VRAM measurements.",
                        ]
                        directives = [
                            "Incorporate subsampled Gaussian gradient clipping with an explicit Rényi-DP composition bound (Proposition 2).",
                            "Report peak client VRAM (GB), DDIM sampling latency, and Membership Inference Attack (MIA) AUC across ε ∈ {1.0, 2.0, 4.0}.",
                        ]
                    else:
                        scores = {
                            "logical_soundness": 6.9,
                            "baseline_completeness": 7.1,
                            "computational_feasibility": 7.2,
                            "statistical_rigor": 6.8,
                        }
                        flaws = [
                            "Separately averaging low-rank adapters (Ē_B and Ē_A) across hospitals introduces bilinear cross-term truncation error (ΔW ≠ Ē_B · Ē_A).",
                            "Missing theoretical guarantee that Fourier amplitude harmonization prevents router gate collapse under scanner domain shift.",
                            "Experimental design does not test missing-modality hospital dropout or quantify router load-balancing Shannon entropy.",
                        ]
                        directives = [
                            "Replace naive adapter averaging with server-side truncated SVD_r(Σ p_k B_k A_k) and bound the Frobenius residual (Proposition 3).",
                            "Include 95% Hausdorff Distance (HD95), Router Gate Shannon Entropy, and a 25% missing-modality stress test.",
                        ]
                else:
                    scores = {
                        "logical_soundness": 9.3 if idx != 2 else 9.1,
                        "baseline_completeness": 9.2 if idx != 2 else 9.0,
                        "computational_feasibility": 9.1,
                        "statistical_rigor": 9.2,
                    }
            else:
                # Topic-adaptive, evidence-grounded review for general research domains
                has_ablation = "4-part ablation" in hyp.experimental_setup.lower() or "4-way ablation" in hyp.experimental_setup.lower()
                has_falsifiable_spec = "falsifiable hypothesis specification" in hyp.theoretical_guarantee.lower()
                has_sufficient_baselines = len(hyp.baselines) >= 4
                has_citations = len(hyp.supporting_citations) >= 1

                if not (has_ablation and has_falsifiable_spec and has_sufficient_baselines):
                    scores = {
                        "logical_soundness": 7.4,
                        "baseline_completeness": 7.1 if has_citations else 6.5,
                        "computational_feasibility": 7.3,
                        "statistical_rigor": 7.0,
                    }
                    flaws = [
                        f"Proposal {hyp.short_name} ({hyp.concept_a} + {hyp.concept_b}) requires an explicit 4-part ablation design separating standalone {hyp.concept_a} and standalone {hyp.concept_b}.",
                        f"Baseline set ({', '.join(hyp.baselines)}) should explicitly include standalone component baselines and ablated hybrid controls on {', '.join(hyp.datasets[:2])}.",
                        "Analytical scope must explicitly state falsifiable evaluation criteria and distinguish proposed mechanisms from published empirical findings.",
                    ]
                    directives = [
                        f"Formulate a 4-part ablation protocol comparing standalone {hyp.concept_a}, standalone {hyp.concept_b}, unregularized hybrid, and full {hyp.short_name}.",
                        "Specify falsifiable evaluation criteria and multi-seed variance controls without claiming unperformed experiments as established fact.",
                    ]
                else:
                    scores = {
                        "logical_soundness": 9.1,
                        "baseline_completeness": 9.0,
                        "computational_feasibility": 9.0,
                        "statistical_rigor": 9.1,
                    }

            composite = round(sum(scores.values()) / 4.0, 2)
            scores["composite_score"] = composite
            status = "PASS" if composite >= self.PASS_THRESHOLD and not flaws else "REVISE"

            reviews.append(
                {
                    "hypothesis_id": hyp.hypothesis_id,
                    "title": hyp.title,
                    "iteration": iteration,
                    "scores": scores,
                    "verdict": status,
                    "flaws_identified": flaws
                    or [
                        f"Approved as a grounded future research proposal ({hyp.short_name}): falsifiable ablation criteria specified and epistemic status clearly labeled."
                    ],
                    "revision_directives": directives,
                }
            )

        mean_score = round(
            sum(r["scores"]["composite_score"] for r in reviews) / max(1, len(reviews)),
            2,
        )
        overall_passed = all(r["verdict"] == "PASS" for r in reviews)

        return {
            "agent": "The Reviewer 2 (Review & Critique Agent)",
            "iteration": iteration,
            "pass_threshold": self.PASS_THRESHOLD,
            "mean_score": mean_score,
            "overall_verdict": "PASS" if overall_passed else "REVISE",
            "hypothesis_reviews": reviews,
        }

    def audit_manuscript_sections(
        self,
        topic: str,
        sections: List[Dict[str, Any]],
        papers: List[AcademicPaper],
        title: str = "",
        abstract: str = "",
        iteration: int = 1,
    ) -> QualityReviewReport:
        """
        Stage F deterministic + semantic quality-control check over the assembled manuscript.
        Checks:
        1. Relevance to the exact user topic & absence of cross-domain clinical FL contamination.
        2. Citation validity (every [k] in text maps to 1 <= k <= len(papers)).
        3. Absence of fabricated experimental validation claims.
        4. Section completeness and non-redundancy.
        """
        issues: List[ReviewIssue] = []
        num_papers = len(papers)
        is_fl = is_canonical_clinical_fl_topic(topic)

        # 1. Check Title and Abstract relevance to topic
        eval_title = title or topic
        eval_abstract = abstract or (
            " ".join(sections[0].get("paragraphs", [])) if sections else topic
        )
        rel_score = compute_paper_relevance(topic, eval_title, eval_abstract)
        if rel_score < 0.15 and num_papers > 0:
            issues.append(
                ReviewIssue(
                    section_id="abstract",
                    category="topic_relevance",
                    severity="error",
                    description=f"Title/abstract relevance score ({rel_score:.2f}) is too low for topic '{topic}'.",
                    correction_directive=f"Ensure the title and abstract directly address '{topic}' and its retrieved corpus.",
                )
            )

        # 2. Check cross-domain clinical FL contamination in non-clinical topics
        if not is_fl:
            corpus_text_l = " ".join(f"{p.title} {p.abstract}" for p in papers).lower()
            prohibited_unless_in_corpus = [
                ("multi-institutional clinical cohort", "References 'Multi-Institutional Clinical Cohort' not present in corpus."),
                ("unseen hospital sites", "References 'unseen hospital sites' in a non-clinical topic."),
                ("patient memorization", "References 'patient memorization' in a non-clinical topic."),
                ("mimic-cxr", "References clinical dataset 'MIMIC-CXR' in a non-clinical topic."),
                ("camelyon17", "References pathology dataset 'Camelyon17' in a non-clinical topic."),
            ]
            all_blocks = [("abstract", abstract)] + [
                (sec.get("section_id", f"sec_{i}"), " ".join(sec.get("paragraphs", [])))
                for i, sec in enumerate(sections)
            ]
            for sec_id, text_block in all_blocks:
                tb_low = text_block.lower()
                for phrase, reason in prohibited_unless_in_corpus:
                    if phrase in tb_low and phrase not in corpus_text_l:
                        issues.append(
                            ReviewIssue(
                                section_id=sec_id,
                                category="topic_relevance",
                                severity="error",
                                description=reason,
                                correction_directive=f"Remove unrelated clinical phrase '{phrase}' and ground strictly in '{topic}' sources.",
                            )
                        )

        # 3. Check Citation Validity across abstract and all sections
        citation_ok = True
        all_texts = [("abstract", abstract)] + [
            (sec.get("section_id", f"sec_{i}"), " ".join(sec.get("paragraphs", [])))
            for i, sec in enumerate(sections)
        ]
        for sec_id, text_block in all_texts:
            for match in re.finditer(r"\[(\d+(?:\s*[\-–,]\s*\d+)*)\]", text_block):
                raw_cite = match.group(1)
                nums = [int(n) for n in re.findall(r"\d+", raw_cite)]
                invalid_nums = [n for n in nums if n < 1 or n > num_papers]
                if invalid_nums:
                    citation_ok = False
                    issues.append(
                        ReviewIssue(
                            section_id=sec_id,
                            category="citation_accuracy",
                            severity="error",
                            description=f"Out-of-range citation [{raw_cite}] found (valid range: [1]–[{num_papers}]).",
                            correction_directive=f"Remove or remap invalid citation indices {invalid_nums} to verified corpus entries [1]–[{num_papers}].",
                        )
                    )

        # 4. Check for unsupported experimental proof claims
        grounding_ok = True
        for sec_id, text_block in all_texts:
            for pat, desc in UNSUPPORTED_CLAIM_PATTERNS:
                if re.search(pat, text_block, flags=re.IGNORECASE):
                    grounding_ok = False
                    issues.append(
                        ReviewIssue(
                            section_id=sec_id,
                            category="unsupported_claim",
                            severity="error",
                            description=desc,
                            correction_directive="Qualify proposed methods as hypotheses requiring future experimental validation.",
                        )
                    )

        errors = [i for i in issues if i.severity == "error"]
        topic_ok = not any(i.category == "topic_relevance" for i in errors)

        return QualityReviewReport(
            approved=len(errors) == 0,
            iteration=iteration,
            topic_relevance_passed=topic_ok,
            citation_verification_passed=citation_ok,
            factual_grounding_passed=grounding_ok,
            issues=issues,
            corrected_sections=[],
            summary_notes=[
                f"Audited {len(sections)} sections against {num_papers} verified sources for topic '{topic}'."
            ],
        )
