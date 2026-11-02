"""
Knowledge Graph Extractor & Structural Hole Analyzer (`NetworkX`).

Extracts Subject-Predicate-Object (S-P-O) triads from retrieved literature chunks,
constructs a topic-isolated directed methodology evolution graph in `NetworkX`,
validates node/edge provenance, identifies candidate structural holes when supported
by graph topology, and renders a domain-accurate visual graph diagram.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from tools.arxiv_search import AcademicPaper
from tools.rag_engine import DocumentChunk


VALID_NODE_TYPES = {"method", "dataset", "limitation", "concept", "metric"}
VALID_PREDICATES = {
    "improves_upon",
    "evaluated_on",
    "exhibits_limitation",
    "outperforms",
    "extends",
    "mitigates",
    "replaces",
}


@dataclass
class KnowledgeTriad:
    """Subject-Predicate-Object triad extracted from academic literature with provenance."""

    subject: str
    subject_type: str  # "method" | "concept"
    predicate: str     # e.g., "improves_upon", "evaluated_on", "exhibits_limitation"
    obj: str
    object_type: str   # "method" | "dataset" | "limitation" | "metric"
    paper_id: str
    year: int
    evidence: str
    source: str = "retrieved_literature"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KnowledgeGraphBuilder:
    """
    Builds and analyzes an in-memory `NetworkX` directed knowledge graph of
    scientific methodologies, evaluation settings, and limitations grounded
    in the current topic's retrieved corpus.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.triads: List[KnowledgeTriad] = []
        self.methodology_records: List[Dict[str, Any]] = []
        self.current_topic: str = ""

    def clear(self) -> None:
        """Reset all graph state to prevent cross-topic contamination."""
        self.graph.clear()
        self.triads.clear()
        self.methodology_records.clear()
        self.current_topic = ""

    def build_from_papers_and_chunks(
        self,
        papers: List[AcademicPaper],
        chunks: Optional[List[DocumentChunk]] = None,
        topic: str = "",
    ) -> Dict[str, Any]:
        """
        Extract S-P-O triads from structured paper metadata and unstructured text chunks,
        then populate `self.graph` with full provenance.
        """
        self.clear()
        self.current_topic = (topic or "").strip()

        for paper in papers:
            method = (paper.primary_method or "").strip()
            parent = (paper.improves_upon or "").strip()
            if not method:
                continue

            self._ensure_node(
                method,
                node_type="method",
                year=paper.year,
                paper_id=paper.paper_id,
                title=paper.title,
                source=paper.source,
                limitations=paper.limitations,
            )

            if parent and parent.lower() not in {
                "conventional domain baselines",
                "prior domain baselines (see abstract)",
            }:
                self._ensure_node(
                    parent,
                    node_type="method",
                    year=max(2010, paper.year - 1),
                    source="baseline_reference",
                )
                self._add_triad(
                    KnowledgeTriad(
                        subject=method,
                        subject_type="method",
                        predicate="improves_upon",
                        obj=parent,
                        object_type="method",
                        paper_id=paper.paper_id,
                        year=paper.year,
                        evidence=f"In {paper.paper_id} ({paper.year}), {method} builds upon or compares against {parent}.",
                        source=paper.source,
                    )
                )

            for ds in paper.datasets:
                ds_clean = ds.strip()
                if not ds_clean or "abstract-only" in ds_clean.lower():
                    continue
                self._ensure_node(
                    ds_clean,
                    node_type="dataset",
                    year=paper.year,
                    paper_id=paper.paper_id,
                    source=paper.source,
                )
                self._add_triad(
                    KnowledgeTriad(
                        subject=method,
                        subject_type="method",
                        predicate="evaluated_on",
                        obj=ds_clean,
                        object_type="dataset",
                        paper_id=paper.paper_id,
                        year=paper.year,
                        evidence=f"{method} ({paper.paper_id}) evaluated on {ds_clean}.",
                        source=paper.source,
                    )
                )

            if paper.limitations:
                lim_short = self._summarize_limitation(paper.limitations)
                self._ensure_node(
                    lim_short,
                    node_type="limitation",
                    year=paper.year,
                    paper_id=paper.paper_id,
                    source=paper.source,
                )
                self._add_triad(
                    KnowledgeTriad(
                        subject=method,
                        subject_type="method",
                        predicate="exhibits_limitation",
                        obj=lim_short,
                        object_type="limitation",
                        paper_id=paper.paper_id,
                        year=paper.year,
                        evidence=paper.limitations,
                        source=paper.source,
                    )
                )

            self.methodology_records.append(
                {
                    "paper_id": paper.paper_id,
                    "year": paper.year,
                    "venue": getattr(paper, "venue", "Scientific Venue"),
                    "category": getattr(paper, "category", "Core Methodology"),
                    "citation": paper.first_author_citation(),
                    "title": paper.title,
                    "authors": ", ".join(paper.authors[:3])
                    + (" et al." if len(paper.authors) > 3 else ""),
                    "method": method,
                    "improves_upon": parent or "Domain Baseline",
                    "datasets": ", ".join(paper.datasets),
                    "metrics": ", ".join(f"{k}: {v}" for k, v in paper.metrics.items()),
                    "limitations": paper.limitations,
                    "source": paper.source,
                    "analysis_scope": getattr(paper, "analysis_scope", "abstract_only"),
                }
            )

        if chunks:
            self._extract_nlp_triads_from_chunks(chunks)

        return self.summary()

    def _ensure_node(self, name: str, node_type: str, **attrs: Any) -> None:
        valid_type = node_type if node_type in VALID_NODE_TYPES else "concept"
        if not self.graph.has_node(name):
            self.graph.add_node(name, node_type=valid_type, **attrs)
        else:
            for k, v in attrs.items():
                if v and not self.graph.nodes[name].get(k):
                    self.graph.nodes[name][k] = v

    def _add_triad(self, triad: KnowledgeTriad) -> None:
        if triad.predicate not in VALID_PREDICATES:
            return
        self.triads.append(triad)
        self.graph.add_edge(
            triad.subject,
            triad.obj,
            predicate=triad.predicate,
            paper_id=triad.paper_id,
            year=triad.year,
            evidence=triad.evidence,
            source=triad.source,
        )

    @staticmethod
    def _summarize_limitation(lim_text: str) -> str:
        words = lim_text.strip().rstrip(".").split()
        return " ".join(words[:7]) + ("..." if len(words) > 7 else "")

    def _extract_nlp_triads_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Pattern-based relation extraction over chunk text for explicit comparative clauses.
        """
        pattern = re.compile(
            r"\b([A-Z][A-Za-z0-9\-]{2,18})\s+"
            r"(outperforms|improves upon|extends|mitigates|replaces)\s+"
            r"([A-Z][A-Za-z0-9\-]{2,18})\b"
        )
        for ch in chunks:
            for match in pattern.finditer(ch.text):
                subj, pred, obj = match.group(1), match.group(2), match.group(3)
                if subj == obj:
                    continue
                pred_norm = pred.replace(" ", "_")
                if not self.graph.has_edge(subj, obj):
                    self._ensure_node(subj, node_type="method", year=ch.year, paper_id=ch.paper_id)
                    self._ensure_node(obj, node_type="method", year=ch.year, paper_id=ch.paper_id)
                    self._add_triad(
                        KnowledgeTriad(
                            subject=subj,
                            subject_type="method",
                            predicate=pred_norm,
                            obj=obj,
                            object_type="method",
                            paper_id=ch.paper_id,
                            year=ch.year,
                            evidence=match.group(0),
                            source=ch.metadata.get("source", "chunk_nlp"),
                        )
                    )

    def find_structural_holes(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Analyze the Knowledge Graph for candidate structural holes (unlinked pairs of
        methodological nodes from the retrieved corpus that address complementary
        challenges or share evaluation settings).
        Returns `[]` if the graph lacks sufficient evidence (< 2 method nodes).
        """
        method_nodes = [
            n
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "method" and d.get("paper_id")
        ]
        if len(method_nodes) < 2:
            return []

        try:
            pagerank = nx.pagerank(self.graph.to_undirected())
        except Exception:
            pagerank = {n: 1.0 / max(1, len(self.graph)) for n in self.graph.nodes}

        method_datasets: Dict[str, set] = {}
        for m in method_nodes:
            ds_neighbors = {
                nbr
                for nbr in self.graph.successors(m)
                if self.graph.nodes[nbr].get("node_type") == "dataset"
            }
            method_datasets[m] = ds_neighbors

        candidates: List[Tuple[float, Dict[str, Any]]] = []
        seen_pairs = set()
        method_subgraph = self.graph.subgraph(method_nodes).to_undirected()

        for i, m_a in enumerate(method_nodes):
            for m_b in method_nodes[i + 1 :]:
                pair_key = tuple(sorted((m_a, m_b)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if self.graph.has_edge(m_a, m_b) or self.graph.has_edge(m_b, m_a):
                    continue

                shared_ds = sorted(method_datasets[m_a] & method_datasets[m_b])
                union_ds = sorted(method_datasets[m_a] | method_datasets[m_b])

                try:
                    path_len = nx.shortest_path_length(method_subgraph, m_a, m_b)
                except nx.NetworkXNoPath:
                    path_len = 6

                year_a = int(self.graph.nodes[m_a].get("year", 2022))
                year_b = int(self.graph.nodes[m_b].get("year", 2022))
                recency_bonus = ((year_a - 2017) + (year_b - 2017)) * 0.08
                centrality_score = (pagerank.get(m_a, 0.01) + pagerank.get(m_b, 0.01)) * 15.0
                distance_bonus = min(path_len, 6) * 0.45
                overlap_bonus = len(shared_ds) * 0.55

                hole_score = round(
                    recency_bonus + centrality_score + distance_bonus + overlap_bonus, 4
                )

                lim_a = self.graph.nodes[m_a].get("limitations", "")
                lim_b = self.graph.nodes[m_b].get("limitations", "")
                eval_context = shared_ds or union_ds[:2] or ["retrieved domain evaluation settings"]

                candidates.append(
                    (
                        hole_score,
                        {
                            "concept_a": m_a,
                            "concept_b": m_b,
                            "year_a": year_a,
                            "year_b": year_b,
                            "paper_a": self.graph.nodes[m_a].get("paper_id", ""),
                            "paper_b": self.graph.nodes[m_b].get("paper_id", ""),
                            "method_graph_distance": path_len,
                            "shared_datasets": eval_context,
                            "limitation_a": lim_a,
                            "limitation_b": lim_b,
                            "hole_bridge_score": hole_score,
                            "claim_status": "proposed_hypothesis",
                            "structural_gap_rationale": (
                                f"Candidate hypothesis: '{m_a}' ({year_a}) and '{m_b}' ({year_b}) "
                                f"are {path_len} hops apart in the retrieved corpus graph without direct integration, "
                                f"while addressing {', '.join(eval_context)}. Investigating their combination "
                                f"is a proposed future direction to address their respective limitations."
                            ),
                        },
                    )
                )

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        used_nodes = set()
        for _, hole in candidates:
            if hole["concept_a"] not in used_nodes and hole["concept_b"] not in used_nodes:
                selected.append(hole)
                used_nodes.add(hole["concept_a"])
                used_nodes.add(hole["concept_b"])
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for _, hole in candidates:
                if hole not in selected:
                    selected.append(hole)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    def export_graph(
        self,
        output_dir: Path,
        structural_holes: Optional[List[Dict[str, Any]]] = None,
        topic: str = "",
    ) -> Dict[str, str]:
        """
        Save the knowledge graph as JSON (`knowledge_graph.json`) and render a
        visual graph plot in both `knowledge_graph.png` and `knowledge_graph.jpg`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "knowledge_graph.json"
        png_path = output_dir / "knowledge_graph.png"
        jpg_path = output_dir / "knowledge_graph.jpg"

        active_topic = (topic or self.current_topic or "Retrieved Research Corpus").strip()
        holes = structural_holes if structural_holes is not None else self.find_structural_holes()

        payload = {
            "topic": active_topic,
            "summary": self.summary(),
            "structural_hole_applicable": len(holes) > 0,
            "nodes": [
                {"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **attrs}
                for u, v, attrs in self.graph.edges(data=True)
            ],
            "structural_holes": holes,
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            fig, ax = plt.subplots(figsize=(14, 6.8), dpi=220)
            fig.patch.set_facecolor("#F8FAFC")
            ax.set_facecolor("#F8FAFC")

            vis_nodes = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("node_type") in {"method", "dataset"}
            ]
            subg = self.graph.subgraph(vis_nodes)

            if len(subg.nodes) > 0:
                pos = nx.spring_layout(subg, k=1.25, iterations=85, seed=42)
                method_list = [
                    n for n in subg.nodes if subg.nodes[n].get("node_type") == "method"
                ]
                dataset_list = [
                    n for n in subg.nodes if subg.nodes[n].get("node_type") == "dataset"
                ]

                nx.draw_networkx_nodes(
                    subg,
                    pos,
                    nodelist=method_list,
                    node_color="#1D4ED8",
                    edgecolors="#0F172A",
                    linewidths=1.2,
                    node_size=680,
                    alpha=0.94,
                    ax=ax,
                )
                nx.draw_networkx_nodes(
                    subg,
                    pos,
                    nodelist=dataset_list,
                    node_color="#0D9488",
                    edgecolors="#0F172A",
                    linewidths=1.0,
                    node_size=460,
                    node_shape="s",
                    alpha=0.88,
                    ax=ax,
                )

                lineage_edges = [
                    (u, v)
                    for u, v, d in subg.edges(data=True)
                    if d.get("predicate") in {"improves_upon", "outperforms", "extends"}
                ]
                eval_edges = [
                    (u, v)
                    for u, v, d in subg.edges(data=True)
                    if d.get("predicate") == "evaluated_on"
                ]

                nx.draw_networkx_edges(
                    subg,
                    pos,
                    edgelist=lineage_edges,
                    edge_color="#2563EB",
                    width=1.8,
                    arrows=True,
                    arrowsize=14,
                    alpha=0.85,
                    ax=ax,
                )
                nx.draw_networkx_edges(
                    subg,
                    pos,
                    edgelist=eval_edges,
                    edge_color="#94A3B8",
                    style="dotted",
                    width=1.1,
                    arrows=False,
                    alpha=0.75,
                    ax=ax,
                )

                hole_edges = [
                    (h["concept_a"], h["concept_b"])
                    for h in holes
                    if h["concept_a"] in pos and h["concept_b"] in pos
                ]
                if hole_edges:
                    nx.draw_networkx_edges(
                        subg,
                        pos,
                        edgelist=hole_edges,
                        edge_color="#E11D48",
                        style="dashed",
                        width=2.8,
                        arrows=False,
                        ax=ax,
                    )

                for node, (x_coord, y_coord) in pos.items():
                    is_method = subg.nodes[node].get("node_type") == "method"
                    ax.text(
                        x_coord,
                        y_coord + (0.045 if is_method else -0.048),
                        node,
                        fontsize=7.8 if is_method else 6.8,
                        fontweight="bold" if is_method else "normal",
                        color="#0F172A" if is_method else "#1E293B",
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "facecolor": "#FFFFFF" if is_method else "#F1F5F9",
                            "edgecolor": "#CBD5E1" if is_method else "none",
                            "alpha": 0.90,
                        },
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Insufficient Literature Entities Extracted for Topic: {active_topic}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#475569",
                )

            legend_handles = [
                Line2D([0], [0], marker="o", color="w", label="Methodology / Algorithm Node (V_method)", markerfacecolor="#1D4ED8", markersize=9),
                Line2D([0], [0], marker="s", color="w", label="Benchmark / Evaluation Setting (V_dataset)", markerfacecolor="#0D9488", markersize=8),
                Line2D([0], [0], color="#2563EB", lw=2, label="Comparative / Lineage Edge (improves_upon)"),
                Line2D([0], [0], color="#94A3B8", lw=1.2, ls="dotted", label="Evaluation Coupling (evaluated_on)"),
                Line2D([0], [0], color="#E11D48", lw=2.6, ls="dashed", label="Candidate Hypothesis Bridge"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=3,
                frameon=True,
                facecolor="#FFFFFF",
                edgecolor="#CBD5E1",
                fontsize=8.0,
            )

            ax.set_title(
                f"Methodological Knowledge Graph G = (V, E) & Candidate Bridges: {active_topic[:75]}",
                color="#0F172A",
                fontsize=11.0,
                fontweight="bold",
                pad=10,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E1")
                spine.set_linewidth(1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(png_path, facecolor=fig.get_facecolor(), dpi=220)
            plt.savefig(jpg_path, facecolor=fig.get_facecolor(), dpi=220, format="jpg")
            plt.close(fig)
        except Exception:
            pass

        return {
            "json_path": str(json_path),
            "png_path": str(png_path),
            "jpg_path": str(jpg_path),
        }

    def summary(self) -> Dict[str, Any]:
        node_types: Dict[str, int] = {}
        primary_methods_count = 0
        for _, attrs in self.graph.nodes(data=True):
            nt = attrs.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1
            if nt == "method" and attrs.get("paper_id"):
                primary_methods_count += 1

        return {
            "topic": self.current_topic,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_triads": len(self.triads),
            "node_type_counts": node_types,
            "methodologies_extracted": len(self.methodology_records),
            "structural_hole_applicable": primary_methods_count >= 2,
        }
