"""
Vector Database & RAG Pipeline (`RecursiveCharacterTextSplitter` + `ChromaDB`).

Supports native `langchain_text_splitters` and `chromadb` when installed, with a
deterministic dense cosine-similarity vector store fallback so semantic chunking
and retrieval work out-of-the-box in any local Python environment.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.arxiv_search import AcademicPaper


@dataclass
class DocumentChunk:
    """A single embedded text chunk with provenance metadata."""

    chunk_id: str
    paper_id: str
    title: str
    year: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively across a hierarchy of separators (`["\\n\\n", "\\n", ". ", " ", ""]`),
    delegating to `langchain_text_splitters.RecursiveCharacterTextSplitter` when available.
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = max(100, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size // 2))
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._lc_splitter = None
        try:
            from langchain_text_splitters import (
                RecursiveCharacterTextSplitter as LCRecursiveSplitter,
            )

            self._lc_splitter = LCRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
        except Exception:
            self._lc_splitter = None

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if self._lc_splitter is not None:
            return self._lc_splitter.split_text(text)
        return self._native_recursive_split(text.strip(), self.separators)

    def _native_recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep = separators[0]
        next_seps = separators[1:]
        splits = text.split(sep) if sep else list(text)

        chunks: List[str] = []
        current = ""

        for part in splits:
            piece = part if not sep or not current else sep + part
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current:
                    chunks.append(current.strip())
                    overlap_tail = (
                        current[-self.chunk_overlap :].strip()
                        if self.chunk_overlap > 0
                        else ""
                    )
                    current = (overlap_tail + (" " if overlap_tail else "") + part).strip()
                else:
                    if len(part) > self.chunk_size and next_seps:
                        chunks.extend(self._native_recursive_split(part, next_seps))
                    else:
                        chunks.append(part[: self.chunk_size].strip())
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if c]


def _deterministic_embed(text: str, dim: int = 256) -> List[float]:
    """
    Compute a normalized n-gram hashed bag-of-concepts embedding vector in R^dim
    for fast local semantic similarity retrieval without external API latency.
    """
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if not tokens:
        vec[0] = 1.0
        return vec

    # Unigrams + bigrams for phrase sensitivity
    features = list(tokens) + [
        f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
    ]
    for feat in features:
        h = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "big") % dim
        sign = 1.0 if (h[2] % 2 == 0) else -1.0
        weight = 1.5 if "_" in feat else 1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class ChromaRAGEngine:
    """
    ChromaDB-compatible RAG Vector Database for the AI Scientist pipeline.
    Uses `chromadb.EphemeralClient` (or `PersistentClient`) when installed,
    and maintains a mirrored in-memory embedding index for deterministic retrieval.
    """

    def __init__(
        self,
        collection_name: str = "ai_scientist_literature",
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        persist_directory: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._chunks: List[DocumentChunk] = []
        self._embeddings: List[List[float]] = []
        self._backend_name = "in_memory_chroma_compatible"
        self._chroma_collection = None

        self._chroma_client = None
        try:
            import chromadb

            self._chroma_client = (
                chromadb.PersistentClient(path=persist_directory)
                if persist_directory
                else chromadb.EphemeralClient()
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            self._backend_name = "chromadb"
        except Exception:
            self._chroma_client = None
            self._chroma_collection = None

    def clear(self, new_collection_name: Optional[str] = None) -> None:
        """
        Clear all indexed chunks and embeddings to guarantee strict topic isolation
        across consecutive runs.
        """
        self._chunks.clear()
        self._embeddings.clear()
        if new_collection_name:
            # Chroma collection names must be 3-63 alphanumeric/underscore/hyphen chars
            clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", new_collection_name).strip("_-")
            if len(clean_name) < 3:
                clean_name = f"topic_{clean_name}_db"
            self.collection_name = clean_name[:60]

        if self._chroma_client is not None:
            try:
                try:
                    self._chroma_client.delete_collection(name=self.collection_name)
                except Exception:
                    pass
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name
                )
            except Exception:
                self._chroma_collection = None

    def ingest_papers(
        self,
        papers: List[AcademicPaper],
        clear_existing: bool = True,
        topic_slug: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Chunk all papers via `RecursiveCharacterTextSplitter` and embed into Vector DB.
        By default clears existing chunks so previous topic runs cannot contaminate retrieval.
        """
        if clear_existing:
            col_name = f"scientist_{topic_slug}" if topic_slug else None
            self.clear(new_collection_name=col_name)

        new_chunks: List[DocumentChunk] = []
        for paper in papers:
            full_doc = paper.to_document_text()
            raw_splits = self.splitter.split_text(full_doc)
            for idx, chunk_text in enumerate(raw_splits):
                cid = f"{paper.paper_id}::chunk-{idx}"
                meta = {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "primary_method": paper.primary_method,
                    "improves_upon": paper.improves_upon,
                    "datasets": ", ".join(paper.datasets),
                    "limitations": paper.limitations,
                }
                doc_chunk = DocumentChunk(
                    chunk_id=cid,
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                    text=chunk_text,
                    metadata=meta,
                )
                new_chunks.append(doc_chunk)
                self._chunks.append(doc_chunk)
                self._embeddings.append(_deterministic_embed(chunk_text))

        if self._chroma_collection is not None and new_chunks:
            try:
                self._chroma_collection.upsert(
                    ids=[c.chunk_id for c in new_chunks],
                    documents=[c.text for c in new_chunks],
                    embeddings=[_deterministic_embed(c.text) for c in new_chunks],
                    metadatas=[c.metadata for c in new_chunks],
                )
            except Exception:
                pass

        return {
            "backend": self._backend_name,
            "papers_ingested": len(papers),
            "chunks_added": len(new_chunks),
            "total_chunks": len(self._chunks),
        }

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most semantically relevant chunks for `query`.
        """
        if not self._chunks:
            return []

        q_emb = _deterministic_embed(query)
        scored = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            sim = _cosine_similarity(q_emb, emb)
            scored.append((sim, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: List[Dict[str, Any]] = []
        for score, chunk in scored[: max(1, int(top_k))]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "year": chunk.year,
                    "similarity": round(score, 4),
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        return results

    def get_all_chunks(self) -> List[DocumentChunk]:
        return list(self._chunks)

    def stats(self) -> Dict[str, Any]:
        unique_papers = {c.paper_id for c in self._chunks}
        return {
            "collection_name": self.collection_name,
            "backend": self._backend_name,
            "total_chunks": len(self._chunks),
            "unique_papers": len(unique_papers),
        }
