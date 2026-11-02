"""
OpenRouter LLM Integration Client for Multi-Agent AI Scientist (`tools/llm_client.py`).

Features:
- Loads `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` from environment or `.env`
  without ever printing, logging, or exposing credentials.
- Bounded retries with backoff for transient rate limits (HTTP 429) and server errors (5xx).
- Validates HTTP status, provider error payloads, empty responses, token truncation,
  and JSON schema compliance locally.
- Handles models that do not support `response_format={"type": "json_object"}` by
  automatically retrying without `response_format` and extracting the JSON object.
- Records structured stage diagnostics (`stage`, `model`, `latency_ms`, `status`,
  `error_category`) without logging secrets.
- Raises explicit `LLMGenerationError` when requested instead of silently masking failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.arxiv_search import AcademicPaper

logger = logging.getLogger("ai_scientist.llm_client")

BASE_DIR = Path(__file__).resolve().parent.parent


class LLMGenerationError(RuntimeError):
    """Explicit exception raised when an OpenRouter generation stage fails."""

    def __init__(
        self,
        stage: str,
        error_category: str,
        message: str,
        model: str = "",
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(f"[{stage}] ({error_category}) {message}")
        self.stage = stage
        self.error_category = error_category
        self.message = message
        self.model = model
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "error_category": self.error_category,
            "message": self.message,
            "model": self.model,
            "status_code": self.status_code,
        }


def _ensure_env_loaded() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env", override=False)
    except Exception:
        pass


def get_llm_config() -> Dict[str, Any]:
    """Retrieve OpenRouter configuration from environment or `.env`."""
    _ensure_env_loaded()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    temp_str = os.getenv("OPENROUTER_TEMPERATURE", "0.2")
    try:
        temperature = float(temp_str)
    except ValueError:
        temperature = 0.2

    return {
        "api_key": api_key.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "model": model.strip(),
        "temperature": temperature,
    }


def is_llm_available() -> bool:
    """Check if a non-empty LLM API key is configured."""
    cfg = get_llm_config()
    return bool(cfg["api_key"])


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    Robustly parse and validate a JSON object from model output, stripping markdown
    code fences if present.
    """
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ValueError("Model returned empty content string.")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to locate the outermost JSON object `{ ... }`
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
        else:
            raise

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object (dict), got {type(parsed).__name__}.")
    return parsed


class OpenRouterClient:
    """
    Production OpenRouter API client with bounded retries, structured JSON validation,
    fallback for models without `response_format` support, and stage telemetry.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: int = 45,
        max_retries: int = 2,
    ) -> None:
        cfg = get_llm_config()
        self.api_key = (api_key if api_key is not None else cfg["api_key"]).strip()
        self.model = (model if model is not None else cfg["model"]).strip()
        self.base_url = (base_url if base_url is not None else cfg["base_url"]).strip().rstrip("/")
        self.temperature = temperature if temperature is not None else cfg["temperature"]
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.diagnostics: List[Dict[str, Any]] = []

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def call_json(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        required_keys: Optional[List[str]] = None,
        max_tokens: int = 2800,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a chat-completion request expecting a validated JSON object.
        Raises `LLMGenerationError` with detailed stage and failure category if all attempts fail.
        """
        if not self.api_key:
            self._log_event(stage, 0.0, "failed", "missing_credentials")
            raise LLMGenerationError(
                stage=stage,
                error_category="missing_credentials",
                message="OPENROUTER_API_KEY is not configured.",
                model=self.model,
            )

        import requests

        req_timeout = timeout or self.timeout
        req_temp = temperature if temperature is not None else self.temperature
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://ai-scientist.local",
            "X-Title": "AI Scientist Grounded Manuscript Pipeline",
            "Content-Type": "application/json",
        }

        use_json_format = True
        last_error: Optional[LLMGenerationError] = None

        for attempt in range(1, self.max_retries + 2):
            t0 = time.monotonic()
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": req_temp,
                "max_tokens": max_tokens,
            }
            if use_json_format:
                payload["response_format"] = {"type": "json_object"}

            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=req_timeout,
                )
                latency_ms = round((time.monotonic() - t0) * 1000.0, 1)

                # If model does not support response_format, retry without it
                if resp.status_code == 400 and use_json_format:
                    use_json_format = False
                    self._log_event(
                        stage, latency_ms, "retry", "unsupported_response_format", resp.status_code, attempt
                    )
                    continue

                if resp.status_code == 429:
                    self._log_event(stage, latency_ms, "retry", "rate_limit", 429, attempt)
                    last_error = LLMGenerationError(
                        stage=stage,
                        error_category="rate_limit",
                        message="OpenRouter rate limit (HTTP 429).",
                        model=self.model,
                        status_code=429,
                    )
                    if attempt <= self.max_retries:
                        time.sleep(min(2.0 * attempt, 5.0))
                        continue
                    raise last_error

                if resp.status_code >= 500:
                    self._log_event(stage, latency_ms, "retry", "server_error", resp.status_code, attempt)
                    last_error = LLMGenerationError(
                        stage=stage,
                        error_category="server_error",
                        message=f"OpenRouter server error (HTTP {resp.status_code}).",
                        model=self.model,
                        status_code=resp.status_code,
                    )
                    if attempt <= self.max_retries:
                        time.sleep(min(1.5 * attempt, 4.0))
                        continue
                    raise last_error

                if resp.status_code != 200:
                    self._log_event(stage, latency_ms, "failed", "http_error", resp.status_code, attempt)
                    raise LLMGenerationError(
                        stage=stage,
                        error_category="http_error",
                        message=f"Unexpected HTTP status {resp.status_code} from OpenRouter.",
                        model=self.model,
                        status_code=resp.status_code,
                    )

                try:
                    data = resp.json()
                except Exception as exc:
                    self._log_event(stage, latency_ms, "failed", "invalid_http_json", 200, attempt)
                    raise LLMGenerationError(
                        stage=stage,
                        error_category="invalid_http_json",
                        message=f"Response body is not valid JSON: {exc}",
                        model=self.model,
                        status_code=200,
                    ) from exc

                if "error" in data and data["error"]:
                    err_msg = str(data["error"].get("message", data["error"])) if isinstance(data["error"], dict) else str(data["error"])
                    self._log_event(stage, latency_ms, "failed", "provider_error", 200, attempt)
                    raise LLMGenerationError(
                        stage=stage,
                        error_category="provider_error",
                        message=f"Provider error: {err_msg[:200]}",
                        model=self.model,
                    )

                choices = data.get("choices") or []
                if not choices:
                    self._log_event(stage, latency_ms, "failed", "empty_choices", 200, attempt)
                    raise LLMGenerationError(
                        stage=stage,
                        error_category="empty_choices",
                        message="OpenRouter response contained no choices.",
                        model=self.model,
                    )

                first_choice = choices[0]
                finish_reason = first_choice.get("finish_reason", "")
                raw_content = (first_choice.get("message") or {}).get("content") or ""

                if not raw_content.strip():
                    self._log_event(stage, latency_ms, "failed", "empty_content", 200, attempt)
                    raise LLMGenerationError(
                        stage=stage,
                        error_category="empty_content",
                        message="Model returned an empty message content.",
                        model=self.model,
                    )

                try:
                    parsed = _extract_json_object(raw_content)
                except Exception as exc:
                    cat = "token_truncation" if finish_reason == "length" else "malformed_json"
                    self._log_event(stage, latency_ms, "failed", cat, 200, attempt)
                    last_error = LLMGenerationError(
                        stage=stage,
                        error_category=cat,
                        message=f"Failed to parse model JSON output ({cat}): {exc}",
                        model=self.model,
                    )
                    if attempt <= self.max_retries:
                        continue
                    raise last_error from exc

                if required_keys:
                    missing = [
                        k for k in required_keys
                        if k not in parsed or parsed[k] is None or (isinstance(parsed[k], str) and not parsed[k].strip())
                    ]
                    if missing:
                        self._log_event(stage, latency_ms, "failed", "schema_validation_error", 200, attempt)
                        last_error = LLMGenerationError(
                            stage=stage,
                            error_category="schema_validation_error",
                            message=f"Model JSON missing required non-empty keys: {missing}",
                            model=self.model,
                        )
                        if attempt <= self.max_retries:
                            continue
                        raise last_error

                self._log_event(stage, latency_ms, "success", "none", 200, attempt)
                return parsed

            except LLMGenerationError:
                raise
            except requests.exceptions.Timeout as exc:
                latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
                self._log_event(stage, latency_ms, "failed", "timeout", None, attempt)
                last_error = LLMGenerationError(
                    stage=stage,
                    error_category="timeout",
                    message=f"OpenRouter request timed out after {req_timeout}s.",
                    model=self.model,
                )
                if attempt <= self.max_retries:
                    continue
                raise last_error from exc
            except requests.exceptions.RequestException as exc:
                latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
                self._log_event(stage, latency_ms, "failed", "network_error", None, attempt)
                last_error = LLMGenerationError(
                    stage=stage,
                    error_category="network_error",
                    message=f"Network error communicating with OpenRouter: {exc.__class__.__name__}",
                    model=self.model,
                )
                if attempt <= self.max_retries:
                    continue
                raise last_error from exc

        if last_error is not None:
            raise last_error
        raise LLMGenerationError(
            stage=stage,
            error_category="unknown_error",
            message="Exhausted retries without completing OpenRouter request.",
            model=self.model,
        )

    def _log_event(
        self,
        stage: str,
        latency_ms: float,
        status: str,
        error_category: str,
        status_code: Optional[int] = None,
        attempt: int = 1,
    ) -> None:
        entry = {
            "stage": stage,
            "model": self.model,
            "latency_ms": latency_ms,
            "status": status,
            "error_category": error_category,
            "status_code": status_code,
            "attempt": attempt,
        }
        self.diagnostics.append(entry)
        logger.info(
            "OpenRouter stage=%s model=%s status=%s category=%s latency_ms=%.1f attempt=%d",
            stage,
            self.model,
            status,
            error_category,
            latency_ms,
            attempt,
        )


def verify_and_sanitize_citations_in_text(text: str, max_citation_index: int) -> str:
    """
    Ensure every bracketed numeric citation `[k]` in `text` resolves to a valid
    retrieved source `1 <= k <= max_citation_index`. Strips out-of-range or
    fabricated citation numbers.
    """
    if not text or max_citation_index <= 0:
        return re.sub(r"\s*\[\d+(?:\s*[\-–,]\s*\d+)*\]", "", text or "")

    def _replace_match(m: re.Match[str]) -> str:
        inner = m.group(1)
        # Handle range e.g. [1-5] or [1–5]
        range_match = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", inner.strip())
        if range_match:
            start_i = int(range_match.group(1))
            end_i = int(range_match.group(2))
            if start_i > end_i or start_i > max_citation_index:
                return ""
            clamped_start = max(1, start_i)
            clamped_end = min(max_citation_index, end_i)
            if clamped_start == clamped_end:
                return f"[{clamped_start}]"
            return f"[{clamped_start}–{clamped_end}]"

        # Handle comma-separated list e.g. [1, 3, 99]
        nums = [int(n) for n in re.findall(r"\d+", inner)]
        valid = [n for n in nums if 1 <= n <= max_citation_index]
        if not valid:
            return ""
        return "[" + ", ".join(str(n) for n in valid) + "]"

    cleaned = re.sub(r"\[(\d+(?:\s*[\-–,]\s*\d+)*)\]", _replace_match, text)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def synthesize_scholarly_prose(
    topic: str,
    papers: List[AcademicPaper],
    miner_report: Dict[str, Any],
    hypotheses: List[Any],
    adversarial_history: List[Dict[str, Any]],
    timeout: int = 45,
    raise_on_error: bool = False,
    client: Optional[OpenRouterClient] = None,
) -> Optional[Dict[str, Any]]:
    """
    Synthesize grounded manuscript sections for `topic` using `OpenRouterClient`.
    Validates schema and sanitizes all in-text citations against `1..len(papers)`.
    """
    if not papers:
        return None

    or_client = client or OpenRouterClient(timeout=timeout)
    if not or_client.is_configured():
        if raise_on_error:
            raise LLMGenerationError(
                stage="manuscript_synthesis",
                error_category="missing_credentials",
                message="OPENROUTER_API_KEY is not configured.",
                model=or_client.model,
            )
        return None

    g_sum = miner_report.get("graph_summary", {})
    holes = miner_report.get("structural_holes", [])

    papers_fact_lines = []
    for idx, p in enumerate(papers, start=1):
        m_str = ", ".join(f"{k}: {v}" for k, v in list(p.metrics.items())[:2])
        ds_str = ", ".join(p.datasets[:2])
        papers_fact_lines.append(
            f"[{idx}] PaperID: {p.paper_id} | Title: \"{p.title}\" | Authors: {p.first_author_citation()} | "
            f"Venue: {p.venue} | Method: '{p.primary_method}' | Compares With: '{p.improves_upon}' | "
            f"Evaluation Setting: {ds_str} | Reported Info: {m_str} | "
            f"Noted Challenge: {p.limitations} | Theme: {p.category} | "
            f"Abstract Excerpt: {p.abstract[:280]}"
        )
    papers_fact_block = "\n".join(papers_fact_lines)

    holes_fact_lines = []
    for idx, h in enumerate(holes[:3], start=1):
        holes_fact_lines.append(
            f"Candidate Gap {idx}: '{h['concept_a']}' <-> '{h['concept_b']}' "
            f"(Graph distance: {h.get('method_graph_distance', 4)} hops, Score: {h.get('hole_bridge_score', 5.0):.4f}) | "
            f"Shared Setting: {', '.join(h.get('shared_datasets', []))} | "
            f"Limitations: {h.get('concept_a')}: {h.get('limitation_a', '')[:100]}; "
            f"{h.get('concept_b')}: {h.get('limitation_b', '')[:100]}"
        )
    holes_fact_block = "\n".join(holes_fact_lines) if holes_fact_lines else "No structural holes identified."

    hyp_fact_lines = []
    for idx, hyp in enumerate(hypotheses[:3], start=1):
        hyp_fact_lines.append(
            f"Proposed Direction H-{idx} ('{hyp.short_name}'): Integrating '{hyp.concept_a}' and '{hyp.concept_b}'\n"
            f"  - Hypothesis Claim: {hyp.core_claim}\n"
            f"  - Proposed Protocol: {hyp.experimental_setup}"
        )
    hyp_fact_block = "\n\n".join(hyp_fact_lines) if hyp_fact_lines else "No separate hypotheses proposed."

    system_prompt = (
        "You are a Senior Academic Researcher and Literature Review Author writing a rigorous, "
        "evidence-grounded literature review and research synthesis.\n\n"
        "STRICT EVIDENCE & ANTI-HALLUCINATION RULES:\n"
        f"1. Write exclusively about the exact topic: \"{topic}\". Never introduce unrelated domains (e.g., do NOT mention hospitals, clinical trials, patients, or federated learning unless the topic and retrieved papers are explicitly about them).\n"
        f"2. Cite ONLY the provided retrieved papers using bracketed numbers [1] to [{len(papers)}]. Never cite numbers outside [1]..[{len(papers)}] and never invent authors, venues, datasets, or experimental results.\n"
        "3. Clearly distinguish established findings reported in the retrieved abstracts from the system's comparative synthesis and proposed future research hypotheses.\n"
        "4. Never claim that proposed hypotheses have been experimentally validated or mathematically proven by this review.\n"
        "5. Acknowledge that the synthesis is grounded in the retrieved abstracts and bibliographic metadata.\n"
        "6. Write in clear, natural, publication-quality academic English without buzzwords or filler."
    )

    user_prompt = f"""Synthesize a structured, evidence-grounded academic literature review for the topic: "{topic}".

--- RETRIEVED EVIDENCE MANIFEST ([1]–[{len(papers)}]) ---
{papers_fact_block}

--- KNOWLEDGE GRAPH & CANDIDATE RESEARCH GAPS ---
Nodes: {g_sum.get('total_nodes', 0)} | Edges: {g_sum.get('total_edges', 0)}
{holes_fact_block}

--- PROPOSED FUTURE RESEARCH DIRECTIONS (HYPOTHESES) ---
{hyp_fact_block}

--- REQUIRED JSON OUTPUT SCHEMA ---
Return a single valid JSON object with EXACTLY these keys:
{{
  "scholarly_title": "string: Clear, accurate academic title for a literature review on {topic} (must explicitly include {topic})",
  "abstract": "string: 140-190 word academic abstract summarizing the scope of {topic}, the {len(papers)} retrieved studies [1]–[{len(papers)}], key thematic paradigms, open challenges identified in the literature, and proposed future research directions.",
  "index_terms": "string: 5-6 comma-separated domain-accurate keywords for {topic}",
  "intro_p1": "string: 65-90 words introducing {topic}, its core technical objectives, and the scope of this review citing relevant foundational works from [1]–[{len(papers)}].",
  "intro_p2": "string: 65-90 words analyzing the primary technical challenges and trade-offs documented across the retrieved studies in {topic}.",
  "intro_p3": "string: 50-75 words explaining the review methodology (systematic metadata/abstract analysis and comparative graph mapping across the {len(papers)} retrieved papers).",
  "thematic_synthesis": "string: 120-160 words synthesizing the methodologies, baselines, and findings across the retrieved studies [1]–[{len(papers)}] in {topic}.",
  "cross_pillar_synthesis": "string: 80-110 words comparing complementary strengths and open limitations across the retrieved approaches in {topic}.",
  "conclusion": "string: 85-115 words summarizing the literature synthesis on {topic}, honest scope limitations (abstract-level evidence analysis), and proposed future directions."
}}"""

    required_keys = [
        "scholarly_title",
        "abstract",
        "intro_p1",
        "intro_p2",
        "cross_pillar_synthesis",
        "conclusion",
    ]

    try:
        parsed = or_client.call_json(
            stage="manuscript_synthesis",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_keys=required_keys,
            max_tokens=2800,
        )
        # Sanitize all citations in every text field against 1..len(papers)
        max_idx = len(papers)
        for k, v in list(parsed.items()):
            if isinstance(v, str) and k != "scholarly_title" and k != "index_terms":
                parsed[k] = verify_and_sanitize_citations_in_text(v, max_idx)

        # Ensure compatibility keys if legacy callers read pillar_a_text..pillar_d_text
        if "thematic_synthesis" in parsed and "pillar_a_text" not in parsed:
            parsed["pillar_a_text"] = parsed["thematic_synthesis"]
        parsed["_llm_model"] = or_client.model
        parsed["_diagnostics"] = list(or_client.diagnostics)
        return parsed
    except LLMGenerationError:
        if raise_on_error:
            raise
        return None
