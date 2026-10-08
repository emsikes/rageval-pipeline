from pydantic import BaseModel, Field
from typing import Literal


class RelevantContext(BaseModel):
    """
    A ground truth passage that answers all or part of an eval question.

    Located by doc_id and an optional character, so labels survive re-chunking.
    chunk_id is static only when chunking is fixed.
    """

    doc_id: str
    span: tuple[int, int] | None = None
    chunk_id: str | None = None
    text: str | None = None
    relevance: int = Field(default=1, ge=0, le=3)


class EvalCase(BaseModel):
    """One evaluation question with its ground truth."""

    case_id: str
    question: str
    reference_answer: str | None = None
    relevant_contexts: list[RelevantContext] = []
    answerable: bool = True
    question_type: Literal["factoid", "multi_hop", "comparison"] = "factoid"
    source: Literal["human", "synthetic"] = "human"
    tags: list[str] = []


class RetrievedChunk(BaseModel):
    """A chunk returned by the RAG system under test."""

    doc_id: str
    chunk_id: str | None = None
    span: tuple[int, int] | None = None
    text: str
    score: float | None = None


class RunRecord(BaseModel):
    """What the RAG system under test returned for one EvalCase."""

    case_id: str
    answer: str
    retrieved: list[RetrievedChunk] = []
    citations: list[str] = []
    latency_ms: float | None = None
    cost_usd: float | None = None


class ScoreRecord(BaseModel):
    """Metric results for one run."""

    case_id: str
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}