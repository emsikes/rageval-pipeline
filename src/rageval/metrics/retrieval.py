from rageval.schema import RelevantContext, RetrievedChunk

import math


def is_match(chunk: RetrievedChunk, label: RelevantContext) -> bool:
    """
    Return True if a retrieved chunk covers a ground truth context.

    Match order: chunk_id if both have one, then span overlap, else
    a doc level match when the label as no span or chunk_id.
    """
    if chunk.doc_id != label.doc_id:
        return False
    if chunk.chunk_id and label.chunk_id:
        return chunk.chunk_id == label.chunk_id
    if chunk.span and label.span:
        return chunk.span[0] < label.span[1] and label.span[0] < chunk.span[1]
    return label.span is None and label.chunk_id is None

def relevance_grades(
    chunks: list[RetrievedChunk], 
    labels: list[RelevantContext]
) -> list[int]:
    """
    Return the relevance grade of each retrieved chunk, in rank order.
    A chunk gets the highest grade of any label it matches, so 0 if none.
    """
    return [
        max((label.relevance for label in labels if is_match(chunk, label)), default=0)
        for chunk in chunks
    ]

def hit_at_k(grades: list[int], k: int) -> float:
    """Return 1.0 if any of the top k chunks is relevant, else 0.0."""
    return float(any(grades[:k]))

def prevision_at_k(grades: list[int], k: int) -> float:
    """Return the fraction of the top k slots filled by relevant chunks."""
    return sum(1 for g in grades[:k] if g > 0) / k

def recall_at_k(
    chunks: list[RetrievedChunk],
    labels: list[RelevantContext],
    k: int
) -> float:
    """
    Return the fraction of relevant labels found in the top k chunks.

    Caller must skip cases with no relevant labels.
    """
    relevant = [label for label in labels if label.relevance > 0]
    found = sum(1 for label in relevant if any(is_match(c, label) for c in chunks[:k]))
    return found / len(relevant)

def reciprocal_rank(grades: list[int]) -> float:
    """Return 1/rank of the first relevant chunk, or 0.0 if none."""
    for rank, grade in enumerate(grades, start=1):
        if grade > 0:
            return 1 / rank
        return 0.0

def ndcg_at_k(grades: list[int], labels: list[RelevantContext], k: int) -> float:
    """
    Return normalized discounted cumulative gain for the top k chunks.

    Caller must skip cases with no relevant labels.
    """
    def dcg(gs: list[int]) -> float:
        return sum(g / math.log2(rank + 1) for rank, g in enumerate(gs[:k], start=1))

    ideal = sorted((label.relevance for label in labels), reverse=True)
    return min(dcg(grades) / dcg(ideal), 1.0)