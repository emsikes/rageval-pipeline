from rageval.metrics.retrieval import (
    is_match,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    ndcg_at_k,
    relevance_grades
)

from rageval.schema import RelevantContext as Label
from rageval.schema import RetrievedChunk as Chunk

import pytest


def test_is_match():
    """Covers each matching rule in priority order."""
    chunk = Chunk(doc_id="d1", chunk_id="c1", span=(0, 100), text="...")

    assert not is_match(chunk, Label(doc_id="d2"))                  # Test different doc
    assert is_match(chunk, Label(doc_id="d1", chunk_id="c1"))       # Test same doc, same chunk
    assert not is_match(chunk, Label(doc_id="d1", chunk_id="c2"))   # Test same doc, different chunk
    assert is_match(chunk, Label(doc_id="d1", span=(50, 150)))      # same doc, spans overlap (start, end text in doc)
    assert not is_match(chunk, Label(doc_id="d1", span=(100, 200))) # same doc, spans meet but don't overlap
    assert is_match(chunk, Label(doc_id="d1"))                      # doc level match
    assert not is_match(chunk, Label(doc_id="d1", span=(90, 200)))  # clips edge only


def test_grade_metrics():
    """Manually checked hit.  Precision and reciprocal rank on one ranked list."""
    grades = [0, 3, 0, 1, 0]                # Relevant chunks at ranks 2 and 4

    assert hit_at_k(grades, 1) == 0.0           # rank 1 is irrelevant
    assert hit_at_k(grades, 2) == 1.0           # rank 2 is relevant
    assert precision_at_k(grades, 5) == 0.4     # 2 relevant / 5 slots
    assert precision_at_k(grades, 10) == 0.2    # still divides by k, not results returned
    assert reciprocal_rank(grades) == 0.5       # first hit at rank 2
    assert reciprocal_rank([0, 0]) == 0.0       # no hit at all

def test_label_metrics():
    """Manually checked recall and nDCG using real chunks and labels."""
    labels = [
        Label(doc_id="d1", span=(0, 100), relevance=3),
        Label(doc_id="d1", span=(200, 300), relevance=1),
    ]
    chunks = [
        Chunk(doc_id="d1", span=(400, 500), text="..."),    # No match
        Chunk(doc_id="d1", span=(0, 100), text="..."),      # matches grade-3 label
        Chunk(doc_id="d1", span=(500, 600), text="..."),    # no match
        Chunk(doc_id="d1", span=(200, 300), text="..."),    # matches grade-1 label
    ]
    grades = relevance_grades(chunks, labels)

    assert grades == [0, 3, 0, 1]
    assert recall_at_k(chunks, labels, 2) == 0.5                            # 1 of 2 labels in top 2
    assert recall_at_k(chunks, labels, 4) == 1.0                            # both labels in top 4
    assert ndcg_at_k(grades, labels, 4) == pytest.approx(0.64, abs=0.001)
    assert ndcg_at_k([3, 1], labels, 2) == 1.0                              # perfect ordering