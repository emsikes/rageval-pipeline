from rageval.metrics.retrieval import is_match
from rageval.schema import RelevantContext as Label
from rageval.schema import RetrievedChunk as Chunk


def test_is_match():
    """Covers each matching rule in priority order."""
    chunk = Chunk(doc_id="d1", chunk_id="c1", span=(0, 100), text="...")

    assert not is_match(chunk, Label(doc_id="d2"))                  # Test different doc
    assert is_match(chunk, Label(doc_id="d1", chunk_id="c1"))       # Test same doc, same chunk
    assert not is_match(chunk, Label(doc_id="d1", chunk_id="c2"))   # Test same doc, different chunk
    assert is_match(chunk, Label(doc_id="d1", span=(50, 150)))      # same doc, spans overlap (start, end text in doc)
    assert not is_match(chunk, Label(doc_id="d1", span=(100, 200))) # same doc, spans meet but don't overlap
    assert is_match(chunk, Label(doc_id="d1"))                      # doc level match