from src.rag import QualityRAG

def test_search_returns_relevant_record():
    records = [
        {"source":"a.csv","record_id":"1","text":"bore oversize caused by incorrect tool offset","metadata":{}},
        {"source":"a.csv","record_id":"2","text":"surface scratch caused by handling","metadata":{}},
    ]
    rag = QualityRAG()
    rag.fit(records)
    hits = rag.search("oversize bore tool offset", top_k=1)
    assert hits and hits[0]["record_id"] == "1"
