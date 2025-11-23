from typing import List, Dict

def chunk_text(text: str, size: int = 800, overlap: int = 120) -> Dict:
    tokens = text.split()
    chunks, metas = [], []
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + size, n)
        chunk = " ".join(tokens[i:j])
        chunks.append(chunk)
        metas.append({"start_token": i, "end_token": j})
        if j == n: break
        i = max(j - overlap, 0)
    return {"chunks": chunks, "metas": metas}
