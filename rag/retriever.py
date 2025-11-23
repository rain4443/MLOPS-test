import numpy as np, faiss
from typing import Dict, List, Tuple
from .embedder import bedrock_embed
from utils.config import EMBED_MODEL_ID

def search_topk(query: str, index: faiss.Index, meta: Dict, k: int = 5) -> Tuple[List[int], List[float]]:
    qvec = bedrock_embed([query], EMBED_MODEL_ID).astype("float32")
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    return I[0].tolist(), D[0].tolist()

def build_context(meta: Dict, idx_list: List[int], chunks: List[str]) -> str:
    ctx = []
    for i in idx_list:
        m = meta["metas"][i]
        ctx.append(f"[chunk#{i} tokens({m['start_token']}-{m['end_token']})]\n{chunks[i]}")
    return "\n\n".join(ctx)
