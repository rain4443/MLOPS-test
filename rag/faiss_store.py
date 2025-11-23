import os, pickle, faiss, numpy as np
from typing import Dict, Tuple
from utils.config import LOCAL_CACHE_DIR
from utils import s3

def build_faiss(embs: np.ndarray) -> faiss.Index:
    # 내적(코사인 유사도용) 사용 → 벡터를 정규화 권장
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def save_local(index, meta: Dict, dirpath: str):
    os.makedirs(dirpath, exist_ok=True)
    faiss.write_index(index, os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

def load_local(dirpath: str) -> Tuple[faiss.Index, Dict]:
    index = faiss.read_index(os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return index, meta

def upload_to_s3(cache_dir: str, s3_prefix: str):
    s3.upload_file(os.path.join(cache_dir, "index.faiss"), f"{s3_prefix}/index.faiss")
    s3.upload_file(os.path.join(cache_dir, "meta.pkl"),   f"{s3_prefix}/meta.pkl")

def download_from_s3(cache_dir: str, s3_prefix: str):
    os.makedirs(cache_dir, exist_ok=True)
    s3.download_file(f"{s3_prefix}/index.faiss", os.path.join(cache_dir, "index.faiss"))
    s3.download_file(f"{s3_prefix}/meta.pkl",   os.path.join(cache_dir, "meta.pkl"))

def ensure_local_index(doc_id: str, s3_bucket_prefix: str) -> Tuple[faiss.Index, Dict]:
    cache_dir = os.path.join(LOCAL_CACHE_DIR, doc_id)
    idx_path = os.path.join(cache_dir, "index.faiss")
    if not os.path.exists(idx_path):
        download_from_s3(cache_dir, f"{s3_bucket_prefix}/{doc_id}")
    return load_local(cache_dir)
