import os, json, numpy as np, boto3
from botocore.config import Config as BotoConfig
from typing import List
from utils.config import AWS_REGION, EMBED_MODEL_ID, model_provider

_brt = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
)


def bedrock_embed(texts: List[str], model_id: str = None) -> np.ndarray:
    """Bedrock 임베딩 모델 통합 호출. Titan / Cohere 모두 지원."""
    model_id = model_id or EMBED_MODEL_ID
    provider = model_provider(model_id)

    # ===== Titan 계열 (예: amazon.titan-embed-text-v2:0) =====
    if provider == "titan":
        # 문자열 1개 / 리스트 모두 처리
        if isinstance(texts, str):
            body = {"inputText": texts}
        else:
            # 일부 버전은 inputText 리스트도 허용, 일부는 texts만 허용 → 둘 다 시도
            body = {"inputText": texts}

        resp = _brt.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())

        if "embeddings" in payload:
            vecs = payload["embeddings"]
        elif "embedding" in payload:  # 단일 벡터
            vecs = [payload["embedding"]]
        else:
            # 구형/특이 스키마 방어 코드
            results = payload.get("results", [])
            if results and "embedding" in results[0]:
                emb = results[0]["embedding"]
                vecs = [emb] if isinstance(emb[0], (int, float)) else emb
            else:
                raise RuntimeError(f"Titan 응답에서 임베딩을 찾을 수 없습니다: {payload}")

        return np.array(vecs, dtype="float32")

    # ===== Cohere 계열 (cohere.embed-v3 / v4) =====
    if provider == "cohere":
        # 항상 리스트 형태로 맞추기
        input_texts = texts if isinstance(texts, list) else [texts]

        # v3/v4 공통 요청 포맷 (v4는 embedding_types 지정 권장)
        body = {
            "texts": input_texts,
            "input_type": "search_document",
            "embedding_types": ["float"],  # v4에서 float 타입 사용
        }

        resp = _brt.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())

        emb = payload.get("embeddings")
        if emb is None:
            raise RuntimeError(f"Cohere 응답에서 'embeddings' 필드를 찾을 수 없습니다: {payload}")

        # ✅ v4: {"embeddings": {"float": [[...], ...], "int8": [[...], ...]}}
        if isinstance(emb, dict):
            if "float" in emb:
                vecs = emb["float"]
            else:
                # 혹시 다른 타입만 넘어오는 경우 첫 번째 타입 사용
                first_key = next(iter(emb.keys()))
                vecs = emb[first_key]
        else:
            # ✅ v3: {"embeddings": [[...], [...]]}
            vecs = emb

        return np.array(vecs, dtype="float32")

    # 여기까지 안 걸리면 아직 지원 안 하는 모델
    raise ValueError(f"Unsupported embedding model: {model_id}")
