import json, boto3
from botocore.config import Config as BotoConfig
from utils.config import AWS_REGION, LLM_MODEL_ID, model_provider

_brt = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
)

SYSTEM_RULE = (
    "당신은 교재 기반 문제 생성기입니다. 업로드된 문서에서 제공된 컨텍스트만을 근거로 "
    "문항(다지선다/단답)과 정답, 해설을 한국어로 생성합니다. 근거가 없으면 '문서 근거 부족'이라고 답합니다."
)

def make_question_prompt(user_query: str, context: str) -> str:
    return (
        f"[규칙]\n{SYSTEM_RULE}\n\n"
        f"[컨텍스트]\n{context}\n\n"
        f"[요청]\n"
        f"- 위 컨텍스트를 바탕으로 학생 평가용 문제를 만들어 주세요.\n"
        f"- 다양한 난이도의 객관식/단답형 문제를 섞어서 3~5문항 정도 생성하세요.\n"
        f"- 각 문항마다 보기(객관식), 정답, 해설을 함께 작성하세요.\n"
        f"- 출력 형식은 번호를 붙인 목록 형태로 깔끔하게 정리하세요.\n\n"
        f"[사용자 지정 요청]\n{user_query}\n"
    )

def invoke_llm(prompt: str, model_id=None, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Bedrock LLM 호출 (Anthropic Claude / Amazon Nova)."""
    if model_id is None:
        model_id = LLM_MODEL_ID

    provider = model_provider(model_id)

    # ===== Anthropic Claude 계열 =====
    if provider == "anthropic":
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = _brt.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())

        txt = ""
        for c in payload.get("content", []):
            if c.get("type") == "text":
                txt += c.get("text", "")
        if not txt:
            txt = payload.get("output_text", "")
        return txt.strip()

    # ===== Amazon Nova (nova-lite / nova-pro 등) =====
    if provider == "nova":
        # Nova는 messages-v1 스키마 사용 :contentReference[oaicite:1]{index=1}
        body = {
            "schemaVersion": "messages-v1",
            "system": [{"text": SYSTEM_RULE}],
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            },
        }

        resp = _brt.invoke_model(modelId=model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())

        # Nova 응답: output.message.content[0].text 형태 :contentReference[oaicite:2]{index=2}
        try:
            return payload["output"]["message"]["content"][0]["text"].strip()
        except Exception:
            # 혹시 스키마가 조금 다르거나 에러 시 전체를 문자열로 반환
            return json.dumps(payload, ensure_ascii=False)

    # 그 외 미지원 LLM
    raise ValueError(f"Unsupported LLM model: {model_id}")
