import os, io, time, uuid, numpy as np, streamlit as st
from utils.config import (AWS_REGION, S3_BUCKET, EMBED_MODEL_ID, LLM_MODEL_ID,
                          CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, LOCAL_CACHE_DIR)
from utils import s3 as s3util
from rag.loader import load_pdf_bytes_to_text
from rag.chunker import chunk_text
from rag.embedder import bedrock_embed
from rag.faiss_store import build_faiss, save_local, upload_to_s3, ensure_local_index
from rag.retriever import search_topk, build_context
from rag.generator import make_question_prompt, invoke_llm

st.set_page_config(page_title="문제집 AI (RAG@Bedrock)", layout="wide")
st.title("📘 학업의 진심이 나라를 위한 문제집 AI")

with st.sidebar:
    st.subheader("환경")
    st.write(f"AWS Region: `{AWS_REGION}`")
    st.write(f"S3 Bucket: `{S3_BUCKET}`")
    st.write(f"Embed: `{EMBED_MODEL_ID}`")
    st.write(f"LLM: `{LLM_MODEL_ID}`")
    st.divider()
    st.caption("🔒 업로드된 PDF와 임베딩 인덱스는 S3에 저장됩니다.")

tab1, tab2 = st.tabs(["1) 문서 업로드 & 인덱싱", "2) 검색/질의 & 문제 생성"])

# ===== 1) 업로드 & 임베딩/FAISS =====
with tab1:
    st.header("문서 업로드 및 인덱스 생성")
    pdf_file = st.file_uploader("PDF 업로드", type=["pdf"])
    colA, colB = st.columns(2)
    with colA:
        chunk_size = st.number_input("Chunk Size", 200, 2000, CHUNK_SIZE, step=50)
    with colB:
        chunk_overlap = st.number_input("Chunk Overlap", 0, 500, CHUNK_OVERLAP, step=10)

    if st.button("인덱스 생성", disabled=(pdf_file is None)):
        with st.spinner("PDF 처리 중..."):
            pdf_bytes = pdf_file.read()
            text = load_pdf_bytes_to_text(pdf_bytes)

        with st.spinner("청크 분할..."):
            cobj = chunk_text(text, size=int(chunk_size), overlap=int(chunk_overlap))
            chunks, metas = cobj["chunks"], cobj

        with st.spinner("임베딩 & 인덱스 생성(Bedrock)..."):
            # 배치 분할로 임베딩 (간단 버전: 통짜 호출)
            embs = bedrock_embed(chunks, EMBED_MODEL_ID).astype("float32")
            # 정규화(코사인 유사도)
            from faiss import normalize_L2
            normalize_L2(embs)
            index = build_faiss(embs)

        # 문서 ID 생성 및 S3 저장
        doc_id = uuid.uuid4().hex
        cache_dir = os.path.join(LOCAL_CACHE_DIR, doc_id)
        # 메타에 원본 chunks 포함(간단화)
        metas["chunks"] = chunks

        with st.spinner("S3 저장 중..."):
            # 원문 PDF 저장
            s3util._s3.put_object(Bucket=S3_BUCKET,
                                  Key=f"pdfs/{doc_id}/input.pdf", Body=pdf_bytes)
            # FAISS + meta 저장
            save_local(index, metas, cache_dir)
            upload_to_s3(cache_dir, f"faiss/{doc_id}")

        st.success("인덱스 생성 & 업로드 완료!")
        st.code(f"doc_id = {doc_id}", language="bash")
        st.session_state["last_doc_id"] = doc_id

# ===== 2) 검색/질의 =====
with tab2:
    st.header("검색/질의 & 문제 생성")
    doc_id = st.text_input("doc_id 입력(또는 좌측 탭에서 생성 후 자동 채움)",
                           value=st.session_state.get("last_doc_id", ""))

    query = st.text_area("요청(예: '중요 개념 3개에 대해 5지선다 2문제씩 만들어줘')", height=120)

    c1, c2, c3 = st.columns(3)
    with c1:
        topk = st.number_input("Top-K", 1, 20, TOP_K)
    with c2:
        max_tokens = st.number_input("Max Tokens", 200, 4000, 1000, step=100)
    with c3:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    if st.button("검색 & 생성 실행", disabled=(len(doc_id.strip()) == 0 or len(query.strip()) == 0)):
        try:
            with st.spinner("인덱스 로딩(S3 캐시)…"):
                index, meta = ensure_local_index(doc_id, "faiss")
                chunks = meta["chunks"]

            with st.spinner("Top-K 검색…"):
                idxs, scores = search_topk(query, index, meta, k=int(topk))
                ctx = build_context(meta, idxs, chunks)

            with st.expander("🔎 검색 컨텍스트(근거)", expanded=False):
                st.write(ctx)

            with st.spinner("LLM 생성(Bedrock)…"):
                prompt = make_question_prompt(query, ctx)
                answer = invoke_llm(prompt, model_id=LLM_MODEL_ID,
                                    max_tokens=int(max_tokens), temperature=float(temperature))

            st.subheader("🧩 생성 결과")
            st.markdown(answer)
        except Exception as e:
            st.error(f"오류: {e}")
