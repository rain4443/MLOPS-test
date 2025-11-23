import fitz  # PyMuPDF

def load_pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """메모리 상의 PDF 바이트를 받아 전체 텍스트를 추출"""
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)
