import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION     = os.getenv("AWS_REGION", "us-west-1")
S3_BUCKET      = os.getenv("S3_BUCKET", "ysu-ml-a-09-s3")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "us.cohere.embed-v4:0")
LLM_MODEL_ID   = os.getenv("LLM_MODEL_ID", "us.amazon.nova-pro-v1:0")

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K          = int(os.getenv("TOP_K", "5"))

LOCAL_CACHE_DIR = os.getenv("LOCAL_CACHE_DIR", "./data/cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

def model_provider(model_id: str) -> str:
    m = model_id.lower()
    if "anthropic" in m: return "anthropic"
    if "titan-embed" in m or "amazon.titan" in m: return "titan"
    if "cohere.embed" in m: return "cohere"
    if "mistral" in m: return "mistral"
    if "meta" in m or "llama" in m: return "meta"
    if "nova" in m or "amazon.nova" in m: return "nova"
    return "unknown"
