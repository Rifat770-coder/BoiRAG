import os

from langchain_huggingface import HuggingFaceEmbeddings

from config import DIMENSION, MODEL


def load_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_PATH") or MODEL,
            model_kwargs={
                "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
                "trust_remote_code": False,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        vector = embeddings.embed_query("বাংলা বই")
    except Exception as exc:
        raise RuntimeError("BGE-M3 could not load or encode. Check model access and device.") from exc
    if len(vector) != DIMENSION:
        raise ValueError(f"Expected {DIMENSION} embedding dimensions, received {len(vector)}.")
    return embeddings
