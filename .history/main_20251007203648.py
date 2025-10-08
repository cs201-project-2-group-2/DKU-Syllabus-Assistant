# main.py (patched)
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from fastapi.responses import FileResponse

# ====== 配置 ======
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
ACTION_API_KEY = os.getenv("ACTION_API_KEY", "replace-me")  # GPT Actions 用

# ====== FastAPI 基础 ======
app = FastAPI(title="Syllabi Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产可收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Pydantic 模型 ======
class SearchItem(BaseModel):
    id: int
    text: str
    score: float

class SearchResponse(BaseModel):
    ok: bool
    data: List[SearchItem]

class CourseItem(BaseModel):
    id: int
    info: str

class CoursesResponse(BaseModel):
    ok: bool
    data: List[CourseItem]

class SearchReq(BaseModel):
    query: str
    top_k: int = 3

# ====== 模型延迟加载 ======
_embed_model: Optional[SentenceTransformer] = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def embed_text(text: str) -> np.ndarray:
    emb = get_embed_model().encode(text, show_progress_bar=False, normalize_embeddings=False)
    # SentenceTransformers 返回 np.ndarray float32
    # 如果你的 FAISS 索引用的是内积（IndexFlatIP / IVF+IP），请把向量做单位化：
    # emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype("float32")

def check_auth(x_api_key: Optional[str]):
    if x_api_key != ACTION_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

# ====== 载入索引与映射 ======
with open("faiss_index.pkl", "rb") as f:
    index_data = pickle.load(f)

faiss_index = None
id_to_text: Dict[int, str] = {}

if isinstance(index_data, dict):
    faiss_index = index_data.get("index")
    id_to_text = index_data.get("id_to_text", {}) or {}
elif isinstance(index_data, (list, tuple)):
    for item in index_data:
        try:
            cid = int(item.get("id")) if isinstance(item, dict) and "id" in item else None
            text = item.get("text") if isinstance(item, dict) else str(item)
        except Exception:
            cid, text = None, str(item)
        if cid is not None:
            id_to_text[cid] = text
else:
    try:
        id_to_text = dict(index_data)
    except Exception:
        id_to_text = {}

if faiss_index is None:
    print("Warning: no FAISS index found in faiss_index.pkl — search will return empty results.")

# ====== 路由 ======
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchReq, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    check_auth(x_api_key)
    q_vec = embed_text(req.query).reshape(1, -1)

    if faiss_index is None:
        return {"ok": True, "data": []}

    # 如索引 metric 是内积（IP），且你构建时对库向量做了单位化，这里也要对 q_vec 做单位化：
    # q_vec /= (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    D, I = faiss_index.search(q_vec, req.top_k)
    out: List[SearchItem] = []
    for idx, score in zip(I[0], D[0]):
        if int(idx) == -1:
            continue
        text = id_to_text.get(int(idx), "")
        out.append(SearchItem(id=int(idx), text=text, score=float(score)))
    return {"ok": True, "data": out}

@app.get("/courses", response_model=CoursesResponse)
def list_courses(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    check_auth(x_api_key)
    courses: List[CourseItem] = []
    for cid, text in id_to_text.items():
        if "Course:" in text:
            courses.append(CourseItem(id=cid, info=text))
    return {"ok": True, "data": courses}
