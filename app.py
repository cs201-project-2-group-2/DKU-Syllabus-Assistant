# app.py
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
import faiss, pickle, numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACTION_API_KEY = os.getenv("ACTION_API_KEY", "replace-me")  # 给 GPT Actions 用
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Syllabi Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 若只对服务器调用可收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 载入索引
with open("faiss_index.pkl", "rb") as f:
    index_data = pickle.load(f)
faiss_index = index_data["index"]
id_to_text: Dict[int, str] = index_data["id_to_text"]

def check_auth(x_api_key: Optional[str]):
    if x_api_key != ACTION_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

class SearchReq(BaseModel):
    query: str
    top_k: int = 3

class SearchHit(BaseModel):
    id: int
    text: str
    score: float

class SearchResp(BaseModel):
    ok: bool
    data: List[SearchHit]

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/search", response_model=SearchResp)
def search(req: SearchReq, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    check_auth(x_api_key)
    q_vec = embed_text(req.query)
    q_vec = np.array([q_vec], dtype="float32")

    # 如果索引是基于内积（IndexFlatIP/IVF+IP），先做单位化
    # q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12

    D, I = faiss_index.search(q_vec, req.top_k)
    hits = []
    for i, d in zip(I[0], D[0]):
        if int(i) in id_to_text:
            hits.append(SearchHit(id=int(i), text=id_to_text[int(i)], score=float(d)))
    return {"ok": True, "data": hits}

@app.get("/courses")
def list_courses(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    check_auth(x_api_key)
    courses = []
    for cid, text in id_to_text.items():
        if "Course:" in text:
            courses.append({"id": cid, "info": text})
    return {"ok": True, "data": courses}

def embed_text(text: str) -> List[float]:
    """文字转向量（OpenAI 官方 embeddings 接口）"""
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return resp.data[0].embedding
