#!/usr/bin/env python3
"""
index_data.py
- 从 parsed_data.csv / parsed_data.json 读取文本 chunk
- 用 sentence-transformers 生成 embeddings（批量）
- 用 FAISS 建索引（cosine via normalize + IndexFlatIP）
- 保存索引和 metadata
- 提供 search_index() 函数示例
"""

import os
import argparse
import json
import pickle
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# huggingface / sentence-transformers
from sentence_transformers import SentenceTransformer

# try import faiss, otherwise set flag to use fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception as e:
    print("Warning: faiss import failed:", e)
    _HAS_FAISS = False

# -------------- utils: 读取数据 --------------
def load_data(csv_path: str = None, json_path: str = None) -> List[Dict[str, Any]]:
    """
    读取 parsed_data.csv / parsed_data.json，返回 list of dict:
      {"id": int, "text": str, "meta": {...}}
    自动适配常见字段：'text', 'content', 'chunk', 'description'，否则尝试拼接其余字段。
    """
    items = []
    next_id = 0

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rowd = row.to_dict()
            # 尝试取 text 字段
            text = None
            for candidate in ("text", "content", "chunk", "body", "description"):
                if candidate in rowd and pd.notna(rowd[candidate]):
                    text = str(rowd[candidate]).strip()
                    break
            if not text:
                # 拼接剩余的字符串字段作为后备
                parts = []
                for k, v in rowd.items():
                    if k.lower() in ("id",) or pd.isna(v):
                        continue
                    parts.append(f"{k}: {v}")
                text = " ; ".join(parts).strip()
            meta = {k: v for k, v in rowd.items() if k not in ("text", "content", "chunk", "body", "description")}
            items.append({"id": next_id, "text": text, "meta": meta})
            next_id += 1

    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)
        # 如果 json 是 dict -> 可能是 {"course":..., "chunks": [...]}
        if isinstance(data, dict):
            # try common layout
            if "chunks" in data and isinstance(data["chunks"], list):
                for c in data["chunks"]:
                    text = c.get("text") or c.get("content") or c.get("chunk") or ""
                    meta = {k: v for k, v in c.items() if k not in ("text", "content", "chunk")}
                    items.append({"id": next_id, "text": text, "meta": meta})
                    next_id += 1
            else:
                # fallback: flatten dict entries
                items.append({"id": next_id, "text": json.dumps(data, ensure_ascii=False), "meta": {}})
                next_id += 1
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    text = entry.get("text") or entry.get("content") or entry.get("chunk") or ""
                    if not text:
                        # 拼接字典中的字符串字段
                        parts = []
                        for k, v in entry.items():
                            if k in ("text",):
                                continue
                            if v is None:
                                continue
                            parts.append(f"{k}: {v}")
                        text = " ; ".join(parts)
                    meta = {k: v for k, v in entry.items() if k not in ("text", "content", "chunk")}
                    items.append({"id": next_id, "text": text, "meta": meta})
                    next_id += 1
                else:
                    items.append({"id": next_id, "text": str(entry), "meta": {}})
                    next_id += 1
        else:
            # raw string
            items.append({"id": next_id, "text": str(data), "meta": {}})
            next_id += 1

    return items


# -------------- embedding & indexing --------------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    使用 sentence-transformers 批量生成 embeddings（numpy array）。
    返回 shape = (N, dim)
    """
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def build_faiss_index(embeddings: np.ndarray, ids: np.ndarray = None):
    """
    使用 FAISS 建立 IndexFlatIP（cosine via normalize），返回 index（IndexIDMap）。
    embeddings: numpy float32 (N, dim)
    ids: np.ndarray (N,) int64 可选
    """
    if not _HAS_FAISS:
        raise RuntimeError("FAISS not available in this environment.")
    # 归一化到单位向量（使得 inner product == cosine similarity）
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors => cosine
    id_index = faiss.IndexIDMap(index)  # allow custom ids
    if ids is None:
        ids = np.arange(embeddings.shape[0], dtype="int64")
    else:
        ids = ids.astype("int64")
    id_index.add_with_ids(embeddings, ids)
    return id_index


# -------------- 保存与加载 --------------
def save_index_and_meta(index, meta_list: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "syllabus.index")
    meta_path = os.path.join(output_dir, "meta.pkl")
    # faiss 写索引
    faiss.write_index(index, index_path)
    # 保存 meta（list），按 id 的顺序或 map
    with open(meta_path, "wb") as f:
        pickle.dump(meta_list, f)
    print("Saved index to", index_path)
    print("Saved meta to", meta_path)


def load_index_and_meta(output_dir: str):
    index_path = os.path.join(output_dir, "syllabus.index")
    meta_path = os.path.join(output_dir, "meta.pkl")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta_list = pickle.load(f)
    # meta_list 假定是一个 list 或者 dict mapping id -> meta
    # 这里我们返回两个对象
    return index, meta_list


# -------------- 查询函数 --------------
def search_index(model: SentenceTransformer, index, meta_list, query: str, top_k: int = 5):
    """
    返回 list of {id, score, meta, text(if present in meta)} 按 score 降序
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    # 归一化
    if _HAS_FAISS:
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)  # D: scores (inner product), I: ids
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = None
            # meta_list 可能是 list indexed by position or dict id->meta
            if isinstance(meta_list, dict):
                meta = meta_list.get(int(idx), {})
            elif isinstance(meta_list, list):
                # 尝试找到 meta with id == idx
                found = next((m for m in meta_list if m.get("id") == int(idx)), None)
                meta = found or {}
            results.append({"id": int(idx), "score": float(score), "meta": meta})
        return results
    else:
        # fallback: brute-force (meta_list must contain embeddings separately)
        raise RuntimeError("FAISS not available - use fallback search implementation.")


# -------------- CLI 主流程 --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="parsed_data.csv")
    parser.add_argument("--json", type=str, default="parsed_data.json")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./index_data")
    args = parser.parse_args()

    print("Loading data...")
    items = load_data(args.csv, args.json)
    if not items:
        print("No data loaded. Check files.")
        return

    texts = [it["text"] for it in items]
    ids = np.array([it["id"] for it in items], dtype="int64")
    meta_list = items  # 保存完整 item（含 id,text,meta）

    print(f"Loaded {len(texts)} chunks. Loading model {args.model} ...")
    model = SentenceTransformer(args.model)

    print("Generating embeddings ...")
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)
    # ensure dtype float32 for faiss
    embeddings = embeddings.astype("float32")

    if _HAS_FAISS:
        print("Building FAISS index ...")
        index = build_faiss_index(embeddings, ids)
        print("Saving index and meta ...")
        save_index_and_meta(index, meta_list, args.output_dir)
    else:
        # fallback: 将 embeddings + meta 保存为 numpy + pickle，供简单暴力检索使用
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta_list, f)
        print("FAISS not available. Saved embeddings.npy and meta.pkl for fallback search.")

    print("Done.")


if __name__ == "__main__":
    main()
