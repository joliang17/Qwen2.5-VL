#!/usr/bin/env python3
import os
CACHE_DIR = '/fs/nexus-faculty/zhou/colorbench/cache'
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_MODULES_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

import json
import re
import sys
import numpy as np
from typing import List, Tuple, Optional
from argparse import ArgumentParser
from collections import defaultdict

import torch
from transformers import CLIPTokenizer, CLIPModel

# ----------------------------
# CLIP utilities
# ----------------------------
def load_clip(model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
    """
    Load CLIP text encoder and tokenizer.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval().to(device)
    return tokenizer, model, device

@torch.no_grad()
def clip_text_embeddings(
    texts: List[str],
    tokenizer: CLIPTokenizer,
    model: CLIPModel,
    device: str,
    max_length: int = 77,
) -> torch.Tensor:
    """
    Get L2-normalized CLIP text embeddings for a list of strings.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    # Use CLIP's convenience method
    feats = model.get_text_features(**inputs)  # [B, D]
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats  # normalized embeddings

@torch.no_grad()
def clip_text_diversity(
    q1: str,
    q2: str,
    tokenizer: CLIPTokenizer,
    model: CLIPModel,
    device: str,
) -> float:
    """
    Compute a similarity score in [0, 1] from CLIP cosine similarity.
    similarity = 1 - ((cos_sim + 1) / 2)
    """
    embs = clip_text_embeddings([q1, q2], tokenizer, model, device)
    similarity = torch.nn.functional.cosine_similarity(embs[0:1], embs[1:2]).item()
    # map cosine similarity [-1, 1] -> [0, 1] similarity, then invert to similarity
    return float(similarity)

# ----------------------------
# Question text extraction
# ----------------------------
_Q_RE = re.compile(r"Question:\s*(.*?)(?:\n|Answer:|$)", flags=re.IGNORECASE | re.DOTALL)

def extract_question(text: str) -> str:
    """
    Try to extract the question string from a blob like:
      'Question: ...\\nAnswer: ...'
    Falls back to the original string if no pattern is found.
    """
    if not text:
        return ""
    m = _Q_RE.search(text)
    if m:
        # Clean up whitespace
        q = m.group(1).strip()
        # If the line still contains 'Answer:' on same line, split it off
        q = q.split("Answer:")[0].strip()
        return q
    return text.strip()

# ----------------------------
# CLI for processing a JSON file (like your keywords_ver.json)
# ----------------------------
def process_file(json_path: str, save_all_path: str, save_score_path: str):
    """
    Expects items that contain:
      - item["result"]["gen"]  (generated QA blob)
      - item["original_answer"]["value"]  (original QA blob)
    Prints per-item similarity and an overall average.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    tokenizer, model, device = load_clip()
    list_scores = []
    bucket = defaultdict(list)  # num_keywords -> [scores]

    def safe_get(d: dict, path: List[str]) -> Optional[str]:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur if isinstance(cur, str) else None

    for i, item in enumerate(data):
        gen_blob = safe_get(item, ["result", "gen"]) or ""
        ori_blob = safe_get(item, ["original_answer", "value"]) or ""

        gen_q = extract_question(gen_blob)
        ori_q = extract_question(ori_blob)

        if not gen_q or not ori_q:
            print(f"[{i}] Missing question text — skipping")
            continue

        score = clip_text_diversity(gen_q, ori_q, tokenizer, model, device)
        nk = int(item.get("num_keywords", -1))
        list_scores.append([item['num_keywords'], score, {'ori_q': ori_q, 'gen_q': gen_q}])
        bucket[nk].append(score)

    # save to folder
    with open(save_all_path, "w") as f: 
        json.dump(list_scores, f, indent=4, ensure_ascii=False)

    # calculate avg scores 
    per_num_keywords = {
        nk: {
            "count": len(scores),
            "avg_similarity": np.round((sum(scores) / len(scores)), 4) if scores else None,
        }
        for nk, scores in sorted(bucket.items(), key=lambda x: x[0])
    }
    overall_count = sum(v["count"] for v in per_num_keywords.values())
    overall_avg = (
        np.round(sum((v["avg_similarity"] * v["count"]) for v in per_num_keywords.values()) / overall_count, 4)
        if overall_count else None
    )
    # print a quick table
    print("\nAverage similarity by num_keywords:")
    for nk, stats in per_num_keywords.items():
        print(f"  {nk}: avg={stats['avg_similarity']:.4f} (n={stats['count']})")
    print(f"\nOverall: avg={overall_avg:.4f} (n={overall_count})" if overall_avg is not None else "\nOverall: n=0")

    # --- save JSON: keep items + add summary ---
    out_obj = {
        "source_json": json_path,
        "count": overall_count,
        "overall_avg_similarity": overall_avg,
        "per_num_keywords": per_num_keywords,
        # "items": list_scores,  # keeps your existing data
    }
    with open(save_score_path, "w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {save_score_path}")
    return 

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/mminstruct_lora_1e4_samples_ori/keywords_ver.json")
    parser.add_argument("--save_folder", type=str, default="/fs/nexus-scratch/yliang17/Research/VLM/Qwen2.5-VL/evaluation/mmmu/similarity/")

    args = parser.parse_args()
    json_path = args.json_path
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    save_all_path = os.path.join(save_folder, f"{json_path.split('/')[-2]}.json")
    save_score_path = os.path.join(save_folder, f"{json_path.split('/')[-2]}_scores.json")

    process_file(json_path=json_path, save_all_path=save_all_path, save_score_path=save_score_path)