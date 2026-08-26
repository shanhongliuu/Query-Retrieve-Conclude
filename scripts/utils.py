from config import * 

import os 
import json 
import re
from typing import Any, Dict, List, Tuple
from rank_bm25 import BM25Okapi
import torch 
from zeroshot import * 

def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        print(f"[load_json] Missing file: {path}")
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)
    
def normalize_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def clean_decoded_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"<\|im_start\|>.*?\n", "", s, flags=re.DOTALL)
    s = s.replace("<|im_end|>", "").strip()
    return s.strip()

def is_no_answer(ans: str) -> bool:
    if not ans:
        return True
    a = normalize_ws(ans).lower()
    patterns = [
        "no answer can be found",
        "no answer could be found",
        "no rankable chunks",
        "[q_item is none]",
        "no evidence found"
    ]
    return any(p in a for p in patterns)

def extract_json_obj(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        return {}

    s = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}

####################################
# REVISED: chunk words instead
# def chunk_text_chars(text: str, length: int = 128) -> List[str]:
#     text = str(text or "")
#     if not text.strip():
#         return []
#     chunks = [text[i:i + length] for i in range(0, len(text), length)]
#     return [normalize_ws(c) for c in chunks if normalize_ws(c)]
def chunk_text_words(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    words = normalize_ws(text).split()
    chunks = []

    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks
####################################

def tokenize_for_bm25(text: str) -> List[str]:
    text = normalize_ws(text).lower()
    return re.findall(r"[a-z0-9']+", text)

def rank_evid_text(query: str, all_texts: List[str], top_k: int = 10) -> List[str]:
    tokenized_corpus = []
    all_corpus = []

    for text in all_texts:
        text = str(text or "")
        if not text.strip():
            continue

        # REVISED: use chunk_text_words instead of chunk_text_chars
        chunks = chunk_text_words(text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS)
        all_corpus.extend(chunks)

        for chunk in chunks:
            tokenized_corpus.append(tokenize_for_bm25(chunk))

    if len(tokenized_corpus) == 0:
        return []

    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_related_evid = [all_corpus[i] for i in top_n]
    return top_related_evid

def gen_retrieved_input(detailed_evid: List[str]) -> str:
    texts = []
    for i in range(len(detailed_evid)):
        texts.append(f"The {i+1}-th retrieved document: {detailed_evid[i]}")
    return "\n".join(texts)

def parse_answer_output(raw_output: str, num_chunks: int) -> Tuple[str, List[int], str]:
    raw_clean = clean_decoded_text(raw_output)
    obj = extract_json_obj(raw_clean)

    if isinstance(obj, dict):
        answer = normalize_ws(obj.get("answer", ""))
        used_chunk_ids = obj.get("used_chunk_ids", [])

        if not isinstance(used_chunk_ids, list):
            used_chunk_ids = []

        parsed_ids = []
        for x in used_chunk_ids:
            try:
                v = int(x)
                if 1 <= v <= num_chunks:
                    parsed_ids.append(v)
            except Exception:
                continue

        parsed_ids = parsed_ids[:MAX_USED_CHUNK_IDS]

        if not answer:
            answer = "No answer can be found."
        if is_no_answer(answer):
            answer = "No answer can be found."

        return answer, parsed_ids, raw_clean

    text = raw_clean.replace("**ANSWER:**", "").strip()
    if not text:
        text = "No answer can be found."
    if is_no_answer(text):
        text = "No answer can be found."

    return text, [], raw_clean



def estimate_used_chunk_ids(answer: str, ranked_chunks: List[str]) -> List[int]:
    if is_no_answer(answer):
        return []

    ans_toks = set(tokenize_for_bm25(answer))
    if not ans_toks:
        return []

    used = []
    for i, chunk in enumerate(ranked_chunks, start=1):
        chunk_toks = set(tokenize_for_bm25(chunk))
        overlap = ans_toks.intersection(chunk_toks)
        if len(overlap) >= 2:
            used.append(i)

    return used[:MAX_USED_CHUNK_IDS]


####################################
# REVISED: use search_snippet for a cleaner evidence text 
def collect_full_texts_for_question(question_item: Dict[str, Any]) -> List[str]:
    texts = []

    for result in question_item.get("results", []):
        if not isinstance(result, dict):
            continue

        full_text = str(result.get("full_text", "") or "").strip()
        if full_text:
            texts.append(full_text)
            continue

        snippet = normalize_ws(result.get("search_snippet", ""))
        if snippet:
            texts.append(snippet)

    out = []
    seen = set()
    for t in texts:
        key = normalize_ws(t).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out

def collect_texts_for_question(q_item: Dict[str, Any]) -> List[str]:
    texts = []

    # 1. Prefer already selected evidence passages
    for field in ["retrieved_evidence_fulltext", "retrieved_evidence", "evidence_passages"]:
        value = q_item.get(field, [])
        if isinstance(value, list):
            texts.extend([normalize_ws(x) for x in value if normalize_ws(x)])
        elif isinstance(value, str) and normalize_ws(value):
            texts.append(normalize_ws(value))

    # 2. Then use result-level evidence passages/snippets
    for result in q_item.get("results", []):
        if not isinstance(result, dict):
            continue

        passages = result.get("evidence_passages", [])
        if isinstance(passages, list):
            texts.extend([normalize_ws(p) for p in passages if normalize_ws(p)])

        snippet = normalize_ws(result.get("search_snippet", ""))
        if snippet:
            texts.append(snippet)

    # 3. Use full_text only as last fallback
    if not texts:
        for result in q_item.get("results", []):
            full_text = normalize_ws(result.get("full_text", ""))
            if full_text:
                texts.append(full_text)

    # Deduplicate
    out = []
    seen = set()
    for t in texts:
        key = t.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t)

    return out
####################################



def is_completed_answer_record(rec: Dict[str, Any]) -> bool:
    if not isinstance(rec, dict) or len(rec) == 0:
        return False

    questions = rec.get("questions", None)
    answers = rec.get("answers", None)

    if isinstance(questions, list) and isinstance(answers, list) and len(questions) == len(answers):
        return True

    return False

def clean_statement_output(text: str) -> str:
    text = clean_decoded_text(text)
    text = text.replace("[STAT]:", "").strip()

    if "[STAT]" in text:
        text = text.split("[STAT]")[-1].strip()

    lines = [normalize_ws(x) for x in text.splitlines() if normalize_ws(x)]
    if not lines:
        return ""

    text = lines[0]
    text = text.strip().strip('"').strip("'").strip()
    return normalize_ws(text)

# REVISED: don't need to check if number of statements is same as number of QAs, since we can have 0 statements for a QA if no evidence was found. Just check if we've attempted this meme at all by looking for the "statement" key.
# just check if we've attempted this meme at all by looking for the "statement" key. Even if statement is empty, we don't want to waste GPU re-running it.
def is_completed_statement_record(rec: Dict[str, Any]) -> bool:
    if not isinstance(rec, dict) or len(rec) == 0:
        return False
    # If the key exists, we've attempted this meme. 
    # Even if statement is empty, we don't want to waste GPU re-running it.
    return "statement" in rec

def gen_incontext_input(ques: str, ans: str, demos: str) -> str:
    texts = []
    texts.append(demos)
    texts.append("[QUES]: " + ques)
    texts.append("[ANS]: " + ans)
    texts.append("[STAT]:")
    return "\n".join(texts)


def initiate_model(selected_model: str):
    MODEL_NAME = ""
    processor = None
    model = None

    if selected_model == "QWEN3-32B":
        MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
        model, processor = load_qwen3_model(MODEL_NAME) 
    elif selected_model == "GEMMA3-27B":
        MODEL_NAME = "google/gemma-3-27b-it"
        model, processor = load_gemma3_model(MODEL_NAME)
    elif selected_model == "GPT4":
        pass

    return MODEL_NAME, model, processor
    

