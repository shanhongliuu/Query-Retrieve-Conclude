#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# dataset
SUPPORTED_DATASETS = (
    "kym",
    "memeintent",
    "memeinterpret",
    "mami",
    "multioff",
    "msd",
    "harmcp",
)


def get_project_root() -> Path:

    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    meme_id: Optional[str],
    input_ris_path: Optional[Path],
    output_ris_summary_path: Optional[Path],
    output_caption_ris_summary_path: Optional[Path],
) -> tuple[Path, Path, Path]:

    project_root = get_project_root()

    if dataset:

        # one-meme test
        if meme_id is not None:

            if input_ris_path is None:
                input_ris_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "ris_raw.json"
                )

            if output_ris_summary_path is None:
                output_ris_summary_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "ris_sum.json"
                )

            if output_caption_ris_summary_path is None:
                output_caption_ris_summary_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "ris_caps.json"
                )

        # full dataset
        else:

            if input_ris_path is None:
                input_ris_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "ris_raw.json"
                )

            if output_ris_summary_path is None:
                output_ris_summary_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "ris_sum.json"
                )

            if output_caption_ris_summary_path is None:
                output_caption_ris_summary_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "ris_caps.json"
                )

    if input_ris_path is None:
        raise ValueError(
            "Missing RIS input. "
            "Use --dataset or --input-ris."
        )

    if output_ris_summary_path is None:
        raise ValueError(
            "Missing RIS summary output. "
            "Use --dataset or --output-ris-summary."
        )

    if output_caption_ris_summary_path is None:
        raise ValueError(
            "Missing caption RIS output. "
            "Use --dataset or --output-caption-ris."
        )

    return (
        input_ris_path.expanduser().resolve(),
        output_ris_summary_path.expanduser().resolve(),
        output_caption_ris_summary_path.expanduser().resolve(),
    )



def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        print(f"[load_json] Missing file: {path}")
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def dedup_texts(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        x = normalize_ws(x)
        if not x:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(x)
    return out


def tokenize_for_anchor_scoring(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\']*", text or "")


def extract_candidate_anchors_from_text(text: str) -> List[str]:
    
    text = normalize_ws(text)
    if not text:
        return []

    anchors = []

    # multiword capitalized entities
    multi_caps = re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b",
        text
    )
    anchors.extend(multi_caps)

    # acronyms / all caps
    acronyms = re.findall(r"\b([A-Z]{2,})\b", text)
    anchors.extend(acronyms)

    # distinctive singletons
    strong_singletons = {
        "trump", "biden", "harris", "clinton", "epstein", "maxwell",
        "mccabe", "facebook", "twitter", "fbi", "isis", "covid",
        "catholics", "muslims", "kkk", "klan", "omar"
    }

    for tok in tokenize_for_anchor_scoring(text):
        t = tok.strip(".,:;!?()[]{}\"'").lower()
        if t in strong_singletons:
            anchors.append(tok)

    return dedup_texts(anchors)


def collect_top_snippets(
    rec: Dict[str, Any],
    max_snippets: int = 5
) -> List[str]:

    snippets = []

    # Prefer bm25 top chunks
    bm25_chunks = rec.get("bm25_top_chunks", [])
    if isinstance(bm25_chunks, list):
        for x in bm25_chunks:
            if isinstance(x, dict):
                txt = normalize_ws(x.get("chunk_text", ""))
                if txt:
                    snippets.append(txt)

    # Fallback: retrieved evidence
    retrieved_evidence = rec.get("retrieved_evidence", [])
    if isinstance(retrieved_evidence, list):
        for x in retrieved_evidence:
            txt = normalize_ws(x)
            if txt:
                snippets.append(txt)

    # Also include Lens snippets/titles
    lens_candidates = rec.get("lens_candidates", [])
    if isinstance(lens_candidates, list):
        for x in lens_candidates:
            if not isinstance(x, dict):
                continue

            snippet = normalize_ws(x.get("snippet", ""))
            title = normalize_ws(x.get("title", ""))

            if snippet:
                snippets.append(snippet)
            elif title:
                snippets.append(title)

    return dedup_texts(snippets)[:max_snippets]


def collect_strong_anchors(
    rec: Dict[str, Any],
    max_anchors: int = 8
) -> List[str]:

    anchors = []

    lens_candidates = rec.get("lens_candidates", [])
    if isinstance(lens_candidates, list):
        for cand in lens_candidates[:10]:
            if not isinstance(cand, dict):
                continue

            title = normalize_ws(cand.get("title", ""))
            snippet = normalize_ws(cand.get("snippet", ""))

            if title:
                anchors.extend(
                    extract_candidate_anchors_from_text(title)
                )

            if snippet:
                anchors.extend(
                    extract_candidate_anchors_from_text(snippet)
                )

    for txt in collect_top_snippets(
        rec,
        max_snippets=8
    ):
        anchors.extend(
            extract_candidate_anchors_from_text(txt)
        )

    return dedup_texts(anchors)[:max_anchors]


def infer_ris_quality(rec: Dict[str, Any]) -> str:

    bm25_chunks = rec.get("bm25_top_chunks", [])
    lens_candidates = rec.get("lens_candidates", [])
    retrieved_evidence = rec.get("retrieved_evidence", [])

    n_chunks = (
        len(bm25_chunks)
        if isinstance(bm25_chunks, list)
        else 0
    )

    n_lens = (
        len(lens_candidates)
        if isinstance(lens_candidates, list)
        else 0
    )

    n_ev = (
        len(retrieved_evidence)
        if isinstance(retrieved_evidence, list)
        else 0
    )

    if n_chunks >= 3 and n_lens >= 3:
        return "high"

    if n_chunks >= 1 or n_ev >= 2 or n_lens >= 2:
        return "medium"

    return "low"



BAD_ANCHOR_SUBSTRINGS = [
    "alec dent",
    "aisle dash katie",
    "vera wang",
    "marie claire",
    "dispatch",
    "newsletter",
    "fact check",
    "factcheck",
    "culture editor",
    "staff writer",
    "member can share",
    "paywall",
    "sign up",
    "subscribed",
    "email",
    "newsletter sign-up",
    "racked sophia",
    "lemondrop",
    "jihan",
    "katie",
]

BAD_SNIPPET_SUBSTRINGS = [
    "newsletter",
    "sign up",
    "you are now subscribed",
    "email to sign up",
    "member can share articles",
    "paywall",
    "culture editor",
    "staff writer",
    "corrections@",
    "factcheck@",
    "dispatch members",
    "marie claire",
    "vera wang wedding dress",
    "newsletter sign-up",
    "racked sophia",
    "lemondrop",
]

GOOD_VISUAL_CUES = [
    "pictured",
    "photograph",
    "photo",
    "walking",
    "standing",
    "wedding",
    "spelled",
    "spell",
    "the word",
    "shows",
    "showing",
    "appears",
    "appearing",
    "visible",
    "robes",
    "hoods",
    "crosses",
]


def looks_like_bad_anchor(anchor: str) -> bool:
    a = normalize_ws(anchor).lower()

    if not a:
        return True

    if any(
        bad in a
        for bad in BAD_ANCHOR_SUBSTRINGS
    ):
        return True

    if len(a.split()) > 5:
        return True

    return False


def looks_like_visual_anchor(anchor: str) -> bool:
    a = normalize_ws(anchor)

    if not a:
        return False

    if len(a.split()) >= 2:
        return True

    strong_singletons = {
        "trump", "biden", "epstein", "maxwell", "clinton",
        "zuckerberg", "omar", "glyphosate", "facebook",
        "harris", "mccabe", "twitter", "fbi", "muslims",
        "kkk", "klan"
    }

    if a.lower() in strong_singletons:
        return True

    return False


def filter_caption_anchors(
    anchors: List[str]
) -> List[str]:

    kept = []

    for a in anchors:
        a = normalize_ws(a)

        if not a:
            continue

        if looks_like_bad_anchor(a):
            continue

        if not looks_like_visual_anchor(a):
            continue

        kept.append(a)

    return dedup_texts(kept)


def clean_snippet_for_caption(snippet: str) -> str:
    s = normalize_ws(snippet)

    if not s:
        return ""

    words = s.split()

    if len(words) > 60:
        s = " ".join(words[:60])

    return s


def keep_caption_snippet(snippet: str) -> bool:
    s = normalize_ws(snippet).lower()

    if not s:
        return False

    if any(
        bad in s
        for bad in BAD_SNIPPET_SUBSTRINGS
    ):
        return False

    if any(
        cue in s
        for cue in GOOD_VISUAL_CUES
    ):
        return True

    if re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
        normalize_ws(snippet)
    ):
        return True

    return False


def build_caption_ris_hints(
    snippets: List[str],
    max_hints: int = 2
) -> List[str]:

    kept = []

    for s in snippets:
        s = clean_snippet_for_caption(s)

        if not s:
            continue

        if not keep_caption_snippet(s):
            continue

        kept.append(s)

    return dedup_texts(kept)[:max_hints]


def add_scene_hints_from_snippets(
    snippets: List[str]
) -> List[str]:

    hints = []
    joined = " ".join(snippets).lower()

    if "wedding" in joined:
        hints.append(
            "wedding photograph"
        )

    if "walking chelsea clinton down the aisle" in joined:
        hints.append(
            "Bill Clinton walking Chelsea Clinton at her wedding"
        )

    if "spelled" in joined and "trump" in joined:
        hints.append(
            "objects arranged to spell 'Trump'"
        )

    if "maxwell" in joined and "wedding" in joined:
        hints.append(
            "Maxwell visible in a wedding photograph"
        )

    if (
        "white robes" in joined
        or "hoods" in joined
        or "crosses" in joined
    ):
        hints.append(
            "robed figures with crosses visible in a historical photograph"
        )

    if "andrew mccabe" in joined:
        hints.append(
            "Andrew McCabe visible in a portrait-style image"
        )

    return dedup_texts(hints)


# =========================================================
# ONE-MEME TEST
# =========================================================
def select_ris_records(
    ris_data: Dict[str, Any],
    meme_id: Optional[str],
) -> Dict[str, Any]:

    if meme_id is None:
        return ris_data

    target_id = normalize_ws(meme_id)
    selected = {}

    for key, rec in ris_data.items():

        if not isinstance(rec, dict):
            continue

        rec_meme_id = normalize_ws(
            rec.get("meme_id", key)
        )

        if rec_meme_id == target_id:
            selected[key] = rec

    if not selected:
        raise ValueError(
            f"Meme ID {target_id} was not found "
            "in the raw RIS input."
        )

    return selected



def prepare_ris(
    input_ris_path: Path,
    output_ris_summary_path: Path,
    output_caption_ris_summary_path: Path,
    meme_id: Optional[str],
) -> None:

    ris_data = load_json(
        str(input_ris_path),
        {}
    )

    if not isinstance(ris_data, dict):
        raise ValueError(
            "Raw RIS input must be a JSON object."
        )

    ris_data = select_ris_records(
        ris_data=ris_data,
        meme_id=meme_id,
    )

    print(
        "Loaded raw RIS entries:",
        len(ris_data)
    )

    ris_summaries = {} #raw ris

    for key, rec in ris_data.items():
        if not isinstance(rec, dict):
            continue

        meme_id_value = str(
            rec.get(
                "meme_id",
                key
            )
        )

        img_name = normalize_ws(
            rec.get("img_path", "")
            or rec.get("img", "")
        )

        top_snippets = collect_top_snippets(
            rec,
            max_snippets=5
        )

        strong_anchors = collect_strong_anchors(
            rec,
            max_anchors=8
        )

        ris_quality = infer_ris_quality(
            rec
        )

        ris_summaries[meme_id_value] = {
            "img": img_name,
            "ris_quality": ris_quality,
            "ris_strong_anchors": strong_anchors,
            "ris_top_snippets": top_snippets,
        }

    save_json(
        str(output_ris_summary_path),
        ris_summaries
    )

    print(
        "Saved RIS summaries to:"
    )

    print(
        output_ris_summary_path
    )

    caption_friendly = {}

    for meme_id_value, rec in ris_summaries.items():

        img_name = normalize_ws(
            rec.get("img", "")
        )

        ris_quality = normalize_ws(
            rec.get(
                "ris_quality",
                "low"
            )
        )

        anchors = rec.get(
            "ris_strong_anchors",
            []
        )

        snippets = rec.get(
            "ris_top_snippets",
            []
        )

        if not isinstance(
            anchors,
            list
        ):
            anchors = []

        if not isinstance(
            snippets,
            list
        ):
            snippets = []

        anchors = [
            normalize_ws(x)
            for x in anchors
            if normalize_ws(x)
        ]

        snippets = [
            normalize_ws(x)
            for x in snippets
            if normalize_ws(x)
        ]

        caption_anchors = filter_caption_anchors(
            anchors
        )

        caption_hints = build_caption_ris_hints(
            snippets,
            max_hints=2
        )

        scene_hints = add_scene_hints_from_snippets(
            snippets
        )

        caption_anchors = dedup_texts(
            caption_anchors
            + scene_hints
        )

        caption_friendly[meme_id_value] = {
            "img": img_name,
            "ris_quality": ris_quality,
            "caption_ris_anchors": caption_anchors,
            "caption_ris_hints": caption_hints,
            "source_ris_strong_anchors": anchors,
            "source_ris_top_snippets": snippets,
        }

    save_json(
        str(output_caption_ris_summary_path),
        caption_friendly
    )

    print(
        "\nSaved caption-friendly RIS summaries to:"
    )

    print(
        output_caption_ris_summary_path
    )

    print(
        "\nSample RIS summaries:"
    )

    for k, v in list(
        ris_summaries.items()
    )[:3]:

        print(f"\n{k}")

        print(
            json.dumps(
                v,
                ensure_ascii=False,
                indent=2
            )
        )

    print(
        "\nSample caption-friendly summaries:"
    )

    for k, v in list(
        caption_friendly.items()
    )[:3]:

        print(f"\n{k}")

        print(
            json.dumps(
                v,
                ensure_ascii=False,
                indent=2
            )
        )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare reverse-image-search evidence "
            "for QRC caption and question generation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help=(
            "Dataset name. When provided, input/output "
            "paths are derived automatically."
        ),
    )

    # one-meme test
    parser.add_argument(
        "--meme-id",
        type=str,
        default=None,
        help=(
            "Process only one meme ID. "
            "Omit this argument for the full dataset."
        ),
    )

    parser.add_argument(
        "--input-ris",
        type=Path,
        default=None,
        help=(
            "Path to ris_raw.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--output-ris-summary",
        type=Path,
        default=None,
        help=(
            "Path to ris_sum.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--output-caption-ris",
        type=Path,
        default=None,
        help=(
            "Path to ris_caps.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    try:

        (
            input_ris_path,
            output_ris_summary_path,
            output_caption_ris_summary_path,
        ) = resolve_paths(
            dataset=args.dataset,
            meme_id=args.meme_id,
            input_ris_path=args.input_ris,
            output_ris_summary_path=args.output_ris_summary,
            output_caption_ris_summary_path=args.output_caption_ris,
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not input_ris_path.exists():

        print(
            f"ERROR: Raw RIS input not found: "
            f"{input_ris_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    print("=" * 80)
    print("Query Stage - Prepare RIS")
    print("=" * 80)

    if args.dataset:
        print(
            f"Dataset     : {args.dataset}"
        )

    if args.meme_id is not None:
        print(
            f"Meme ID     : {args.meme_id}"
        )

    print(
        f"Input RIS   : {input_ris_path}"
    )

    print(
        f"RIS summary : {output_ris_summary_path}"
    )

    print(
        f"Caption RIS : {output_caption_ris_summary_path}"
    )

    try:

        prepare_ris(
            input_ris_path=input_ris_path,
            output_ris_summary_path=output_ris_summary_path,
            output_caption_ris_summary_path=output_caption_ris_summary_path,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: RIS preparation failed: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()