#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm



try:
    from serpapi import search as google_search_api_search
except Exception:
    google_search_api_search = None

try:
    from serpapi import GoogleSearch
except Exception:
    try:
        from serpapi.google_search import GoogleSearch
    except Exception:
        GoogleSearch = None



SUPPORTED_DATASETS = (
    "kym",
    "memeintent",
    "memeinterpret",
    "mami",
    "multioff",
    "msd",
    "harmcp",
)



SLEEP_SEC = 0.7
K_LENS_CANDIDATES = 10
K_URL_SCRAPE = 5
CHUNK_WORDS = 128
TOP_CHUNKS_TOTAL = 5
REQUEST_TIMEOUT = 20



UNSCRAPABLE_DOMAINS = [
    "www.nytimes.com",
    "www.washingtonpost.com",
    "twitter.com",
    "x.com",
    "www.youtube.com",
]

RIS_BLOCK_DOMAINS = [
    "redbubble.com",
    "etsy.com",
    "amazon.",
    "ebay.",
    "aliexpress.",
    "pinterest.",
    "teepublic.com",
    "spreadshirt.",
    "zazzle.com",
    "tiktok.com",
    "facebook.com",
    "instagram.com",
]

PREFERRED_DOMAINS = [
    "wikipedia.org",
    "cnn.com",
    "bbc.com",
    "foxnews.com",
    "pewresearch.org",
    "theguardian.com",
    "knowyourmeme.com",
    "en.meming.world",
]



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    meme_id: Optional[str],
    url_map_json: Optional[Path],
    output_json: Optional[Path],
) -> tuple[Path, Path]:

    project_root = get_project_root()

    if dataset:

        # one-meme test
        if meme_id is not None:

            if url_map_json is None:
                url_map_json = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "url_map_imgbb.json"
                )

            if output_json is None:
                output_json = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "ris_raw.json"
                )

        # full dataset
        else:

            if url_map_json is None:
                url_map_json = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "url_map_imgbb.json"
                )

            if output_json is None:
                output_json = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "ris_raw.json"
                )

    if url_map_json is None:
        raise ValueError(
            "Missing ImgBB URL map. "
            "Use --dataset or --url-map-json."
        )

    if output_json is None:
        raise ValueError(
            "Missing output JSON. "
            "Use --dataset or --output-json."
        )

    return (
        url_map_json.expanduser().resolve(),
        output_json.expanduser().resolve(),
    )



def load_json(
    path: Path,
    default: Any = None,
) -> Any:

    if not path.exists():
        if default is not None:
            return default

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        str(path) + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        tmp_path,
        path,
    )



def normalize_ws(
    s: Any,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(s or ""),
    ).strip()


def get_domain(
    url: str,
) -> str:

    try:
        return urlparse(
            url
        ).netloc.lower()

    except Exception:
        return ""



def is_blocked_domain(
    dom: str,
) -> bool:

    dom = dom.lower()

    return any(
        b in dom
        for b in RIS_BLOCK_DOMAINS
    )


def is_preferred_domain(
    dom: str,
) -> bool:

    dom = dom.lower()

    return any(
        p in dom
        for p in PREFERRED_DOMAINS
    )


def is_unscrapable_domain(
    dom: str,
) -> bool:

    dom = dom.lower()

    return any(
        b in dom
        for b in UNSCRAPABLE_DOMAINS
    )



def google_search_api_dict(
    params: Dict[str, Any],
) -> Dict[str, Any]:

    if google_search_api_search is not None:
        return google_search_api_search(
            params
        )

    if GoogleSearch is not None:
        return GoogleSearch(
            params
        ).get_dict()

    raise RuntimeError(
        "No usable Google Search API client import found."
    )



def ris_rank_score(
    dom: str,
    text: str,
) -> int:

    score = 0

    if is_preferred_domain(dom):
        score += 80

    if "knowyourmeme.com" in dom:
        score += 50

    if "wikipedia.org" in dom:
        score += 35

    if "en.meming.world" in dom:
        score += 30

    if "reddit.com" in dom:
        score += 10

    score += (
        min(
            len(text or ""),
            250,
        )
        // 25
    )

    return score


def lens_candidates(
    image_url: str,
    google_search_api_key: str,
) -> List[Dict[str, Any]]:

    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": google_search_api_key,
        "hl": "en",
        "gl": "us",
    }

    results = google_search_api_dict(
        params
    )

    if "error" in results:
        print(
            "Google Search API Lens Error:",
            results["error"],
        )
        return []

    cands = []

    for sec in [
        "exact_matches",
        "visual_matches",
        "organic_results",
    ]:

        for item in (
            results.get(sec, [])
            or []
        ):

            link = normalize_ws(
                item.get(
                    "link",
                    "",
                )
            )

            if not link:
                continue

            dom = get_domain(
                link
            )

            if is_blocked_domain(
                dom
            ):
                continue

            snippet = normalize_ws(
                item.get(
                    "snippet",
                    "",
                )
            )

            title = normalize_ws(
                item.get(
                    "title",
                    "",
                )
            )

            text_for_score = (
                snippet
                or title
            )[:250]

            cands.append(
                {
                    "url": link,
                    "domain": dom,
                    "lens_section": sec,
                    "snippet": snippet,
                    "title": title,
                    "score": ris_rank_score(
                        dom,
                        text_for_score,
                    ),
                    "is_preferred_domain": (
                        is_preferred_domain(
                            dom
                        )
                    ),
                    "is_unscrapable_domain": (
                        is_unscrapable_domain(
                            dom
                        )
                    ),
                }
            )

    seen = set()
    deduped = []

    for c in cands:

        if c["url"] in seen:
            continue

        seen.add(
            c["url"]
        )

        deduped.append(
            c
        )

    deduped.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return deduped[
        :K_LENS_CANDIDATES
    ]


def pick_top_urls_for_scrape(
    lens_cands: List[Dict[str, Any]],
    k: int = K_URL_SCRAPE,
) -> List[str]:

    picked = []

    for c in lens_cands:

        dom = c.get(
            "domain",
            "",
        )

        url = c.get(
            "url",
            "",
        )

        if is_unscrapable_domain(
            dom
        ):
            continue

        picked.append(
            url
        )

        if len(picked) >= k:
            break

    return picked



def scrape_url_text(
    url: str,
    timeout: int = 15,
) -> str:

    def clean_text(
        t: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            t or "",
        ).strip()

    def soup_extract(
        html: str,
    ) -> str:

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "header",
                "footer",
                "nav",
                "aside",
                "form",
            ]
        ):
            tag.decompose()

        title = (
            soup.title.get_text(
                " ",
                strip=True,
            )
            if soup.title
            else ""
        )

        paras = [
            p.get_text(
                " ",
                strip=True,
            )
            for p in soup.find_all(
                "p"
            )
        ]

        text = " ".join(
            [title]
            + paras[:40]
        )

        return clean_text(
            text
        )

    try:

        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        if (
            r.status_code == 200
            and r.text
        ):

            text = soup_extract(
                r.text
            )

            lowered = text.lower()

            if any(
                m in lowered
                for m in [
                    "enable javascript",
                    "access denied",
                    "verify you are a human",
                    "captcha",
                ]
            ):
                text = ""

            if len(text) >= 400:
                return text

    except Exception:
        pass

    try:

        jina_url = (
            "https://r.jina.ai/http://"
            + url.replace(
                "https://",
                "",
            ).replace(
                "http://",
                "",
            )
        )

        r2 = requests.get(
            jina_url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        if (
            r2.status_code == 200
            and r2.text
        ):

            text2 = clean_text(
                r2.text
            )

            if len(text2) >= 400:
                return text2

    except Exception:
        pass

    return ""


def chunk_text_words(
    text: str,
    chunk_words: int = CHUNK_WORDS,
) -> List[str]:

    words = re.findall(
        r"\S+",
        text,
    )

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_words,
    ):

        chunk = " ".join(
            words[
                i:i + chunk_words
            ]
        ).strip()

        if chunk:
            chunks.append(
                chunk
            )

    return chunks


def tokenize(
    s: str,
) -> List[str]:

    return re.findall(
        r"[a-z0-9]+",
        (s or "").lower(),
    )



def bm25_rank(
    chunks: List[str],
    queries: List[str],
    top_k: int = TOP_CHUNKS_TOTAL,
) -> List[Dict[str, Any]]:

    if not chunks:
        return []

    docs = [
        tokenize(c)
        for c in chunks
    ]

    N = len(
        docs
    )

    avgdl = (
        sum(
            len(d)
            for d in docs
        )
        / max(
            N,
            1,
        )
    )

    df = {}

    for d in docs:

        for t in set(d):

            df[t] = (
                df.get(
                    t,
                    0,
                )
                + 1
            )

    k1 = 1.5
    b = 0.75

    def idf(t):

        return math.log(
            1
            + (
                N
                - df.get(t, 0)
                + 0.5
            )
            / (
                df.get(t, 0)
                + 0.5
            )
        )

    q_tokens_list = [
        tokenize(q)
        for q in (
            queries
            or []
        )
        if q
        and q.strip()
    ]

    if not q_tokens_list:

        return [
            {
                "chunk_text": c,
                "score": 0.0,
            }
            for c in chunks[
                :top_k
            ]
        ]

    scores = []

    for i, doc in enumerate(
        docs
    ):

        dl = len(
            doc
        )

        tf = {}

        for t in doc:

            tf[t] = (
                tf.get(
                    t,
                    0,
                )
                + 1
            )

        score = 0.0

        for q_tokens in q_tokens_list:

            for t in q_tokens:

                if t not in tf:
                    continue

                denom = (
                    tf[t]
                    + k1
                    * (
                        1
                        - b
                        + b
                        * (
                            dl
                            / (
                                avgdl
                                + 1e-9
                            )
                        )
                    )
                )

                score += (
                    idf(t)
                    * (
                        tf[t]
                        * (
                            k1
                            + 1
                        )
                    )
                    / (
                        denom
                        + 1e-9
                    )
                )

        scores.append(
            (
                score,
                i,
            )
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    top = []

    for s, idx in scores[
        :top_k
    ]:

        top.append(
            {
                "chunk_text": chunks[
                    idx
                ],
                "score": float(
                    s
                ),
            }
        )

    return top



def build_lens_query_texts(
    lens_cands: List[Dict[str, Any]],
    fallback_text: str = "",
    fallback_name: str = "",
    max_cands: int = 6,
) -> List[str]:

    qs = []

    for c in lens_cands[
        :max_cands
    ]:

        t = normalize_ws(
            c.get(
                "title",
                "",
            )
        )

        s = normalize_ws(
            c.get(
                "snippet",
                "",
            )
        )

        if t:
            qs.append(
                t
            )

        if s:
            qs.append(
                s
            )

    if fallback_text:
        qs.append(
            fallback_text
        )

    if fallback_name:
        qs.append(
            fallback_name
        )

    seen = set()
    out = []

    for q in qs:

        q2 = normalize_ws(
            q
        )

        if len(q2) < 3:
            continue

        if q2.lower() in seen:
            continue

        seen.add(
            q2.lower()
        )

        out.append(
            q2
        )

    return out[:10]



def select_img_items(
    url_map: Dict[str, Any],
    meme_id: Optional[str],
) -> List[tuple[str, Dict[str, Any]]]:

    img_items = []

    for img_path, rec in url_map.items():

        if not isinstance(
            rec,
            dict,
        ):
            continue

        url = normalize_ws(
            rec.get(
                "url",
                "",
            )
        )

        if not url:
            continue

        if meme_id is not None:

            rec_meme_id = normalize_ws(
                rec.get(
                    "meme_id",
                    "",
                )
            )

            if rec_meme_id != normalize_ws(
                meme_id
            ):
                continue

        img_items.append(
            (
                img_path,
                rec,
            )
        )

    img_items.sort(
        key=lambda x: x[0]
    )

    if (
        meme_id is not None
        and not img_items
    ):
        raise ValueError(
            f"Meme ID {meme_id} was not found "
            "in the ImgBB URL map."
        )

    return img_items



def run_reverse_image_search(
    url_map_json: Path,
    output_json: Path,
    google_search_api_key: str,
    meme_id: Optional[str],
) -> None:

    url_map = load_json(
        url_map_json,
        default={},
    )

    if not isinstance(
        url_map,
        dict,
    ):
        raise ValueError(
            "Expected url_map_imgbb.json "
            "to be a dict."
        )

    print(
        "Loaded ImgBB map entries:",
        len(url_map),
    )

    final_ris = load_json(
        output_json,
        default={},
    )

    if not isinstance(
        final_ris,
        dict,
    ):
        final_ris = {}

    print(
        "Loaded existing RIS output entries:",
        len(final_ris),
    )

    img_items = select_img_items(
        url_map=url_map,
        meme_id=meme_id,
    )

    print(
        "Images to process for RIS:",
        len(img_items),
    )

    for img_path, rec in tqdm(
        img_items,
        desc="RIS (images-only)",
    ):

        if (
            img_path in final_ris
            and final_ris[
                img_path
            ].get(
                "retrieved_evidence"
            )
        ):
            continue

        imgbb_url = normalize_ws(
            rec.get(
                "url",
                "",
            )
        )

        meme_text = normalize_ws(
            rec.get(
                "text",
                "",
            )
        )

        meme_id_value = normalize_ws(
            rec.get(
                "meme_id",
                "",
            )
        )

        fallback_name = (
            os.path.splitext(
                os.path.basename(
                    img_path
                )
            )[0]
            .replace(
                "_",
                " ",
            )
        )

        ris = {
            "meme_id": meme_id_value,
            "img_path": img_path,
            "text": meme_text,
            "imgbb_url": imgbb_url,
            "retrieved_evidence": [],
            "url": [],
            "lens_candidates": [],
            "top_urls_scraped": [],
            "bm25_top_chunks": [],
            "error": "",
        }

        try:

            lens_cands = lens_candidates(
                imgbb_url,
                google_search_api_key,
            )

            ris[
                "lens_candidates"
            ] = lens_cands

            top_urls = (
                pick_top_urls_for_scrape(
                    lens_cands,
                    k=K_URL_SCRAPE,
                )
            )

            ris[
                "top_urls_scraped"
            ] = top_urls

            lens_query_texts = (
                build_lens_query_texts(
                    lens_cands,
                    fallback_text=meme_text,
                    fallback_name=fallback_name,
                    max_cands=6,
                )
            )

            all_ranked_chunks = []

            for url in top_urls:

                page_text = (
                    scrape_url_text(
                        url,
                        timeout=REQUEST_TIMEOUT,
                    )
                )

                if not page_text:
                    continue

                chunks = (
                    chunk_text_words(
                        page_text,
                        CHUNK_WORDS,
                    )
                )

                if not chunks:
                    continue

                ranked = bm25_rank(
                    chunks,
                    lens_query_texts,
                    top_k=TOP_CHUNKS_TOTAL,
                )

                for r in ranked:

                    all_ranked_chunks.append(
                        {
                            "url": url,
                            "bm25_score": r[
                                "score"
                            ],
                            "chunk_len_words": (
                                CHUNK_WORDS
                            ),
                            "chunk_text": r[
                                "chunk_text"
                            ],
                        }
                    )

                time.sleep(
                    0.2
                )

            all_ranked_chunks.sort(
                key=lambda x: x[
                    "bm25_score"
                ],
                reverse=True,
            )

            top_global = (
                all_ranked_chunks[
                    :TOP_CHUNKS_TOTAL
                ]
            )

            ris[
                "bm25_top_chunks"
            ] = top_global

            ris[
                "retrieved_evidence"
            ] = [
                x["chunk_text"]
                for x in top_global
            ]

            ris[
                "url"
            ] = [
                x["url"]
                for x in top_global
            ]

            if not ris[
                "retrieved_evidence"
            ]:

                fallback_e = []
                fallback_u = []

                for c in lens_cands[
                    :TOP_CHUNKS_TOTAL
                ]:

                    u = c.get(
                        "url",
                        "",
                    )

                    txt = (
                        normalize_ws(
                            c.get(
                                "snippet",
                                "",
                            )
                        )
                        or normalize_ws(
                            c.get(
                                "title",
                                "",
                            )
                        )
                    )

                    if txt:
                        fallback_e.append(
                            txt
                        )
                        fallback_u.append(
                            u
                        )

                ris[
                    "retrieved_evidence"
                ] = fallback_e

                ris[
                    "url"
                ] = fallback_u

        except Exception as e:

            ris["error"] = repr(
                e
            )

        final_ris[
            img_path
        ] = ris

        save_json(
            final_ris,
            output_json,
        )

        time.sleep(
            SLEEP_SEC
        )

    print(
        "\nSaved RIS evidence to:",
        output_json,
    )

    print(
        "Total RIS entries:",
        len(final_ris),
    )



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run Google Lens reverse image search "
            "and retrieve RIS evidence."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help=(
            "Dataset name. When provided, paths "
            "are derived automatically."
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
        "--url-map-json",
        type=Path,
        default=None,
        help=(
            "Path to url_map_imgbb.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path for ris_raw.json. "
            "Overrides the path derived from --dataset."
        ),
    )

    return parser.parse_args()



def main() -> None:

    args = parse_args()

    google_search_api_key = normalize_ws(
        os.environ.get(
            "GoogleSearchAPI_KEY",
            "",
        )
    )

    if not google_search_api_key:

        print(
            "ERROR: GoogleSearchAPI_KEY is not set.",
            file=sys.stderr,
        )

        print(
            'export GoogleSearchAPI_KEY="YOUR_GOOGLE_SEARCH_API_KEY"',
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        url_map_json, output_json = (
            resolve_paths(
                dataset=args.dataset,
                meme_id=args.meme_id,
                url_map_json=args.url_map_json,
                output_json=args.output_json,
            )
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not url_map_json.exists():

        print(
            "ERROR: ImgBB URL map not found: "
            f"{url_map_json}",
            file=sys.stderr,
        )

        sys.exit(1)

    print("=" * 80)
    print("Query Stage - Reverse Image Search")
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
        f"ImgBB map   : {url_map_json}"
    )

    print(
        f"Output JSON : {output_json}"
    )

    print(
        f"Lens top-k  : {K_LENS_CANDIDATES}"
    )

    print(
        f"Scrape top-k: {K_URL_SCRAPE}"
    )

    print(
        f"Chunk words : {CHUNK_WORDS}"
    )

    print(
        f"Evidence top: {TOP_CHUNKS_TOTAL}"
    )

    try:

        run_reverse_image_search(
            url_map_json=url_map_json,
            output_json=output_json,
            google_search_api_key=google_search_api_key,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: Reverse image search failed: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()