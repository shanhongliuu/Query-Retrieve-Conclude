#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from serpapi import GoogleSearch
except Exception:
    from serpapi.google_search import GoogleSearch



SUPPORTED_DATASETS = (
    "kym",
    "memeintent",
    "memeinterpret",
    "mami",
    "multioff",
    "msd",
    "harmcp",
)

SUPPORTED_MODELS = (
    "gemma",
    "qwen",
)

SEARCH_ENGINE = "google"
GOOGLE_DOMAIN = "google.com"
GL = "us"
HL = "en"

TOP_SEARCH_RESULTS_PER_QUESTION = 10
SAVE_EVERY_N_MEMES = 1

REQUEST_TIMEOUT = 12
SLEEP_BETWEEN_REQUESTS = 1.0

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

DEBUG = False



UNSCRAPABLE_DOMAINS = [
    "www.nytimes.com",
    "www.washingtonpost.com",
    "twitter.com",
    "x.com",
    "www.youtube.com",
]

BLOCK_DOMAINS = [
    "redbubble.com", "etsy.com", "amazon.", "ebay.", "aliexpress.",
    "pinterest.", "teepublic.com", "spreadshirt.", "zazzle.com",
    "tiktok.com", "facebook.com", "instagram.com",
]

PREFERRED_DOMAINS = [
    "wikipedia.org",
    "knowyourmeme.com",
    "bbc.com",
    "cnn.com",
    "foxnews.com",
    "theguardian.com",
    "pewresearch.org",
    "en.meming.world",
]



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    model_name: str,
    meme_id: Optional[str],
    question_path: Optional[Path],
    evidence_path: Optional[Path],
) -> Tuple[Path, Path]:

    project_root = get_project_root()

    if dataset:

        # one-meme test
        if meme_id is not None:

            if question_path is None:

                if model_name == "gemma":
                    question_name = "question_gemma.json"
                else:
                    question_name = "questions.json"

                question_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / question_name
                )

            if evidence_path is None:

                if model_name == "gemma":
                    evidence_name = "wst_fulltext_gemma.json"
                else:
                    evidence_name = "wst_fulltext.json"

                evidence_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "retrieve"
                    / evidence_name
                )

        # full dataset
        else:

            if question_path is None:

                if model_name == "gemma":
                    question_path = (
                        project_root
                        / "outputs"
                        / dataset
                        / "question_generation"
                        / "que"
                        / "question_gemma.json"
                    )

                else:
                    question_path = (
                        project_root
                        / "outputs"
                        / dataset
                        / "question_generation"
                        / "que"
                        / "questions.json"
                    )

            if evidence_path is None:

                if model_name == "gemma":
                    evidence_path = (
                        project_root
                        / "outputs"
                        / dataset
                        / "wst"
                        / "gemma"
                        / "wst_fulltext_gemma.json"
                    )

                else:
                    evidence_path = (
                        project_root
                        / "outputs"
                        / dataset
                        / "wst"
                        / "wst_fulltext.json"
                    )

    if question_path is None:
        raise ValueError(
            "Missing question JSON. "
            "Use --dataset or --question-json."
        )

    if evidence_path is None:
        raise ValueError(
            "Missing WST output JSON. "
            "Use --dataset or --output-json."
        )

    return (
        question_path.expanduser().resolve(),
        evidence_path.expanduser().resolve(),
    )



def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return {} if default is None else default

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    tmp_path = path + ".tmp"

    with open(
        tmp_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        tmp_path,
        path
    )


def normalize_ws(s: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(s or "")
    ).strip()


def domain_of(url: str) -> str:
    try:
        return urlparse(
            url
        ).netloc.lower()

    except Exception:
        return ""


def dedup_texts(
    items: List[str]
) -> List[str]:

    out = []
    seen = set()

    for x in items:

        x = normalize_ws(
            x
        )

        if not x:
            continue

        key = re.sub(
            r"[^a-z0-9]+",
            " ",
            x.lower()
        ).strip()

        if (
            key
            and key not in seen
        ):
            seen.add(
                key
            )

            out.append(
                x
            )

    return out


def clean_questions_list(
    qs_any: Any
) -> List[str]:

    if not qs_any:
        return []

    if isinstance(
        qs_any,
        str
    ):
        qs_any = [
            qs_any
        ]

    if not isinstance(
        qs_any,
        list
    ):
        return []

    joined = "".join(
        [
            str(x)
            for x in qs_any
        ]
    )

    if (
        '"questions"' in joined
        or joined.strip().startswith(
            "{"
        )
    ):

        m = re.search(
            r"\{.*\}",
            joined,
            flags=re.DOTALL
        )

        if m:

            try:

                obj = json.loads(
                    m.group(0)
                )

                if (
                    isinstance(
                        obj,
                        dict
                    )
                    and isinstance(
                        obj.get(
                            "questions"
                        ),
                        list
                    )
                ):

                    out = [
                        normalize_ws(
                            x
                        )
                        for x
                        in obj[
                            "questions"
                        ]
                        if (
                            isinstance(
                                x,
                                str
                            )
                            and normalize_ws(
                                x
                            )
                        )
                    ]

                    return dedup_texts(
                        out
                    )

            except Exception:
                pass

    out: List[str] = []
    buf = ""

    for x in qs_any:

        s = normalize_ws(
            x
        )

        if not s:
            continue

        buf = (
            (
                buf
                + " "
                + s
            ).strip()
            if buf
            else s
        )

        if "?" in buf:

            first_q = (
                buf.split(
                    "?",
                    1
                )[0].strip()
                + "?"
            )

            out.append(
                first_q
            )

            buf = ""

    if (
        buf
        and len(
            buf
        ) > 5
    ):
        out.append(
            buf
        )

    return dedup_texts(
        out
    )



def domain_matches_any(
    domain: str,
    patterns: List[str]
) -> bool:

    return any(
        p in domain
        for p in patterns
    )


def is_unscrapable_domain(
    domain: str
) -> bool:

    return domain_matches_any(
        domain,
        UNSCRAPABLE_DOMAINS
    )


def is_block_domain(
    domain: str
) -> bool:

    return domain_matches_any(
        domain,
        BLOCK_DOMAINS
    )


def is_preferred_domain(
    domain: str
) -> bool:

    return domain_matches_any(
        domain,
        PREFERRED_DOMAINS
    )


def score_result_domain(
    url: str
) -> float:

    domain = domain_of(
        url
    )

    if is_block_domain(
        domain
    ):
        return -100.0

    score = 0.0

    if is_preferred_domain(
        domain
    ):
        score += 10.0

    if is_unscrapable_domain(
        domain
    ):
        score -= 3.0

    return score


def filter_and_rank_results(
    results: List[
        Dict[
            str,
            str
        ]
    ]
) -> List[
    Dict[
        str,
        Any
    ]
]:

    kept = []

    for r in results:

        url = normalize_ws(
            r.get(
                "url",
                ""
            )
        )

        if not url:
            continue

        domain = domain_of(
            url
        )

        if is_block_domain(
            domain
        ):
            continue

        kept.append(
            {
                **r,
                "domain": domain,
                "domain_score": score_result_domain(
                    url
                ),
                "is_preferred_domain": is_preferred_domain(
                    domain
                ),
                "is_unscrapable_domain": is_unscrapable_domain(
                    domain
                ),
            }
        )

    kept.sort(
        key=lambda x: x.get(
            "domain_score",
            0.0
        ),
        reverse=True
    )

    return kept[
        :TOP_SEARCH_RESULTS_PER_QUESTION
    ]



def search_web_serpapi(
    query: str,
    max_results: int = 30
) -> List[
    Dict[
        str,
        str
    ]
]:

    out = []

    num_pages = (
        max_results
        + 9
    ) // 10

    for page_num in range(
        num_pages
    ):

        params = {
            "engine": SEARCH_ENGINE,
            "q": query,
            "api_key": GoogleSearchAPI_KEY,
            "google_domain": GOOGLE_DOMAIN,
            "gl": GL,
            "hl": HL,
            "num": 10,
            "start": page_num * 10,
        }

        try:

            search = GoogleSearch(
                params
            )

            results = (
                search.get_dict()
            )

            organic = (
                results.get(
                    "organic_results",
                    []
                )
            )

            for r in organic:

                title = normalize_ws(
                    r.get(
                        "title",
                        ""
                    )
                )

                url = normalize_ws(
                    r.get(
                        "link",
                        ""
                    )
                )

                snippet = normalize_ws(
                    r.get(
                        "snippet",
                        ""
                    )
                )

                if url:

                    out.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                        }
                    )

        except Exception as e:

            if DEBUG:
                print(
                    "SerpAPI search error:",
                    repr(e)
                )

    deduped = []
    seen = set()

    for r in out:

        url = r[
            "url"
        ]

        if url in seen:
            continue

        seen.add(
            url
        )

        deduped.append(
            r
        )

    return deduped[
        :max_results
    ]



def fetch_html(
    url: str
) -> str:

    try:

        headers = {
            "User-Agent": USER_AGENT
        }

        resp = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        if (
            resp.status_code
            != 200
        ):
            return ""

        ctype = resp.headers.get(
            "Content-Type",
            ""
        )

        if (
            "text/html"
            not in ctype
            and "application/xhtml"
            not in ctype
        ):
            return ""

        return resp.text

    except Exception as e:

        if DEBUG:
            print(
                "fetch_html error:",
                url,
                repr(e)
            )

        return ""


def html_to_text(
    html: str
) -> Tuple[
    str,
    str
]:

    if not html:
        return "", ""

    soup = BeautifulSoup(
        html,
        "html.parser"
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

    title = normalize_ws(
        soup.title.get_text(
            " ",
            strip=True
        )
        if soup.title
        else ""
    )

    main = soup.find(
        "article"
    )

    if main is None:
        main = soup.find(
            "main"
        )

    if main is None:
        main = (
            soup.body
            if soup.body
            else soup
        )

    text = main.get_text(
        "\n",
        strip=True
    )

    text = normalize_ws(
        text
    )

    return (
        title,
        text
    )



def retrieve_fulltext_for_question(
    question: str
) -> Dict[
    str,
    Any
]:

    raw_results = search_web_serpapi(
        question,
        max_results=TOP_SEARCH_RESULTS_PER_QUESTION
    )

    search_results = filter_and_rank_results(
        raw_results
    )

    retrieved_items = []

    for rank, sr in enumerate(
        search_results,
        start=1
    ):

        url = sr.get(
            "url",
            ""
        )

        title_from_search = sr.get(
            "title",
            ""
        )

        snippet = sr.get(
            "snippet",
            ""
        )

        domain = sr.get(
            "domain",
            domain_of(
                url
            )
        )

        is_unscrapable = sr.get(
            "is_unscrapable_domain",
            False
        )

        is_preferred = sr.get(
            "is_preferred_domain",
            False
        )

        page_title = ""
        page_text = ""

        if not is_unscrapable:

            html = fetch_html(
                url
            )

            (
                page_title,
                page_text,
            ) = html_to_text(
                html
            )

        item = {
            "rank": rank,
            "question": question,
            "title": (
                page_title
                if page_title
                else title_from_search
            ),
            "url": url,
            "domain": domain,
            "is_preferred_domain": is_preferred,
            "is_unscrapable_domain": is_unscrapable,
            "search_snippet": snippet,
            "full_text": page_text,
        }

        retrieved_items.append(
            item
        )

        time.sleep(
            SLEEP_BETWEEN_REQUESTS
        )

    return {
        "question": question,
        "results": retrieved_items,
    }


def flatten_fulltext_evidence(
    question_evidence: List[
        Dict[
            str,
            Any
        ]
    ]
) -> Tuple[
    List[str],
    List[str]
]:

    flat_texts = []
    flat_urls = []

    for qrec in question_evidence:

        for item in qrec.get(
            "results",
            []
        ):

            txt = normalize_ws(
                item.get(
                    "full_text",
                    ""
                )
            )

            url = normalize_ws(
                item.get(
                    "url",
                    ""
                )
            )

            if txt:

                flat_texts.append(
                    txt
                )

                flat_urls.append(
                    url
                )

    return (
        flat_texts,
        flat_urls
    )


def is_completed_wst_record(
    rec: Dict[
        str,
        Any
    ]
) -> bool:

    if (
        not isinstance(
            rec,
            dict
        )
        or len(
            rec
        ) == 0
    ):
        return False

    question_evidence = rec.get(
        "question_evidence",
        None
    )

    questions = rec.get(
        "questions",
        None
    )

    if (
        isinstance(
            question_evidence,
            list
        )
        and len(
            question_evidence
        ) > 0
    ):
        return True

    # Only treat as completed-empty if the record actually exists
    # and explicitly stores an empty questions list.
    if (
        "questions" in rec
        and isinstance(
            questions,
            list
        )
        and len(
            questions
        ) == 0
    ):
        return True

    return False



def select_question_records(
    qs: Dict[
        str,
        Any
    ],
    meme_id: Optional[str],
) -> Dict[
    str,
    Any
]:

    if meme_id is None:
        return qs

    target_id = normalize_ws(
        meme_id
    )

    if target_id not in qs:
        raise ValueError(
            f"Meme ID {target_id} was not found "
            "in the question JSON."
        )

    return {
        target_id: qs[
            target_id
        ]
    }



def run_wst(
    question_path: Path,
    evidence_path: Path,
    meme_id: Optional[str],
) -> None:

    qs = load_json(
        str(
            question_path
        ),
        {}
    )

    if not isinstance(
        qs,
        dict
    ):
        raise ValueError(
            "Question JSON must be a JSON object."
        )

    qs = select_question_records(
        qs=qs,
        meme_id=meme_id,
    )

    ev = load_json(
        str(
            evidence_path
        ),
        {}
    )

    if not isinstance(
        ev,
        dict
    ):
        ev = {}

    print(
        "Questions :",
        len(
            qs
        )
    )

    print(
        "Evidence  :",
        len(
            ev
        )
    )

    meme_ids = list(
        qs.keys()
    )

    print(
        "Meme IDs to process:",
        len(
            meme_ids
        )
    )

    for idx, meme_id_value in enumerate(
        meme_ids,
        start=1
    ):

        existing_rec = ev.get(
            meme_id_value,
            {}
        )

        if is_completed_wst_record(
            existing_rec
        ):

            print(
                f"[{idx}/{len(meme_ids)}] "
                f"{meme_id_value}: "
                "already completed, skipped"
            )

            continue

        qrec = qs.get(
            meme_id_value,
            {}
        )

        img_name = qrec.get(
            "img",
            ""
        )

        meme_text = qrec.get(
            "text",
            ""
        )

        gen_caption = qrec.get(
            "generated_caption",
            ""
        )

        recognized_people = qrec.get(
            "recognized_people",
            []
        )

        question_types_used = qrec.get(
            "question_types_used",
            []
        )

        questions = clean_questions_list(
            qrec.get(
                "questions",
                []
            )
        )

        if not questions:

            ev[
                meme_id_value
            ] = {
                "img": img_name,
                "text": meme_text,
                "generated_caption": gen_caption,
                "recognized_people": recognized_people,
                "question_types_used": question_types_used,
                "questions": [],
                "question_evidence": [],
                "retrieved_evidence_fulltext": [],
                "retrieved_evidence_urls": [],
            }

            save_json(
                str(
                    evidence_path
                ),
                ev
            )

            print(
                f"[{idx}/{len(meme_ids)}] "
                f"{meme_id_value}: "
                "no questions, saved empty record"
            )

            continue

        print(
            f"[{idx}/{len(meme_ids)}] "
            f"{meme_id_value}: "
            f"{len(questions)} questions"
        )

        per_question_evidence = []

        for qi, question in enumerate(
            questions,
            start=1
        ):

            print(
                f"   Q{qi}: "
                f"{question}"
            )

            qev = retrieve_fulltext_for_question(
                question
            )

            per_question_evidence.append(
                qev
            )

        (
            flat_texts,
            flat_urls,
        ) = flatten_fulltext_evidence(
            per_question_evidence
        )

        ev[
            meme_id_value
        ] = {
            "img": img_name,
            "text": meme_text,
            "generated_caption": gen_caption,
            "recognized_people": recognized_people,
            "question_types_used": question_types_used,
            "questions": questions,
            "question_evidence": per_question_evidence,
            "retrieved_evidence_fulltext": flat_texts,
            "retrieved_evidence_urls": flat_urls,
        }

        if (
            idx
            % SAVE_EVERY_N_MEMES
            == 0
        ):

            save_json(
                str(
                    evidence_path
                ),
                ev
            )

            print(
                "   saved ->",
                evidence_path
            )

    save_json(
        str(
            evidence_path
        ),
        ev
    )

    print(
        "\nFinal saved evidence to:",
        evidence_path
    )



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve full-text web evidence for "
            "QRC-generated questions using the "
            "original WST implementation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help=(
            "Dataset name. When provided, "
            "input/output paths are derived automatically."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        required=True,
        help=(
            "Question-generation branch: "
            "qwen or gemma."
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
        "--question-json",
        type=Path,
        default=None,
        help=(
            "Path to generated question JSON."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path to WST full-text evidence JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:

    global GoogleSearchAPI_KEY

    args = parse_args()

    GoogleSearchAPI_KEY = (
        os.environ.get(
            "GoogleSearchAPI_KEY",
            ""
        )
        or ""
    ).strip()

    if not GoogleSearchAPI_KEY:

        print(
            "ERROR: Missing GoogleSearchAPI_KEY environment variable.",
            file=sys.stderr,
        )

        print(
            'export GoogleSearchAPI_KEY="YOUR_GOOGLE_SEARCH_API_KEY"',
            file=sys.stderr,
        )

        sys.exit(
            1
        )

    try:

        (
            question_path,
            evidence_path,
        ) = resolve_paths(
            dataset=args.dataset,
            model_name=args.model,
            meme_id=args.meme_id,
            question_path=args.question_json,
            evidence_path=args.output_json,
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(
            1
        )

    if not question_path.exists():

        print(
            f"ERROR: Question JSON not found: "
            f"{question_path}",
            file=sys.stderr,
        )

        sys.exit(
            1
        )

    print(
        "=" * 80
    )

    print(
        "Retrieve Stage - WST"
    )

    print(
        "=" * 80
    )

    if args.dataset:

        print(
            f"Dataset      : {args.dataset}"
        )

    print(
        f"Model        : {args.model}"
    )

    if args.meme_id is not None:

        print(
            f"Meme ID      : {args.meme_id}"
        )

    print(
        f"Question JSON: {question_path}"
    )

    print(
        f"Output JSON  : {evidence_path}"
    )

    print(
        f"Search engine: {SEARCH_ENGINE}"
    )

    print(
        "Results/question:",
        TOP_SEARCH_RESULTS_PER_QUESTION
    )

    try:

        run_wst(
            question_path=question_path,
            evidence_path=evidence_path,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: WST retrieval failed: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()