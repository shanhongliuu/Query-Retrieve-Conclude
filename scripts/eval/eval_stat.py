import sys
from pathlib import Path

# Shared scripts are one directory above scripts/evaluation/
SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from config import *
from utils import *
from prompt_templates import EVAL_PROMPT



from google import genai
from google.genai import types


GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    ""
).strip()

if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. "
        "Please set it before running."
    )

client = genai.Client(
    http_options=genai.types.HttpOptions(
        api_version="v1"
    ),
    api_key=GEMINI_API_KEY
)

# GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"



def strip_leading_number(s: str) -> str:
    return re.sub(
        r"^\s*\d+\.\s*",
        "",
        str(s or "")
    ).strip()


def split_items_from_text(text: str) -> List[str]:
    if not text:
        return []

    parts = [
        normalize_ws(x)
        for x in str(text).split("\n")
    ]

    return [
        x
        for x in parts
        if x
    ]


def split_items_from_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []

    cleaned = []

    for x in items:
        x = normalize_ws(x)
        x = strip_leading_number(x)

        if x:
            cleaned.append(x)

    return cleaned


def format_numbered_block(
    label: str,
    items: List[str]
) -> str:

    if not items:
        return f"[{label}]:"

    numbered = " ".join(
        [
            f"{i + 1}. {item}"
            for i, item in enumerate(items)
        ]
    )

    return f"[{label}]: {numbered}"


def build_eval_input(
    pred_items: List[str],
    ref_items: List[str]
) -> str:

    pred_block = format_numbered_block(
        "PRED",
        pred_items
    )

    ref_block = format_numbered_block(
        "REF",
        ref_items
    )

    return (
        EVAL_PROMPT
        + "\n\n"
        + pred_block
        + "\n"
        + ref_block
    )



def call_gemini(prompt_text: str) -> str:

    # Safety config
    config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt_text,
        # config=config
    )


    if not response.text:
        print(
            "Warning: Gemini returned an empty response. "
            f"Finish Reason: "
            f"{response.candidates[0].finish_reason}"
        )
        return ""

    return response.text.strip()



def parse_count(
    feedback: str,
    label: str
) -> Optional[int]:

    pattern = (
        rf"\[{re.escape(label)}\]"
        rf"\s*:\s*(-?\d+)"
    )

    m = re.search(
        pattern,
        feedback
    )

    if m:
        try:
            return int(
                m.group(1)
            )
        except Exception:
            return None

    return None


def compute_prf(
    pred_in_ref: int,
    ref_in_pred: int,
    num_pred_items: int,
    num_ref_items: int
):

    precision = (
        0.0
        if num_pred_items == 0
        else pred_in_ref / num_pred_items
    )

    recall = (
        0.0
        if num_ref_items == 0
        else ref_in_pred / num_ref_items
    )

    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


def convert_reference_to_dict(
    reference_data_raw: Any
) -> Dict[str, Dict[str, Any]]:


    if isinstance(
        reference_data_raw,
        dict
    ):
        return reference_data_raw

    if isinstance(
        reference_data_raw,
        list
    ):
        converted = {}

        for idx, rec in enumerate(
            reference_data_raw,
            start=1
        ):
            if not isinstance(
                rec,
                dict
            ):
                continue

            meme_id = rec.get(
                "meme_id",
                idx
            )

            converted[
                str(meme_id)
            ] = rec

        return converted

    raise ValueError(
        "Reference bks json must be either "
        "a dict or a list of records."
    )


def extract_reference_items(
    ref_rec: Dict[str, Any]
) -> List[str]:
    """
    Supports several possible reference formats.

    Priority:
    1. background_knowledge_list
    2. bks
    3. statement
    4. text (fallback only if needed)
    """

    if "background_knowledge_list" in ref_rec:
        return split_items_from_list(
            ref_rec.get(
                "background_knowledge_list",
                []
            )
        )

    if "bks" in ref_rec:
        return split_items_from_text(
            ref_rec.get(
                "bks",
                ""
            )
        )

    if "statement" in ref_rec:
        return split_items_from_text(
            ref_rec.get(
                "statement",
                ""
            )
        )

    return split_items_from_text(
        ref_rec.get(
            "text",
            ""
        )
    )


def pivot_dict_by_image(
    original_dict
):
    new_dict = {}

    for item_id, content in original_dict.items():

        # Get the image path to use as the new key
        img_path = content.get(
            "img"
        )

        if img_path:

            # We store the rest of the data
            # (text, bks) under this path.
            # We can also keep the original ID
            # inside the dict if needed.
            new_dict[
                img_path
            ] = {
                "original_id": item_id,
                **content
            }

    return new_dict



MAX_RETRIES = 3
SAVE_EVERY = 10


def run_evaluation(
    pred_path,
    ref_path,
    save_path,
    avg_save_path
):

    reference_data_raw = load_json(
        ref_path,
        {}
    )

    reference_data = convert_reference_to_dict(
        reference_data_raw
    )

    pred_data = load_json(
        pred_path,
        {}
    )

    # Pivot both dictionaries by image path
    reference_data = pivot_dict_by_image(
        reference_data
    )

    pred_data = pivot_dict_by_image(
        pred_data
    )

    common_paths = sorted(
        set(pred_data.keys())
        & set(reference_data.keys())
    )

    results = {}

    for img_path in common_paths:

        pred_rec = pred_data[
            img_path
        ]

        ref_rec = reference_data[
            img_path
        ]

        # Note: in run_zeroshot we saved it
        # as 'pred_statement'
        pred_statement = pred_rec.get(
            "pred_statement",
            ""
        )

        pred_items = split_items_from_text(
            pred_statement
        )

        ref_items = extract_reference_items(
            ref_rec
        )

        print(
            f"Evaluating meme at {img_path}"
        )

        print(
            ">>> Pred items:",
            pred_items
        )

        print(
            ">>> Ref items:",
            ref_items
        )

        if not pred_items or not ref_items:

            results[
                img_path
            ] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_pred_items": len(
                    pred_items
                ),
                "num_ref_items": len(
                    ref_items
                )
            }

            print(
                ">>> No items to compare, "
                "skipping Gemini evaluation."
            )

            continue

        prompt_text = build_eval_input(
            pred_items,
            ref_items
        )

        judge_feedback = ""

        for attempt in range(
            MAX_RETRIES
        ):
            try:

                judge_feedback = call_gemini(
                    prompt_text
                )

                # print(
                #     f">>> Gemini feedback:\n"
                #     f"{judge_feedback}"
                # )

                break

            except Exception as e:

                print(
                    "Error calling Gemini "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES}): "
                    f"{repr(e)}"
                )

                continue

        p_in_r = max(
            0,
            min(
                parse_count(
                    judge_feedback,
                    "PRED in REF"
                )
                or 0,
                len(pred_items)
            )
        )

        r_in_p = max(
            0,
            min(
                parse_count(
                    judge_feedback,
                    "REF in PRED"
                )
                or 0,
                len(ref_items)
            )
        )

        precision, recall, f1 = compute_prf(
            p_in_r,
            r_in_p,
            len(pred_items),
            len(ref_items)
        )

        print(
            f">>> Precision: "
            f"{precision:.4f}, "
            f"Recall: "
            f"{recall:.4f}, "
            f"F1: "
            f"{f1:.4f}"
        )

        results[
            img_path
        ] = {
            "ref_img": ref_rec.get(
                "img",
                ""
            ),
            "pred_img": pred_rec.get(
                "img",
                ""
            ),
            "pred_items": pred_items,
            "ref_items": ref_items,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "judge_feedback": judge_feedback
        }

        # time.sleep(1000)
        # brief pause to avoid hitting rate limits

    # Save results
    save_json(
        save_path,
        results
    )

    # Calculate Averages
    valid = list(
        results.values()
    )

    avg_results = {
        "avg_precision": (
            sum(
                v["precision"]
                for v in valid
            )
            / len(valid)
        ),
        "avg_recall": (
            sum(
                v["recall"]
                for v in valid
            )
            / len(valid)
        ),
        "avg_f1": (
            sum(
                v["f1"]
                for v in valid
            )
            / len(valid)
        ),
        "count": len(
            valid
        )
    }

    save_json(
        avg_save_path,
        avg_results
    )

    return avg_results



def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated background knowledge "
            "statements against reference BKS using Gemini."
        )
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help=(
            "Path to the JSON file containing "
            "predicted background knowledge statements."
        )
    )

    parser.add_argument(
        "--ref_path",
        type=str,
        required=True,
        help=(
            "Path to the JSON file containing "
            "reference background knowledge statements."
        )
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help=(
            "Path to save per-meme "
            "evaluation results."
        )
    )

    parser.add_argument(
        "--avg_save_path",
        type=str,
        required=True,
        help=(
            "Path to save averaged "
            "precision, recall, and F1."
        )
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print(
        "============== BKS Evaluation =============="
    )

    print(
        f"Gemini model: {GEMINI_MODEL_NAME}"
    )

    print(
        f"Prediction path: {args.pred_path}"
    )

    print(
        f"Reference path: {args.ref_path}"
    )

    print(
        f"Detail output: {args.save_path}"
    )

    print(
        f"Average output: {args.avg_save_path}"
    )

    print(
        "============================================"
    )

    avg_results = run_evaluation(
        pred_path=args.pred_path,
        ref_path=args.ref_path,
        save_path=args.save_path,
        avg_save_path=args.avg_save_path
    )

    print(
        "\n============== Evaluation Complete =============="
    )

    print(
        f"Average Precision: "
        f"{avg_results['avg_precision']:.4f}"
    )

    print(
        f"Average Recall:    "
        f"{avg_results['avg_recall']:.4f}"
    )

    print(
        f"Average F1:        "
        f"{avg_results['avg_f1']:.4f}"
    )

    print(
        f"Evaluated memes:   "
        f"{avg_results['count']}"
    )

    print(
        "================================================="
    )


if __name__ == "__main__":
    main()