import sys
from pathlib import Path

# Shared scripts are one directory above scripts/conclude/
SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import os
from typing import Any, Dict, List

from config import *
from utils import *
from qwen_helpers import qa_to_evid_qwen


def main():
    parser = argparse.ArgumentParser(
        description="Convert QA pairs into descriptive statements."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="memeintent",
        help="Name of the dataset folder"
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="v2_withBM",
        help="Suffix matching the answer JSON file"
    )

    parser.add_argument(
        "--meme-id",
        type=str,
        default=None,
        help="Process only one meme ID for testing"
    )

    args = parser.parse_args()

    # Get paths from config
    IMAGES_DIR, DATA_JSON_PATH, \
        OUT_DIR, RETRIEVED_JSON_PATH, \
        ANSWER_JSON_PATH, STATEMENT_JSON_PATH, \
        EVAL_RESULTS_PATH, EVAL_AVG_PATH \
        = set_retrieve_paths(
            dataset_name=args.dataset,
            answer_suffix=args.output_suffix,
            output_suffix=args.output_suffix,
            selected_model="QWEN3-32B"
        )

    # =========================================================
    # ONE-MEME TEST PATHS
    # =========================================================
    if args.meme_id is not None:

        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        TEST_RETRIEVE_DIR = (
            PROJECT_ROOT
            / "outputs"
            / "test_runs"
            / args.dataset
            / "retrieve"
        )

        TEST_CONCLUDE_DIR = (
            PROJECT_ROOT
            / "outputs"
            / "test_runs"
            / args.dataset
            / "conclude"
        )

        TEST_CONCLUDE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Keep the original answer filename from set_retrieve_paths(),
        # but read it from the one-meme retrieve test directory.
        ANSWER_JSON_PATH = str(
            TEST_RETRIEVE_DIR
            / Path(ANSWER_JSON_PATH).name
        )

        # Keep the original statement filename from set_retrieve_paths(),
        # but save it inside the one-meme conclude test directory.
        STATEMENT_JSON_PATH = str(
            TEST_CONCLUDE_DIR
            / Path(STATEMENT_JSON_PATH).name
        )

    # Load data
    qa_data = load_json(
        ANSWER_JSON_PATH,
        {}
    )

    statement_data = load_json(
        STATEMENT_JSON_PATH,
        {}
    )

    if not isinstance(qa_data, dict) or not qa_data:
        print(
            f"Error: Could not load QA data from "
            f"{ANSWER_JSON_PATH}"
        )
        return

    # =========================================================
    # ONE-MEME TEST SELECTION
    # =========================================================
    if args.meme_id is not None:

        meme_id = str(args.meme_id)

        if meme_id not in qa_data:
            print(
                f"Error: meme_id={meme_id} was not found in "
                f"{ANSWER_JSON_PATH}"
            )
            return

        qa_data = {
            meme_id: qa_data[meme_id]
        }

    print(
        "============== Generate Statements from QA =============="
    )
    print(
        f"Dataset: {args.dataset}"
    )

    if args.meme_id is not None:
        print(
            f"Meme ID: {args.meme_id}"
        )

    print(
        f"Input QA path: {ANSWER_JSON_PATH}"
    )
    print(
        f"Output Stat path: {STATEMENT_JSON_PATH}"
    )
    print(
        f"Loaded: {len(qa_data)} records"
    )
    print(
        "=========================================================="
    )

    meme_items = list(
        qa_data.items()
    )

    # Metrics for the final print
    new_memes = 0
    total_qa_pairs = 0
    total_statements = 0

    for meme_idx, (meme_id, rec) in enumerate(
        meme_items,
        start=1
    ):

        # Check if already processed
        existing_rec = statement_data.get(
            str(meme_id),
            {}
        )

        if is_completed_statement_record(existing_rec):
            print(
                f"Meme {meme_idx}/{len(meme_items)} | "
                f"ID: {rec.get('img', 'N/A')} - "
                f"already completed, skipping."
            )
            continue

        if not isinstance(rec, dict):
            continue

        print(
            f"Processing meme "
            f"{meme_idx}/{len(meme_items)}: "
            f"{rec.get('img', 'N/A')}"
        )

        new_memes += 1

        img = rec.get(
            "img",
            ""
        )

        text = rec.get(
            "text",
            ""
        )

        gen_caption = rec.get(
            "generated_caption",
            ""
        )

        questions = rec.get(
            "questions",
            []
        )

        answers = rec.get(
            "answers",
            []
        )

        per_meme_statements: List[str] = []

        for qa_idx, ans_item in enumerate(
            answers,
            start=1
        ):

            if not isinstance(ans_item, dict):
                continue

            ques = normalize_ws(
                ans_item.get(
                    "question",
                    ""
                )
            )

            ans = normalize_ws(
                ans_item.get(
                    "answer",
                    ""
                )
            )

            print(
                f"  - QA {qa_idx}/{len(answers)} "
                f"| Q: {ques} "
                f"| A: {ans}"
            )

            if not ques or not ans:
                continue

            total_qa_pairs += 1

            print(
                f"  - Total QA pairs processed so far: "
                f"{total_qa_pairs}"
            )

            # Use the original Qwen helper
            statement = qa_to_evid_qwen(
                ques,
                ans
            )

            # REVISED: Only append if we actually got a factual string back
            if statement and statement.strip():

                print(
                    f"  - Generated statement: "
                    f"{statement}"
                )

                per_meme_statements.append(
                    statement
                )

                total_statements += 1

                print(
                    f"  - Stat "
                    f"{qa_idx}/{len(answers)} generated."
                )

            else:

                print(
                    f"  - Stat "
                    f"{qa_idx}/{len(answers)} "
                    f"skipped (No factual info)."
                )

        # Save record
        statement_data[str(meme_id)] = {
            "img": img,
            "text": text,
            "generated_caption": gen_caption,
            "questions": questions,
            "answers": answers,
            "statement": "\n".join(
                per_meme_statements
            )
        }

        # Save progress according to config setting
        if (
            SAVE_EVERY_MEME > 0
            and meme_idx % SAVE_EVERY_MEME == 0
        ):

            save_json(
                STATEMENT_JSON_PATH,
                statement_data
            )

    # Final Save
    save_json(
        STATEMENT_JSON_PATH,
        statement_data
    )

    print(
        "\n" + "=" * 30
    )

    print(
        "Processing Complete."
    )

    print(
        f"Total memes processed this run: "
        f"{new_memes}"
    )

    print(
        f"Total QA pairs converted:       "
        f"{total_qa_pairs}"
    )

    print(
        f"Total statements generated:     "
        f"{total_statements}"
    )

    print(
        f"Results saved to: "
        f"{STATEMENT_JSON_PATH}"
    )

    print(
        "=" * 30
    )


if __name__ == "__main__":
    main()