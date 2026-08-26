import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import argparse
import os

from config import *
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Generate answers from meme evidence.")
    parser.add_argument("--dataset", type=str, default="memeintent", help="Name of the dataset folder")
    parser.add_argument("--output_suffix", type=str, default="v2", help="Suffix for the output JSON file")
    parser.add_argument("--top_k", type=int, default=10, help="Top K chunks to use for ranking")
    parser.add_argument("--no_rank", action="store_true", help="If set, skip BM25 and pass all evidence (up to limit)")
    parser.add_argument("--selected_model", type=str, default="QWEN3-32B", help="Model name for generation")

    # one-meme test
    parser.add_argument(
        "--meme-id",
        type=str,
        default=None,
        help="Process only one meme ID for testing."
    )
    
    args = parser.parse_args()
    
    if args.selected_model == "QWEN3-32B":
        import qwen_helpers as helper
    elif args.selected_model == "GEMMA3-27B":
        import gemma_helpers as helper

    # Override paths based on args
    IMAGES_DIR, DATA_JSON_PATH, \
    OUT_DIR, RETRIEVED_JSON_PATH, \
        ANSWER_JSON_PATH, STATEMENT_JSON_PATH, \
            EVAL_RESULTS_PATH, EVAL_AVG_PATH \
            = set_retrieve_paths(
                dataset_name = args.dataset, 
                answer_suffix = args.output_suffix, 
                output_suffix = args.output_suffix,
                selected_model = args.selected_model)


    if args.meme_id is not None:

        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        TEST_RETRIEVE_DIR = (
            PROJECT_ROOT
            / "outputs"
            / "test_runs"
            / args.dataset
            / "retrieve"
        )

        if args.selected_model == "QWEN3-32B":
            RETRIEVED_JSON_PATH = str(
                TEST_RETRIEVE_DIR / "wst_fulltext.json"
            )

        elif args.selected_model == "GEMMA3-27B":
            RETRIEVED_JSON_PATH = str(
                TEST_RETRIEVE_DIR / "wst_fulltext_gemma.json"
            )

        # Keep the original answer filename from set_retrieve_paths(),
        # but save it inside the one-meme test directory.
        ANSWER_JSON_PATH = str(
            TEST_RETRIEVE_DIR / Path(ANSWER_JSON_PATH).name
        )

    final_answers = load_json(ANSWER_JSON_PATH, {})
    retrieved_data = load_json(RETRIEVED_JSON_PATH, {})
    
    if not isinstance(retrieved_data, dict):
        print(f"Error: Could not load data from {RETRIEVED_JSON_PATH}")
        return


    if args.meme_id is not None:

        meme_id = str(args.meme_id)

        if meme_id not in retrieved_data:
            print(
                f"Error: meme_id={meme_id} was not found in "
                f"{RETRIEVED_JSON_PATH}"
            )
            return

        retrieved_data = {
            meme_id: retrieved_data[meme_id]
        }

    print(f"============== Generate answer from Evidence ==============")
    print(f"Dataset: {args.dataset} | Loaded: {len(retrieved_data)} records")

    if args.meme_id is not None:
        print(f"Meme ID: {args.meme_id}")

    print(f"Retrieved evidence path: {RETRIEVED_JSON_PATH}")
    print(f"Output path: {ANSWER_JSON_PATH}")
    print(f"Ranking: {'Disabled' if args.no_rank else 'BM25 (Top ' + str(args.top_k) + ')'}")
    print("==========================================================")


    meme_items = list(retrieved_data.items())
    for meme_idx, (meme_id, rec) in enumerate(meme_items, start=1):
        print(f"\nProcessing meme {meme_idx}/{len(meme_items)}: meme_id={rec.get('img', 'N/A')}")
        existing_rec = final_answers.get(str(meme_id), {})
        if is_completed_answer_record(existing_rec):
            print(f"  - already completed, skipped")
            continue

        img = normalize_ws(rec.get("img", ""))
        text = normalize_ws(rec.get("text", ""))
        gen_caption = normalize_ws(rec.get("generated_caption", "") or rec.get("image_caption", ""))
        
        questions = rec.get("questions", [])
        question_evidence = rec.get("question_evidence", [])

        # Build lookup
        qev_lookup = {normalize_ws(qev.get("question", "")): qev for qev in question_evidence if isinstance(qev, dict)}

        answer_records = []
        for q_idx, question in enumerate(questions, start=1):
            question = normalize_ws(question)
            q_item = qev_lookup.get(question)
            print(f"  - Question {q_idx}/{len(questions)}: {question}")

            if not q_item:
                answer_records.append({"question": question, "answer": "[q_item is None]  No evidence found.", "used_chunk_ids": []})
                print(f"  - [meme {meme_idx}/{len(meme_items)}] Question {q_idx}/{len(questions)}: No evidence record found for question.")
                continue

            # 1. Collect evidence
            evidences = collect_texts_for_question(q_item)
            
            # 2. Ranking Logic
            if args.no_rank:
                # Bypass ranking: just take the first X snippets
                ranked_chunks = evidences[:args.top_k]
            else:
                # Standard BM25 ranking
                ranked_chunks = rank_evid_text(question, evidences, top_k=args.top_k)

            if not ranked_chunks:
                answer_records.append({"question": question, "answer": "No rankable chunks.", "used_chunk_ids": []})
                print(f"  - [meme {meme_idx}/{len(meme_items)}] Question {q_idx}/{len(questions)}: No rankable chunks found.")
                continue

            # 3. Generation
            answer, used_ids, raw_out, used_k = helper.answer_with_fixed_top_k(question, ranked_chunks)
            print(f"  - [meme {meme_idx}/{len(meme_items)}] Question {q_idx}/{len(questions)}: Generated answer: {answer}")

            answer_records.append({
                "question": question,
                "answer": answer,
                "used_chunk_ids": used_ids,
                "raw_model_output": raw_out,
                "used_top_k_chunks": used_k
            })

        final_answers[str(meme_id)] = {
            "img": img,
            "text": text,
            "generated_caption": gen_caption,
            "questions": questions,
            "answers": answer_records
        }

        if SAVE_EVERY_MEME > 0 and meme_idx % SAVE_EVERY_MEME == 0:
            save_json(ANSWER_JSON_PATH, final_answers)
            print(f"Progress: {meme_idx}/{len(meme_items)} memes saved.")

    save_json(ANSWER_JSON_PATH, final_answers)
    print("Total memes processed:", len(final_answers))
    print(f"Finished. Saved to {ANSWER_JSON_PATH}")

if __name__ == "__main__":
    main()