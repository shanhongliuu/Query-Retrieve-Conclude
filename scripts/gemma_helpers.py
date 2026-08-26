import json
from typing import List, Tuple, Optional, Any

import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

from utils import *


MODEL_ID = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loaded:", MODEL_ID)
print("Model device:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)


def get_model_device():
    """
    Returns the device of the first model parameter.
    This follows the same style as the Qwen helper.
    """
    return next(model.parameters()).device


def build_gemma_messages(prompt_text: str):
    """
    Gemma 3 chat template expects the user content to be a list of typed blocks.
    This is slightly different from the Qwen string-only content format.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]


def build_qa_to_statement_prompt(ques: str, ans: str) -> str:
    """
    """

    ques = normalize_ws(ques)
    ans = normalize_ws(ans)

    prompt_lines = [
        "You are a expert writer. Given a question ([QUES]) and its answer [ANS], your goal is to convert the caption and the QA pair into a statement [STAT].",

        "Below are some examples:",
        "",
        "[QUES]: What event or incident is referenced by the image of the Statue of Liberty with dark smoke rising behind a city skyline?",
        "[ANS]: The image of the Statue of Liberty with dark smoke rising behind a city skyline refers to the events following the 9/11 terrorist attacks in Lower Manhattan.",
        "[STAT]: The event referenced by the image of the Statue of Liberty with dark smoke rising behind a city skyline is the 9/11 terrorist attacks in Lower Manhattan.",
        "",
        "[QUES]: What is the significance of \"MN\" in the context of this meme?",
        "[ANS]: In the context of the meme, \"MN\" refers to Minnesota, as evidenced by references to Minneapolis, Minnesota.",
        "[STAT]: \"MN\" refers to Minnesota, a U.S. state in the context of this meme.",
        "",
        "……",
        "",
        "[QUES]: Who is George Floyd, and how is him relevant to Minnesota in this meme?",
        "[ANS]: George Floyd was an African American man. He was killed during an arrest by Minneapolis police in May 2020. His death led to large protests in Minneapolis.",
        "[STAT]: George Floyd  was an African man. He is relevant to Minnesota in the meme because he died during a police arrest in Minneapolis, and the incident became widely associated with protests against police violence.",
        "",
        "Please convert the QA pair below into its statement:",
        f"[QUES]: {ques}",
        f"[ANS]: {ans}",
        "[STAT]:"
    ]

    return "\n".join(prompt_lines)
@torch.inference_mode()
def gemma_generate_text(prompt_text: str, max_new_tokens: int = 220) -> str:
    """
    General Gemma text generation helper.
    """

    messages = build_gemma_messages(prompt_text)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = processor.tokenizer(
        [text],
        return_tensors="pt"
    ).to(get_model_device())

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return clean_decoded_text(response)


@torch.inference_mode()
def synthesize_meme_statement(meme_text, caption, qa_pairs_list):
    """
    Converts each QA pair into one statement using the prompt in this script.

    Args:
        meme_text: Meme text. Kept for compatibility. Not directly used by the current prompt.
        caption: Meme caption. Kept for compatibility. Not directly used by the current prompt.
        qa_pairs_list: list of (question, answer) tuples.

    Returns:
        One string containing all generated statements separated by newlines.
    """

    if not qa_pairs_list:
        return ""

    statements = []

    for item in qa_pairs_list:
        if isinstance(item, dict):
            ques = normalize_ws(item.get("question", ""))
            ans = normalize_ws(item.get("answer", ""))
        else:
            ques, ans = item
            ques = normalize_ws(ques)
            ans = normalize_ws(ans)

        if not ques or not ans:
            continue

        if is_no_answer(ans):
            continue

        prompt_text = build_qa_to_statement_prompt(ques, ans)

        raw_output = gemma_generate_text(
            prompt_text=prompt_text,
            max_new_tokens=MAX_NEW_TOKENS_STATEMENT
        )

        statement = clean_statement_output(raw_output)

        if statement and not is_no_answer(statement):
            statements.append(statement)

    return "\n".join(statements)


@torch.inference_mode()
def gemma_answer_generate(prompt_text: str, max_new_tokens: int = 220) -> str:
    """
    Gemma version of qwen_answer_generate().
    Keeps the same input/output behavior.
    """

    return gemma_generate_text(
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens
    )


def gen_answer_with_gemma(detailed_evid: List[str], question: str) -> Tuple[str, List[int], str]:
    """
    Gemma version of gen_answer_with_qwen().
    """

    evidence_block = gen_retrieved_input(detailed_evid)

    prompt_text = ANSWER_PROMPT.format(
        question=question,
        evidence_block=evidence_block
    )

    print(" - .... Gemma processing ...")

    raw_output = gemma_answer_generate(
        prompt_text=prompt_text,
        max_new_tokens=MAX_NEW_TOKENS_ANSWER
    )

    answer, used_chunk_ids, raw_clean = parse_answer_output(
        raw_output,
        len(detailed_evid)
    )

    return answer, used_chunk_ids, raw_clean


def answer_with_fixed_top_k(
    question: str,
    ranked_chunks_full: List[str],
) -> Tuple[str, List[int], str, int]:
    """
    """

    chunks = ranked_chunks_full[:TOP_K_CHUNKS]

    if not chunks:
        return (
            "No answer can be found.",
            [],
            json.dumps(
                {
                    "answer": "No answer can be found.",
                    "used_chunk_ids": []
                },
                ensure_ascii=False
            ),
            TOP_K_CHUNKS
        )

    answer, used_chunk_ids, raw_output = gen_answer_with_gemma(
        chunks,
        question
    )


    if not used_chunk_ids:
        used_chunk_ids = estimate_used_chunk_ids(answer, chunks)

    return answer, used_chunk_ids, raw_output, TOP_K_CHUNKS


@torch.inference_mode()
def qa_to_evid_gemma(ques: str, ans: str):
    """
    Gemma version of qa_to_evid_qwen().
    Converts one QA pair into one clean background-knowledge statement.
    This version uses the prompt defined inside this script.
    """

    ques = normalize_ws(ques)
    ans = normalize_ws(ans)

    if not ques or not ans:
        return None

    if is_no_answer(ans):
        return None

    prompt_text = build_qa_to_statement_prompt(ques, ans)

    raw_output = gemma_generate_text(
        prompt_text=prompt_text,
        max_new_tokens=MAX_NEW_TOKENS_STATEMENT
    )

    statement = clean_statement_output(raw_output)

    if not statement:
        return normalize_ws(ans)

    if is_no_answer(statement):
        return None

    return statement