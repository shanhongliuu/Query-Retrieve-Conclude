#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor


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

GEMMA_MODEL_ID = "google/gemma-3-27b-it"
QWEN_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

MAX_NEW_TOKENS_QUESTION = 320
DEBUG = False

SUBSET_ROOT = ""



QUESTION_PROMPT = r"""
You are a Knowledge Gap Analyst for meme understanding.

You are given ONLY:
- Meme text
- Generated image caption
- Recognized people list (may be empty)

Your goal:
Generate a small set of useful background-knowledge questions that a curious human viewer might ask
to better understand the meme.

IMPORTANT:
The goal is NOT to explain the meme directly.
The goal is NOT to fill a fixed number of slots.
The goal is to ask the most likely useful questions a person would search for in order to understand the meme.

The meme may involve:
- people
- objects
- symbols
- logos
- phrases
- quotes
- slogans
- characters
- events
- cultural references
- meme templates
- screenshots
- brands
- places
- organizations
- internet formats

A good set of questions should focus on useful missing background knowledge, such as:
1) entity / identity / reference
   - Who is this person?
   - What is this logo / symbol / character / brand / object?
2) phrase / quote / slogan
   - What does this phrase refer to?
   - Where does this quote come from?
3) association / relationship
   - Why is this person / object / phrase linked together here?
   - What is the connection between the image and the text?
4) event / topic / context
   - What event, issue, or discourse is being referenced?
5) meme template / cultural format
   - Is this a known meme format or reaction image?
   - What is the original context of this image format?

### INPUTS 
Recognized people:
{recognized_people}

Generated image caption:
{gen_caption}

Meme text:
{text}

### OUTPUT (JSON ONLY)
{{
  "question_types_used": ["entity", "phrase", "association"],
  "questions": ["...", "...", "..."]
}}

### HARD CONSTRAINTS
A) Use ONLY information in the inputs.
   - Do NOT invent outside facts, years, names, events, or claims unless they are explicitly present in the inputs.
B) Each question must be grounded in at least one input anchor from:
   - recognized people
   - generated caption
   - meme text
C) Each array element must be ONE complete question ending with exactly ONE '?'.
   - No merged questions in one string.
   - No numbering like "1) ...".
D) Generate ONLY useful, non-redundant questions.
   - Do not ask generic filler questions.
   - Do not repeat the same question in different wording.
E) Prefer diversity of question types when possible.
F) Usually generate 2 to 4 questions.
   - Generate only as many as are useful.
   - Do NOT force a question type if it is not relevant.
G) Keep questions neutral and retrieval-friendly.
   - Ask what a human would search to understand the meme better.
H) Do NOT explain the meme. Only ask questions.
I) If recognized_people is non-empty, then for any question about that person’s identity, relevance, or association, explicitly use the recognized name string.
   - Example: use “John F. Kennedy” instead of “the man” or “this person”.
   - Example: use “Donald Trump” instead of “the man with light-colored hair”.
J) Do NOT use vague person references such as:
   - “the man”
   - “this person”
   - “the person in the image”
   - “the black-and-white portrait”
   when a recognized person name is available.
K) When recognized_people is non-empty, prefer name-based questions that are more useful for retrieval.

Return JSON ONLY.
""".strip()



def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    model_name: str,
    meme_id: Optional[str],
    caption_path: Optional[Path],
    image_dir: Optional[Path],
    output_path: Optional[Path],
) -> Tuple[Path, Path, Path]:

    project_root = get_project_root()

    if dataset:

        if image_dir is None:
            image_dir = (
                project_root
                / "data"
                / dataset
                / "images"
            )

        # one-meme test
        if meme_id is not None:

            if caption_path is None:
                if model_name == "gemma":
                    caption_name = "captions_gemma.json"
                else:
                    caption_name = "captions.json"

                caption_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / caption_name
                )

            if output_path is None:
                if model_name == "gemma":
                    output_name = "question_gemma.json"
                else:
                    output_name = "questions.json"

                output_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / output_name
                )

        # full dataset
        else:

            if caption_path is None:
                if model_name == "gemma":
                    caption_name = "captions_gemma.json"
                else:
                    caption_name = "captions.json"

                caption_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "caps"
                    / caption_name
                )

            if output_path is None:
                if model_name == "gemma":
                    output_name = "question_gemma.json"
                else:
                    output_name = "questions.json"

                output_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "que"
                    / output_name
                )

    if caption_path is None:
        raise ValueError(
            "Missing caption JSON. "
            "Use --dataset or --caption-json."
        )

    if image_dir is None:
        raise ValueError(
            "Missing image directory. "
            "Use --dataset or --image-dir."
        )

    if output_path is None:
        raise ValueError(
            "Missing output JSON. "
            "Use --dataset or --output-json."
        )

    return (
        caption_path.expanduser().resolve(),
        image_dir.expanduser().resolve(),
        output_path.expanduser().resolve(),
    )


# =========================================================
# HELPERS
# =========================================================
def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        print(f"[load_json] Missing file: {path}")
        return {} if default is None else default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False
        )


def normalize_ws(s: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(s or "")
    ).strip()


def resolve_image_path(
    img_name: str,
    image_path: str = ""
) -> str:

    image_path = normalize_ws(
        image_path
    )

    if image_path:

        if os.path.exists(
            image_path
        ):
            return image_path

        rel_candidate = os.path.join(
            SUBSET_ROOT,
            image_path
        )

        if os.path.exists(
            rel_candidate
        ):
            return rel_candidate

    img_name = normalize_ws(
        img_name
    )

    if not img_name:
        return ""

    candidate = os.path.join(
        SUBSET_ROOT,
        img_name
    )

    if os.path.exists(
        candidate
    ):
        return candidate

    return ""


def clean_decoded_text_qwen(
    s: str
) -> str:

    if not isinstance(
        s,
        str
    ):
        return ""

    s = s.strip()

    s = re.sub(
        r"<\|im_start\|>.*?\n",
        "",
        s,
        flags=re.DOTALL
    )

    s = s.replace(
        "<|im_end|>",
        ""
    ).strip()

    return s.strip()


def clean_decoded_text_gemma(
    s: str
) -> str:

    if not isinstance(
        s,
        str
    ):
        return ""

    s = s.strip()

    # Qwen cleanup
    s = re.sub(
        r"<\|im_start\|>.*?\n",
        "",
        s,
        flags=re.DOTALL
    )

    s = s.replace(
        "<|im_end|>",
        ""
    ).strip()

    # Gemma cleanup
    s = (
        s.replace(
            "<bos>",
            ""
        )
        .replace(
            "<eos>",
            ""
        )
        .strip()
    )

    s = re.sub(
        r"<start_of_turn>.*?\n",
        "",
        s,
        flags=re.DOTALL
    )

    s = s.replace(
        "<end_of_turn>",
        ""
    ).strip()

    return s.strip()


def extract_json_obj(
    raw: str
) -> Dict[str, Any]:

    if not isinstance(
        raw,
        str
    ):
        return {}

    s = (
        raw.strip()
        .replace(
            "```json",
            ""
        )
        .replace(
            "```",
            ""
        )
        .strip()
    )

    try:
        obj = json.loads(
            s
        )

        if isinstance(
            obj,
            dict
        ):
            return obj

    except Exception:
        pass

    start = s.find(
        "{"
    )

    end = s.rfind(
        "}"
    )

    if (
        start != -1
        and end != -1
        and end > start
    ):

        candidate = s[
            start:end + 1
        ]

        try:
            obj = json.loads(
                candidate
            )

            if isinstance(
                obj,
                dict
            ):
                return obj

        except Exception:
            pass

    return {}


def dedup_questions(
    questions: List[str]
) -> List[str]:

    out = []
    seen = set()

    for q in questions:

        q = normalize_ws(
            q
        )

        if not q:
            continue

        if (
            "?" in q
            and not q.endswith("?")
        ):
            q = (
                q.split(
                    "?",
                    1
                )[0].strip()
                + "?"
            )

        elif (
            q
            and not q.endswith("?")
        ):
            q = q + "?"

        key = re.sub(
            r"[^a-z0-9]+",
            " ",
            q.lower()
        ).strip()

        if (
            key
            and key not in seen
        ):
            seen.add(
                key
            )

            out.append(
                q
            )

    return out


def parse_question_output(
    raw_text: str,
    parsed_obj: Dict[str, Any]
) -> Tuple[List[str], List[str]]:

    if isinstance(
        parsed_obj,
        dict
    ):

        qtypes = parsed_obj.get(
            "question_types_used",
            []
        )

        questions = parsed_obj.get(
            "questions",
            []
        )

        if not isinstance(
            qtypes,
            list
        ):
            qtypes = []

        if not isinstance(
            questions,
            list
        ):
            questions = []

        qtypes = [
            normalize_ws(x)
            for x in qtypes
            if normalize_ws(x)
        ]

        questions = [
            normalize_ws(x)
            for x in questions
            if normalize_ws(x)
        ]

        questions = dedup_questions(
            questions
        )

        return (
            qtypes,
            questions
        )

    raw = normalize_ws(
        raw_text
    )

    qtypes = []

    m_types = re.search(
        r'"question_types_used"\s*:\s*\[(.*?)\]',
        raw
    )

    if m_types:

        qtypes = [
            normalize_ws(x)
            for x in re.findall(
                r'"([^"]+)"',
                m_types.group(1)
            )
            if normalize_ws(x)
        ]

    questions = re.findall(
        r'"([^"]+\?)"',
        raw
    )

    questions = [
        normalize_ws(x)
        for x in questions
        if normalize_ws(x)
    ]

    questions = dedup_questions(
        questions
    )

    return (
        qtypes,
        questions
    )


def build_question_prompt(
    meme_text: str,
    generated_caption: str,
    recognized_people: List[str]
) -> str:

    recognized_people_json = json.dumps(
        recognized_people,
        ensure_ascii=False
    )

    extra_rule = ""

    if recognized_people:

        extra_rule = (
            f"\nIMPORTANT PERSON RULE:\n"
            f"The recognized_people list is non-empty: {recognized_people_json}\n"
            f"If you ask about the person’s identity or relevance, you must explicitly use the exact recognized name in the question wording.\n"
            f"Do not use generic wording like 'the man', 'this person', or 'the portrait'.\n"
        )

    return QUESTION_PROMPT.format(
        recognized_people=recognized_people_json,
        gen_caption=generated_caption,
        text=meme_text
    ) + extra_rule



def select_caption_records(
    captions: Dict[str, Dict[str, Any]],
    meme_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:

    if meme_id is None:
        return captions

    target_id = normalize_ws(
        meme_id
    )

    if target_id not in captions:
        raise ValueError(
            f"Meme ID {target_id} was not found "
            "in the caption JSON."
        )

    return {
        target_id: captions[target_id]
    }



def load_model(
    model_name: str
):

    if model_name == "gemma":

        from transformers import (
            Gemma3ForConditionalGeneration,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                GEMMA_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=bnb_config,
            )
            .eval()
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                GEMMA_MODEL_ID,
                padding_side="left"
            )
        )

        print(
            "Loaded:",
            GEMMA_MODEL_ID
        )

        print(
            "cuda_available:",
            torch.cuda.is_available()
        )

        print(
            "GPU count:",
            torch.cuda.device_count()
        )

        print(
            "Model device:",
            next(
                model.parameters()
            ).device
        )

        return (
            model,
            processor
        )

    if model_name == "qwen":

        from transformers import (
            Qwen3VLForConditionalGeneration,
        )

        model = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                QWEN_MODEL_ID,
                dtype="auto",
                device_map="auto",
            )
            .eval()
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                QWEN_MODEL_ID
            )
        )

        print(
            "Loaded:",
            QWEN_MODEL_ID
        )

        print(
            "Model device:",
            next(
                model.parameters()
            ).device
        )

        return (
            model,
            processor
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )



def gemma_text_generate(
    model,
    processor,
    prompt_text: str,
    max_new_tokens: int = 320
) -> Tuple[str, Dict[str, Any]]:

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_text
            }
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = inputs.to(
        model_device
    )

    input_len = inputs[
        "input_ids"
    ].shape[-1]

    with torch.inference_mode():

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    out_text = processor.decode(
        gen_ids[0][
            input_len:
        ],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    out_text = clean_decoded_text_gemma(
        out_text
    )

    parsed = extract_json_obj(
        out_text
    )

    return (
        out_text,
        parsed
    )


def qwen_text_generate(
    model,
    processor,
    prompt_text: str,
    max_new_tokens: int = 320
) -> Tuple[str, Dict[str, Any]]:

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt_text
            }
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        k: (
            v.to(
                model_device
            )
            if hasattr(
                v,
                "to"
            )
            else v
        )
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=getattr(
                processor.tokenizer,
                "eos_token_id",
                None
            ),
            eos_token_id=getattr(
                processor.tokenizer,
                "eos_token_id",
                None
            ),
        )

    out_text = processor.batch_decode(
        gen_ids[
            :,
            inputs[
                "input_ids"
            ].shape[1]:
        ],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )[0]

    out_text = clean_decoded_text_qwen(
        out_text
    )

    parsed = extract_json_obj(
        out_text
    )

    return (
        out_text,
        parsed
    )


def generate_questions(
    model,
    processor,
    model_name: str,
    caption_path: Path,
    output_path: Path,
    meme_id: Optional[str],
) -> None:

    captions: Dict[
        str,
        Dict[str, Any]
    ] = load_json(
        str(caption_path),
        {}
    )

    if not isinstance(
        captions,
        dict
    ):
        raise ValueError(
            "Caption JSON must be a JSON object."
        )

    captions = select_caption_records(
        captions=captions,
        meme_id=meme_id,
    )

    output: Dict[
        str,
        Dict[str, Any]
    ] = load_json(
        str(output_path),
        {}
    )

    if not isinstance(
        output,
        dict
    ):
        output = {}

    print(
        "Caption entries loaded:",
        len(captions)
    )

    print(
        "Existing question output:",
        len(output)
    )


    for i, (
        meme_id_value,
        rec
    ) in enumerate(
        captions.items(),
        start=1
    ):

        if meme_id_value in output:

            print(
                f"[{i}/{len(captions)}] "
                f"{meme_id_value}: "
                "already done, skip"
            )

            continue

        img_name = normalize_ws(
            rec.get(
                "img",
                ""
            )
        )

        image_path = resolve_image_path(
            img_name=img_name,
            image_path=rec.get(
                "image_path",
                ""
            )
        )

        meme_text = normalize_ws(
            rec.get(
                "text",
                ""
            )
        )

        generated_caption = normalize_ws(
            rec.get(
                "generated_caption",
                ""
            )
        )

        recognized_people = rec.get(
            "recognized_people",
            []
        )

        if not isinstance(
            recognized_people,
            list
        ):
            recognized_people = []

        recognized_people = [
            normalize_ws(x)
            for x in recognized_people
            if normalize_ws(x)
        ]

        prompt = build_question_prompt(
            meme_text=meme_text,
            generated_caption=generated_caption,
            recognized_people=recognized_people
        )

        if model_name == "gemma":

            (
                raw_question_output,
                parsed_question,
            ) = gemma_text_generate(
                model=model,
                processor=processor,
                prompt_text=prompt,
                max_new_tokens=MAX_NEW_TOKENS_QUESTION
            )

        else:

            (
                raw_question_output,
                parsed_question,
            ) = qwen_text_generate(
                model=model,
                processor=processor,
                prompt_text=prompt,
                max_new_tokens=MAX_NEW_TOKENS_QUESTION
            )

        (
            question_types_used,
            questions,
        ) = parse_question_output(
            raw_question_output,
            parsed_question
        )

        output[
            meme_id_value
        ] = {
            "img": img_name,
            "image_path": image_path,
            "text": meme_text,
            "generated_caption": generated_caption,
            "recognized_people": recognized_people,
            "question_types_used": question_types_used,
            "questions": questions,
            "raw_question_output": raw_question_output,
        }

        save_json(
            str(output_path),
            output
        )

        print(
            f"[{i}/{len(captions)}] "
            f"{meme_id_value}: saved "
            f"({len(questions)} questions)"
        )

        if DEBUG:

            print(
                "Prompt:"
            )

            print(
                prompt
            )

            print(
                "Question types:",
                question_types_used
            )

            print(
                "Questions:",
                questions
            )

            print(
                "Raw output:",
                raw_question_output
            )

            print(
                "-" * 80
            )

    print(
        "Done. Saved to:",
        output_path
    )



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate background-knowledge questions "
            "from QRC caption outputs."
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

    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        required=True,
        help=(
            "Question-generation model: gemma or qwen."
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
        "--caption-json",
        type=Path,
        default=None,
        help=(
            "Path to caption output JSON."
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing meme images."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Question output JSON path."
        ),
    )

    return parser.parse_args()


def main() -> None:

    global SUBSET_ROOT

    args = parse_args()

    try:

        (
            caption_path,
            image_dir,
            output_path,
        ) = resolve_paths(
            dataset=args.dataset,
            model_name=args.model,
            meme_id=args.meme_id,
            caption_path=args.caption_json,
            image_dir=args.image_dir,
            output_path=args.output_json,
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not caption_path.exists():

        print(
            f"ERROR: Caption JSON not found: "
            f"{caption_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not image_dir.exists():

        print(
            f"ERROR: Image directory not found: "
            f"{image_dir}",
            file=sys.stderr,
        )

        sys.exit(1)

    SUBSET_ROOT = str(
        image_dir
    )

    print(
        "=" * 80
    )

    print(
        "Query Stage - Generate Questions"
    )

    print(
        "=" * 80
    )

    if args.dataset:

        print(
            f"Dataset     : {args.dataset}"
        )

    print(
        f"Model       : {args.model}"
    )

    if args.meme_id is not None:

        print(
            f"Meme ID     : {args.meme_id}"
        )

    print(
        f"Caption JSON: {caption_path}"
    )

    print(
        f"Image dir   : {image_dir}"
    )

    print(
        f"Output JSON : {output_path}"
    )

    try:

        model, processor = load_model(
            args.model
        )

        generate_questions(
            model=model,
            processor=processor,
            model_name=args.model,
            caption_path=caption_path,
            output_path=output_path,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: Question generation failed: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()