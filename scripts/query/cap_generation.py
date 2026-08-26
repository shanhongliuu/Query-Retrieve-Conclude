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
from PIL import Image
from transformers import AutoProcessor


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

SUPPORTED_MODELS = (
    "gemma",
    "qwen",
)

GEMMA_MODEL_ID = "google/gemma-3-27b-it"
QWEN_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

MAX_NEW_TOKENS_CAPTION = 260
DEBUG = False

SUBSET_ROOT = ""


# =========================================================
# PROMPT
# =========================================================
CAPTION_PROMPT = r"""
You are an Image Captioner for memes.

You are given:
- A meme image
- Meme text (context only; do NOT repeat it)
- RIS visual hints (support only)

Your task:
Generate a literal description of what is visibly shown.
Use RIS only to help identify ambiguous visible words, people, symbols, or known scenes.

STRICT OUTPUT:
Return JSON ONLY in this format:
{{
  "image_caption": "...",
  "recognized_people": ["..."]
}}

CAPTION RULES:
1) 2–3 sentences total.
2) Describe only visible entities, layout, and simple actions/poses.
3) NO intent, NO opinions, NO meme explanation.
4) If the image contains multiple panels or portraits, summarize the overall structure and the most salient visible elements.
5) You MAY mention short visible labels or words inside the image if they help identify what is shown.
6) Public figure naming:
   - If you are VERY confident a visible person is a known public figure, include their name.
   - RIS may help increase confidence, but only if the person is plausibly visible in the image.
   - If not confident, DO NOT guess. Use neutral descriptions such as "a man", "a woman", "a politician".
7) "recognized_people" should contain ONLY names you are very confident are visibly present.
8) If the image implies speech, you may use exactly: "appearing to say the text".
9) Do NOT say "this meme" or "in the image".
10) Use RIS carefully:
   - RIS is only a support signal for identifying visible elements.
   - If RIS repeatedly supports a visible word, person, symbol, or known photo context, you may use that to strengthen the caption.
   - Do NOT include facts that are not visually supported.
   - Do NOT mention hidden background facts, controversy details, or events unless they are needed only to identify what is visibly shown.
11) Prefer visual certainty over RIS speculation.
12) If RIS strongly suggests that a visible arrangement forms a specific word or slogan, mention that visible word directly.
13) If RIS strongly suggests the identities of visible public figures in a known photograph, you may name them if the match is visually plausible.

Use RIS for:
- visible words or slogans formed by objects
- visible public figure identification
- visually obvious known photo or scene identification

Do not use RIS for:
- controversy explanation
- hidden context
- speculation
- unseen details

Meme text (context only; do NOT repeat it):
{text}

RIS visual hints:
{ris_summary}
""".strip()


# =========================================================
# PATHS
# =========================================================
def get_project_root() -> Path:

    return Path(__file__).resolve().parents[2]


def resolve_paths(
    dataset: Optional[str],
    model_name: str,
    meme_id: Optional[str],
    meme_path: Optional[Path],
    ris_path: Optional[Path],
    image_dir: Optional[Path],
    output_path: Optional[Path],
) -> tuple[Path, Path, Path, Path]:

    project_root = get_project_root()

    if dataset:

        if meme_path is None:
            meme_path = (
                project_root
                / "data"
                / dataset
                / f"{dataset}_qg.json"
            )

        if image_dir is None:
            image_dir = (
                project_root
                / "data"
                / dataset
                / "images"
            )

        # one-meme test
        if meme_id is not None:

            if ris_path is None:
                ris_path = (
                    project_root
                    / "outputs"
                    / "test_runs"
                    / dataset
                    / "query"
                    / "ris_caps.json"
                )

            if output_path is None:

                if model_name == "gemma":
                    output_name = "captions_gemma.json"
                else:
                    output_name = "captions.json"

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

            if ris_path is None:
                ris_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "ris"
                    / "ris_caps.json"
                )

            if output_path is None:

                if model_name == "gemma":
                    output_name = "captions_gemma.json"
                else:
                    output_name = "captions.json"

                output_path = (
                    project_root
                    / "outputs"
                    / dataset
                    / "question_generation"
                    / "caps"
                    / output_name
                )

    if meme_path is None:
        raise ValueError(
            "Missing meme JSON. "
            "Use --dataset or --meme-json."
        )

    if ris_path is None:
        raise ValueError(
            "Missing caption RIS JSON. "
            "Use --dataset or --ris-json."
        )

    if image_dir is None:
        raise ValueError(
            "Missing image directory. "
            "Use --dataset or --image-dir."
        )

    if output_path is None:
        raise ValueError(
            "Missing output path. "
            "Use --dataset or --output-json."
        )

    return (
        meme_path.expanduser().resolve(),
        ris_path.expanduser().resolve(),
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
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
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


def clean_decoded_text(s: str) -> str:
    if not isinstance(s, str):
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


def extract_json_obj(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str):
        return {}

    s = (
        raw.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        obj = json.loads(s)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")

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

            if isinstance(obj, dict):
                return obj

        except Exception:
            pass

    return {}


def dedup_texts(
    items: List[str]
) -> List[str]:

    out = []
    seen = set()

    for x in items:

        x = normalize_ws(x)

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
            seen.add(key)
            out.append(x)

    return out


def build_caption_ris_block(
    caption_ris_summary: Dict[str, Any]
) -> str:

    anchors = caption_ris_summary.get(
        "caption_ris_anchors",
        []
    )

    hints = caption_ris_summary.get(
        "caption_ris_hints",
        []
    )

    quality = caption_ris_summary.get(
        "ris_quality",
        "low"
    )

    if not isinstance(
        anchors,
        list
    ):
        anchors = []

    if not isinstance(
        hints,
        list
    ):
        hints = []

    anchors = [
        normalize_ws(x)
        for x in anchors
        if normalize_ws(x)
    ]

    hints = [
        normalize_ws(x)
        for x in hints
        if normalize_ws(x)
    ]

    lines = [
        f"RIS quality: {quality}"
    ]

    if anchors:

        lines.append(
            "Caption-support visual anchors:"
        )

        for a in anchors:
            lines.append(
                f"- {a}"
            )

    else:

        lines.append(
            "Caption-support visual anchors: (none)"
        )

    if hints:

        lines.append(
            "Caption-support hints:"
        )

        for h in hints:
            lines.append(
                f"- {h}"
            )

    else:

        lines.append(
            "Caption-support hints: (none)"
        )

    return "\n".join(
        lines
    )


def normalize_memes(
    raw_memes: Any
) -> Dict[str, Dict[str, Any]]:

    normalized = {}

    if isinstance(
        raw_memes,
        list
    ):

        for idx, rec in enumerate(
            raw_memes,
            start=1
        ):

            if not isinstance(
                rec,
                dict
            ):
                continue

            meme_id = str(
                rec.get(
                    "meme_id",
                    idx
                )
            )

            normalized[meme_id] = {
                "img_path": normalize_ws(
                    rec.get(
                        "img_path",
                        ""
                    )
                    or rec.get(
                        "img",
                        ""
                    )
                ),
                "text": normalize_ws(
                    rec.get(
                        "text",
                        ""
                    )
                ),
            }

    elif isinstance(
        raw_memes,
        dict
    ):

        for meme_id, rec in raw_memes.items():

            if not isinstance(
                rec,
                dict
            ):
                continue

            normalized[
                str(meme_id)
            ] = {
                "img_path": normalize_ws(
                    rec.get(
                        "img_path",
                        ""
                    )
                    or rec.get(
                        "img",
                        ""
                    )
                ),
                "text": normalize_ws(
                    rec.get(
                        "text",
                        ""
                    )
                ),
            }

    return normalized


def candidate_image_paths(
    img_name: str
) -> List[str]:
    """
    Generate candidate local paths for both png/jpg/jpeg.
    Supports:
    - 7.png
    - 7.jpg
    - images/7.png
    - images/7.jpg
    - 7   (no extension)
    """

    img_name = normalize_ws(
        img_name
    )

    if not img_name:
        return []

    candidates = []

    if os.path.isabs(
        img_name
    ):

        candidates.append(
            img_name
        )

        stem, ext = os.path.splitext(
            img_name
        )

        if ext.lower() not in {
            ".png",
            ".jpg",
            ".jpeg"
        }:

            candidates.extend(
                [
                    stem + ".png",
                    stem + ".jpg",
                    stem + ".jpeg"
                ]
            )

        return list(
            dict.fromkeys(
                candidates
            )
        )

    variants = [
        img_name
    ]

    if img_name.startswith("./"):
        variants.append(
            img_name[2:]
        )

    if img_name.startswith(
        "images/"
    ):
        variants.append(
            img_name[
                len("images/"):
            ]
        )

    else:
        variants.append(
            f"images/{img_name}"
        )

    expanded_variants = []

    for v in variants:

        v = normalize_ws(v)

        if not v:
            continue

        stem, ext = os.path.splitext(
            v
        )

        if ext.lower() in {
            ".png",
            ".jpg",
            ".jpeg"
        }:

            expanded_variants.append(
                v
            )

            # also try swapping extension
            expanded_variants.append(
                stem + ".png"
            )

            expanded_variants.append(
                stem + ".jpg"
            )

            expanded_variants.append(
                stem + ".jpeg"
            )

        else:

            expanded_variants.append(
                v
            )

            expanded_variants.append(
                v + ".png"
            )

            expanded_variants.append(
                v + ".jpg"
            )

            expanded_variants.append(
                v + ".jpeg"
            )

    for rel in expanded_variants:

        rel = normalize_ws(
            rel
        )

        if not rel:
            continue

        # if rel begins with images/, strip it because
        # SUBSET_ROOT already points to images/
        if rel.startswith(
            "images/"
        ):
            rel2 = rel[
                len("images/"):
            ]

        else:
            rel2 = rel

        candidates.append(
            os.path.join(
                SUBSET_ROOT,
                rel2
            )
        )

    return list(
        dict.fromkeys(
            candidates
        )
    )


def resolve_image_path(
    img_name: str
) -> str:
    """
    Return the first existing path among png/jpg/jpeg candidates.
    If nothing exists, return the first candidate for debugging.
    """

    candidates = candidate_image_paths(
        img_name
    )

    for p in candidates:

        if os.path.exists(
            p
        ):
            return p

    return (
        candidates[0]
        if candidates
        else ""
    )


def should_skip_existing(
    existing_rec: Dict[str, Any]
) -> bool:

    if (
        not isinstance(
            existing_rec,
            dict
        )
        or not existing_rec
    ):
        return False

    generated_caption = normalize_ws(
        existing_rec.get(
            "generated_caption",
            ""
        )
    )

    raw_caption_output = normalize_ws(
        existing_rec.get(
            "raw_caption_output",
            ""
        )
    ).lower()

    if generated_caption:
        return True

    if (
        "image not found"
        in raw_caption_output
    ):
        return False

    if (
        "caption generation failed"
        in raw_caption_output
    ):
        return False

    return False


# =========================================================
# ONE-MEME TEST
# =========================================================
def select_memes(
    memes: Dict[str, Dict[str, Any]],
    meme_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:

    if meme_id is None:
        return memes

    target_id = normalize_ws(
        meme_id
    )

    if target_id not in memes:
        raise ValueError(
            f"Meme ID {target_id} was not found."
        )

    return {
        target_id: memes[target_id]
    }


# =========================================================
# MODEL
# =========================================================
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

        return model, processor

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

        return model, processor

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


# =========================================================
# MODEL CALL: GEMMA 3
# =========================================================
def gemma_caption_generate(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 260
) -> Tuple[str, Dict[str, Any]]:

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": prompt_text
            },
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
            v.to(model_device)
            if hasattr(v, "to")
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

    out_text = clean_decoded_text(
        out_text
    )

    parsed = extract_json_obj(
        out_text
    )

    return (
        out_text,
        parsed
    )


# =========================================================
# MODEL CALL: QWEN
# =========================================================
def qwen_caption_generate(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 260
) -> Tuple[str, Dict[str, Any]]:

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": prompt_text
            },
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
            v.to(model_device)
            if hasattr(v, "to")
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

    out_text = clean_decoded_text(
        out_text
    )

    parsed = extract_json_obj(
        out_text
    )

    return (
        out_text,
        parsed
    )


def parse_caption_output(
    raw_text: str,
    parsed_obj: Dict[str, Any]
) -> Tuple[str, List[str]]:

    if isinstance(
        parsed_obj,
        dict
    ):

        caption = normalize_ws(
            parsed_obj.get(
                "image_caption",
                ""
            )
        )

        people = parsed_obj.get(
            "recognized_people",
            []
        )

        if not isinstance(
            people,
            list
        ):
            people = []

        people = [
            normalize_ws(x)
            for x in people
            if normalize_ws(x)
        ]

        people = dedup_texts(
            people
        )

        return (
            caption,
            people
        )

    raw = normalize_ws(
        raw_text
    )

    m = re.search(
        r'"image_caption"\s*:\s*"([^"]+)"',
        raw
    )

    caption = (
        normalize_ws(
            m.group(1)
        )
        if m
        else ""
    )

    return (
        caption,
        []
    )


# =========================================================
# CAPTION GENERATION
# =========================================================
def generate_captions(
    model,
    processor,
    model_name: str,
    meme_path: Path,
    ris_path: Path,
    output_path: Path,
    meme_id: Optional[str],
) -> None:

    raw_memes = load_json(
        str(meme_path),
        []
    )

    memes: Dict[
        str,
        Dict[str, Any]
    ] = normalize_memes(
        raw_memes
    )

    memes = select_memes(
        memes=memes,
        meme_id=meme_id,
    )

    caption_ris_summaries: Dict[
        str,
        Dict[str, Any]
    ] = load_json(
        str(ris_path),
        {}
    )

    output: Dict[
        str,
        Dict[str, Any]
    ] = load_json(
        str(output_path),
        {}
    )

    if not isinstance(
        caption_ris_summaries,
        dict
    ):
        caption_ris_summaries = {}

    if not isinstance(
        output,
        dict
    ):
        output = {}

    print(
        "Meme entries loaded:",
        len(memes)
    )

    print(
        "Caption-friendly RIS summaries loaded:",
        len(caption_ris_summaries)
    )

    print(
        "Existing output:",
        len(output)
    )

    # =========================================================
    # MAIN LOOP
    # =========================================================
    for i, (
        meme_id_value,
        rec
    ) in enumerate(
        memes.items(),
        start=1
    ):

        existing_rec = output.get(
            meme_id_value,
            {}
        )

        if should_skip_existing(
            existing_rec
        ):

            print(
                f"[{i}/{len(memes)}] "
                f"{meme_id_value}: "
                "already done, skip"
            )

            continue

        img_name = normalize_ws(
            rec.get(
                "img_path",
                ""
            )
        )

        meme_text = normalize_ws(
            rec.get(
                "text",
                ""
            )
        )

        image_path = resolve_image_path(
            img_name
        )

        if DEBUG:

            print(
                f"\n[{meme_id_value}] "
                f"img_name = {img_name}"
            )

            print(
                f"[{meme_id_value}] "
                "candidate paths:"
            )

            for p in candidate_image_paths(
                img_name
            ):

                print(
                    "   ",
                    p,
                    "| exists =",
                    os.path.exists(p)
                )

            print(
                f"[{meme_id_value}] "
                f"resolved image_path = {image_path}"
            )

        caption_ris_summary = (
            caption_ris_summaries.get(
                meme_id_value,
                {
                    "img": img_name,
                    "ris_quality": "low",
                    "caption_ris_anchors": [],
                    "caption_ris_hints": [],
                }
            )
        )

        ris_block = build_caption_ris_block(
            caption_ris_summary
        )

        generated_caption = ""
        recognized_people: List[str] = []
        raw_caption_output = ""

        if (
            image_path
            and os.path.exists(
                image_path
            )
        ):

            try:

                image = (
                    Image.open(
                        image_path
                    )
                    .convert(
                        "RGB"
                    )
                )

                prompt = CAPTION_PROMPT.format(
                    text=meme_text,
                    ris_summary=ris_block
                )

                if model_name == "gemma":

                    (
                        raw_caption_output,
                        parsed_caption,
                    ) = gemma_caption_generate(
                        model=model,
                        processor=processor,
                        image=image,
                        prompt_text=prompt,
                        max_new_tokens=MAX_NEW_TOKENS_CAPTION
                    )

                else:

                    (
                        raw_caption_output,
                        parsed_caption,
                    ) = qwen_caption_generate(
                        model=model,
                        processor=processor,
                        image=image,
                        prompt_text=prompt,
                        max_new_tokens=MAX_NEW_TOKENS_CAPTION
                    )

                (
                    generated_caption,
                    recognized_people,
                ) = parse_caption_output(
                    raw_caption_output,
                    parsed_caption
                )

            except Exception as e:

                raw_caption_output = (
                    "Caption generation failed: "
                    f"{repr(e)}"
                )

                generated_caption = ""
                recognized_people = []

        else:

            raw_caption_output = (
                "Caption generation skipped: "
                "image not found at "
                f"{image_path}"
            )

            generated_caption = ""
            recognized_people = []

        output[
            meme_id_value
        ] = {
            "img": img_name,
            "image_path": image_path,
            "text": meme_text,
            "caption_ris_quality": (
                caption_ris_summary.get(
                    "ris_quality",
                    "low"
                )
            ),
            "caption_ris_anchors_used": (
                caption_ris_summary.get(
                    "caption_ris_anchors",
                    []
                )
            ),
            "caption_ris_hints_used": (
                caption_ris_summary.get(
                    "caption_ris_hints",
                    []
                )
            ),
            "generated_caption": generated_caption,
            "recognized_people": recognized_people,
            "raw_caption_output": raw_caption_output,
        }

        save_json(
            str(output_path),
            output
        )

        print(
            f"[{i}/{len(memes)}] "
            f"{meme_id_value}: saved "
            f"(caption="
            f"{'yes' if generated_caption else 'no'}, "
            f"people={len(recognized_people)})"
        )

        if DEBUG:

            print(
                "RIS block:"
            )

            print(
                ris_block
            )

            print(
                "Caption:",
                generated_caption
            )

            print(
                "People:",
                recognized_people
            )

            print(
                "Raw output:",
                raw_caption_output
            )

            print(
                "-" * 80
            )

    print(
        "Done. Saved to:",
        output_path
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate image captions for QRC "
            "using the original Gemma or Qwen implementation."
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
            "Caption model: gemma or qwen."
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
        "--meme-json",
        type=Path,
        default=None,
        help=(
            "Path to *_qg.json."
        ),
    )

    parser.add_argument(
        "--ris-json",
        type=Path,
        default=None,
        help=(
            "Path to ris_caps.json."
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
            "Caption output JSON path."
        ),
    )

    return parser.parse_args()


def main() -> None:

    global SUBSET_ROOT

    args = parse_args()

    try:

        (
            meme_path,
            ris_path,
            image_dir,
            output_path,
        ) = resolve_paths(
            dataset=args.dataset,
            model_name=args.model,
            meme_id=args.meme_id,
            meme_path=args.meme_json,
            ris_path=args.ris_json,
            image_dir=args.image_dir,
            output_path=args.output_json,
        )

    except ValueError as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not meme_path.exists():

        print(
            f"ERROR: Meme JSON not found: "
            f"{meme_path}",
            file=sys.stderr,
        )

        sys.exit(1)

    if not ris_path.exists():

        print(
            f"ERROR: Caption RIS JSON not found: "
            f"{ris_path}",
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
        "Query Stage - Generate Captions"
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
        f"Meme JSON   : {meme_path}"
    )

    print(
        f"Image dir   : {image_dir}"
    )

    print(
        f"Caption RIS : {ris_path}"
    )

    print(
        f"Output JSON : {output_path}"
    )

    try:

        model, processor = load_model(
            args.model
        )

        generate_captions(
            model=model,
            processor=processor,
            model_name=args.model,
            meme_path=meme_path,
            ris_path=ris_path,
            output_path=output_path,
            meme_id=args.meme_id,
        )

    except Exception as exc:

        print(
            f"ERROR: Caption generation failed: "
            f"{exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()