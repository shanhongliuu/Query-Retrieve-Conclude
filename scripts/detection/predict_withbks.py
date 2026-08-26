import os
import re
import json
import argparse
from typing import Any, Dict

import torch
from PIL import Image
from tqdm import tqdm


TASK_DEFINITIONS = {
    "hatefulness": (
        "A meme is hateful if it attacks, degrades, humiliates, or expresses hostility "
        "toward a person or group based on protected characteristics such as race, "
        "religion, gender, nationality, or disability."
    ),
    "misogyny": (
        "A meme is misogynistic if it expresses sexism or hatred toward women through "
        "shaming, stereotyping, objectification, or violence."
    ),
    "offensiveness": (
        "A meme is offensive if it contains insulting, rude, or disrespectful content "
        "that may harm the social identity or dignity of individuals or groups."
    ),
    "sarcasm": (
        "A meme is sarcastic if it conveys meaning through irony, exaggeration, or "
        "incongruity between literal and intended meaning, often arising from mismatched "
        "image-text semantics."
    ),
    "harmfulness": (
        "A meme is harmful if it has the potential to negatively impact individuals, "
        "communities, or society, including subtle or implicit harmful implications "
        "beyond explicit abuse."
    ),
    "humor": (
        "A meme is humorous if it presents humor, sarcasm, or satire. The humor label "
        "should be determined based on whether the image creator intended the meme to "
        "be humorous, rather than whether the annotator personally found it funny."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict meme labels using generated background knowledge statements."
    )

    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        choices=["qwen_vl", "llava", "gemma"],
        help="Model family to use."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local or HuggingFace model path."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name, e.g., memeinterpret, mami, multioff, msd, harm, pridemm."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=list(TASK_DEFINITIONS.keys()),
        help="Detection task name."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to original input JSON containing img/text. No label is required."
    )
    parser.add_argument(
        "--statement_json",
        type=str,
        default=None,
        help=(
            "Path to generated background knowledge statement JSON. "
            "If omitted, defaults to outputs/{dataset_name}/stat/statements_v2_withBM_cap_qa.json"
        )
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root folder containing meme images."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save prediction JSON."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum new tokens for binary prediction."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu. If omitted, auto-detect."
    )
    parser.add_argument(
        "--use_small_subset",
        action="store_true",
        help="Use only the first N examples."
    )
    parser.add_argument(
        "--small_subset_size",
        type=int,
        default=10,
        help="Subset size when --use_small_subset is enabled."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling during generation."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions instead of resuming."
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default="34GiB",
        help=(
            "Maximum memory per visible GPU for device_map='auto'. "
            "For 44GiB GPUs, try 34GiB or 32GiB if OOM continues."
        )
    )
    parser.add_argument(
        "--cpu_offload_memory",
        type=str,
        default="180GiB",
        help="Maximum CPU memory available for model offloading."
    )

    return parser.parse_args()


def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        if default is not None:
            return default
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, path)


def normalize_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def normalize_mapping(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}

    if isinstance(data, list):
        return {str(i): rec for i, rec in enumerate(data, start=1)}

    raise ValueError("JSON must be a dict or list.")



def resolve_image_name(rec: Dict[str, Any]) -> str:
    for key in ["img", "image", "img_path", "image_path", "name", "file_name", "filename"]:
        value = normalize_ws(rec.get(key, ""))
        if value:
            return os.path.basename(value)
    return ""


def resolve_image_path(rec: Dict[str, Any], image_root: str) -> str:
    img_name = resolve_image_name(rec)
    return os.path.join(image_root, img_name)


def extract_text(rec: Dict[str, Any]) -> str:
    return normalize_ws(rec.get("text", ""))


def extract_statement_from_rec(rec: Dict[str, Any]) -> str:
    """
    Extract generated background knowledge from a statement record.

    Supported keys:
        pred_statement
        statement
        background_knowledge
        background_knowledge_statement
        bks
    """
    for key in [
        "pred_statement",
        "statement",
        "background_knowledge",
        "background_knowledge_statement",
        "bks",
    ]:
        if key in rec:
            value = rec.get(key)

            if isinstance(value, str):
                return normalize_ws(value)

            if isinstance(value, list):
                return " ".join(normalize_ws(x) for x in value if normalize_ws(x))

    return ""


def build_statement_index(statement_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build index by both entry_id and image filename.

    This supports two matching modes:
    1. same entry_id between input_json and statement_json;
    2. same image filename.
    """
    index = {}

    for entry_id, rec in statement_data.items():
        if not isinstance(rec, dict):
            continue

        index[str(entry_id)] = rec

        img = resolve_image_name(rec)
        if img:
            index[img] = rec

    return index


def attach_statement(
    entry_id: str,
    dataset_rec: Dict[str, Any],
    statement_index: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge generated background knowledge statement into original input record.
    """
    rec = dict(dataset_rec)

    img_name = resolve_image_name(dataset_rec)

    st_rec = None

    if str(entry_id) in statement_index:
        st_rec = statement_index[str(entry_id)]
    elif img_name and img_name in statement_index:
        st_rec = statement_index[img_name]

    if st_rec is None:
        rec["background_knowledge"] = ""
        rec["statement_found"] = False
        return rec

    statement = extract_statement_from_rec(st_rec)

    rec["background_knowledge"] = statement
    rec["statement_found"] = bool(statement)

    if "generated_caption" in st_rec:
        rec["generated_caption"] = st_rec.get("generated_caption", "")

    if "questions" in st_rec:
        rec["statement_questions"] = st_rec.get("questions", [])

    if "answers" in st_rec:
        rec["statement_answers"] = st_rec.get("answers", [])

    return rec


# =========================================================
# PROMPT
# =========================================================
LLAVA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_user_prompt(task_name: str, rec: Dict[str, Any]) -> str:
    task_def = TASK_DEFINITIONS[task_name]

    img_name = resolve_image_name(rec)
    text = extract_text(rec)
    background_knowledge = normalize_ws(rec.get("background_knowledge", ""))

    prompt = (
        "You are given a meme image, its embedded text, and generated background knowledge.\n\n"
        f"The detection task is: {task_name}.\n\n"
        f"Task definition:\n{task_def}\n\n"
        f"### Image: {img_name}\n"
        f"### Embedded text: {text}\n"
        f"### Generated background knowledge:\n{background_knowledge}\n\n"
        f"Question: Based on the meme and the generated background knowledge, "
        f"does this meme satisfy the task '{task_name}'?\n\n"
        "Return exactly one character:\n"
        "1 if yes\n"
        "0 if no\n\n"
        "Do not output any other words."
    )

    return prompt


def build_llava_full_prompt(task_name: str, rec: Dict[str, Any]) -> str:
    return (
        f"{LLAVA_SYSTEM_PROMPT} "
        f"USER: <image>\n{build_user_prompt(task_name, rec)}\n"
        f"ASSISTANT:"
    )


def build_max_memory(max_memory_per_gpu: str, cpu_offload_memory: str) -> Dict[Any, str]:

    num_visible_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory_per_gpu for i in range(num_visible_gpus)}
    max_memory["cpu"] = cpu_offload_memory
    return max_memory


def print_cuda_info():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print("Visible CUDA GPUs:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gib = props.total_memory / (1024 ** 3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gib = free_bytes / (1024 ** 3)
        print(
            f"  cuda:{i}: {props.name}, "
            f"total_memory={total_gib:.2f} GiB, "
            f"free_memory={free_gib:.2f} GiB"
        )


def load_model_and_processor(
    model_family: str,
    model_path: str,
    device: str,
    torch_dtype,
    max_memory_per_gpu: str = "34GiB",
    cpu_offload_memory: str = "180GiB",
):


    if model_family == "llava":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path)

        if device == "cuda" and torch.cuda.device_count() > 1:
            print_cuda_info()
            max_memory = build_max_memory(max_memory_per_gpu, cpu_offload_memory)
            print("MAX_MEMORY:", max_memory)

            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

        model.eval()

        if hasattr(model, "hf_device_map"):
            print("HF_DEVICE_MAP:", model.hf_device_map)

        return model, processor

    if model_family in ["qwen_vl", "gemma"]:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(model_path)

        if device == "cuda":
            if torch.cuda.device_count() < 1:
                raise RuntimeError("device='cuda' was requested, but no CUDA GPU is visible.")

            print_cuda_info()
            max_memory = build_max_memory(max_memory_per_gpu, cpu_offload_memory)
            print("MAX_MEMORY:", max_memory)

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)

        model.eval()

        if hasattr(model, "hf_device_map"):
            print("HF_DEVICE_MAP:", model.hf_device_map)

        return model, processor

    raise ValueError(f"Unsupported model family: {model_family}")


def get_model_input_device(model, fallback_device: str):

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass

    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(fallback_device)


def move_inputs_to_model_device(inputs: Dict[str, Any], model, fallback_device: str) -> Dict[str, Any]:
    model_input_device = get_model_input_device(model, fallback_device)

    return {
        k: v.to(model_input_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }



def clean_decoded_text(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.strip()


    s = re.sub(r"<\|im_start\|>.*?\n", "", s, flags=re.DOTALL)
    s = s.replace("<|im_end|>", "").strip()

    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[-1].strip()

    return s.strip()


def parse_binary_prediction(raw_output: str) -> int:
    if not isinstance(raw_output, str):
        return -1

    text = raw_output.strip().lower()
    text = text.replace("answer:", "").strip()

    if text in {"1", "yes", "true"}:
        return 1

    if text in {"0", "no", "false"}:
        return 0


    m = re.search(r"\b([01])\b", text)
    if m:
        return int(m.group(1))

    return -1



def generate_with_llava(
    model,
    processor,
    image: Image.Image,
    task_name: str,
    rec: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    do_sample: bool
) -> str:
    full_prompt = build_llava_full_prompt(task_name, rec)

    inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = move_inputs_to_model_device(inputs, model, device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    decoded_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return clean_decoded_text(decoded_output)


def generate_with_chat_template(
    model,
    processor,
    image: Image.Image,
    task_name: str,
    rec: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    do_sample: bool
) -> str:
    prompt_text = build_user_prompt(task_name, rec)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
            padding=True
        )

    except Exception:

        inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt"
        )

    inputs = move_inputs_to_model_device(inputs, model, device)

    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else None

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    # Decode only newly generated tokens if possible.
    if input_len is not None:
        gen_ids = output_ids[:, input_len:]
        decoded_output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    else:
        decoded_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return clean_decoded_text(decoded_output)


def generate_one(
    model_family: str,
    model,
    processor,
    image: Image.Image,
    task_name: str,
    rec: Dict[str, Any],
    device: str,
    max_new_tokens: int,
    do_sample: bool
) -> str:
    if model_family == "llava":
        return generate_with_llava(
            model=model,
            processor=processor,
            image=image,
            task_name=task_name,
            rec=rec,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    if model_family in ["qwen_vl", "gemma"]:
        return generate_with_chat_template(
            model=model,
            processor=processor,
            image=image,
            task_name=task_name,
            rec=rec,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    raise ValueError(f"Unsupported model family: {model_family}")


def predict_one(
    entry_id: str,
    rec: Dict[str, Any],
    model_family: str,
    model,
    processor,
    task_name: str,
    image_root: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool
) -> Dict[str, Any]:
    img_name = resolve_image_name(rec)
    text = extract_text(rec)
    image_path = resolve_image_path(rec, image_root)
    background_knowledge = normalize_ws(rec.get("background_knowledge", ""))

    base_out = {
        "entry_id": str(entry_id),
        "img": img_name,
        "text": text,
        "task_name": task_name,
        "background_knowledge": background_knowledge,
        "statement_found": bool(rec.get("statement_found", False)),
    }

    if not img_name:
        return {
            **base_out,
            "pred": -1,
            "raw_output": "ERROR: missing image name"
        }

    if not os.path.exists(image_path):
        return {
            **base_out,
            "pred": -1,
            "raw_output": f"ERROR: image not found at {image_path}"
        }

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            **base_out,
            "pred": -1,
            "raw_output": f"ERROR: cannot open image: {repr(e)}"
        }

    raw_output = generate_one(
        model_family=model_family,
        model=model,
        processor=processor,
        image=image,
        task_name=task_name,
        rec=rec,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample
    )

    pred = parse_binary_prediction(raw_output)

    return {
        **base_out,
        "pred": pred,
        "raw_output": raw_output
    }


def main():
    args = parse_args()

    if args.statement_json is None:
        args.statement_json = os.path.join(
            "outputs",
            args.dataset_name,
            "stat",
            "statements_v2_withBM_cap_qa.json"
        )

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print("MODEL_FAMILY:", args.model_family)
    print("MODEL_PATH:", args.model_path)
    print("DATASET_NAME:", args.dataset_name)
    print("TASK_NAME:", args.task_name)
    print("INPUT_JSON:", args.input_json)
    print("STATEMENT_JSON:", args.statement_json)
    print("IMAGE_ROOT:", args.image_root)
    print("OUTPUT_JSON:", args.output_json)
    print("DEVICE:", device)
    print("TORCH_DTYPE:", torch_dtype)
    print("MAX_NEW_TOKENS:", args.max_new_tokens)
    print("OVERWRITE:", args.overwrite)
    print("MAX_MEMORY_PER_GPU:", args.max_memory_per_gpu)
    print("CPU_OFFLOAD_MEMORY:", args.cpu_offload_memory)

    data_raw = load_json(args.input_json)
    statement_raw = load_json(args.statement_json, default={})

    data = normalize_mapping(data_raw)
    statements = normalize_mapping(statement_raw)
    statement_index = build_statement_index(statements)

    items = list(data.items())

    print("Loaded input entries:", len(items))
    print("Loaded statement entries:", len(statements))

    if args.use_small_subset:
        items = items[:args.small_subset_size]
        print("Using small subset:", len(items))

    model, processor = load_model_and_processor(
        model_family=args.model_family,
        model_path=args.model_path,
        device=device,
        torch_dtype=torch_dtype,
        max_memory_per_gpu=args.max_memory_per_gpu,
        cpu_offload_memory=args.cpu_offload_memory,
    )
    print("Loaded model successfully.")

    predictions: Dict[str, Any] = {}


    if os.path.exists(args.output_json) and not args.overwrite:
        try:
            predictions = normalize_mapping(load_json(args.output_json, default={}))
            print("Loaded existing predictions:", len(predictions))
        except Exception:
            predictions = {}

    for entry_id, rec in tqdm(items, desc=f"Predicting {args.dataset_name}/{args.task_name}"):
        entry_id = str(entry_id)

        if not isinstance(rec, dict):
            continue

        if (
            not args.overwrite
            and entry_id in predictions
            and predictions[entry_id].get("pred", -1) in [0, 1]
        ):
            if args.debug:
                print(f"[SKIP] entry_id={entry_id} already predicted")
            continue

        merged_rec = attach_statement(
            entry_id=entry_id,
            dataset_rec=rec,
            statement_index=statement_index
        )

        out = predict_one(
            entry_id=entry_id,
            rec=merged_rec,
            model_family=args.model_family,
            model=model,
            processor=processor,
            task_name=args.task_name,
            image_root=args.image_root,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample
        )

        predictions[entry_id] = out
        save_json(predictions, args.output_json)

        if args.debug:
            print("\nentry_id:", entry_id)
            print("img:", out["img"])
            print("pred:", out["pred"])
            print("statement_found:", out["statement_found"])
            print("background_knowledge:", out["background_knowledge"][:300])
            print("raw_output:", out["raw_output"][:300])
            print("-" * 100)

    save_json(predictions, args.output_json)
    print("\nSaved prediction file to:", args.output_json)


if __name__ == "__main__":
    main()