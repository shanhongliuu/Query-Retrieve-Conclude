
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from prompt_templates import zeroshot_prompt


DEFAULT_SAMPLE_IMAGE = str(
    Path(__file__).resolve().parents[1] / "data" / "kym" / "images" / "1.jpg"
)

def load_qwen3_model(
    MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct",  #others: "Qwen/Qwen3-VL-8B-Instruct",
):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        dtype="auto",
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Loaded:", MODEL_NAME)
    print("Model device:", next(model.parameters()).device)
    
    return model, processor

def qwen3_zeroshot(
    model, processor,
    prompt_text = zeroshot_prompt,
    image_path = DEFAULT_SAMPLE_IMAGE,
    max_new_tokens=128
):
    """
    Run the frozen Qwen3-VL model to generate background given the prompt and image.

    Args:
        model: Loaded Qwen3VLForConditionalGeneration
        processor: Loaded AutoProcessor
        max_new_tokens: Generation length

    Returns:
        background statement
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Prepare inputs using Qwen chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # ✅ disable sampling
            eos_token_id=processor.tokenizer.eos_token_id
        )

    # Trim prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def load_gemma3_model(MODEL_NAME="google/gemma-3-27b-it"):
    # Using bfloat16 for the RTX 4090 to stay within 24GB VRAM
    # Note: 27B at bfloat16 takes ~54GB. For a 24GB card, you MUST use 4-bit quantization.
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config, # Critical for 24GB VRAM
        device_map="auto"
    )
    
    print("Loaded:", MODEL_NAME)
    print("Model device:", next(model.parameters()).device)
    return model, processor

import torch
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

def load_gemma3_model(model_id="google/gemma-3-27b-it"):
    # Using bfloat16 for the RTX 4090 to stay within 24GB VRAM
    # Note: 27B at bfloat16 takes ~54GB. For a 24GB card, you MUST use 4-bit quantization.
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config, # Critical for 24GB VRAM
        device_map="auto"
    )
    return model, processor

def gemma3_zeroshot(model, processor, prompt_text, image_path, max_new_tokens=128):
    image = Image.open(image_path).convert("RGB")
    
    # Gemma 3 expects a specific message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    
    # Slice to remove input tokens
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()



def generate_with_zeroshot(selected_model, model, processor, prompt_text, image_path, max_new_tokens=128):
    if selected_model == "QWEN3-32B":
        return qwen3_zeroshot(model, processor, prompt_text, image_path, max_new_tokens)
    elif selected_model == "GEMMA3-27B":
        return gemma3_zeroshot(model, processor, prompt_text, image_path, max_new_tokens)
    else:
        raise ValueError(f"Unsupported model: {selected_model}")
