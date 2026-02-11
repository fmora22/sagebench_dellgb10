#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

# repo root: Benchmarking/
BASE_DIR = Path(__file__).resolve().parent.parent

# image path: env override OR repo-relative default
image_path_str = os.environ.get("IMAGE_PATH", "images/image2.jpg")
IMAGE_PATH = Path(image_path_str)
if not IMAGE_PATH.is_absolute():
    IMAGE_PATH = (BASE_DIR / IMAGE_PATH).resolve()

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

# outputs: env override OR repo-relative outputs/
output_dir_str = os.environ.get("BENCH_OUTPUT_DIR", "outputs")
OUTPUT_DIR = Path(output_dir_str)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (BASE_DIR / OUTPUT_DIR).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "phi35_vision_caption.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# bf16 is usually fine; if you hit issues, switch to float16
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {device} (dtype={dtype})")

# note: set eager unless you installed flash_attn
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    _attn_implementation="eager",
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    num_crops=16,  # per model card suggestion for single image
)

image = Image.open(IMAGE_PATH).convert("RGB")
question = "Describe this image in detail."

# model-card format:
# <|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n
messages = [
    {"role": "user", "content": "<|image_1|>\n" + question},
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(prompt, [image], return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

generation_args = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "do_sample": False,
}

with torch.no_grad():
    out = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args,
    )

# remove input tokens (matches model card)
out = out[:, inputs["input_ids"].shape[1]:]
caption = processor.batch_decode(
    out,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\n=== PHI-3.5 VISION CAPTION ===")
print(caption)

result = {
    "model_id": MODEL_ID,
    "image_path": str(IMAGE_PATH),
    "device": device,
    "dtype": str(dtype),
    "question": question,
    "caption": caption,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nsaved caption json to: {OUTPUT_JSON}")
print("=== END ===")
