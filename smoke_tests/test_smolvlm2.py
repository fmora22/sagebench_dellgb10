#!/usr/bin/env python3
# smolvlm2 image caption test (instruct)
# usage:
#   python3 test_smolvlm2.py
# env overrides:
#   IMAGE_PATH=images/image2.jpg
#   BENCH_OUTPUT_DIR=outputs

import os
import json
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText

# keep consistent with your other scripts
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

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
OUTPUT_JSON = OUTPUT_DIR / "smolvlm2_caption.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# bf16 is usually best on modern nvidia gpus; fall back cleanly on cpu
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"loading {MODEL_ID} on {device} (dtype={dtype})")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    _attn_implementation="sdpa" if device == "cuda" else "eager",
).to(device)

image = Image.open(IMAGE_PATH).convert("RGB")
question = "Describe this image in detail."

# smolvlm2 uses a chat template with image placeholders
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n=== SMOLVLM2 CAPTION ===")
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
