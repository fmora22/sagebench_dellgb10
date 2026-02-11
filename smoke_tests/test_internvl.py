#!/usr/bin/env python3
import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# keep consistent with your other scripts
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

MODEL_ID = "OpenGVLab/InternVL2-2B"

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
OUTPUT_JSON = OUTPUT_DIR / "internvl2_2b_caption.json"

# imagenet normalization (from internvl examples)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def load_image(image_file: Path, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, len(tiles)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    max_num_tiles = int(os.environ.get("MAX_NUM_TILES", "12"))

    # optional: pin remote code revision
    REVISION = os.environ.get("HF_REVISION", None)

    print(f"loading {MODEL_ID} on {device} (dtype={dtype})")
    if REVISION:
        print(f"using hf revision: {REVISION}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
        revision=REVISION,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        revision=REVISION,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()

    if device == "cuda":
        model = model.cuda()

    pixel_values, n_tiles = load_image(IMAGE_PATH, max_num=max_num_tiles)
    pixel_values = pixel_values.to(dtype)
    if device == "cuda":
        pixel_values = pixel_values.cuda()

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    question = "<image>\nPlease describe the image in detail."
    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    print("\n=== INTERNVL2-2B OUTPUT ===")
    print(response)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "image_path": str(IMAGE_PATH),
                "device": device,
                "dtype": str(dtype),
                "question": question,
                "caption": response,
                "max_new_tokens": max_new_tokens,
                "max_num_tiles": max_num_tiles,
                "num_tiles_used": n_tiles,
            },
            f,
            indent=2,
        )

    print(f"\nsaved caption json to: {OUTPUT_JSON}")
    print("=== END ===")

if __name__ == "__main__":
    main()
