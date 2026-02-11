#!/usr/bin/env python3
# official vlm benchmark (week 5 baseline)
# - manifest-driven image set
# - standardized prompt + max_new_tokens
# - fp16 baseline by default (override with --dtype bf16/fp16)
# - warmup + correct cuda timing
# - per-run jsonl + per-model summary json
#
# usage:
#   python3 vision_benchmark.py moondream2 smolvlm2 --manifest data/testsets/coco_val2017_100.txt

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch
from PIL import Image

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# -------------------------
# general utils
# -------------------------

def read_manifest(path: Path) -> list[str]:
    paths = []
    with open(path, "r") as f:
        for line in f:
            p = line.strip()
            if p:
                paths.append(p)
    return paths

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")

def system_snapshot() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0),
        "ram_used_mb": round(mem.used / (1024**2), 1),
        "ram_available_mb": round(mem.available / (1024**2), 1),
        "ram_percent": mem.percent,
    }

def reset_cuda_peaks():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def cuda_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_mem_alloc_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
        "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
        "gpu_max_mem_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "gpu_max_mem_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
    }

def timed_call(fn, device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0

def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]

# -------------------------
# model adapters (from your smoke tests)
# each model implements:
# - load_<model>(device, dtype) -> handle dict
# - generate_<model>(handle, image, prompt, max_new_tokens, device) -> text
# -------------------------

def load_moondream2(device: str, dtype: torch.dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    return {"model_id": model_id, "model": model, "tokenizer": tokenizer}

def generate_moondream2(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    # moondream uses its own api: encode_image + answer_question
    model = handle["model"]
    tokenizer = handle["tokenizer"]
    with torch.inference_mode():
        image_emb = model.encode_image(image)
        text = model.answer_question(image_emb, prompt, tokenizer)
    return str(text).strip()

def load_gemma3n(device: str, dtype: torch.dtype):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "google/gemma-3n-E4B-it"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    return {"model_id": model_id, "model": model, "processor": processor}

def generate_gemma3n(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    processor = handle["processor"]
    model = handle["model"]

    tokenizer = processor.tokenizer
    bos = tokenizer.bos_token or ""
    full_prompt = (
        f"{bos}"
        "<start_of_turn>user\n"
        f"<image_soft_token>{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    inputs = processor(text=full_prompt, images=[image], return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0, input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

def load_smolvlm2(device: str, dtype: torch.dtype):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        _attn_implementation="sdpa" if device == "cuda" else "eager",
    ).to(device)
    return {"model_id": model_id, "model": model, "processor": processor}

def generate_smolvlm2(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    processor = handle["processor"]
    model = handle["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # note: this includes the full conversation in some configs; keep consistent across runs
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return str(text).strip()

def load_llava_mini(device: str, dtype: torch.dtype):
    # llava-mini uses the llavamini package, not hf transformers directly
    from llavamini.model.builder import load_pretrained_model
    from llavamini.mm_utils import get_model_name_from_path

    model_path = "ICTNLP/llava-mini-llama-3.1-8b"
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    return {
        "model_id": model_path,
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
    }

def generate_llava_mini(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    from llavamini.mm_utils import process_images, tokenizer_image_token
    from llavamini.constants import IMAGE_TOKEN_INDEX

    model = handle["model"]
    tokenizer = handle["tokenizer"]
    image_processor = handle["image_processor"]

    # llava expects <image> token + question
    llava_prompt = f"<image>\n{prompt}"

    input_ids = tokenizer_image_token(
        llava_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)

    image_tensor = process_images([image], image_processor, model.config).unsqueeze(1).to(device, dtype=model.dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return str(text)

def load_internvl2_2b(device: str, dtype: torch.dtype):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    model_id = "OpenGVLab/InternVL2-2B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()

    if device == "cuda":
        model = model.cuda()

    # store preprocess config + transform in handle
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    input_size = 448
    max_num_tiles = 12

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return {
        "model_id": model_id,
        "model": model,
        "tokenizer": tokenizer,
        "transform": transform,
        "input_size": input_size,
        "max_num_tiles": max_num_tiles,
    }

def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
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

    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
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

def generate_internvl2_2b(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    model = handle["model"]
    tokenizer = handle["tokenizer"]
    transform = handle["transform"]
    input_size = handle["input_size"]
    max_num_tiles = handle["max_num_tiles"]

    tiles = _internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num_tiles,
    )
    pixel_values = torch.stack([transform(t) for t in tiles]).to(dtype=model.dtype)
    if device == "cuda":
        pixel_values = pixel_values.cuda()

    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}

    # internvl expects <image> tag in the question
    question = "<image>\n" + prompt
    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, question, generation_config)
    return str(response).strip()

def load_phi3_5_vision(device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        _attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=16,
    )
    return {"model_id": model_id, "model": model, "processor": processor}

def generate_phi3_5_vision(handle, image, prompt: str, max_new_tokens: int, device: str) -> str:
    model = handle["model"]
    processor = handle["processor"]

    messages = [{"role": "user", "content": "<|image_1|>\n" + prompt}]
    chat = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(chat, [image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_args = {"max_new_tokens": max_new_tokens, "temperature": 0.0, "do_sample": False}

    with torch.inference_mode():
        out = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **gen_args)

    out = out[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return str(text).strip()

# model key -> (load_fn, gen_fn)
MODEL_REGISTRY = {
    "moondream2": (load_moondream2, generate_moondream2),
    "smolvlm2": (load_smolvlm2, generate_smolvlm2),
    "llava-mini": (load_llava_mini, generate_llava_mini),
    "gemma3n": (load_gemma3n, generate_gemma3n),
    "internvl2-2b": (load_internvl2_2b, generate_internvl2_2b),
    "phi-3.5-vision": (load_phi3_5_vision, generate_phi3_5_vision),
}

# -------------------------
# runner
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help=f"valid: {list(MODEL_REGISTRY.keys())}")
    p.add_argument("--manifest", required=True, help="text file with one image path per line")
    p.add_argument("--output_dir", default="outputs", help="base output dir")
    p.add_argument("--prompt", default="Write one detailed sentence describing the image.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    return p.parse_args()

def resolve_dtype(device: str, dtype_flag: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    if dtype_flag == "fp16":
        return torch.float16
    if dtype_flag == "bf16":
        return torch.bfloat16
    return torch.float32

def run_model(model_key: str, image_paths: list[str], args, device: str, dtype: torch.dtype, host: str, run_id: str):
    if model_key not in MODEL_REGISTRY:
        print(f"skipping unknown model: {model_key}")
        return

    load_fn, gen_fn = MODEL_REGISTRY[model_key]

    out_dir = Path(args.output_dir).resolve() / host / model_key / run_id
    ensure_dir(out_dir)

    handle = load_fn(device=device, dtype=dtype)

    meta = {
        "run_id": run_id,
        "host": host,
        "device": device,
        "dtype": str(dtype),
        "model_key": model_key,
        "model_id": handle.get("model_id"),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "num_images": len(image_paths),
        "manifest": args.manifest,
        "torch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # warmup
    if args.warmup > 0:
        try:
            img0 = load_image(image_paths[0])
            for _ in range(args.warmup):
                _ = gen_fn(handle, img0, args.prompt, args.max_new_tokens, device)
        except Exception as e:
            print(f"warmup failed for {model_key}: {e}")

    latencies = []
    errors = 0

    runs_path = out_dir / "runs.jsonl"
    with open(runs_path, "w") as f:
        for i, image_path in enumerate(image_paths):
            img = load_image(image_path)
            before_sys = system_snapshot()
            reset_cuda_peaks()

            def call():
                try:
                    txt = gen_fn(handle, img, args.prompt, args.max_new_tokens, device)
                    return txt, None
                except Exception as e:
                    return "", str(e)

            (txt, err), ms = timed_call(call, device=device)

            after_sys = system_snapshot()
            cuda_stats = cuda_snapshot()

            latencies.append(ms)
            if err is not None:
                errors += 1

            rec = {
                "image_index": i,
                "image_name": Path(image_path).name,
                "image_path": image_path,
                "latency_ms": round(ms, 3),
                "output_text": txt,
                "error": err,
                "sys_before": before_sys,
                "sys_after": after_sys,
                "cuda_stats": cuda_stats,
                "timestamp": datetime.now().isoformat(),
            }
            f.write(json.dumps(rec) + "\n")

            status = "ok" if err is None else "error"
            print(f"[{model_key}] {i+1}/{len(image_paths)} {status} {ms:.1f} ms")

    lat_sorted = sorted(latencies)
    summary = {
        "model_key": model_key,
        "model_id": handle.get("model_id"),
        "num_runs": len(latencies),
        "num_errors": errors,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "latency_ms_min": round(min(latencies), 3) if latencies else None,
        "latency_ms_max": round(max(latencies), 3) if latencies else None,
        "latency_ms_p50": round(percentile(lat_sorted, 50), 3) if lat_sorted else None,
        "latency_ms_p90": round(percentile(lat_sorted, 90), 3) if lat_sorted else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    del handle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"saved: {out_dir}")

def main():
    args = parse_args()

    image_paths = read_manifest(Path(args.manifest))
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("no images in manifest")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(device, args.dtype)

    host = platform.node() or "unknown_host"
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for model_key in args.models:
        run_model(model_key, image_paths, args, device, dtype, host, run_id)

if __name__ == "__main__":
    main()
