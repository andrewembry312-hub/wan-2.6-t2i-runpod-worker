"""
RunPod serverless handler for Wan 2.6 T2I — text prompt → image generation.

Architecture:
  - Wan 2.6 transformer backbone with advanced prompt understanding
  - 2.6B parameters, ~5GB weights in fp16
  - Supports up to 2048×2048 resolution
  - Runs on any GPU with >=8GB VRAM
  - Diffusers-native pipeline (from_pretrained)

Environment (set by RunPod template):
  - RUNPOD_POD_ID       — auto
  - RUNPOD_AI_API_KEY   — auto

Input schema (via RunPod serverless job):
  {
    "input": {
      "prompt": "a serene mountain landscape",  // REQUIRED — text prompt
      "negative_prompt": "",                     // optional — negative prompt
      "height": 1024,                            // optional — image height (512-2048)
      "width": 1024,                             // optional — image width (512-2048)
      "num_inference_steps": 20,                 // optional — fewer = faster (10-step possible)
      "guidance_scale": 5.0,                     // optional — CFG scale
      "seed": null                               // optional — random seed (null = random)
    }
  }

Output:
  {
    "image_b64": "<base64-encoded PNG>",
    "prompt": "a serene mountain landscape",
    "seed": 42,
    "wall_time_s": 3.2
  }
"""

import base64
import os
import random
import time
import traceback
from io import BytesIO

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

# Disable flash/mem-efficient SDPA — avoids "no kernel image" CUDA errors
# that occur when the GPU compute capability isn't in the precompiled kernels.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Model path (baked into image at BUILD TIME) ──────────────────────────────
MODEL_ID = "/models/wan26-t2i"

# ── Global pipeline (loaded once, reused across jobs) ─────────────────────────
_pipe = None
_device = None


def load_pipeline():
    """Load Wan 2.6 T2I pipeline once and cache globally."""
    global _pipe, _device
    if _pipe is not None:
        return _pipe, _device

    print("[Cold Start] Loading Wan 2.6 T2I pipeline...", flush=True)
    t0 = time.time()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"  Device: {_device}, dtype: {dtype}", flush=True)

    # Import Wan 2.6 pipeline from diffusers or custom package
    # The actual import path depends on how the model is packaged
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    # Move fully to GPU — cpu_offload causes issues in serverless containers
    pipe = pipe.to(_device)

    print(f"[Cold Start] Pipeline ready in {time.time() - t0:.1f}s", flush=True)

    _pipe = pipe
    return _pipe, _device


def image_to_b64(image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_inference(
    prompt: str,
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int | None = None,
) -> tuple:
    """
    Run Wan 2.6 inference.
    Returns (PIL Image, actual_seed, wall_time_s).
    """
    pipe, device = load_pipeline()

    # Set seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[Inference] Generating: prompt='{prompt[:80]}'", flush=True)
    print(
        f"  size={width}x{height}, steps={num_inference_steps}, "
        f"cfg={guidance_scale}, seed={seed}",
        flush=True,
    )

    t_start = time.time()

    # Run inference — use autocast for mixed precision
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
            ).images

    wall_time = time.time() - t_start
    print(f"[Done] Generation took {wall_time:.1f}s", flush=True)

    return images[0], seed, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ═══════════════════════════════════════════════════════════════════════════════


def handler(job):
    """
    RunPod serverless handler: text prompt → base64 PNG.

    Called once per job. The pipeline stays loaded across jobs (global).
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")

    if not prompt:
        return {"error": "Missing required field: prompt"}

    negative_prompt = str(job_input.get("negative_prompt", ""))
    height = int(job_input.get("height", 1024))
    width = int(job_input.get("width", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 20))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))
    seed_raw = job_input.get("seed", None)
    seed = int(seed_raw) if seed_raw is not None else None

    # Validate dimensions (512-2048 in 64px multiples)
    height = max(512, min(2048, height // 64 * 64))
    width = max(512, min(2048, width // 64 * 64))

    # Validate steps
    num_inference_steps = max(1, min(100, num_inference_steps))

    # Validate guidance scale
    guidance_scale = max(1.0, min(20.0, guidance_scale))

    try:
        # Run inference
        image, actual_seed, wall_time = run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Encode as base64 PNG
        print("[Worker] Encoding image as base64 PNG...", flush=True)
        image_b64 = image_to_b64(image)

        return {
            "image_b64": image_b64,
            "prompt": prompt,
            "seed": actual_seed,
            "wall_time_s": round(wall_time, 1),
            "width": width,
            "height": height,
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": f"Wan 2.6 T2I inference failed: {str(exc)}",
            "traceback": traceback.format_exc(),
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
