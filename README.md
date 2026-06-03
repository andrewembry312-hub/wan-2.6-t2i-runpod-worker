# elabs / Wan 2.6 T2I — Text-to-Image Generation

[![Run on RunPod](https://runpod.io/badge/runpod-hub)](https://runpod.io/console/hub)

High-quality **text-to-image** generation powered by **Wan 2.6**. Advanced prompt understanding, multiple aspect ratios, style control, and fast generation. Weights baked into the Docker image — no network volume or cold-download delays.

## Highlights

- **2.6B parameters** — excellent prompt understanding and image quality
- **Multi-aspect ratio support**: square, portrait, landscape (up to 2048px)
- **Style control** via prompt engineering and guidance scale tuning
- **Seed control** for reproducible generations
- **Weights baked in** — no cold-download delays
- **8GB+ VRAM required**
- **Apache-2.0** licensed

## API

### Input

```json
{
  "input": {
    "prompt": "a serene mountain landscape at sunset, photorealistic",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": -1
  }
}
```

### Output

```json
{
  "image_b64": "<base64 PNG>",
  "prompt": "a serene mountain landscape at sunset, photorealistic",
  "seed": 374969113,
  "wall_time_s": 3.2
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Text prompt |
| `negative_prompt` | string | `""` | Negative prompt |
| `width` | int | `1024` | Output width (512–2048, multiples of 64) |
| `height` | int | `1024` | Output height (512–2048, multiples of 64) |
| `num_inference_steps` | int | `20` | Denoising steps (10 = ultra-fast, 50 = quality) |
| `guidance_scale` | float | `5.0` | CFG guidance scale (1.0–20.0) |
| `seed` | int | `null` | Fixed seed for reproducibility (`-1` or `null` = random) |

## GPU Requirements

- **Recommended**: RTX 4090 / RTX 6000 Ada / L40S / A5000
- **Minimum**: Any GPU with ≥8GB VRAM (RTX 3080, A5000, L4, etc.)
- **CUDA**: 12.0+

## Benchmark

| GPU | Steps | Resolution | Time |
|---|---|---|---|
| RTX 4090 | 20 | 1024×1024 | ~3.2s |
| RTX 4090 | 50 | 1024×1024 | ~7.5s |
| RTX A5000 | 20 | 1024×1024 | ~5.0s |
| L40S | 50 | 1024×1024 | ~6.0s |

## License

Apache-2.0
