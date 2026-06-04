# elabs / Wan 2.6 Text-to-Image

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy-orange?logo=runpod)](https://console.runpod.io/hub)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

**Text-to-image generation** with Wan 2.6 -- a 2.6B parameter transformer model. High-quality, detail-rich images in multiple aspect ratios up to 2048x2048.

![Wan T2I](https://pub-796a08821c1c483aaf5e274e0d03e350.r2.dev/hub-icons/wan-t2i.svg)

## Highlights

- 2.6B parameters -- high-quality, detail-rich generation
- Native 2048x2048 -- full resolution without upscaling
- Fast draft mode -- 20 steps for rapid iteration
- Multiple aspect ratios -- 1:1, 16:9, 9:16, 4:3
- Weights baked in -- no network volume, instant cold start

## Quick Start

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A sunset over misty mountains, oil painting style"}}'
```

## API

### Input

```json
{
  "input": {
    "prompt": "A sunset over misty mountains, oil painting style",
    "negative_prompt": "blurry, low quality, watermark",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Output

```json
{
  "image_b64": "<base64 PNG>",
  "prompt": "A sunset over misty mountains, oil painting style",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "wall_time_s": 4.5
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text description of the image |
| `negative_prompt` | string | `""` | Elements to exclude |
| `width` | int | `1024` | Width (256-2048, multiple of 64) |
| `height` | int | `1024` | Height (256-2048, multiple of 64) |
| `num_inference_steps` | int | `20` | Steps (20=fast, 50=quality) |
| `guidance_scale` | float | `7.5` | Prompt adherence (1.0-20.0) |
| `seed` | int | random | Seed (-1 = random) |

## GPU Requirements

- Minimum: >=8GB VRAM
- Recommended: RTX 4090, L40S, A5000 (>=16GB)
- CUDA: 12.4+

## Benchmarks

| GPU | 1024x1024 @20 steps | 1024x1024 @50 steps |
|---|---|---|
| RTX 4090 | ~3s | ~7s |
| L40S | ~4s | ~10s |
| A5000 | ~6s | ~15s |

## License

Apache-2.0. Based on [Wan-AI/Wan2.1](https://huggingface.co/Wan-AI).
