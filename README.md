# vramio

**Know your VRAM before you run.**

A dead-simple API to estimate GPU memory requirements for any HuggingFace model.

[![Live Demo](https://img.shields.io/badge/demo-vramio.ksingh.in-blue)](https://vramio.ksingh.in)

---

## The Problem

You found a cool model on HuggingFace. Now what?

- "Will it fit on my 24GB GPU?"
- "What quantization do I need?"
- "How much VRAM for inference?"

**The answers are buried** — scattered across model cards, config files, or simply missing. You either dig through safetensors metadata yourself, or download the model and pray.

## The Solution

One API call. Instant answer.

```bash
curl "https://vramio.ksingh.in/model?hf_id=meta-llama/Llama-2-7b"
```

```json
{
  "model": "meta-llama/Llama-2-7b",
  "total_parameters": "6.74B",
  "memory_required": "12.55 GB",
  "current_dtype": "F16",
  "recommended_vram": "15.06 GB",
  "other_precisions": {
    "fp32": "25.10 GB",
    "fp16": "12.55 GB",
    "int8": "6.27 GB",
    "int4": "3.14 GB"
  },
  "overhead_note": "Includes 20% for activations/KV cache (2K context)"
}
```

**`recommended_vram`** = what you actually need (includes 20% overhead for inference).

## How It Works

1. Fetches safetensors metadata from HuggingFace (just headers, not weights)
2. Parses tensor shapes and dtypes
3. Calculates memory for each precision
4. Adds 20% overhead for activations + KV cache

No model downloads. No GPU required. Just math.

## Self-Host

```bash
# Clone and run
git clone https://github.com/ksingh-scogo/vramio.git
cd vramio
pip install httpx[http2]
python server_embedded.py
```

Or deploy free on [Render](https://render.com) using the included `render.yaml`.

## Tech Stack

- **160 lines** of Python
- **Zero frameworks** — just stdlib `http.server` + `httpx`
- **1 dependency** — `httpx[http2]`

## Credits

Built on memory estimation logic from [hf-mem](https://github.com/alvarobartt/hf-mem) by [@alvarobartt](https://github.com/alvarobartt).

## License

MIT
