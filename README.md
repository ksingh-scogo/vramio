# vram.io

Minimal API to estimate VRAM requirements for HuggingFace models.

## Usage

```
GET /model?hf_id=microsoft/phi-2
```

```json
{
  "model": "microsoft/phi-2",
  "total_parameters": "2.78B",
  "native_dtype": "F16",
  "native_memory": "5.18 GB",
  "memory_requirements": {
    "full_precision_fp32": "10.36 GB",
    "half_precision_fp16_bf16": "5.18 GB",
    "quantized_int8": "2.59 GB",
    "quantized_int4": "1.29 GB"
  }
}
```

## Run Locally

```bash
pip install httpx[http2]
python server_embedded.py
```

## Deploy to Render (Free)

1. Push to GitHub
2. Connect repo at [render.com/new](https://render.com/new)
3. Select **Web Service** → **Python** runtime
4. Set start command: `python server_embedded.py`

Or use the included `render.yaml` for one-click deploy.

## Credits

Built on the memory estimation logic from [hf-mem](https://github.com/alvarobartt/hf-mem) by [@alvarobartt](https://github.com/alvarobartt).

## License

MIT
