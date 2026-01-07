#!/usr/bin/env python3
"""Ultra-minimal API for HuggingFace model memory estimation.

Zero framework overhead - uses only stdlib http.server + httpx.
Embeds hf-mem logic directly to avoid subprocess overhead.
"""

import json
import os
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import httpx

PORT = int(os.environ.get("PORT", 8080))
HF_TOKEN = os.environ.get("HF_TOKEN")

# Bytes per dtype for memory calculation
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4,
    "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def get_safetensor_metadata(client: httpx.Client, url: str) -> dict | None:
    """Fetch safetensors header metadata via range request."""
    try:
        # First get the header size (first 8 bytes)
        resp = client.get(url, headers={"Range": "bytes=0-7"})
        if resp.status_code not in (200, 206):
            return None
        header_size = struct.unpack("<Q", resp.content)[0]

        # Now fetch the actual header
        resp = client.get(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
        if resp.status_code not in (200, 206):
            return None
        return json.loads(resp.content)
    except Exception:
        return None


def estimate_memory(model_id: str, revision: str = "main") -> dict:
    """Estimate VRAM requirements for a HuggingFace model."""
    # Validate model_id format
    if not model_id or "/" not in model_id:
        return {"error": "Invalid model ID. Expected format: owner/model-name"}

    # Sanitize model_id (basic protection)
    model_id = model_id.strip()
    if any(c in model_id for c in ["<", ">", '"', "'", ";", "&", "|"]):
        return {"error": "Invalid characters in model ID"}

    headers = {"Accept": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        with httpx.Client(http2=True, headers=headers, timeout=30, follow_redirects=True) as client:
            # Get file listing
            api_url = f"https://huggingface.co/api/models/{model_id}/tree/{revision}"
            resp = client.get(api_url)
            if resp.status_code != 200:
                return {"error": f"Model not found or inaccessible: {model_id}"}

            files = resp.json()
            safetensor_files = [f for f in files if f["path"].endswith(".safetensors")]

            if not safetensor_files:
                return {"error": "No safetensors files found in model"}

            # Check for index file (sharded models)
            index_files = [f for f in files if f["path"].endswith("model.safetensors.index.json")]
            if index_files:
                idx_url = f"https://huggingface.co/{model_id}/resolve/{revision}/{index_files[0]['path']}"
                idx_resp = client.get(idx_url)
                if idx_resp.status_code == 200:
                    idx_data = idx_resp.json()
                    shard_files = list(set(idx_data.get("weight_map", {}).values()))
                    safetensor_files = [f for f in files if f["path"] in shard_files]

            # Calculate total memory from all safetensor files
            total_params = 0
            total_bytes = 0
            dtype_counts = {}

            for sf in safetensor_files:
                url = f"https://huggingface.co/{model_id}/resolve/{revision}/{sf['path']}"
                meta = get_safetensor_metadata(client, url)
                if not meta:
                    continue

                for key, tensor_info in meta.items():
                    if key == "__metadata__":
                        continue
                    dtype = tensor_info.get("dtype", "F32")
                    shape = tensor_info.get("shape", [])
                    params = 1
                    for dim in shape:
                        params *= dim
                    total_params += params
                    byte_size = DTYPE_SIZES.get(dtype, 4)
                    total_bytes += params * byte_size
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params

            if total_params == 0:
                return {"error": "Could not parse model metadata"}

            def fmt_size(bytes_val: float) -> str:
                if bytes_val >= 1024**3:
                    return f"{bytes_val / (1024**3):.2f} GB"
                return f"{bytes_val / (1024**2):.2f} MB"

            def fmt_params(p: int) -> str:
                if p >= 1e9:
                    return f"{p / 1e9:.2f}B"
                return f"{p / 1e6:.2f}M"

            return {
                "model": model_id,
                "total_parameters": fmt_params(total_params),
                "memory_required": fmt_size(total_bytes),
                "current_dtype": max(dtype_counts, key=dtype_counts.get),
                "recommended_vram": fmt_size(total_bytes * 1.2),
                "other_precisions": {
                    "fp32": fmt_size(total_params * 4),
                    "fp16": fmt_size(total_params * 2),
                    "int8": fmt_size(total_params * 1),
                    "int4": fmt_size(total_params * 0.5),
                },
                "overhead_note": "Includes 20% for activations/KV cache (2K context)",
            }
    except httpx.TimeoutException:
        return {"error": "Request to HuggingFace timed out. Try again."}
    except Exception as e:
        return {"error": f"Failed to fetch model info: {str(e)}"}


class Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler with error resilience."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, data: dict, status: int = 200):
        try:
            body = json.dumps(data, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass  # Client disconnected, ignore
        except ConnectionResetError:
            pass  # Client reset connection, ignore

    def do_GET(self):
        try:
            path = urlparse(self.path)
            params = parse_qs(path.query)
            hf_id = params.get("hf_id", [None])[0]

            if path.path in ("/", "/model") and hf_id:
                result = estimate_memory(hf_id)
                status = 500 if "error" in result else 200
                return self._json(result, status)

            self._json({"error": "Usage: /model?hf_id=microsoft/phi-2"}, 400)
        except Exception:
            pass  # Catch-all to prevent server crash


if __name__ == "__main__":
    print(f"vram.io | http://localhost:{PORT}/model?hf_id=<model>")
    HTTPServer(("", PORT), Handler).serve_forever()
