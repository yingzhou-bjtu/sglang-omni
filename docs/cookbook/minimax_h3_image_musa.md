# MiniMax-H3-Image on Moore Threads MUSA

This recipe runs MiniMax-H3-Image text-to-image inference on one MTT S5000.

## Prerequisites

- A MUSA 5.2 environment with `torch_musa` and SGLang-Omni installed.
- A local MiniMax-H3-Image checkpoint.
- The staged SGLang MUSA compatibility changes:
  - [Tolerate Triton versions without TensorDescriptor](https://github.com/yingzhou-bjtu/sglang/pull/4)
  - [Add torch fallback for fused activation operators](https://github.com/yingzhou-bjtu/sglang/pull/5)

The optional [MATE FlashAttention 3 adapter](https://github.com/yingzhou-bjtu/sglang/pull/6) is not required by the H3 text-to-image path.

## Start the server

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1

sglang serve \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 30000 \
  --num-gpus 1 \
  --attention-backend fa \
  --warmup-mode off \
  --dit-cpu-offload \
  --vae-cpu-offload \
  --text-encoder-cpu-offload \
  --dit-offload-prefetch-size 16
```

The CPU offload options are required on a 96 GB device because the model components do not fit resident at the same time. `TORCHDYNAMO_DISABLE=1` keeps this example on the functional eager path.

## Generate an image

```bash
curl -sS -X POST http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON' -o response.json
{
  "prompt": "A sunlit still life of a cobalt-blue ceramic vase holding magenta peonies on a warm oak table",
  "size": "2048x2048",
  "seed": 42,
  "generator_device": "cpu",
  "response_format": "b64_json",
  "output_format": "png"
}
JSON
```

Decode the returned image:

```bash
python - <<'PY'
import base64
import json

with open("response.json", encoding="utf-8") as response_file:
    response = json.load(response_file)

with open("output.png", "wb") as output_file:
    output_file.write(base64.b64decode(response["data"][0]["b64_json"]))
PY
```

`generator_device: "cpu"` fixes the initial-noise generator so the same seed produces the same initial latent across machines. It does not change the inference device.

## Validate

1. `GET /health` returns HTTP 200.
2. `/v1/images/generations` returns one base64 PNG.
3. Logs contain `Pipeline instantiated` and `Using FlashAttention (FA3) backend`.
4. Logs do not contain `Failed to build JIT module`, `Pipeline class 'MiniMaxH3ImagePipeline' not found`, or `Segfault`.

## Known limitations

- Use TP1 for reproducibility checks. TP4 can run but does not produce reproducible images with the same seed.
- Do not enable `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` during component loading.
- The fused activation path uses the torch fallback on MUSA. Expert-filtered MoE activation variants are not covered by this recipe.
- Image editing through `/v1/images/edits` is not covered by this recipe.

This recipe uses the eager functional path and is not a throughput or compiled-serving benchmark.
