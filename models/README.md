# Model weights

Model binaries are **not** tracked in git (see `.gitignore`). Fetch them here.

## Layout detector (required from Stage 2)

**PP-DocLayoutV3** — ONNX, Apache-2.0, 131 MB.

```bash
curl -L -o models/layout/pp-doclayoutv3.onnx \
  https://huggingface.co/alex-dinh/PP-DocLayoutV3-ONNX/resolve/main/PP-DocLayoutV3.onnx
```

The path is configured in the layout detector profile YAML, not hardcoded — see
`src/processing/strategies/config/layout_profiles/`.

### Why V3 and not the planned PP-DocLayout-S

`plan.md` specified **PP-DocLayout-S** (14.5 ms/page). PaddlePaddle publishes it
only in native Paddle inference format — using it would mean either installing
`paddlex`/`paddle2onnx` (the dependency surface the plan explicitly set out to
avoid) or converting it ourselves. **PP-DocLayoutV3 is already exported to ONNX
under Apache-2.0**, so it needs one `curl` and one wheel, which is what the plan
actually wanted.

The cost is speed: measured **~600 ms/page on CPU**, not the 15 ms budgeted —
about 5 s across the 8-page benchmark. That is still ~28x cheaper than the
~17 s/page VLM grounding it replaces, so the 85 s grounding bucket becomes ~5 s
and the Stage 2 target holds. Revisit only if the pre-pass budget gets tight.

**Verified against the parity baseline** (`tests/test_onnx_layout.py`): all 5
known-good figure boxes at IoU 0.90–0.95 (gate is 0.70), correct figure count
per page, and both page-4 tables detected.

> **Do not substitute DocLayout-YOLO.** It is AGPL-3.0 (built on
> Ultralytics/YOLOv10) and is incompatible with this project's
> permissive-licence rule. See `CLAUDE.md`.

## VLM weights (GGUF)

VLM weights are **not tracked in git** (tens of gigabytes) and are loaded by
llama-server via the `config/vlm_profiles/model.ini` presets, not by this
codebase directly.

**Where to put them.** The released `model.ini` names each file by a path
relative to the repository root:

```
models/allenAI/olmOCR-2-7B-1025.Q5_K_M.gguf
models/allenAI/olmOCR-2-7B-1025.mmproj-f16.gguf
models/GLM/GLM-OCR-f16.gguf
models/GLM/GLM-OCR-mmproj-GLM-OCR-Q8_0.gguf
models/Deepseek/deepseek-ocr-q8_0.gguf
models/Deepseek/mmproj-deepseek-ocr-q8_0.gguf
```

So download each pair into `models/<vendor>/` beneath this repository and the
configuration needs no editing. `python run_llama_server.py --check` lists any
file it cannot find, before starting anything.

Keeping the weights elsewhere — a shared store used by several projects, which
is how the development machine is set up — works too: point the `model` and
`mmproj` keys at absolute paths in `model.ini`. Strategy Lab's **Model Presets**
tab edits that file directly.

| Model | Role | Licence |
|---|---|---|
| GLM-OCR | `extraction` | MIT (weights), Apache-2.0 (code) |
| olmOCR-2-7B | title page, scanned fallback | Apache-2.0 |
| DeepSeek-OCR-2 | `layout_detector` fallback | Apache-2.0 |
