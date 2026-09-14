# SOP — Adding a New Model

Standard procedure for downloading a new model, characterizing its performance, and making it
available through `llama-swap` and OpenCode. Follows the conventions established in `PLAN.md` and
`record.md`; see those for the full worked history (Gemma 4, Qwen3.6/3.8, Qwen3-Coder, Granite 4.2).

## 0. Prerequisites

- Both llama.cpp backends built: `~/llama.cpp/build-vulkan/bin/` and `~/llama.cpp/build-hip/bin/`.
- `uvx` available (for ephemeral `hf` CLI runs).
- `~/.config/llama-swap/config.yaml` and `~/.config/opencode/opencode.json` exist.

## 1. Find the GGUF source

Preference order:
1. `unsloth/<model>-GGUF`, quant `UD-Q6_K_XL` (Unsloth Dynamic mixed-precision, this project's
   default quality/size tradeoff). Check the repo's exact file list first — unsloth repos often
   ship both a plain `Q6_K.gguf` and the `UD-*` dynamic quant; use the exact filename, never a
   wildcard like `--include "*Q6_K*"` (it pulls every matching file and wastes bandwidth/disk).
2. If no unsloth repo exists for the model, fall back to the model publisher's own official GGUF
   repo (e.g. `ibm-granite/granite-4.2-3b-GGUF`), using the plain `Q6_K.gguf` — closest available
   match to the `UD-Q6_K_XL` quality level.
3. If neither exists, a well-regarded third-party quantizer (e.g. `bartowski`) is an acceptable
   fallback — ask the user to confirm the choice before downloading, since this is a deviation from
   the established source.

Check what exists via the HF API before downloading anything:
```bash
curl -s "https://huggingface.co/api/models/<org>/<model>-GGUF" | python3 -c "
import json,sys
d=json.load(sys.stdin)
for s in d.get('siblings', []):
    print(s['rfilename'])
"
```
If unsure whether an unsloth repo exists at all, search:
```bash
curl -s "https://huggingface.co/api/models?search=<model-name>%20GGUF" | python3 -c "
import json,sys
for m in json.load(sys.stdin): print(m['id'])
"
```

**If the quant source deviates from the `unsloth` UD-Q6_K_XL default, ask the user to confirm before
downloading** (via `AskUserQuestion` or equivalent) — don't silently substitute.

## 2. Download

```bash
uvx --from "huggingface_hub" hf download <repo> \
  --include "<exact-filename>.gguf" --local-dir ~/models/<model-short-name>
```
- `~/models/<model-short-name>` — short, lowercase, hyphenated, matching the pattern already in use
  (`gemma-4-31b`, `qwen3-coder-30b-a3b`, `granite-4.2-3b`, ...).
- `hf download` verifies checksums and resumes automatically. Sanity-check the resulting file size
  against the repo's listed size, and confirm only one `.gguf` landed in the directory.
- Large downloads (>~1GB) will exceed a foreground command's default timeout — run in the
  background and wait for the completion notification rather than polling.

## 3. Read GGUF metadata

Get the model's native context length (and sanity-check param count) before benchmarking:
```bash
python3 ~/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py ~/models/<model>/<file>.gguf 2>&1 \
  | grep -i "context_length\|block_count"
```
Note the `<arch>.context_length` value — this becomes the `-c` flag in step 5, following the
project's "serve at full native context" convention (see the 2026-09-09 context-window sweep in
`record.md`: this hardware has enough GTT/RAM headroom that native context, not system resources,
is the binding constraint for every model tried so far).

## 4. Benchmark both backends

```bash
~/llama.cpp/build-vulkan/bin/llama-bench -m ~/models/<model>/<file>.gguf -ngl 999 -fa 1
~/llama.cpp/build-hip/bin/llama-bench -m ~/models/<model>/<file>.gguf -ngl 999 -fa 1
```
Compare `pp512` (prefill) and `tg128` (generation) throughput. **Do not assume a pattern from other
models** — this hardware has shown dense models mostly favor HIP on prefill (generation usually
tied), while MoE and, empirically, small (<~10GiB) dense models tend to favor Vulkan — but every
model in `record.md` was benchmarked individually because the pattern isn't universal. Pick
whichever backend wins generation throughput when the two disagree (interactive/agentic use cares
more about generation speed than prefill).

Optionally verify GPU (not CPU) is doing the work: run `amdgpu_top` in a second terminal during the
benchmark — GPU busy% should be near 100.

## 5. Decide KV cache quantization (optional)

Only needed if the model's KV cache at full context is large relative to available headroom (in
practice: only used so far for the two largest-context MoE/hybrid models, `qwen3.6-35b-a3b` and
`qwen3.8-27b`, via `--cache-type-k q8_0 --cache-type-v q8_0`). Not needed for smaller models or
models with native context ≤131072 given the sweep's measured headroom — skip by default, add only
if you hit a measured memory constraint.

## 6. Add to `llama-swap` config

Append a new entry to `~/.config/llama-swap/config.yaml`, matching the existing block style exactly
(absolute paths, `--port ${PORT} -ngl 999 -fa 1 -c <native-context>`):
```yaml
  <model-key>:
    cmd: >
      /home/acraig78/llama.cpp/build-<vulkan|hip>/bin/llama-server
      --model /home/acraig78/models/<model>/<file>.gguf
      --port ${PORT} -ngl 999 -fa 1 -c <native-context>
```
`<model-key>` should be short and match the `~/models/` directory naming.

Validate the YAML before moving on:
```bash
python3 -c "import yaml; d=yaml.safe_load(open('/home/acraig78/.config/llama-swap/config.yaml')); print(list(d['models'].keys()))"
```

## 7. Add to OpenCode

Append the corresponding entry to `~/.config/opencode/opencode.json`'s
`provider["llama.cpp"].models`:
```json
"<model-key>": {
  "name": "<Human-readable name> (local)",
  "limit": { "context": <native-context>, "output": 8192 }
}
```
`<model-key>` must be *character-for-character identical* to the `llama-swap` config key — a
mismatch gets a silent 404 from llama-swap instead of a response.

**Skip variant/utility configs** (e.g. a `-minimal` reasoning-off variant, a `-judge` low-effort
variant) unless the user asks for them explicitly — established preference is that these stay
available directly via the llama-swap API but are not cluttering the OpenCode model picker.

Validate:
```bash
python3 -c "import json; json.load(open('/home/acraig78/.config/opencode/opencode.json'))"
```

## 8. Cross-check and restart

Confirm the `llama-swap` config keys and OpenCode config keys that should match actually do:
```bash
curl -s localhost:8080/v1/models | python3 -c "import json,sys; print(sorted(m['id'] for m in json.load(sys.stdin)['data']))"
python3 -c "import json; print(sorted(json.load(open('/home/acraig78/.config/opencode/opencode.json'))['provider']['llama.cpp']['models'].keys()))"
```

- **`llama-swap` restart**: only needed if `config.yaml` changed *after* the running instance last
  read it. If you edited `config.yaml` and the service was already running, restart it:
  ```bash
  systemctl --user restart llama-swap
  ```
- **OpenCode restart**: `opencode.json` is read by the OpenCode client at session start, not by
  `llama-swap`. Editing it never requires restarting `llama-swap` — but any *already-running*
  OpenCode session needs to be restarted (or a new session started) to pick up the change.

Smoke-test the new model end-to-end:
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model-key>","messages":[{"role":"user","content":"say hi"}]}'
```

## 9. Document

Update `PLAN.md` and `record.md` together (both are kept in sync as a pair, not just one):
- `PLAN.md`: models table, architecture diagram (add a backend node + wire it to the right GPU
  backend), Phase 5 download commands, Phase 5 backend-pick summary sentence, Phase 6 `config.yaml`
  code block + context note, Phase 7 `opencode.json` code block + `limit.context` note, "Expected
  results" section.
- `record.md`: per-model benchmark table (`### <Model Name>`), cross-model comparison table row,
  backend-selection summary table row, and a sentence in the pattern-analysis prose if the new
  model's backend result confirms or breaks the established pattern (worth calling out either way —
  breaking a stated pattern is exactly the kind of thing a future reader needs flagged).
