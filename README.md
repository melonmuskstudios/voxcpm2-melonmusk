# audico-cloud-models

One folder per text-to-speech model. Every model answers the same three web addresses, so Audico needs no model-specific code.

```
common/        the 3 endpoints, token check, tunnel start     (shared, written once)
models/<id>/   wrapper.py (load + speak), model.json, requirements.txt, README
```

## The contract (version 1)

All calls need the header `X-Audico-Token: <token>` except `/audico_health`.

### GET /audico_health
```json
{"ok": true, "contract": 1, "loaded": true, "loading": false, "gpu": [{"index":0,"used_mb":8100,"total_mb":15360}], "note": "ready in 190s on 2 GPUs"}
```
`loaded=false, loading=true` means "wait and ask again".

### GET /audico_info
```json
{
  "id": "voxcpm2", "name": "VoxCPM2", "version": "2", "contract": 1, "sample_rate": 48000,
  "languages": ["English", "Spanish", "..."],
  "supports_voice_clone": true, "needs_reference_text": false,
  "max_chars_per_call": 600, "max_parallel": 2,
  "settings": [
    {"key": "cfg", "label": "Guidance scale (CFG)", "type": "slider", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0, "help": "..."},
    {"key": "use_ref_text", "label": "Match the sample closely", "type": "toggle", "default": true, "help": "..."},
    {"key": "style", "label": "Voice style", "type": "text", "default": "", "help": "..."},
    {"key": "quality", "label": "Quality", "type": "select", "options": ["fast", "best"], "default": "fast"}
  ]
}
```
`type` is `slider | toggle | select | text`. `max_parallel` = how many `/audico_speak` calls Audico may send at the same time (one per GPU).

### POST /audico_speak   (multipart/form-data)
| field | meaning |
|---|---|
| `text` | the part to speak (up to `max_chars_per_call`) |
| `settings_json` | `{"cfg":2.0,"steps":10,...}` keys from `/audico_info` (missing keys use the model default) |
| `seed` | integer, or `-1` for random |
| `ref_text` | words spoken in the voice sample (optional) |
| `ref_audio` | the voice sample file. Send it **once**, then send only `ref_hash` |
| `ref_hash` | sha1 (hex) of the sample file bytes; the server keeps samples by this hash |

Answer: the WAV file as the body. Headers: `X-Audico-Seconds` (length of the audio), `X-Audico-Seed` (seed used), `X-Audico-Took` (seconds of work).
Errors: `401` bad token, `503 {"error":"loading"}` model not ready (retry), `409 {"error":"need_ref"}` unknown `ref_hash` (send `ref_audio` again), `400/500` with a `detail` text.

## Add a model
Copy `models/voxcpm2`, change `wrapper.py` (`load(work_dir)` returns an `Adapter` with `info()`, `health()`, `speak()`), add `model.json` and `requirements.txt`. Run it with `python -m common.run <folder>`.
