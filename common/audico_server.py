"""The three Audico endpoints, the same for every model. A model folder only supplies an Adapter (see models/voxcpm2/wrapper.py).

  GET  /audico_health   open: {"ok": true, "loading": bool, ...}          (no secret needed, tells nothing private)
  GET  /audico_info     token: what the model is and which settings it has  (Audico builds its settings panel from this)
  POST /audico_speak    token: multipart form -> a WAV file; seconds/seed in the response headers

The token goes in the header  X-Audico-Token  (or ?token=...).
"""
import hashlib
import hmac
import json
import os
import random
import shutil
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

CONTRACT_VERSION = 1


class Adapter:
    """What a model wrapper must provide."""
    def info(self) -> dict: raise NotImplementedError          # see README for the fields
    def health(self) -> dict: raise NotImplementedError        # {"loaded": bool, "loading": bool, "gpu": [...], "note": str}
    def speak(self, text, ref_path, ref_text, settings, seed) -> tuple:   # -> (wav_path, seconds_of_audio)
        raise NotImplementedError


def build_app(adapter: Adapter, token: str, work_dir: Path) -> FastAPI:
    app = FastAPI(title="Audico cloud model", docs_url=None, redoc_url=None, openapi_url=None)
    refs = work_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)

    def auth(x_audico_token: str = Header(default=""), token_q: str = Query(default="", alias="token")):
        given = x_audico_token or token_q
        if not token or not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="bad or missing token")

    @app.get("/audico_health")
    def health():
        h = adapter.health()
        return {"ok": True, "contract": CONTRACT_VERSION, "loaded": bool(h.get("loaded")), "loading": bool(h.get("loading")),
                "gpu": h.get("gpu", []), "note": h.get("note", "")}

    @app.get("/audico_info", dependencies=[Depends(auth)])
    def info():
        return dict(adapter.info(), contract=CONTRACT_VERSION)

    @app.post("/audico_speak", dependencies=[Depends(auth)])
    def speak(text: str = Form(...), settings_json: str = Form("{}"), seed: int = Form(-1), ref_text: str = Form(""),
              ref_hash: str = Form(""), ref_audio: UploadFile = File(None)):
        h = adapter.health()
        if not h.get("loaded"):
            return JSONResponse({"error": "loading", "message": h.get("note") or "The model is still loading."}, status_code=503)
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty text")
        try:
            settings = json.loads(settings_json or "{}")
            if not isinstance(settings, dict):
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="settings_json must be a JSON object")
        # the voice sample: sent once, then reused by its hash (a few MB per call would be wasteful)
        ref_path = None
        if ref_audio is not None:
            data = ref_audio.file.read()
            if data:
                ref_hash = hashlib.sha1(data).hexdigest()
                ref_path = refs / (ref_hash + ".wav")
                if not ref_path.exists():
                    ref_path.write_bytes(data)
        elif ref_hash:
            ref_path = refs / (ref_hash + ".wav") if ref_hash.isalnum() else None
            if ref_path is None or not ref_path.exists():
                return JSONResponse({"error": "need_ref", "message": "Send the voice sample once (ref_audio)."}, status_code=409)
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)
        t0 = time.time()
        try:
            wav, seconds = adapter.speak(text, str(ref_path) if ref_path else None, ref_text.strip(), settings, int(seed))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        return FileResponse(wav, media_type="audio/wav", headers={
            "X-Audico-Seconds": f"{seconds:.3f}", "X-Audico-Seed": str(seed), "X-Audico-Took": f"{time.time() - t0:.2f}",
            "X-Audico-Ref-Hash": ref_hash or ""})

    return app
